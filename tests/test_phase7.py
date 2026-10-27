"""Unit and integration tests for Phase 7 (Evaluation, Decoding & Checkpoint Averaging).

Tests:
1. WER and CER calculation on hand-computed examples and edge cases.
2. Greedy CTC decoding on hand-crafted synthetic posteriors (repeats, blanks between repeats, edge blanks).
3. CTC word timestamp calculation: frame-to-time conversion and SentencePiece subword boundary grouping.
4. Checkpoint weight averaging: parameter arithmetic mean and buffer preservation.
5. End-to-end evaluation runner producing markdown reports and error analysis.
"""

import json

import numpy as np
import pytest
import soundfile as sf
import torch

from awaaz.average import average_checkpoints
from awaaz.decode import ctc_greedy_decode, extract_word_timestamps
from awaaz.evaluate import evaluate_manifest
from awaaz.metrics import compute_wer_cer, format_error_report
from awaaz.model import ConformerConfig, ConformerCTC
from awaaz.text import Tokenizer, train_tokenizer


@pytest.fixture(scope="module")
def trained_toy_tokenizer(tmp_path_factory):
    """Generates a small bilingual tokenizer for decoding and timestamp tests."""
    root = tmp_path_factory.mktemp("phase7_tok")
    corpus = root / "corpus.txt"
    with open(corpus, "w", encoding="utf-8") as f:
        for _ in range(30):
            f.write("hello world testing awaaz\n")
            f.write("ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਦੋਸਤੋ\n")

    prefix = str(root / "tok")
    train_tokenizer(output_prefix=prefix, pa_corpus=[str(corpus)], vocab_size=64, character_coverage=1.0)
    tok = Tokenizer(f"{prefix}.model")
    return {"tokenizer": tok, "model_path": f"{prefix}.model"}


def test_metrics_calculation_hand_computed():
    """Verifies WER and CER on exact mathematical examples and edge cases."""
    # 1. Exact match
    res = compute_wer_cer(["hello world"], ["hello world"])
    assert res["wer"] == 0.0
    assert res["cer"] == 0.0

    # 2. 1 substitution out of 2 words -> WER = 0.5
    res = compute_wer_cer(["hello world"], ["hello earth"])
    assert res["wer"] == 0.5

    # 3. 1 deletion out of 2 words -> WER = 0.5
    res = compute_wer_cer(["hello world"], ["hello"])
    assert res["wer"] == 0.5

    # 4. Safe handling of empty references
    res_empty_both = compute_wer_cer([""], [""])
    assert res_empty_both["wer"] == 0.0
    assert res_empty_both["cer"] == 0.0

    res_empty_ref = compute_wer_cer([""], ["unexpected words"])
    assert res_empty_ref["wer"] == 1.0
    assert res_empty_ref["cer"] == 1.0


def test_format_error_report():
    """Verifies error analysis report ranks utterances by descending WER."""
    refs = ["sat sri akaal", "hello world", "perfect transcript"]
    hyps = ["sat sri", "completely wrong sentence emitted", "perfect transcript"]
    langs = ["pa", "en", "en"]

    errors = format_error_report(refs, hyps, languages=langs, top_k=2)
    assert len(errors) == 2
    assert errors[0]["wer"] >= errors[1]["wer"]
    assert errors[-1]["reference"] != "perfect transcript"


def test_ctc_greedy_decode_handcrafted_posteriors():
    """Verifies greedy decoding collapse rules on synthetic frame posteriors."""
    vocab_size = 10

    # Test sequence of token IDs:
    # [0, 0, 3, 3, 3, 0, 3, 0, 4, 4, 0, 5, 0]
    # Expected collapsed IDs: [3, 3, 4, 5]
    seq = [0, 0, 3, 3, 3, 0, 3, 0, 4, 4, 0, 5, 0]
    log_probs = torch.zeros(1, len(seq), vocab_size)
    for t, tok in enumerate(seq):
        log_probs[0, t, tok] = 10.0

    # Decode without tokenizer (returns ID lists)
    decoded_ids = ctc_greedy_decode(log_probs)
    assert decoded_ids == [[3, 3, 4, 5]]


def test_word_timestamp_extraction_exact_frames(trained_toy_tokenizer):
    """Verifies exact frame-to-time conversion and subword piece concatenation."""
    tok = trained_toy_tokenizer["tokenizer"]

    # Choose two distinct words from the vocabulary
    w1_ids = tok.encode("hello")
    w2_ids = tok.encode("world")
    assert len(w1_ids) > 0 and len(w2_ids) > 0

    # Frame duration: (10ms / 1000) * 4 = 0.040s
    hop_ms = 10.0
    subsample_factor = 4
    frame_dur = (hop_ms / 1000.0) * subsample_factor

    # Construct frame sequence:
    total_frames = 35
    log_probs = torch.zeros(total_frames, tok.vocab_size)

    # Emit w1 tokens starting at frame 5 (inserting blank between consecutive identical tokens)
    cur_frame = 5
    w1_start = cur_frame
    for idx, t_id in enumerate(w1_ids):
        if idx > 0 and t_id == w1_ids[idx - 1]:
            cur_frame += 1  # Leave intermediate frame as blank (0)
        log_probs[cur_frame, t_id] = 10.0
        cur_frame += 1
    w1_last_frame = cur_frame - 1

    # Emit w2 tokens starting at frame 20
    cur_frame = 20
    w2_start = cur_frame
    for idx, t_id in enumerate(w2_ids):
        if idx > 0 and t_id == w2_ids[idx - 1]:
            cur_frame += 1
        log_probs[cur_frame, t_id] = 10.0
        cur_frame += 1
    w2_last_frame = cur_frame - 1

    words = extract_word_timestamps(
        log_probs,
        tokenizer=tok,
        subsample_factor=subsample_factor,
        hop_ms=hop_ms,
    )

    assert len(words) == 2
    assert words[0]["word"] == "hello"
    assert words[0]["start"] == pytest.approx(w1_start * frame_dur, abs=1e-3)
    assert words[0]["end"] == pytest.approx((w1_last_frame + 1) * frame_dur, abs=1e-3)

    assert words[1]["word"] == "world"
    assert words[1]["start"] == pytest.approx(w2_start * frame_dur, abs=1e-3)
    assert words[1]["end"] == pytest.approx((w2_last_frame + 1) * frame_dur, abs=1e-3)


def test_checkpoint_weight_averaging(tmp_path):
    """Verifies that average_checkpoints computes the exact arithmetic mean of weights."""
    ckpt1_p = tmp_path / "step_0001000.pt"
    ckpt2_p = tmp_path / "step_0002000.pt"
    out_ckpt = tmp_path / "averaged.pt"

    w1 = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    w2 = torch.tensor([[4.0, 8.0], [12.0, 16.0]])

    torch.save({"model": {"weight": w1, "step_count": torch.tensor(1000)}}, ckpt1_p)
    torch.save({"model": {"weight": w2, "step_count": torch.tensor(2000)}}, ckpt2_p)

    average_checkpoints([ckpt1_p, ckpt2_p], out_ckpt)
    assert out_ckpt.is_file()

    avg_data = torch.load(out_ckpt, map_location="cpu")
    avg_weight = avg_data["model"]["weight"]

    # Expected: (w1 + w2) / 2 = [[3.0, 6.0], [9.0, 12.0]]
    expected_weight = torch.tensor([[3.0, 6.0], [9.0, 12.0]])
    assert torch.allclose(avg_weight, expected_weight)
    assert avg_data["num_checkpoints"] == 2


def test_evaluate_manifest_end_to_end(tmp_path, trained_toy_tokenizer):
    """Verifies evaluate_manifest executes offline & streaming modes and writes reports."""
    tok = trained_toy_tokenizer["tokenizer"]
    tok_path = trained_toy_tokenizer["model_path"]

    # 1. Create a tiny synthetic checkpoint
    cfg = ConformerConfig(
        vocab_size=tok.vocab_size,
        n_mels=80,
        d_model=64,
        num_blocks=2,
        num_heads=2,
        ffn_dim=128,
        conv_kernel_size=15,
        dropout=0.0,
    )
    model = ConformerCTC(cfg)
    ckpt_path = tmp_path / "toy_model.pt"
    torch.save({"model": model.state_dict(), "config": {"model": {"n_mels": 80, "d_model": 64, "num_blocks": 2, "num_heads": 2, "ffn_dim": 128, "conv_kernel_size": 15}}}, ckpt_path)

    # 2. Create synthetic audio fixture and manifest
    manifest_p = tmp_path / "test_manifest.jsonl"
    sr = 16000
    with open(manifest_p, "w", encoding="utf-8") as f:
        for i, text in enumerate(["hello world", "testing awaaz"]):
            audio_p = tmp_path / f"audio_{i}.flac"
            audio_sig = np.zeros(int(1.0 * sr), dtype=np.float32)
            sf.write(str(audio_p), audio_sig, sr)
            f.write(json.dumps({
                "audio_path": audio_p.as_posix(),
                "text": text,
                "duration": 1.0,
                "language": "en",
                "domain": "general",
            }) + "\n")

    # 3. Run evaluation
    out_dir = tmp_path / "eval_reports"
    results = evaluate_manifest(
        manifest_path=str(manifest_p),
        checkpoint_path=str(ckpt_path),
        tokenizer_path=tok_path,
        output_dir=str(out_dir),
        device="cpu",
    )

    # 4. Verify all modes were evaluated
    assert "offline" in results["modes"]
    assert "stream_0.64s" in results["modes"]
    assert "stream_0.32s" in results["modes"]

    # 5. Verify report files were created
    assert (out_dir / "results.md").is_file()
    assert (out_dir / "results.json").is_file()
    assert (out_dir / "error_analysis.txt").is_file()

    md_content = (out_dir / "results.md").read_text(encoding="utf-8")
    assert "Metric Summary Table" in md_content
    assert "Offline (Full Context)" in md_content
    assert "Stream 0.64s" in md_content
