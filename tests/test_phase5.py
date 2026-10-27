"""Unit and integration tests for Phase 5 (Training Loop).

Tests:
1. Linear warmup + cosine decay learning rate scheduler.
2. Dynamic-chunk attention parameter sampling distribution.
3. Atomic checkpoint save, auto-resume, and rolling checkpoint window.
4. Greedy CTC argmax decoding with repeat-collapse and blank removal.
5. Fixture overfit test: miniature Conformer converges on synthetic audio.
6. Verification of docs/TRAINING.md documentation deliverable.
"""

import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from awaaz.dataset import AwaazDataset, DurationBucketedBatchSampler, ctc_collate_fn
from awaaz.features import LogMelSpectrogram
from awaaz.model import ConformerConfig, ConformerCTC
from awaaz.text import Tokenizer, train_tokenizer
from awaaz.train import (
    build_lr_scheduler,
    greedy_decode,
    load_checkpoint,
    sample_chunk_params,
    save_checkpoint,
)


@pytest.fixture(scope="module")
def synthetic_training_setup(tmp_path_factory):
    """Sets up a tiny synthetic dataset and trained tokenizer for rapid overfit testing."""
    root = tmp_path_factory.mktemp("phase5_setup")
    tok_dir = root / "tokenizer"
    tok_dir.mkdir()
    tok_prefix = str(tok_dir / "tok")

    # Train a minimal bilingual tokenizer
    corpus = tok_dir / "corpus.txt"
    with open(corpus, "w", encoding="utf-8") as f:
        for _ in range(50):
            f.write("sat sri akaal ji\nhello world testing conformer\n")
            f.write("ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਜੀ\nਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖਾਲਸਾ\n")

    train_tokenizer(
        output_prefix=tok_prefix,
        pa_corpus=[str(corpus)],
        vocab_size=64,
        character_coverage=1.0,
    )
    tokenizer = Tokenizer(f"{tok_prefix}.model")

    # Generate 4 small audio clips with repeating phrases
    phrases = [
        ("sat sri akaal ji", "pa"),
        ("hello world testing", "en"),
        ("sat sri akaal ji", "pa"),
        ("hello world testing", "en"),
    ]
    manifest_path = root / "train_manifest.jsonl"
    sr = 16000

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for i, (text, lang) in enumerate(phrases):
            audio_path = root / f"clip_{i}.flac"
            # 1.5 seconds tone
            t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False)
            freq = 220.0 + i * 80.0
            sig = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            sf.write(str(audio_path), sig, sr)

            mf.write(
                f'{{"audio_path": "{audio_path.as_posix()}", "text": "{text}", '
                f'"duration": 1.5, "language": "{lang}", "domain": "general"}}\n'
            )

    return {
        "root": root,
        "manifest": str(manifest_path),
        "tokenizer": tokenizer,
        "tok_model_path": f"{tok_prefix}.model",
    }


def test_lr_scheduler_schedule():
    """Verifies linear warmup to peak_lr and cosine decay to min_lr."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=100,
        total_steps=1000,
        peak_lr=1e-3,
        min_lr=1e-5,
    )

    # At step 0: LR is 0
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0, abs=1e-7)

    # Step to warmup midpoint
    for _ in range(50):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(5e-4, rel=1e-2)

    # Step to end of warmup
    for _ in range(50):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(1e-3, rel=1e-2)

    # Step to total_steps
    for _ in range(900):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(1e-5, rel=1e-2)


def test_sample_chunk_params_distribution():
    """Verifies chunk sampling produces both chunked and full-context samples."""
    random.seed(42)
    chunk_counts = 0
    full_counts = 0
    num_samples = 500

    for _ in range(num_samples):
        chunk_size, left = sample_chunk_params(chunk_prob=0.6, chunk_sizes=[8, 16, 32], left_chunks=[2, 4, -1])
        if chunk_size is not None:
            chunk_counts += 1
            assert chunk_size in [8, 16, 32]
            assert left in [2, 4, -1]
        else:
            full_counts += 1
            assert left == -1

    # Empirical probability near 0.6 / 0.4
    prob_chunked = chunk_counts / num_samples
    assert 0.50 <= prob_chunked <= 0.70
    assert full_counts > 0


def test_atomic_checkpoint_save_and_resume(tmp_path):
    """Verifies atomic last.pt saving, load_checkpoint restoration, and rolling window."""
    run_dir = tmp_path / "checkpoints"
    model = nn.Linear(8, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = None

    # Set model weights to 42.0 before saving
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(42.0)

    # Save initial checkpoint at step 10
    save_checkpoint(
        run_dir,
        model,
        optimizer,
        scaler,
        step=10,
        epoch=1,
        config={"test": True},
        is_periodic=False,
    )

    last_file = run_dir / "last.pt"
    assert last_file.is_file()
    assert not (run_dir / "last.pt.tmp").exists()  # Atomic temp cleaned up

    # Restore from last.pt into new_model (which starts with random weights)
    new_model = nn.Linear(8, 8)
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    resumed_step, resumed_epoch = load_checkpoint(last_file, new_model, new_opt)

    assert resumed_step == 10
    assert resumed_epoch == 1
    # Check that restored weights match 42.0
    for p in new_model.parameters():
        assert torch.all(p == 42.0)

    # Test rolling window periodic saves (keep newest 2)
    for s in [100, 200, 300, 400]:
        save_checkpoint(
            run_dir,
            model,
            optimizer,
            scaler,
            step=s,
            epoch=1,
            config={},
            is_periodic=True,
            max_keep=2,
        )

    saved_steps = sorted([p.name for p in run_dir.glob("step_*.pt")])
    assert saved_steps == ["step_0000300.pt", "step_0000400.pt"]


def test_greedy_decode(synthetic_training_setup):
    """Verifies greedy CTC decode collapses repeats and removes blank tokens."""
    tokenizer = synthetic_training_setup["tokenizer"]

    # Construct synthetic logits for batch of 1
    # Target tokens: [5, 5, 0, 5, 8] -> collapsed: [5, 5, 8]
    tok_ids = [5, 5, 0, 5, 8]
    log_probs = torch.zeros(1, len(tok_ids), tokenizer.vocab_size)
    for t_idx, tok in enumerate(tok_ids):
        log_probs[0, t_idx, tok] = 10.0  # high confidence

    out_lengths = torch.tensor([len(tok_ids)])
    hyps = greedy_decode(log_probs, out_lengths, tokenizer)
    assert len(hyps) == 1
    # Result must match decoding of [5, 5, 8]
    expected = tokenizer.decode([5, 5, 8])
    assert hyps[0] == expected


def test_fixture_overfit_convergence(synthetic_training_setup):
    """Mandatory Overfit Check: Train miniature Conformer on synthetic clips to verify loss drop."""
    tokenizer = synthetic_training_setup["tokenizer"]
    manifest = synthetic_training_setup["manifest"]

    dataset = AwaazDataset(
        manifest,
        tokenizer=tokenizer,
        exclude_sacred=True,
        training=False,
        speed_perturb=False,
    )
    sampler = DurationBucketedBatchSampler(dataset, max_batch_seconds=100.0, shuffle=False)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=ctc_collate_fn)

    # Miniature Conformer for ultra-fast CPU test (< 8 seconds)
    cfg = ConformerConfig(
        vocab_size=tokenizer.vocab_size,
        n_mels=80,
        d_model=64,
        num_heads=2,
        num_blocks=2,
        ffn_dim=128,
        conv_kernel_size=15,
        dropout=0.0,
    )
    model = ConformerCTC(cfg)
    frontend = LogMelSpectrogram(sample_rate=16000, n_mels=80)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    losses = []
    model.train()

    # Train for 75 steps
    step = 0
    while step < 75:
        for batch in loader:
            if step >= 75:
                break
            waveforms = batch["waveforms"]
            waveform_lens = batch["waveform_lengths"]
            targets = batch["targets"]
            target_lens = batch["target_lengths"]

            with torch.no_grad():
                features = frontend(waveforms)
                feat_lens = waveform_lens // 160 + 1

            optimizer.zero_grad()
            log_probs, out_lens = model(features, feat_lens, chunk_size=None)
            loss = ctc_loss(log_probs.transpose(0, 1), targets, out_lens, target_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(loss.item())
            step += 1

    initial_loss = np.mean(losses[:5])
    final_loss = np.mean(losses[-5:])

    # Loss must fall substantially (> 70% decrease)
    assert final_loss < initial_loss * 0.35, f"Loss did not fall sufficiently: {initial_loss:.3f} -> {final_loss:.3f}"

    # Verify model produces reasonable greedy predictions on the overfit batch
    model.eval()
    with torch.no_grad():
        log_probs, out_lens = model(features, feat_lens, chunk_size=None)
        hyps = greedy_decode(log_probs, out_lens, tokenizer)
        assert len(hyps) == len(batch["texts"])
        # At least one hypothesis matches the reference or has valid text tokens
        assert any(len(h) > 0 for h in hyps)


def test_training_documentation_exists():
    """Verifies that docs/TRAINING.md exists and contains the failure diagnosis table."""
    doc_path = Path("docs/TRAINING.md")
    assert doc_path.is_file(), "docs/TRAINING.md must exist."

    content = doc_path.read_text(encoding="utf-8")
    assert "Common Failure Symptom Diagnosis Table" in content
    assert "Loss stays flat from the start" in content
    assert "inf" in content and "NaN" in content
    assert "total_steps = budget_hours" in content
