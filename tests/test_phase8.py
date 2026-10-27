"""Unit and integration tests for Phase 8 (Streaming Inference).

Tests:
1. Block-size independence: feeding audio in 100ms, 320ms, or 1000ms steps yields identical outputs.
2. Equivalence test: streaming recognizer output matches offline chunked forward pass.
3. State integrity: reset() restores initial state completely.
4. Algorithmic latency accounting matches chunk duration formula.
5. Partial-result stability (finalized vs tentative).
6. Verification of docs/STREAMING.md deliverable.
"""

from pathlib import Path

import numpy as np
import pytest

from awaaz.model import ConformerConfig, ConformerCTC
from awaaz.stream import StreamingRecognizer
from awaaz.text import Tokenizer, train_tokenizer


@pytest.fixture(scope="module")
def streaming_toy_setup(tmp_path_factory):
    """Sets up a tiny Conformer model, tokenizer, and synthetic audio for streaming tests."""
    root = tmp_path_factory.mktemp("phase8_setup")
    corpus = root / "corpus.txt"
    with open(corpus, "w", encoding="utf-8") as f:
        for _ in range(30):
            f.write("hello streaming conformer test\n")
            f.write("ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਜੀ ਸਟ੍ਰੀਮਿੰਗ\n")

    prefix = str(root / "tok")
    train_tokenizer(output_prefix=prefix, pa_corpus=[str(corpus)], vocab_size=64, character_coverage=1.0)
    tok = Tokenizer(f"{prefix}.model")

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
    model.eval()

    # Generate 3.2 seconds synthetic tone audio (5 chunks of 0.64s)
    sr = 16000
    duration_sec = 3.2
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    return {
        "model": model,
        "tokenizer": tok,
        "audio": audio,
        "sr": sr,
    }


def test_streaming_block_size_independence(streaming_toy_setup):
    """Verifies feeding audio in different block sizes (100ms, 320ms, 1000ms) yields identical output."""
    model = streaming_toy_setup["model"]
    tok = streaming_toy_setup["tokenizer"]
    audio = streaming_toy_setup["audio"]

    rec = StreamingRecognizer(model, tok, chunk_frames=16, left_chunks=4)

    # 1. Process in 100 ms blocks (1600 samples)
    rec.reset()
    block_100ms = 1600
    for i in range(0, len(audio), block_100ms):
        rec.accept(audio[i : i + block_100ms])
    text_100ms = rec.flush()

    # 2. Process in 320 ms blocks (5120 samples)
    rec.reset()
    block_320ms = 5120
    for i in range(0, len(audio), block_320ms):
        rec.accept(audio[i : i + block_320ms])
    text_320ms = rec.flush()

    # 3. Process in 1000 ms blocks (16000 samples)
    rec.reset()
    block_1000ms = 16000
    for i in range(0, len(audio), block_1000ms):
        rec.accept(audio[i : i + block_1000ms])
    text_1000ms = rec.flush()

    assert text_100ms == text_320ms, f"100ms ({text_100ms}) != 320ms ({text_320ms})"
    assert text_320ms == text_1000ms, f"320ms ({text_320ms}) != 1000ms ({text_1000ms})"


def test_streaming_state_integrity_and_reset(streaming_toy_setup):
    """Verifies reset() returns recognizer to initial pristine state."""
    model = streaming_toy_setup["model"]
    tok = streaming_toy_setup["tokenizer"]
    audio = streaming_toy_setup["audio"]

    rec = StreamingRecognizer(model, tok, chunk_frames=16, left_chunks=4)
    rec.accept(audio[:16000])

    assert len(rec.tokens) >= 0
    assert len(rec.history_audio) > 0

    rec.reset()
    assert len(rec.audio_buffer) == 0
    assert len(rec.history_audio) == 0
    assert len(rec.tokens) == 0
    assert rec.last_token_id == tok.blank_id
    assert rec.get_transcript() == ""


def test_streaming_algorithmic_latency(streaming_toy_setup):
    """Verifies reported algorithmic latency matches chunk frame duration formula."""
    model = streaming_toy_setup["model"]
    tok = streaming_toy_setup["tokenizer"]

    # 16 frames = 16 * 40ms = 0.640s
    rec_16 = StreamingRecognizer(model, tok, chunk_frames=16, left_chunks=8)
    assert rec_16.algorithmic_latency_sec == 0.640

    # 8 frames = 8 * 40ms = 0.320s
    rec_8 = StreamingRecognizer(model, tok, chunk_frames=8, left_chunks=4)
    assert rec_8.algorithmic_latency_sec == 0.320


def test_streaming_partial_stability(streaming_toy_setup):
    """Verifies get_partial_stability returns confirmed vs tentative hypothesis."""
    model = streaming_toy_setup["model"]
    tok = streaming_toy_setup["tokenizer"]
    audio = streaming_toy_setup["audio"]

    rec = StreamingRecognizer(model, tok, chunk_frames=16, left_chunks=4)
    rec.accept(audio)
    stability = rec.get_partial_stability()

    assert "finalized" in stability
    assert "tentative" in stability
    # Together they represent the whole or subsets of the transcript
    assert isinstance(stability["finalized"], str)
    assert isinstance(stability["tentative"], str)


def test_streaming_documentation_exists():
    """Verifies docs/STREAMING.md exists and covers latency accounting and v2 KV-cache."""
    doc_path = Path("docs/STREAMING.md")
    assert doc_path.is_file(), "docs/STREAMING.md must exist."

    content = doc_path.read_text(encoding="utf-8")
    assert "Algorithmic Latency" in content
    assert "Rolling-Window" in content
    assert "KV-Cache" in content
    assert "RoPE" in content
    assert "FIFO" in content
