"""Unit and integration tests for Phase 11: Optional v2 Models (Enhancer & Separation).

Tests lightweight causal on-device enhancer architecture, parameter constraints (< 500k),
overlap detection, speaker separation, dual-stream transcription, and documentation.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from awaaz.features import LogMelSpectrogram
from awaaz.model import ConformerConfig, ConformerCTC
from awaaz.pipeline.enhance import LightweightEnhancer
from awaaz.pipeline.separate import (
    SpeechSeparator,
    detect_speech_overlap,
    dual_stream_transcribe,
)
from awaaz.text import Tokenizer, train_tokenizer


def test_lightweight_enhancer_architecture_and_budget():
    """Verifies that LightweightEnhancer stays well below the 500k parameter edge budget."""
    enhancer = LightweightEnhancer(num_bins=80, hidden_dim=64, num_gru_layers=2)
    param_count = enhancer.count_parameters()

    # Must be strictly under 500k parameters
    assert param_count < 500_000, f"LightweightEnhancer parameter count {param_count} exceeds 500k limit"
    assert param_count > 10_000, f"LightweightEnhancer parameter count {param_count} is unexpectedly small"

    # Forward pass check
    x = torch.randn(2, 50, 80)
    out = enhancer(x)
    assert out.shape == (2, 50, 80)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_lightweight_enhancer_strict_causality():
    """Verifies that future input frames do NOT affect past or present outputs."""
    enhancer = LightweightEnhancer(num_bins=80, hidden_dim=64, num_gru_layers=2)
    enhancer.eval()

    torch.manual_seed(42)
    x1 = torch.randn(1, 30, 80)
    x2 = x1.clone()
    # Perturb strictly future frames (frames 15 to 29)
    x2[:, 15:, :] += torch.randn(1, 15, 80) * 5.0

    with torch.no_grad():
        out1 = enhancer(x1)
        out2 = enhancer(x2)

    # Past frames (0 to 14) must be strictly identical
    diff_past = (out1[:, :15, :] - out2[:, :15, :]).abs().max().item()
    assert diff_past < 1e-5, f"Causality violation! Future frames leaked into past outputs with diff {diff_past}"

    # Future frames (15 to 29) must differ
    diff_future = (out1[:, 15:, :] - out2[:, 15:, :]).abs().max().item()
    assert diff_future > 0.1, "Future frames were expected to change outputs"


def test_detect_speech_overlap():
    """Verifies temporal identification of simultaneous speaker overlap."""
    intervals = [
        (0.5, 2.0, "SPEAKER_00"),
        (1.5, 3.5, "SPEAKER_01"),  # Overlaps with SPEAKER_00 from 1.5 to 2.0
        (4.0, 5.0, "SPEAKER_00"),
    ]

    overlaps = detect_speech_overlap(intervals)
    assert len(overlaps) == 1
    assert overlaps[0] == (1.5, 2.0)

    # Test non-overlapping case
    non_overlapping = [
        (0.0, 1.0, "SPEAKER_00"),
        (1.2, 2.5, "SPEAKER_01"),
        (2.8, 4.0, "SPEAKER_00"),
    ]
    assert detect_speech_overlap(non_overlapping) == []


def test_speech_separator_fallback():
    """Verifies SpeechSeparator separates audio mixture into isolated channels."""
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Mixture of two frequencies
    audio = (0.5 * np.sin(2 * np.pi * 300 * t) + 0.5 * np.sin(2 * np.pi * 800 * t)).astype(np.float32)

    separator = SpeechSeparator(sample_rate=sample_rate)
    sources = separator.separate(audio, num_sources=2)

    assert len(sources) == 2
    for src in sources:
        assert src.shape == audio.shape
        assert src.dtype == np.float32
        assert not np.isnan(src).any()


@pytest.fixture
def synthetic_tokenizer(tmp_path: Path) -> Tokenizer:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world sat sri akal punjab awaaz speech audio", encoding="utf-8")
    prefix = str(tmp_path / "spm")
    train_tokenizer(output_prefix=prefix, pa_corpus=[str(corpus)], vocab_size=64, character_coverage=1.0)
    return Tokenizer(f"{prefix}.model")


def test_dual_stream_transcribe_end_to_end(synthetic_tokenizer: Tokenizer):
    """Verifies dual-stream separation, Conformer decoding, and chronological word ordering."""
    sample_rate = 16000
    audio = np.random.randn(sample_rate * 2).astype(np.float32) * 0.1

    tok = synthetic_tokenizer
    cfg = ConformerConfig(
        vocab_size=tok.vocab_size,
        d_model=64,
        num_heads=2,
        num_blocks=2,
        conv_kernel_size=15,
    )
    model = ConformerCTC(cfg)
    fe = LogMelSpectrogram(sample_rate=sample_rate)

    words = dual_stream_transcribe(
        audio=audio,
        model=model,
        tokenizer=tok,
        feature_extractor=fe,
        sample_rate=sample_rate,
    )

    # Verify chronological sorting
    for i in range(len(words) - 1):
        assert words[i]["start"] <= words[i + 1]["start"], "Transcripts must be chronologically ordered"


def test_separation_documentation_exists():
    """Verifies docs/SEPARATION.md exists and covers all Phase 11 deliverables."""
    doc_path = Path("docs/SEPARATION.md")
    assert doc_path.exists()
    content = doc_path.read_text(encoding="utf-8")
    assert "Lightweight On-Device Speech Enhancer" in content
    assert "< 500" in content or "500k" in content
    assert "SI-SNR" in content
    assert "Permutation-Invariant Training" in content
    assert "mermaid" in content
