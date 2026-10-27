"""Phase 4 Acceptance Tests — Conformer-CTC Model Architecture and Causality."""

import pytest
import torch

from awaaz.model import (
    ConformerConfig,
    ConformerCTC,
    build_chunk_attention_mask,
)


@pytest.fixture
def small_config():
    """Provides a small Conformer configuration for testing."""
    return ConformerConfig(
        n_mels=80,
        d_model=144,
        num_blocks=4,
        num_heads=4,
        ffn_dim=288,
        conv_kernel_size=15,
        dropout=0.0,
        subsample_factor=4,
        vocab_size=1024,
        causal_conv=True,
    )


def test_model_shapes_and_subsampling(small_config):
    """Verify output shapes, length reduction formula, and absence of NaNs."""
    model = ConformerCTC(small_config)
    model.eval()

    batch_size = 3
    lengths = torch.tensor([101, 160, 201])  # Feature frame lengths
    max_len = int(lengths.max().item())
    features = torch.randn(batch_size, max_len, 80)

    with torch.no_grad():
        log_probs, out_lengths = model(features, lengths)

    # Subsampling factor 4 exact formula: (L - 1) // 2 + 1 applied twice
    expected_l1 = (lengths - 1) // 2 + 1
    expected_out_lengths = (expected_l1 - 1) // 2 + 1

    assert torch.equal(out_lengths, expected_out_lengths)
    assert log_probs.shape == (batch_size, expected_out_lengths.max().item(), 1024)
    assert not torch.isnan(log_probs).any()
    assert not torch.isinf(log_probs).any()


def test_padding_invariance(small_config):
    """Verify single utterance output matches padded batch output within tolerance."""
    model = ConformerCTC(small_config)
    model.eval()

    short_len = 80
    long_len = 160

    feat_short = torch.randn(1, short_len, 80)
    feat_long = torch.randn(1, long_len, 80)

    # Batched with zero-padding
    batch_feat = torch.zeros(2, long_len, 80)
    batch_feat[0, :short_len] = feat_short[0]
    batch_feat[1, :long_len] = feat_long[0]
    batch_lengths = torch.tensor([short_len, long_len])

    with torch.no_grad():
        out_single, len_single = model(feat_short, torch.tensor([short_len]))
        out_batched, len_batched = model(batch_feat, batch_lengths)

    # Compare unpadded portion of the first utterance
    valid_frames = len_single[0].item()
    diff = torch.max(torch.abs(out_single[0, :valid_frames] - out_batched[0, :valid_frames])).item()
    assert diff < 1e-4, f"Padding invariance violated! Max absolute diff: {diff}"


def test_chunk_causality(small_config):
    """CRITICAL TEST: Mutating future frames must NOT affect earlier chunks."""
    model = ConformerCTC(small_config)
    model.eval()

    chunk_size = 8  # 8 encoder frames per chunk
    left_chunks = 2
    total_frames = 120  # In acoustic feature frames
    features1 = torch.randn(1, total_frames, 80)

    with torch.no_grad():
        out1, out_lens = model(features1, torch.tensor([total_frames]), chunk_size=chunk_size, left_chunks=left_chunks)

    # Mutate features starting from acoustic frame 80 (well into later chunks)
    features2 = features1.clone()
    features2[:, 80:, :] = torch.randn_like(features2[:, 80:, :])

    with torch.no_grad():
        out2, _ = model(features2, torch.tensor([total_frames]), chunk_size=chunk_size, left_chunks=left_chunks)

    # Frame 80 in input corresponds to frame 20 in encoder after factor 4 subsampling
    # Chunk 0 and 1 correspond to encoder frames 0..15.
    # Allowing small 2-frame margin for 2D Conv receptive field (conv kernel 3 stride 2)
    safe_encoder_frames = 14  # Safe frames strictly before mutation boundary
    diff = torch.max(torch.abs(out1[:, :safe_encoder_frames] - out2[:, :safe_encoder_frames])).item()

    assert diff < 1e-5, f"Causality leak detected! Earlier frames mutated: diff = {diff}"


def test_gradient_flow(small_config):
    """Verify backward pass produces finite, non-zero gradients across all parameters."""
    model = ConformerCTC(small_config)
    model.train()

    features = torch.randn(2, 60, 80, requires_grad=True)
    lengths = torch.tensor([50, 60])

    log_probs, out_lens = model(features, lengths)
    loss = log_probs.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient (dead branch)"
        assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"
        assert not torch.isinf(param.grad).any(), f"Inf in gradient for {name}"
        assert (param.grad.abs().sum() > 0).item(), f"Zero gradient for {name}"


def test_parameter_counts():
    """Verify parameter counts for presets match specification within +-15%."""
    # Preset target parameters from §1.6:
    # tiny: ~6M
    # small: ~19M
    # base: ~55M
    targets = {
        "tiny": 6.0e6,
        "small": 19.0e6,
        "base": 55.0e6,
    }

    for preset, target in targets.items():
        cfg = ConformerConfig.from_preset(preset)
        model = ConformerCTC(cfg)
        counts = model.count_parameters()
        total = counts["total"]

        deviation = abs(total - target) / target
        assert deviation <= 0.15, (
            f"Preset '{preset}' parameter count {total:,} ({total/1e6:.2f}M) "
            f"deviates {deviation*100:.1f}% from target {target/1e6:.1f}M (>15%)"
        )


def test_attention_mask_properties():
    """Verify dynamic chunk attention mask shape and non-NaN guarantees."""
    seq_len = 16
    lengths = torch.tensor([12, 16])
    mask = build_chunk_attention_mask(seq_len, lengths, chunk_size=4, left_chunks=1)

    assert mask.shape == (2, 1, seq_len, seq_len)
    # Check that diagonal is allowed (0.0) everywhere
    diag_vals = torch.diagonal(mask, dim1=-2, dim2=-1)
    assert torch.all(diag_vals == 0.0), "Attention mask diagonal is not 0.0"
