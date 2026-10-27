"""Phase 2 Acceptance Tests — Audio Feature Pipeline, CMVN, and Augmentation."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from awaaz.features import (
    LogMelSpectrogram,
    SpecAugment,
    WaveformAugmenter,
    compute_cmvn,
)
from scripts.make_fixture import create_fixture


def test_features_shape_and_frame_formula():
    """Verify LogMelSpectrogram output shape (batch, frames, 80) and frame-count formula."""
    frontend = LogMelSpectrogram(sample_rate=16000, n_mels=80)
    durations = [0.25, 0.5, 1.234, 3.0, 5.0]

    for dur in durations:
        samples = int(dur * 16000)
        expected_frames = samples // 160 + 1
        audio = torch.randn(2, samples)

        features = frontend(audio)
        assert features.shape == (2, expected_frames, 80), (
            f"Failed on duration {dur}s (samples={samples}): "
            f"Expected shape (2, {expected_frames}, 80), got {features.shape}"
        )
        assert not torch.isnan(features).any(), "NaN found in features"
        assert not torch.isinf(features).any(), "Inf found in features"


def test_cmvn_computation_and_normalization(tmp_path):
    """Verify CMVN calculation and zero-mean / unit-variance normalization on audio."""
    train_m, _ = create_fixture(str(tmp_path), num_train=16, num_dev=4)
    cmvn_path = tmp_path / "test_cmvn.npz"

    mean, std = compute_cmvn(train_m, cmvn_path, max_utterances=16)
    assert mean.shape == (80,)
    assert std.shape == (80,)
    assert np.all(std > 0)
    assert cmvn_path.is_file()

    # Load into LogMelSpectrogram
    frontend = LogMelSpectrogram()
    frontend.load_cmvn(cmvn_path)
    assert frontend.cmvn_loaded is True

    # Test normalization on one clip from fixture
    with open(train_m, "r", encoding="utf-8") as f:
        import json
        item = json.loads(f.readline())

    audio, sr = sf.read(item["audio_path"], dtype="float32")
    tensor_audio = torch.from_numpy(audio)
    norm_features = frontend(tensor_audio)

    # Output features should be normalized (mean near 0, reasonable std)
    feat_np = norm_features.squeeze(0).cpu().numpy()
    assert abs(np.mean(feat_np)) < 1.0, f"Mean too far from 0: {np.mean(feat_np)}"
    assert np.std(feat_np) > 0.1, "Standard deviation collapsed to 0"


def test_cmvn_save_and_load(tmp_path):
    """Verify saving and loading CMVN stats preserves exact values."""
    frontend = LogMelSpectrogram()
    dummy_mean = torch.randn(80)
    dummy_std = torch.rand(80) + 0.5
    frontend.cmvn_mean.copy_(dummy_mean)
    frontend.cmvn_std.copy_(dummy_std)

    out_file = tmp_path / "saved_cmvn.npz"
    frontend.save_cmvn(out_file)

    loader_frontend = LogMelSpectrogram()
    loader_frontend.load_cmvn(out_file)

    assert torch.allclose(loader_frontend.cmvn_mean, dummy_mean)
    assert torch.allclose(loader_frontend.cmvn_std, dummy_std)


def test_specaugment_masking_and_eval_mode():
    """Verify SpecAugment applies masks in training mode and acts as identity in eval mode."""
    spec_aug = SpecAugment(freq_mask_max_width=27, num_freq_masks=2, time_mask_max_ratio=0.05, num_time_masks=2)

    features = torch.ones(2, 100, 80)
    lengths = torch.tensor([80, 100])

    # 1. Training mode
    spec_aug.train()
    masked = spec_aug(features, lengths)
    assert masked.shape == features.shape
    # Some values must be masked to 0.0
    assert (masked == 0.0).any()
    # Unpadded region beyond length (80 to 100 for batch 0) must NOT have been touched by time mask
    assert (masked[0, 80:, :] == 1.0).all(), "SpecAugment modified padded frames"

    # 2. Eval mode (must be strict identity pass-through)
    spec_aug.eval()
    eval_out = spec_aug(features, lengths)
    assert torch.equal(eval_out, features), "SpecAugment mutated features during eval mode"


def test_features_determinism():
    """Verify identical random seed produces bitwise identical features."""
    frontend = LogMelSpectrogram()
    audio = torch.randn(1, 16000)

    torch.manual_seed(1234)
    feat1 = frontend(audio)

    torch.manual_seed(1234)
    feat2 = frontend(audio)

    assert torch.equal(feat1, feat2), "Feature extraction is not deterministic with fixed seed"


def test_waveform_augmentations():
    """Verify speed perturbation, random gain, and peak clamping."""
    audio = np.sin(np.linspace(0, 100, 16000, dtype=np.float32))

    augmenter = WaveformAugmenter(speed_perturb=True, random_gain_db=6.0)

    # Speed test
    faster = augmenter.apply_speed(audio, speed=1.1)
    assert len(faster) < len(audio)
    slower = augmenter.apply_speed(audio, speed=0.9)
    assert len(slower) > len(audio)

    # Gain test
    louder = augmenter.apply_gain(audio, gain_db=6.0)
    assert np.max(np.abs(louder)) > np.max(np.abs(audio))

    # Stochastic pipeline
    aug_audio = augmenter.augment(audio)
    assert len(aug_audio) > 0
    assert np.max(np.abs(aug_audio)) <= 0.96


def test_docs_features_exists():
    """Verify docs/FEATURES.md exists and contains the language-neutral spec."""
    root = Path(__file__).resolve().parent.parent
    features_doc = root / "docs" / "FEATURES.md"
    assert features_doc.is_file()
    content = features_doc.read_text(encoding="utf-8")
    assert "Mel Filterbanks" in content
    assert "Global Cumulative Mean" in content
