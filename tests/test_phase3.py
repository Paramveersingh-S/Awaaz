"""Phase 3 Acceptance Tests — Data Pipeline, Manifests, and Duration Batching."""

import json

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from awaaz.dataset import (
    AwaazDataset,
    DurationBucketedBatchSampler,
    ctc_collate_fn,
)
from awaaz.text import Tokenizer, train_tokenizer
from scripts.make_fixture import create_fixture


@pytest.fixture
def test_manifest_and_tokenizer(tmp_path):
    """Sets up fixture manifest and trained SentencePiece tokenizer."""
    train_m, dev_m = create_fixture(str(tmp_path), num_train=16, num_dev=4)
    model_prefix = tmp_path / "spm_tok"
    model_path = train_tokenizer(
        output_prefix=model_prefix,
        manifests=[train_m],
        vocab_size=200,
        character_coverage=1.0,
    )
    tokenizer = Tokenizer(model_path)
    return train_m, tokenizer


def test_dataset_loading_and_collation(test_manifest_and_tokenizer):
    """Verify AwaazDataset loads waveforms and ctc_collate_fn produces valid batches."""
    manifest_path, tokenizer = test_manifest_and_tokenizer
    dataset = AwaazDataset(manifest_path, tokenizer=tokenizer, exclude_sacred=True)

    assert len(dataset) > 0

    # Test single item
    item0 = dataset[0]
    assert "waveform" in item0
    assert "tokens" in item0
    assert "text" in item0
    assert item0["waveform"].ndim == 1
    assert item0["waveform"].dtype == torch.float32
    assert len(item0["waveform"]) > 0
    assert len(item0["tokens"]) > 0

    # Test collation
    batch_items = [dataset[i] for i in range(min(4, len(dataset)))]
    batch = ctc_collate_fn(batch_items)

    assert "waveforms" in batch
    assert "waveform_lengths" in batch
    assert "targets" in batch
    assert "target_lengths" in batch

    assert batch["waveforms"].shape[0] == len(batch_items)
    assert batch["waveforms"].ndim == 2
    assert batch["waveform_lengths"].shape[0] == len(batch_items)
    assert batch["targets"].ndim == 1
    assert batch["targets"].shape[0] == batch["target_lengths"].sum().item()

    # Verify zero-padding: beyond unpadded length, values should be 0.0
    for i in range(len(batch_items)):
        valid_len = batch["waveform_lengths"][i].item()
        max_len = batch["waveforms"].shape[1]
        if valid_len < max_len:
            assert torch.all(batch["waveforms"][i, valid_len:] == 0.0)


def test_batch_sampler_coverage_and_budget(test_manifest_and_tokenizer):
    """Verify sampler visits all samples exactly once and respects duration budget."""
    manifest_path, tokenizer = test_manifest_and_tokenizer
    dataset = AwaazDataset(manifest_path, tokenizer=tokenizer, exclude_sacred=True)

    max_batch_sec = 6.0  # Compact budget for tiny fixture
    sampler = DurationBucketedBatchSampler(dataset, max_batch_seconds=max_batch_sec, shuffle=True, seed=100)

    visited_indices = []
    for batch_indices in sampler:
        assert len(batch_indices) > 0
        batch_durs = [dataset.items[i]["duration"] for i in batch_indices]
        batch_max_dur = max(batch_durs)
        padded_sec = batch_max_dur * len(batch_indices)

        # First item in batch can exceed if single clip > budget, but when multi-item, must respect
        if len(batch_indices) > 1:
            assert padded_sec <= max_batch_sec + 0.1, f"Budget exceeded: {padded_sec} > {max_batch_sec}"

        visited_indices.extend(batch_indices)

    # Every item must be visited exactly once
    assert sorted(visited_indices) == list(range(len(dataset))), "Sampler failed coverage check"

    # Test deterministic shuffling across epochs
    sampler.set_epoch(1)
    epoch1_order = list(sampler)
    sampler.set_epoch(2)
    epoch2_order = list(sampler)
    sampler.set_epoch(1)
    epoch1_repeat = list(sampler)

    assert epoch1_order == epoch1_repeat, "Sampler is not deterministic with same epoch"
    assert epoch1_order != epoch2_order, "Sampler failed to shuffle across different epochs"


def test_dataloader_end_to_end(test_manifest_and_tokenizer):
    """Verify integration with PyTorch DataLoader."""
    manifest_path, tokenizer = test_manifest_and_tokenizer
    dataset = AwaazDataset(manifest_path, tokenizer=tokenizer, exclude_sacred=True)

    sampler = DurationBucketedBatchSampler(dataset, max_batch_seconds=8.0, shuffle=True)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=ctc_collate_fn)

    batch_count = 0
    total_samples = 0
    for batch in loader:
        batch_count += 1
        total_samples += batch["waveforms"].shape[0]
        assert batch["waveforms"].shape[0] == len(batch["texts"])

    assert batch_count == len(sampler)
    assert total_samples == len(dataset)


def test_sacred_text_policy_filtering(tmp_path):
    """Verify sacred domain clips are excluded when exclude_sacred=True and kept when False."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy 1.0s audio
    dummy_wav = audio_dir / "sample.flac"
    sf.write(str(dummy_wav), np.zeros(16000, dtype=np.float32), 16000, format="FLAC")

    manifest_file = tmp_path / "sacred_test.jsonl"
    entries = [
        {"audio_path": str(dummy_wav), "text": "ਆਮ ਗੱਲਬਾਤ ਹੈ", "duration": 1.0, "language": "pa", "domain": "general"},
        {"audio_path": str(dummy_wav), "text": "ਇਕ ਓਅੰਕਾਰ ਸਤਿਨਾਮੁ", "duration": 1.0, "language": "pa", "domain": "sacred"},
    ]
    with open(manifest_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Exclude sacred (default)
    ds_excluded = AwaazDataset(manifest_file, exclude_sacred=True)
    assert len(ds_excluded) == 1
    assert ds_excluded.items[0]["domain"] == "general"

    # Include sacred (explicit opt-in)
    ds_included = AwaazDataset(manifest_file, exclude_sacred=False)
    assert len(ds_included) == 2


def test_duration_and_digit_filtering(tmp_path):
    """Verify clips with digits, empty text, or out-of-bounds duration are dropped."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    dummy_wav = audio_dir / "sample.flac"
    sf.write(str(dummy_wav), np.zeros(16000, dtype=np.float32), 16000, format="FLAC")

    manifest_file = tmp_path / "filter_test.jsonl"
    entries = [
        {"audio_path": str(dummy_wav), "text": "valid utterance", "duration": 2.0, "language": "en"},
        {"audio_path": str(dummy_wav), "text": "contains 101 digits", "duration": 2.0, "language": "en"},  # Drop: digits
        {"audio_path": str(dummy_wav), "text": "short clip", "duration": 0.2, "language": "en"},  # Drop: duration < 0.5s
        {"audio_path": str(dummy_wav), "text": "long clip", "duration": 25.0, "language": "en"},  # Drop: duration > 20.0s
        {"audio_path": str(dummy_wav), "text": "!!! ???", "duration": 2.0, "language": "en"},  # Drop: empty after norm
        {"audio_path": "non_existent.flac", "text": "missing file", "duration": 2.0, "language": "en"},  # Drop: missing
    ]
    with open(manifest_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    ds = AwaazDataset(manifest_file, min_duration=0.5, max_duration=20.0)
    assert len(ds) == 1
    assert ds.items[0]["text"] == "valid utterance"
