"""Dataset loading, collation, and duration-bucketed batch sampling for Awaaz (Phase 3).

Reads JSON Lines manifests, applies normalization filtering and sacred text exclusions,
and forms audio duration batches.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

from awaaz.text import normalize

logger = logging.getLogger("awaaz.dataset")


class AwaazDataset(Dataset):
    """Loads 16 kHz FLAC audio and text targets from JSONL manifests.

    Applies text normalization, filters utterances with digits or empty text,
    enforces the Sacred-Text Policy (§1.5), and filters clips by duration bounds.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path, List[Union[str, Path]]],
        tokenizer=None,
        exclude_sacred: bool = True,
        min_duration: float = 0.5,
        max_duration: float = 20.0,
        training: bool = False,
        speed_perturb: bool = False,
    ):
        self.tokenizer = tokenizer
        self.exclude_sacred = exclude_sacred
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.training = training
        self.speed_perturb = speed_perturb

        self.manifest_paths = [Path(manifest_path)] if not isinstance(manifest_path, list) else [Path(p) for p in manifest_path]
        self.items: List[Dict[str, Any]] = []

        self._load_and_filter()

    def _load_and_filter(self) -> None:
        """Reads manifests and applies normalization and policy filtering."""
        dropped_sacred = 0
        dropped_norm = 0
        dropped_duration = 0
        dropped_missing = 0
        total_read = 0

        for m_path in self.manifest_paths:
            if not m_path.is_file():
                logger.warning("Manifest file does not exist: %s", m_path)
                continue

            with open(m_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    total_read += 1
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # 1. Sacred text policy check (§1.5)
                    domain = entry.get("domain", "general")
                    if self.exclude_sacred and domain == "sacred":
                        dropped_sacred += 1
                        continue

                    # 2. Duration bounds check [0.5, 20.0]s
                    duration = float(entry.get("duration", 0.0))
                    if duration < self.min_duration or duration > self.max_duration:
                        dropped_duration += 1
                        continue

                    # 3. Audio file existence check
                    audio_path = entry.get("audio_path", "")
                    if not audio_path or not Path(audio_path).is_file():
                        dropped_missing += 1
                        continue

                    # 4. Text normalization check (drops if contains digits or empty)
                    raw_text = entry.get("text", "")
                    norm_text = normalize(raw_text)
                    if not norm_text:
                        dropped_norm += 1
                        continue

                    # 5. Tokenization (if tokenizer provided)
                    tokens: List[int] = []
                    if self.tokenizer is not None:
                        tokens = self.tokenizer.encode(norm_text)
                        # Sanity check: tokens must be fewer than acoustic frames with subsampling
                        # Factor 4 subsampling produces frames = samples // 160 + 1 (25 frames/sec)
                        approx_encoder_frames = int(duration * 25.0)
                        if len(tokens) >= approx_encoder_frames:
                            dropped_norm += 1
                            continue

                    self.items.append({
                        "audio_path": audio_path,
                        "text": norm_text,
                        "duration": duration,
                        "language": entry.get("language", "pa"),
                        "domain": domain,
                        "source": entry.get("source", ""),
                        "tokens": tokens,
                    })

        logger.info(
            "Loaded %d valid items (Read: %d, Dropped: %d sacred, %d invalid text/digits, %d duration, %d missing audio)",
            len(self.items),
            total_read,
            dropped_sacred,
            dropped_norm,
            dropped_duration,
            dropped_missing,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        audio_path = item["audio_path"]

        # Read audio via soundfile
        audio, sr = sf.read(audio_path, dtype="float32")
        if sr != 16000:
            import scipy.signal
            num_samples = int(len(audio) * 16000 / sr)
            audio = scipy.signal.resample(audio, num_samples).astype(np.float32)

        # Average multi-channel to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        # Speed perturbation (training only)
        if self.training and self.speed_perturb:
            speed = random.choice([0.9, 1.0, 1.1])
            if speed != 1.0:
                import scipy.signal
                num_target = int(len(audio) / speed)
                audio = scipy.signal.resample(audio, num_target).astype(np.float32)

        waveform_tensor = torch.from_numpy(audio).float()
        token_tensor = torch.tensor(item["tokens"], dtype=torch.long)

        return {
            "waveform": waveform_tensor,
            "tokens": token_tensor,
            "text": item["text"],
            "language": item["language"],
            "domain": item["domain"],
            "source": item["source"],
            "duration": item["duration"],
        }


def ctc_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates variable-length waveforms and token sequences for CTC training.

    Pads waveforms to batch maximum length and concatenates token targets
    into a 1D tensor compatible with PyTorch nn.CTCLoss.

    Args:
        batch: List of item dictionaries from AwaazDataset.

    Returns:
        Dictionary containing:
            - waveforms: (batch, max_audio_samples) float32 tensor
            - waveform_lengths: (batch,) int64 tensor of unpadded audio lengths
            - targets: (total_tokens,) 1D int64 concatenated target tensor
            - target_lengths: (batch,) int64 tensor of token counts per utterance
            - texts: list of normalized strings
            - languages: list of language codes ('en', 'pa')
    """
    batch_size = len(batch)
    waveform_lengths = torch.tensor([b["waveform"].shape[0] for b in batch], dtype=torch.long)
    max_waveform_len = int(waveform_lengths.max().item())

    # Pad waveforms to batch max length with zeros
    waveforms = torch.zeros(batch_size, max_waveform_len, dtype=torch.float32)
    for i, b in enumerate(batch):
        waveforms[i, : b["waveform"].shape[0]] = b["waveform"]

    # Target tokens
    target_lengths = torch.tensor([len(b["tokens"]) for b in batch], dtype=torch.long)
    if target_lengths.sum() > 0:
        targets = torch.cat([b["tokens"] for b in batch], dim=0)
    else:
        targets = torch.empty(0, dtype=torch.long)

    texts = [b["text"] for b in batch]
    languages = [b["language"] for b in batch]

    return {
        "waveforms": waveforms,
        "waveform_lengths": waveform_lengths,
        "targets": targets,
        "target_lengths": target_lengths,
        "texts": texts,
        "languages": languages,
    }


class DurationBucketedBatchSampler(Sampler):
    """Batches audio utterances by duration up to a maximum padded seconds budget.

    Sorts examples by duration to minimize padding waste, packs each batch until
    (max_duration_in_batch * batch_size) reaches max_batch_seconds, and shuffles
    the batch order each epoch deterministically using seed + epoch.
    """

    def __init__(
        self,
        dataset: AwaazDataset,
        max_batch_seconds: float = 400.0,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.max_batch_seconds = max_batch_seconds
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.batches: List[List[int]] = []
        self._build_batches()

    def _build_batches(self) -> None:
        """Groups indices by duration and builds duration-constrained batches."""
        if len(self.dataset) == 0:
            self.batches = []
            return

        # Sort indices by audio duration
        indices_with_dur = [(i, self.dataset.items[i]["duration"]) for i in range(len(self.dataset))]
        indices_with_dur.sort(key=lambda x: x[1])

        batches: List[List[int]] = []
        current_batch: List[int] = []

        for idx, dur in indices_with_dur:
            if not current_batch:
                current_batch.append(idx)
                continue

            # Candidate duration if adding this item
            batch_max_dur = dur  # Since sorted ascending, new item has largest duration
            padded_seconds = batch_max_dur * (len(current_batch) + 1)

            if padded_seconds <= self.max_batch_seconds:
                current_batch.append(idx)
            else:
                batches.append(current_batch)
                current_batch = [idx]

        if current_batch:
            batches.append(current_batch)

        self.batches = batches

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for deterministic reproducible shuffling."""
        self.epoch = epoch

    def __iter__(self):
        batch_indices = list(range(len(self.batches)))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(batch_indices)

        for b_idx in batch_indices:
            yield self.batches[b_idx]

    def __len__(self) -> int:
        return len(self.batches)
