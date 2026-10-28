"""Dataset preparation tool for Hugging Face audio sources (Phase 3).

Streams Hugging Face audio datasets, converts audio to 16 kHz mono FLAC,
applies duration filtering [0.5, 20.0]s, and generates JSONL manifests.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def extract_waveform_and_sr(audio_data) -> Tuple[Optional[np.ndarray], int]:
    """Extracts a 1D float32 mono waveform and sample rate from dict or AudioDecoder."""
    arr = None
    sr = 16000

    if hasattr(audio_data, "get_all_samples"):
        # TorchCodec AudioDecoder object (datasets >= 3.x / 4.x)
        try:
            samples = audio_data.get_all_samples()
            arr = samples.data
            sr = getattr(samples, "sample_rate", 16000)
        except Exception as e:
            logging.debug("TorchCodec decoding failed: %s", e)
    elif isinstance(audio_data, dict):
        arr = audio_data.get("array")
        sr = audio_data.get("sampling_rate", 16000)
    elif hasattr(audio_data, "array"):
        arr = audio_data.array
        sr = getattr(audio_data, "sampling_rate", 16000)
    elif hasattr(audio_data, "__getitem__"):
        try:
            arr = audio_data["array"]
            sr = audio_data.get("sampling_rate", 16000) if hasattr(audio_data, "get") else 16000
        except (KeyError, TypeError):
            pass

    if arr is None:
        return None, sr

    # Convert PyTorch tensor to numpy if applicable
    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        arr = arr.numpy()

    arr = np.asarray(arr, dtype=np.float32)

    # Convert multi-channel to mono (channels, samples) or (samples, channels)
    if arr.ndim > 1:
        if arr.shape[0] < arr.shape[1]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=1)

    arr = np.squeeze(arr)
    # Clip float values to [-1.0, 1.0] to prevent FLAC encoding overflows
    arr = np.clip(arr, -1.0, 1.0)
    return arr, sr


def prepare_hf_dataset(
    dataset_id: str,
    output_dir: str,
    language: str,
    split: str = "train",
    subset: Optional[str] = None,
    max_items: Optional[int] = None,
    text_column: str = "transcription",
    audio_column: str = "audio",
    source_name: Optional[str] = None,
    domain: str = "general",
) -> str:
    """Prepares a Hugging Face dataset into 16 kHz FLAC files and a JSONL manifest."""
    try:
        from datasets import load_dataset
    except ImportError:
        logging.error("The 'datasets' package is required. Install via `pip install datasets`.")
        return ""

    out_path = Path(output_dir)
    audio_dir = out_path / "audio" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path / f"{split}_manifest.jsonl"

    logging.info("Streaming %s (%s, split=%s)...", dataset_id, subset or "default", split)
    ds = load_dataset(dataset_id, subset, split=split, streaming=True)

    count = 0
    with open(manifest_path, "a", encoding="utf-8") as f_out:
        for item in ds:
            if max_items and count >= max_items:
                break

            audio_data = item.get(audio_column)
            if not audio_data:
                for cand in ["audio", "speech"]:
                    if cand in item and item[cand]:
                        audio_data = item[cand]
                        break

            text = item.get(text_column, "")
            if not text:
                for cand in ["transcription", "raw_transcription", "sentence", "text"]:
                    if cand in item and item[cand]:
                        text = item[cand]
                        break

            if not audio_data or not text:
                continue

            clip_id = f"{source_name or dataset_id.replace('/', '_')}_{split}_{count:06d}"
            flac_path = audio_dir / f"{clip_id}.flac"

            # Avoid re-encoding if already present
            if not flac_path.exists():
                arr, sr = extract_waveform_and_sr(audio_data)
                if arr is None or len(arr) == 0:
                    continue

                # Resample to 16 kHz if necessary
                if sr != 16000:
                    import scipy.signal
                    num_samples = int(len(arr) * 16000 / sr)
                    arr = scipy.signal.resample(arr, num_samples)

                duration = len(arr) / 16000.0
                if duration < 0.5 or duration > 20.0:
                    continue

                sf.write(str(flac_path), arr, 16000, format="FLAC")
            else:
                info = sf.info(str(flac_path))
                duration = info.duration

            entry = {
                "audio_path": str(flac_path.resolve()).replace("\\", "/"),
                "text": text,
                "duration": round(duration, 3),
                "language": language,
                "domain": domain,
                "source": source_name or dataset_id,
            }
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
            if count % 200 == 0:
                logging.info("Progress: %d clips prepared for %s (%s)", count, dataset_id, split)

    logging.info("Prepared %d clips to %s", count, manifest_path)
    return str(manifest_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare audio datasets from Hugging Face.")
    parser.add_argument("--dataset_id", type=str, required=True, help="HF dataset identifier.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--language", type=str, required=True, help="Language code ('en' or 'pa').")
    parser.add_argument("--split", type=str, default="train", help="Dataset split.")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset/config.")
    parser.add_argument("--max_items", type=int, default=None, help="Maximum number of items for sanity check.")
    parser.add_argument("--text_column", type=str, default="transcription", help="Transcript column name.")
    args = parser.parse_args()

    prepare_hf_dataset(
        dataset_id=args.dataset_id,
        output_dir=args.output_dir,
        language=args.language,
        split=args.split,
        subset=args.subset,
        max_items=args.max_items,
        text_column=args.text_column,
    )


if __name__ == "__main__":
    main()
