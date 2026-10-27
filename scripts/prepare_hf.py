"""Dataset preparation tool for Hugging Face audio sources (Phase 3).

Streams Hugging Face audio datasets, converts audio to 16 kHz mono FLAC,
applies duration filtering [0.5, 20.0]s, and generates JSONL manifests.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def prepare_hf_dataset(
    dataset_id: str,
    output_dir: str,
    language: str,
    split: str = "train",
    subset: Optional[str] = None,
    max_items: Optional[int] = None,
    text_column: str = "sentence",
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
            text = item.get(text_column, "")
            if not audio_data or not text:
                continue

            clip_id = f"{source_name or dataset_id.replace('/', '_')}_{split}_{count:06d}"
            flac_path = audio_dir / f"{clip_id}.flac"

            # Avoid re-encoding if already present
            if not flac_path.exists():
                arr = audio_data.get("array")
                sr = audio_data.get("sampling_rate", 16000)
                if arr is None:
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
    parser.add_argument("--text_column", type=str, default="sentence", help="Transcript column name.")
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
