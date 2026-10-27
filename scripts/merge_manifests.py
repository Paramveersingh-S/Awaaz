"""Manifest merging and language balancing utility for Awaaz (Phase 3).

Merges multiple JSONL manifests with configurable source repeat factors,
caps English hours, and prints distinct vs repeated duration analytics.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def merge_manifests(
    manifest_configs: List[Dict],
    output_manifest: str,
    max_en_hours: float = 0.0,
    target_pa_ratio: float = 0.5,
) -> None:
    """Merges manifests applying repeat multipliers and language balancing."""
    out_path = Path(output_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    combined_entries = []
    stats = {"en_distinct_sec": 0.0, "pa_distinct_sec": 0.0, "en_total_sec": 0.0, "pa_total_sec": 0.0}

    for cfg in manifest_configs:
        m_path = cfg["path"]
        repeat = cfg.get("repeat", 1)
        lang = cfg.get("language")

        with open(m_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        distinct_sec = sum(item.get("duration", 0.0) for item in lines)
        if lang == "en":
            stats["en_distinct_sec"] += distinct_sec
        elif lang == "pa":
            stats["pa_distinct_sec"] += distinct_sec

        for _ in range(repeat):
            for item in lines:
                item_lang = item.get("language", lang)
                if item_lang == "en" and max_en_hours > 0:
                    if stats["en_total_sec"] / 3600.0 >= max_en_hours:
                        continue
                combined_entries.append(item)
                if item_lang == "en":
                    stats["en_total_sec"] += item.get("duration", 0.0)
                elif item_lang == "pa":
                    stats["pa_total_sec"] += item.get("duration", 0.0)

    with open(out_path, "w", encoding="utf-8") as f:
        for entry in combined_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    en_hrs = stats["en_total_sec"] / 3600.0
    pa_hrs = stats["pa_total_sec"] / 3600.0
    total_hrs = en_hrs + pa_hrs
    pa_ratio = (pa_hrs / total_hrs) if total_hrs > 0 else 0.0

    print("=" * 60)
    print("AWAAZ MANIFEST MERGE REPORT")
    print("=" * 60)
    print(f"Total Clips Written: {len(combined_entries):,}")
    print(f"English: {en_hrs:.2f} hrs (Distinct: {stats['en_distinct_sec']/3600.0:.2f} hrs)")
    print(f"Punjabi: {pa_hrs:.2f} hrs (Distinct: {stats['pa_distinct_sec']/3600.0:.2f} hrs)")
    print(f"Punjabi Share (with repeats): {pa_ratio*100:.1f}%")
    print(f"Output File: {output_manifest}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Merge JSONL manifests with balancing.")
    parser.add_argument("--manifests", nargs="+", required=True, help="Paths to JSONL manifests.")
    parser.add_argument("--repeats", nargs="+", type=int, default=None, help="Repeat count per manifest.")
    parser.add_argument("--languages", nargs="+", required=True, help="Language code per manifest ('en' or 'pa').")
    parser.add_argument("--output", type=str, required=True, help="Output manifest path.")
    parser.add_argument("--max_en_hours", type=float, default=0.0, help="Cap English total hours (0 = no cap).")
    args = parser.parse_args()

    repeats = args.repeats or [1] * len(args.manifests)
    configs = [
        {"path": m, "repeat": r, "language": lang}
        for m, r, lang in zip(args.manifests, repeats, args.languages)
    ]
    merge_manifests(configs, args.output, args.max_en_hours)


if __name__ == "__main__":
    main()
