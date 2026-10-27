"""Evaluation metrics (WER and CER) calculation module for Awaaz (Phase 7).

Computes Word Error Rate and Character Error Rate on normalized hypotheses and references,
prioritizing CER for Punjabi orthography.
"""

from typing import Any, Dict, List, Optional

import jiwer

from awaaz.text import normalize


def compute_wer_cer(
    references: List[str],
    hypotheses: List[str],
    auto_normalize: bool = False,
) -> Dict[str, float]:
    """Computes Word Error Rate and Character Error Rate using JiWER.

    Args:
        references: List of reference ground-truth transcripts.
        hypotheses: List of predicted hypothesis transcripts.
        auto_normalize: If True, applies Awaaz text normalization to both.

    Returns:
        Dict with "wer" and "cer" as float values in [0.0, 1.0+] (not percentage).
    """
    if len(references) != len(hypotheses):
        raise ValueError(f"Length mismatch: {len(references)} references vs {len(hypotheses)} hypotheses.")

    if not references:
        return {"wer": 0.0, "cer": 0.0}

    clean_refs: List[str] = []
    clean_hyps: List[str] = []

    for ref, hyp in zip(references, hypotheses):
        r = normalize(ref) if auto_normalize else (ref or "").strip()
        h = normalize(hyp) if auto_normalize else (hyp or "").strip()
        clean_refs.append(r or "")
        clean_hyps.append(h or "")

    # Handle all-empty references safely
    non_empty_pairs = [(r, h) for r, h in zip(clean_refs, clean_hyps) if r]
    if not non_empty_pairs:
        # All references are empty
        has_hyp = any(len(h) > 0 for h in clean_hyps)
        return {"wer": 1.0 if has_hyp else 0.0, "cer": 1.0 if has_hyp else 0.0}

    valid_refs = [p[0] for p in non_empty_pairs]
    valid_hyps = [p[1] for p in non_empty_pairs]

    try:
        wer = float(jiwer.wer(valid_refs, valid_hyps))
    except Exception:
        wer = 1.0

    try:
        cer = float(jiwer.cer(valid_refs, valid_hyps))
    except Exception:
        cer = 1.0

    return {"wer": round(wer, 4), "cer": round(cer, 4)}


def format_error_report(
    references: List[str],
    hypotheses: List[str],
    audio_paths: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """Identifies the top-K worst utterances by sentence-level WER for error triage."""
    records: List[Dict[str, Any]] = []

    for idx, (ref, hyp) in enumerate(zip(references, hypotheses)):
        r = (ref or "").strip()
        h = (hyp or "").strip()
        if not r:
            sent_wer = 1.0 if h else 0.0
            sent_cer = 1.0 if h else 0.0
        else:
            try:
                sent_wer = float(jiwer.wer(r, h))
                sent_cer = float(jiwer.cer(r, h))
            except Exception:
                sent_wer = 1.0
                sent_cer = 1.0

        records.append({
            "index": idx,
            "reference": r,
            "hypothesis": h,
            "wer": round(sent_wer, 4),
            "cer": round(sent_cer, 4),
            "language": languages[idx] if languages else "unknown",
            "audio_path": audio_paths[idx] if audio_paths else "",
        })

    # Sort descending by WER then CER
    records.sort(key=lambda x: (x["wer"], x["cer"]), reverse=True)
    return records[:top_k]
