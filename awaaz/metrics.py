"""Evaluation metrics (WER and CER) calculation module for Awaaz (Phase 7).

Computes Word Error Rate and Character Error Rate on normalized hypotheses and references,
prioritizing CER for Punjabi orthography.
"""

from typing import Dict, List

import jiwer


def compute_wer_cer(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Computes Word Error Rate and Character Error Rate using JiWER."""
    if not references:
        return {"wer": 0.0, "cer": 0.0}
    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)
    return {"wer": float(wer), "cer": float(cer)}
