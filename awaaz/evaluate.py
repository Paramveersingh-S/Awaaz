"""Evaluation runner for Awaaz (Phase 7).

Computes offline and streaming WER/CER on English and Punjabi manifests,
and outputs markdown report tables and JSON diagnostics.
"""

from typing import Any, Dict


def evaluate_manifest(manifest_path: str, model_path: str, tokenizer_path: str, cmvn_path: str) -> Dict[str, Any]:
    """Evaluates a manifest across offline, 0.64s stream, and 0.32s stream modes."""
    raise NotImplementedError("Evaluation runner will be implemented in Phase 7.")
