"""Decoding strategies and CTC word timestamp extraction for Awaaz (Phase 7).

Includes greedy CTC decoding, token collapse, boundary identification, and optional LM rescoring.
"""

from typing import List, Tuple

import torch


def ctc_greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, tokenizer) -> List[str]:
    """Decodes frame-level log-probabilities via greedy CTC argmax and detokenization."""
    raise NotImplementedError("CTC greedy decoding will be implemented in Phase 7.")


def extract_word_timestamps(
    log_probs: torch.Tensor,
    subsample_factor: int = 4,
    hop_ms: float = 10.0,
) -> List[Tuple[str, float, float]]:
    """Extracts approximate word start and end timestamps from CTC emission spikes."""
    raise NotImplementedError("Timestamp extraction will be implemented in Phase 7.")
