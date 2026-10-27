"""Decoding strategies and CTC word timestamp extraction for Awaaz (Phase 7).

Includes:
1. Greedy CTC argmax decoding with repeat-collapsing and blank filtering.
2. Word timestamp extraction using SentencePiece subword boundary markers.
3. Feature-flagged N-gram Language Model rescoring interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from awaaz.text import Tokenizer


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    tokenizer: Optional[Tokenizer] = None,
) -> Union[List[str], List[List[int]]]:
    """Decodes frame-level log-probabilities via greedy CTC argmax.

    Args:
        log_probs: Float tensor of shape (batch, time, vocab) or (time, vocab).
        lengths: Optional int tensor of shape (batch,) with valid frame counts.
        tokenizer: Optional Tokenizer instance. If provided, returns detokenized
            strings; otherwise returns lists of integer token IDs.

    Returns:
        List of decoded strings (if tokenizer given) or list of token ID lists.
    """
    if log_probs.ndim == 2:
        log_probs = log_probs.unsqueeze(0)
        if lengths is not None and lengths.ndim == 0:
            lengths = lengths.unsqueeze(0)

    batch_size, max_time, _ = log_probs.shape
    argmax_ids = log_probs.argmax(dim=-1).detach().cpu()
    blank_id = tokenizer.blank_id if tokenizer is not None else 0

    hypotheses: List[Any] = []

    for b in range(batch_size):
        seq_len = int(lengths[b].item()) if lengths is not None else max_time
        frame_tokens = argmax_ids[b, :seq_len].tolist()

        collapsed_tokens: List[int] = []
        prev_token = -1
        for tok in frame_tokens:
            if tok != prev_token:
                if tok != blank_id:
                    collapsed_tokens.append(tok)
                prev_token = tok

        if tokenizer is not None:
            hypotheses.append(tokenizer.decode(collapsed_tokens))
        else:
            hypotheses.append(collapsed_tokens)

    return hypotheses


def extract_word_timestamps(
    log_probs: torch.Tensor,
    tokenizer: Tokenizer,
    subsample_factor: int = 4,
    hop_ms: float = 10.0,
) -> List[Dict[str, Any]]:
    """Extracts approximate word start and end timestamps from CTC emission spikes.

    Uses the frame-to-time conversion:
        frame_duration = (hop_ms / 1000.0) * subsample_factor = 40 ms (for factor 4).

    Identifies word boundaries using SentencePiece leading whitespace marker ('\u2581').

    Args:
        log_probs: Float tensor of shape (time, vocab) or (1, time, vocab).
        tokenizer: Initialized Tokenizer for piece and word decoding.
        subsample_factor: Front-end convolutional subsampling factor (default 4).
        hop_ms: STFT hop length in milliseconds (default 10.0).

    Returns:
        List of word dictionaries:
            [{"word": "sat", "start": 0.32, "end": 0.48}, ...]
    """
    if log_probs.ndim == 3:
        if log_probs.shape[0] != 1:
            raise ValueError("extract_word_timestamps operates on a single utterance.")
        log_probs = log_probs.squeeze(0)

    num_frames, _ = log_probs.shape
    frame_duration_sec = (hop_ms / 1000.0) * subsample_factor
    blank_id = tokenizer.blank_id

    argmax_ids = log_probs.argmax(dim=-1).detach().cpu().tolist()

    # 1. Identify non-blank, non-repeated emission spikes with their frame indices
    emissions: List[Tuple[int, int]] = []  # (frame_idx, token_id)
    prev_id = -1

    for t, tok in enumerate(argmax_ids):
        if tok != prev_id:
            if tok != blank_id:
                emissions.append((t, tok))
            prev_id = tok

    if not emissions:
        return []

    # 2. Group emissions into words using SentencePiece boundary marker ('\u2581' or ' ')
    words: List[Dict[str, Any]] = []
    current_word_tokens: List[int] = []
    current_start_frame: Optional[int] = None
    current_last_frame: Optional[int] = None

    def flush_word():
        nonlocal current_word_tokens, current_start_frame, current_last_frame
        if current_word_tokens and current_start_frame is not None and current_last_frame is not None:
            word_str = tokenizer.decode(current_word_tokens).strip()
            if not word_str:
                word_str = "".join(tokenizer.id_to_piece(t).replace("\u2581", "").replace(" ", "") for t in current_word_tokens).strip()
            if word_str:
                start_sec = round(current_start_frame * frame_duration_sec, 3)
                end_sec = round((current_last_frame + 1) * frame_duration_sec, 3)
                words.append({
                    "word": word_str,
                    "start": start_sec,
                    "end": max(end_sec, round(start_sec + frame_duration_sec, 3)),
                })
        current_word_tokens = []
        current_start_frame = None
        current_last_frame = None

    for frame_idx, tok_id in emissions:
        piece = tokenizer.id_to_piece(tok_id)
        is_word_start = piece.startswith("\u2581") or piece.startswith(" ") or (len(words) == 0 and len(current_word_tokens) == 0)

        if is_word_start and current_word_tokens:
            flush_word()

        if not current_word_tokens:
            current_start_frame = frame_idx

        current_last_frame = frame_idx
        current_word_tokens.append(tok_id)

    flush_word()
    return words


def ctc_beam_search_decode(
    log_probs: torch.Tensor,
    tokenizer: Tokenizer,
    lm_path: Optional[str] = None,
    alpha: float = 0.5,
    beta: float = 1.0,
    beam_width: int = 100,
) -> List[str]:
    """Feature-flagged CTC beam search decoding with optional KenLM rescoring.

    If pyctcdecode is not installed or lm_path is None, falls back gracefully
    to greedy CTC decoding.
    """
    if lm_path is None:
        return ctc_greedy_decode(log_probs, tokenizer=tokenizer)

    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError:
        import logging
        logging.getLogger("awaaz.decode").warning(
            "pyctcdecode not installed; falling back to greedy decoding."
        )
        return ctc_greedy_decode(log_probs, tokenizer=tokenizer)

    # Build pyctcdecoder using vocabulary pieces
    labels = [tokenizer.id_to_piece(i) for i in range(tokenizer.vocab_size)]
    # Replace blank representation
    labels[tokenizer.blank_id] = ""

    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=lm_path,
        alpha=alpha,
        beta=beta,
    )

    if log_probs.ndim == 2:
        log_probs = log_probs.unsqueeze(0)

    # Convert log-probs to probabilities for pyctcdecode
    probs = log_probs.exp().cpu().numpy()
    results = [decoder.decode(probs[b], beam_width=beam_width) for b in range(probs.shape[0])]
    return results
