"""Text normalization and SentencePiece tokenizer module for Awaaz (Phase 1).

Handles deterministic NFC Unicode normalization, digit rejection, punctuation stripping,
Gurmukhi + English balance, and SentencePiece unigram tokenization (vocab 1024).
"""

from typing import Optional


def normalize(text: str) -> Optional[str]:
    """Applies Phase-1 deterministic normalization to bilingual text.

    Rules in order:
    1. Unicode NFC normalization.
    2. Lowercase (English).
    3. Drop utterance if it contains any digit (ASCII 0-9 or Gurmukhi U+0A66-U+0A6F).
    4. Replace Gurmukhi danda/double danda (U+0964/U+0965) with space.
    5. Strip all characters except a-z, Gurmukhi block U+0A00-U+0A7F, apostrophe, space.
    6. Collapse whitespace and strip. Return None if empty.

    Args:
        text: Raw input transcript.

    Returns:
        Normalized string or None if utterance should be rejected.
    """
    raise NotImplementedError("Text normalization will be implemented in Phase 1.")


def train_tokenizer(text_files, model_prefix: str, vocab_size: int = 1024) -> None:
    """Trains a language-balanced 1024-vocabulary SentencePiece unigram model.

    Args:
        text_files: List of paths to normalized text corpora.
        model_prefix: Path prefix to write .model and .vocab files.
        vocab_size: Total vocabulary size (fixed at 1024, pad_id=0 as CTC blank).
    """
    raise NotImplementedError("Tokenizer training will be implemented in Phase 1.")
