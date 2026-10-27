"""Text normalization and SentencePiece tokenizer module for Awaaz (Phase 1).

Handles deterministic Unicode NFC normalization, digit rejection, punctuation stripping,
Gurmukhi + English balance, and SentencePiece unigram tokenization (vocab 1024).
"""

import argparse
import json
import logging
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import sentencepiece as spm

logger = logging.getLogger("awaaz.text")

# Unicode codepoints
DANDA = "\u0964"  # ।
DOUBLE_DANDA = "\u0965"  # ॥

# Digits regex: ASCII 0-9 and Gurmukhi digits U+0A66-U+0A6F
DIGIT_PATTERN = re.compile(r"[0-9\u0A66-\u0A6F]")

# Whitelist pattern: keep only a-z, Gurmukhi block (\u0A00-\u0A7F), apostrophe ('), space
# All other characters are removed.
WHITELIST_CLEAN_PATTERN = re.compile(r"[^a-z\u0A00-\u0A7F' ]")

# Collapse consecutive whitespace
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize(text: Optional[str]) -> Optional[str]:
    """Applies Phase-1 deterministic normalization to bilingual text.

    Rules applied in strictly deterministic order:
    1. Unicode NFC normalization (splits composed Gurmukhi nukta letters into base + nukta).
    2. Lowercase (converts Latin characters to lowercase).
    3. Drop utterance if it contains any digit (ASCII 0-9 or Gurmukhi U+0A66-U+0A6F).
    4. Replace Gurmukhi danda / double danda (U+0964, U+0965) with space.
    5. Strip all characters except a-z, Gurmukhi block U+0A00-U+0A7F, apostrophe ('), space.
       (Standardizes typographic curly apostrophes to ASCII ' before stripping).
    6. Collapse whitespace and strip leading/trailing spaces.
    7. Return None if empty or rejected.

    Args:
        text: Raw input transcript.

    Returns:
        Normalized string, or None if utterance should be rejected.
    """
    if text is None:
        return None

    # Step 1: Unicode NFC Normalization
    text = unicodedata.normalize("NFC", text)

    # Step 2: Lowercase (English only)
    text = text.lower()

    # Step 3: Digit rejection (drop utterance containing any digit)
    if DIGIT_PATTERN.search(text):
        return None

    # Replace curly apostrophes with ASCII apostrophe U+0027 before filtering
    text = text.replace("’", "'").replace("‘", "'")

    # Step 4: Replace Gurmukhi danda and double danda with space
    text = text.replace(DANDA, " ").replace(DOUBLE_DANDA, " ")

    # Step 5: Character whitelist filtering
    text = WHITELIST_CLEAN_PATTERN.sub("", text)

    # Step 6: Collapse whitespace and strip
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    # Step 7: Reject empty strings
    if not text:
        return None

    return text


class Tokenizer:
    """SentencePiece Tokenizer wrapper for Awaaz Conformer-CTC models."""

    def __init__(self, model_file: Union[str, Path]):
        self.model_path = str(model_file)
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Tokenizer model file not found: {self.model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_path)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        """Pad token ID, strictly used as the CTC blank."""
        return self.sp.pad_id()

    @property
    def blank_id(self) -> int:
        """Alias for pad_id (CTC Blank)."""
        return self.sp.pad_id()

    @property
    def unk_id(self) -> int:
        """Unknown token ID."""
        return self.sp.unk_id()

    def encode(self, text: str) -> List[int]:
        """Encodes text to a list of subword token IDs."""
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: Sequence[int]) -> str:
        """Decodes token IDs back to a text string, ignoring pad/blank."""
        filtered_ids = [int(i) for i in ids if int(i) != self.pad_id]
        return self.sp.decode(filtered_ids)

    def encode_as_pieces(self, text: str) -> List[str]:
        """Encodes text to subword piece strings."""
        return self.sp.encode_as_pieces(text)

    def piece_to_id(self, piece: str) -> int:
        """Returns integer ID for a subword piece."""
        return self.sp.piece_to_id(piece)

    def id_to_piece(self, piece_id: int) -> str:
        """Returns piece string for a token ID."""
        return self.sp.id_to_piece(piece_id)


def train_tokenizer(
    output_prefix: Union[str, Path],
    manifests: Optional[List[str]] = None,
    en_corpus: Optional[List[str]] = None,
    pa_corpus: Optional[List[str]] = None,
    text_pairs: Optional[List[Tuple[str, str]]] = None,
    vocab_size: int = 1024,
    max_lines_per_lang: int = 300000,
    character_coverage: float = 1.0,
    seed: int = 42,
) -> str:
    """Trains a language-balanced 1024-vocabulary SentencePiece unigram model.

    The training data is strictly balanced 50/50 between English and Punjabi
    to ensure English subwords do not monopolize the vocabulary.

    Args:
        output_prefix: File prefix for the generated .model and .vocab files.
        manifests: Optional paths to JSONL manifests containing 'text' and 'language'.
        en_corpus: Optional paths to raw text files for English.
        pa_corpus: Optional paths to raw text files for Punjabi.
        text_pairs: Optional direct list of (text, language_code) tuples.
        vocab_size: Target vocabulary size (default 1024).
        max_lines_per_lang: Maximum number of normalized sentences per language.
        character_coverage: Character coverage (1.0 for Gurmukhi script fidelity).
        seed: Random seed for shuffling.

    Returns:
        Path to the trained .model file.
    """
    en_lines: List[str] = []
    pa_lines: List[str] = []

    # 1. Ingest from manifests
    if manifests:
        for m_path in manifests:
            with open(m_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    raw_text = item.get("text", "")
                    lang = item.get("language", "pa")
                    norm_text = normalize(raw_text)
                    if norm_text:
                        if lang == "en":
                            en_lines.append(norm_text)
                        else:
                            pa_lines.append(norm_text)

    # 2. Ingest from text corpus files
    if en_corpus:
        for f_path in en_corpus:
            with open(f_path, "r", encoding="utf-8") as f:
                for line in f:
                    norm = normalize(line)
                    if norm:
                        en_lines.append(norm)

    if pa_corpus:
        for f_path in pa_corpus:
            with open(f_path, "r", encoding="utf-8") as f:
                for line in f:
                    norm = normalize(line)
                    if norm:
                        pa_lines.append(norm)

    # 3. Ingest from direct text pairs
    if text_pairs:
        for raw_text, lang in text_pairs:
            norm = normalize(raw_text)
            if norm:
                if lang == "en":
                    en_lines.append(norm)
                else:
                    pa_lines.append(norm)

    # Deduplicate while preserving order
    en_lines = list(dict.fromkeys(en_lines))
    pa_lines = list(dict.fromkeys(pa_lines))

    # Balance datasets
    rng = random.Random(seed)
    rng.shuffle(en_lines)
    rng.shuffle(pa_lines)

    sample_count = min(len(en_lines), len(pa_lines), max_lines_per_lang)
    if sample_count == 0:
        # Fallback if only one language provided in small fixtures
        balanced_corpus = en_lines[:max_lines_per_lang] + pa_lines[:max_lines_per_lang]
    else:
        balanced_corpus = en_lines[:sample_count] + pa_lines[:sample_count]

    rng.shuffle(balanced_corpus)

    out_prefix_path = Path(output_prefix)
    out_prefix_path.parent.mkdir(parents=True, exist_ok=True)

    temp_corpus_file = out_prefix_path.parent / f"{out_prefix_path.name}_temp_corpus.txt"
    with open(temp_corpus_file, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in balanced_corpus)

    try:
        spm.SentencePieceTrainer.train(
            input=str(temp_corpus_file),
            model_prefix=str(out_prefix_path),
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=character_coverage,
            pad_id=0,
            unk_id=1,
            bos_id=-1,
            eos_id=-1,
            normalization_rule_name="identity",
            shuffle_input_sentence=True,
            hard_vocab_limit=False,
        )
    finally:
        if temp_corpus_file.exists():
            temp_corpus_file.unlink()

    model_file = f"{out_prefix_path}.model"
    logger.info("SentencePiece model successfully trained: %s", model_file)
    return model_file


def main():
    parser = argparse.ArgumentParser(description="Awaaz Tokenizer CLI")
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommands")

    train_parser = subparsers.add_parser("train-tokenizer", help="Train a SentencePiece tokenizer.")
    train_parser.add_argument("--output_prefix", type=str, required=True, help="Prefix for .model and .vocab.")
    train_parser.add_argument("--manifests", nargs="+", default=[], help="Paths to JSONL manifests.")
    train_parser.add_argument("--en_corpus", nargs="+", default=[], help="Paths to English text files.")
    train_parser.add_argument("--pa_corpus", nargs="+", default=[], help="Paths to Punjabi text files.")
    train_parser.add_argument("--vocab_size", type=int, default=1024, help="Vocabulary size (default: 1024).")
    train_parser.add_argument("--character_coverage", type=float, default=1.0, help="Character coverage.")

    args = parser.parse_args()

    if args.subcommand == "train-tokenizer":
        train_tokenizer(
            output_prefix=args.output_prefix,
            manifests=args.manifests,
            en_corpus=args.en_corpus,
            pa_corpus=args.pa_corpus,
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
