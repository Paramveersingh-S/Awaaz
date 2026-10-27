"""Phase 1 Acceptance Tests — Text Normalization and SentencePiece Tokenizer."""

import json
from pathlib import Path

from awaaz.text import Tokenizer, normalize, train_tokenizer
from scripts.make_fixture import ENGLISH_SENTENCES, PUNJABI_SENTENCES, create_fixture


def test_normalize_mixed_case():
    """Verify English uppercase is converted to lowercase."""
    assert normalize("Hello WORLD") == "hello world"
    assert normalize("SPEECH Recognition") == "speech recognition"


def test_normalize_apostrophe():
    """Verify standard ASCII and typographic apostrophes are preserved as ASCII apostrophe."""
    assert normalize("don't stop") == "don't stop"
    assert normalize("don’t stop") == "don't stop"  # Right single quotation mark U+2019
    assert normalize("‘awaaz’") == "'awaaz'"


def test_normalize_danda():
    """Verify Gurmukhi danda (।) and double danda (॥) are replaced with a single space."""
    assert normalize("ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ।") == "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ"
    assert normalize("ਪਹਿਲਾ ਵਾਕ। ਦੂਜਾ ਵਾਕ॥") == "ਪਹਿਲਾ ਵਾਕ ਦੂਜਾ ਵਾਕ"


def test_normalize_reject_digits():
    """Verify any utterance containing ASCII (0-9) or Gurmukhi (੦-੯) digits is dropped (None)."""
    # ASCII digits
    assert normalize("flight 101") is None
    assert normalize("number 7 is lucky") is None
    assert normalize("2026") is None

    # Gurmukhi digits (U+0A66 to U+0A6F)
    assert normalize("ਕਮਰਾ ਨੰਬਰ ੧੨") is None
    assert normalize("੩੪੫") is None
    assert normalize("ਗੁਰਮੁਖੀ ੦੧੨੩੪੫੬੭੮੯") is None


def test_normalize_punctuation_and_emojis():
    """Verify non-whitelisted punctuation and emojis are stripped cleanly."""
    assert normalize("Hello! 🎉 How are you?") == "hello how are you"
    assert normalize("audio (high quality, mono); clean...") == "audio high quality mono clean"
    assert normalize("ਪੰਜਾਬੀ! ਬੋਲੀ? ਬਹੁਤ ਮਿੱਠੀ ਹੈ।") == "ਪੰਜਾਬੀ ਬੋਲੀ ਬਹੁਤ ਮਿੱਠੀ ਹੈ"


def test_normalize_empty_and_whitespace():
    """Verify empty strings, whitespace, and punctuation-only strings return None."""
    assert normalize("") is None
    assert normalize("   ") is None
    assert normalize(None) is None
    assert normalize("!@#$%^&*()_+{}[]:;<>/?") is None


def test_normalize_nukta_idempotency():
    """Verify Gurmukhi nukta letters (ਸ਼, ਖ਼, ਗ਼, ਜ਼, ਫ਼, ਲ਼) are stable under repeated normalization."""
    nukta_text = "ਸ਼ੇਰ ਖ਼ਬਰ ਗ਼ਜ਼ਲ ਜ਼ਿੰਦਗੀ ਫ਼ਿਲਮ ਲ਼ੱਸੀ"
    norm1 = normalize(nukta_text)
    assert norm1 is not None

    norm2 = normalize(norm1)
    norm3 = normalize(norm2)

    assert norm1 == norm2
    assert norm2 == norm3


def test_tokenizer_training_and_roundtrip(tmp_path):
    """Verify SentencePiece training, vocabulary properties, and round-trip consistency."""
    # Generate temporary fixture manifests
    train_m, dev_m = create_fixture(str(tmp_path), num_train=20, num_dev=6)

    model_prefix = tmp_path / "test_tokenizer"
    model_path = train_tokenizer(
        output_prefix=model_prefix,
        manifests=[train_m, dev_m],
        vocab_size=256,
        character_coverage=1.0,
    )

    assert Path(model_path).is_file()
    assert (tmp_path / "test_tokenizer.vocab").is_file()

    # Load tokenizer
    tokenizer = Tokenizer(model_path)
    assert tokenizer.pad_id == 0
    assert tokenizer.blank_id == 0
    assert tokenizer.unk_id == 1
    assert tokenizer.vocab_size > 0

    # Test round-trip on all fixture sentences
    with open(train_m, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    for item in items:
        norm = normalize(item["text"])
        if norm:
            encoded_ids = tokenizer.encode(norm)
            # Blank token (0) must NEVER be produced by encoding real text
            assert 0 not in encoded_ids, f"Blank ID 0 found in encoding of '{norm}'"

            decoded_text = tokenizer.decode(encoded_ids)
            assert decoded_text == norm, (
                f"Round-trip mismatch!\nOriginal:  '{norm}'\nDecoded:   '{decoded_text}'"
            )


def test_tokenizer_language_balance(tmp_path):
    """Verify tokenizer encodes both English and Punjabi words effectively."""
    model_prefix = tmp_path / "balanced_tokenizer"
    en_pairs = [(s, "en") for s in ENGLISH_SENTENCES]
    pa_pairs = [(s, "pa") for s in PUNJABI_SENTENCES]

    model_path = train_tokenizer(
        output_prefix=model_prefix,
        text_pairs=en_pairs + pa_pairs,
        vocab_size=200,
        character_coverage=1.0,
    )

    tokenizer = Tokenizer(model_path)

    # English sentence
    en_sample = normalize("hello world this is an awaaz test")
    en_ids = tokenizer.encode(en_sample)
    assert len(en_ids) > 0
    assert 0 not in en_ids
    assert tokenizer.decode(en_ids) == en_sample

    # Punjabi sentence
    pa_sample = normalize("ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਦੋਸਤੋ")
    pa_ids = tokenizer.encode(pa_sample)
    assert len(pa_ids) > 0
    assert 0 not in pa_ids
    assert tokenizer.decode(pa_ids) == pa_sample
