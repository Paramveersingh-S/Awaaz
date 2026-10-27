# PHASE 1 — Text Normalization and Tokenizer

## 1. Goal
A single, deterministic, mathematically sound text pipeline used identically for training, evaluation, and edge inference across English and Gurmukhi Punjabi.

## 2. Technical Specifications
### 2.1 Text Normalization (`awaaz/text.py:normalize(text: str) -> Optional[str]`)
Rules applied in strictly deterministic order:
1. **Unicode NFC Normalization**: Apply `unicodedata.normalize('NFC', text)`. Note: NFC splits a few Gurmukhi letters (ਸ਼, ਖ਼, ਗ਼, ਜ਼, ਫ਼, ਲ਼) into base consonant + nukta (U+0A3C) due to Unicode composition exclusions. This must be applied uniformly everywhere.
2. **Lowercase**: Converts uppercase Latin characters to lowercase (affects English only).
3. **Digit Rejection**: Drop the entire utterance (return `None`) if it contains any digit (ASCII `0–9` or Gurmukhi digits `U+0A66–U+0A6F`). Number verbalization is explicitly out of scope for v1.
4. **Punctuation & Danda Handling**: Replace Gurmukhi danda (`U+0964`, `।`) and double danda (`U+0965`, `॥`) with a single ASCII space.
5. **Character Whitelist Filtering**: Strip all characters except:
   - English lower-case letters: `a–z`
   - Gurmukhi Unicode block: `\u0A00–\u0A7F`
   - Apostrophe: `'` (ASCII `U+0027`)
   - Space: ` ` (ASCII `U+0020`)
6. **Whitespace Consolidation**: Collapse all contiguous spaces into a single space; strip leading and trailing whitespace.
7. **Empty Rejection**: If the final string is empty, return `None`.

### 2.2 Tokenizer Training (`awaaz/text.py:train_tokenizer(...)`)
- **Algorithm**: SentencePiece unigram model.
- **Vocabulary Size**: Exactly `1024` tokens.
- **Character Coverage**: `1.0` (critical for Gurmukhi script fidelity).
- **Special IDs**:
  - `pad_id = 0` (reserved strictly as the **CTC Blank** token).
  - `unk_id = 1` (unknown token).
  - `bos_id = -1` (disabled).
  - `eos_id = -1` (disabled).
- **Normalization**: Identity normalization inside SentencePiece (normalization handled exclusively by `awaaz.text.normalize`).
- **Language Balance**: Input corpus must be balanced with an equal number of lines (e.g., up to 300k lines each) from English and Punjabi text sources to prevent English subwords from dominating the 1024 vocabulary slots.
- **CLI Command**: `python -m awaaz.text train-tokenizer --config configs/small.yaml --output_dir models/tokenizer`

## 3. Mandatory Tests & Acceptance Checks
- `normalize` handles mixed-case English, apostrophes, dandas, emojis, and punctuation stripping.
- Digits (ASCII and Gurmukhi) cause `normalize` to return `None`.
- Gurmukhi nukta characters are idempotent under repeated `normalize()` passes.
- Tokenizer round-trip test: for all normalized sentences, `decode(encode(text)) == text`.
- Token ID `0` is never emitted when encoding valid text.
- Tokenizer trained on fixture text executes end-to-end without warnings.

## 4. Human Step
Run tokenizer training on full corpora in Colab; commit the resulting `.model` and `.vocab` files (small binary/text files) to the repository.

## 5. Definition of Done
All unit tests pass; normalization rules and token mappings are documented in `docs/DECISIONS.md`.
