# Awaaz Development Progress Log

This document records the exact chronological status of Awaaz development across all 14 phases according to Appendix C of `docs/SPEC.md`.

---

## Phase 0: Repo Bootstrap
- **Date**: 2026-10-27
- **Status**: COMPLETED
- **What was built**:
  - Modularized `docs/SPEC.md` into 14 distinct phase documents under `docs/phases/`.
  - `AGENTS.md` containing global workspace rules from §2.
  - `pyproject.toml` with packaging configuration and entrypoints (`awaaz = awaaz.cli:main`).
  - `DATA_LICENSES.md` with candidate dataset licensing audit.
  - `LICENSE` (Apache-2.0).
  - Enterprise `README.md` with badges, architecture diagrams, model preset table, quickstart, and sacred text policy.
  - Model presets configs (`configs/tiny.yaml`, `configs/small.yaml`, `configs/base.yaml`, `configs/data_sources.yaml`).
  - Modular architecture skeletons across `awaaz/` and `awaaz/pipeline/`.
  - `scripts/make_fixture.py` for synthetic CPU fixture generation in under 4 seconds.
  - GitHub Actions CI workflow `.github/workflows/ci.yml`.
  - Comprehensive unit test suite in `tests/test_phase0.py`.
- **What was verified**:
  - `pip install -e .` succeeded with package entry point registered.
  - `awaaz --help` and `awaaz --version` ran cleanly.
  - `scripts/make_fixture.py` generated 16 kHz FLAC audio and valid JSONL manifests in under 4 seconds.
  - `pytest -v` ran all 6 tests in `tests/test_phase0.py` and passed (100%).
  - `ruff check .` ran across the entire repository with all checks passing.
- **What was NOT verified**:
  - Real multi-hour GPU training runs on Colab (designated as Phase 5/6 HUMAN STEP).
- **Decisions made / deviations from spec**:
  - Configured Windows terminal UTF-8 output handling in `awaaz/cli.py` to prevent cp1252 charmap encoding errors.
- **Open issues and risks**:
  - None for Phase 0.
- **HUMAN STEPS now required**:
  - None for Phase 0.
- **Exact next step**:
  - Proceed to Phase 1 (Text Normalization and Tokenizer).

---

## Phase 1: Text Normalization and Tokenizer
- **Date**: 2026-10-27
- **Status**: COMPLETED
- **What was built**:
  - `awaaz/text.py`:
    - `normalize(text: Optional[str]) -> Optional[str]` implementing deterministic NFC Unicode normalization, lowercase, digit rejection (0-9 and Gurmukhi ੦-੯), curly-quote conversion, danda/double-danda substitution, character whitelist filtering, and whitespace collapsing.
    - `Tokenizer` wrapper around `SentencePieceProcessor` with `encode`, `decode`, `encode_as_pieces`, `piece_to_id`, `id_to_piece`, `pad_id` (0), `blank_id` (0), `unk_id` (1).
    - `train_tokenizer(...)` supporting JSONL manifests, text files, and text pairs with language balancing (50% English / 50% Punjabi) and SentencePiece unigram training (vocab 1024, character coverage 1.0, pad_id 0, unk_id 1, identity normalization).
    - CLI subcommand: `python -m awaaz.text train-tokenizer`.
  - `tests/test_phase1.py`:
    - 9 comprehensive unit tests covering case normalization, apostrophe handling, danda replacement, digit rejection, emoji/punctuation filtering, empty string handling, nukta idempotency, SentencePiece training, round-trip equivalence, and language balance.
  - `docs/DECISIONS.md`: Added ADR-007 detailing exact normalization rules and SentencePiece parameters.
- **What was verified**:
  - `pytest -v tests/test_phase1.py`: 9/9 passed in 0.46s.
  - Entire test suite `pytest -v tests/`: 15/15 passed in 0.53s.
  - `ruff check .`: all checks passed cleanly.
  - `python -m awaaz.text train-tokenizer --help` executed with code 0.
- **What was NOT verified**:
  - Training the final production SentencePiece model on the multi-gigabyte Colab corpus (HUMAN STEP in Phase 6).
- **Decisions made / deviations from spec**:
  - Mapped typographic curly apostrophes (`’`, `‘`) to ASCII apostrophe `'` before filtering to prevent breaking contractions.
- **Open issues and risks**:
  - None for Phase 1.
- **HUMAN STEPS now required**:
  - Colab execution for real corpus tokenizer training (scheduled in Phase 6).
- **Exact next step**:
  - Commit Phase 1 and push to `origin main`.
  - Proceed to Phase 2 (Audio Features and Augmentation).

---

## Phase Status Summary Table

| Phase | Title | Status | Completion Date |
|---|---|---|---|
| Phase 0 | Repo Bootstrap | COMPLETED | 2026-10-27 |
| Phase 1 | Text Normalization & Tokenizer | COMPLETED | 2026-10-27 |
| Phase 2 | Audio Features & Augmentation | QUEUED | - |
| Phase 3 | Data Pipeline | QUEUED | - |
| Phase 4 | The Model (Conformer-CTC) | QUEUED | - |
| Phase 5 | Training Loop | QUEUED | - |
| Phase 6 | Colab Runner & Persistence | QUEUED | - |
| Phase 7 | Evaluation & Decoding | QUEUED | - |
| Phase 8 | Streaming Inference | QUEUED | - |
| Phase 9 | Export, Quantization & Edge Budget | QUEUED | - |
| Phase 10 | Full Pipeline (VAD, Diarization, Enhance) | QUEUED | - |
| Phase 11 | Optional v2 Models (Enhancer & Separation) | QUEUED | - |
| Phase 12 | Demo, Docs, Model Card & Release | QUEUED | - |
| Phase 13 | Edge Applications | QUEUED | - |
