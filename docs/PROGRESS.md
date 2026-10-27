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
  - Proceed to Phase 2 (Audio Features and Augmentation).

---

## Phase 2: Audio Features and Augmentation
- **Date**: 2026-10-27
- **Status**: COMPLETED
- **What was built**:
  - `awaaz/features.py`:
    - `LogMelSpectrogram(nn.Module)`: 16,000 Hz, 80-bin triangular mel filterbanks (20–7600 Hz), 25 ms periodic Hann window (400 samples), 10 ms hop (160 samples), centered reflection padding, natural log with 1e-5 floor. Output tensor shape `(batch, frames, 80)` matching exact frame formula `samples // 160 + 1`. Registered buffers for global CMVN `mean` and `std` of shape `(80,)` with `load_cmvn` and `save_cmvn` support.
    - `compute_cmvn(...)`: CLI-runnable utility computing global mean and standard deviation vectors across sampled audio in JSONL manifests and exporting `cmvn.npz`.
    - `SpecAugment(nn.Module)`: Frequency masking (2 masks, max width 27) and time masking (2 masks, max width 5% of valid length). Mask fill value is 0.0 (global normalized mean). Applied only in training mode and restricted strictly to valid unpadded frames. Strict identity pass-through in eval mode.
    - `WaveformAugmenter`: Speed perturbation (0.9x, 1.0x, 1.1x via resampling), additive noise at random SNR 5–20 dB, room impulse response (RIR) convolution, random gain (±6 dB), and peak amplitude clamping.
    - CLI subcommand: `python -m awaaz.features compute-cmvn`.
  - `docs/FEATURES.md`: Language-neutral mathematical specification for Android (Kotlin/C++) and iOS (Swift) engineers detailing windowing, reflection padding, filterbank math, and numerical parity targets.
  - `tests/test_phase2.py`: 7 acceptance unit tests covering output shapes, frame count formula, CMVN computation, save/load persistence, SpecAugment masking and eval mode identity, determinism, waveform speed/gain, and spec documentation existence.
- **What was verified**:
  - `pytest -v tests/test_phase2.py`: 7/7 passed.
  - Entire test suite `pytest -q`: 22/22 passed in 6.44s.
  - `ruff check .`: all checks passed cleanly.
  - `python -m awaaz.features compute-cmvn --help` executed with code 0.
- **What was NOT verified**:
  - Computing CMVN on the multi-thousand-hour production dataset (Colab step).
- **Decisions made / deviations from spec**:
  - None; strictly implemented §1.4 and Phase 2 requirements.
- **Open issues and risks**:
  - None for Phase 2.
- **HUMAN STEPS now required**:
  - Compute CMVN on full training dataset on Colab (scheduled in Phase 6).
- **Exact next step**:
  - Proceed to Phase 3 (Data Pipeline).

---

## Phase 3: Data Pipeline
- **Date**: 2026-10-27
- **Status**: COMPLETED
- **What was built**:
  - `awaaz/dataset.py`:
    - `AwaazDataset(Dataset)`: Manifest loader supporting single or multiple JSONL paths. Enforces Phase-1 text normalization, automatically drops utterances with digits or empty text, filters out durations outside [0.5, 20.0]s, drops missing audio files gracefully, and enforces the Sacred-Text Policy (§1.5) by excluding `"domain": "sacred"` clips when `exclude_sacred=True`. Loads 16 kHz mono float32 waveforms with soundfile.
    - `ctc_collate_fn(batch)`: Pads audio waveforms with zeros to the maximum duration in batch, creates 1D concatenated target tensor and target lengths compatible with PyTorch `nn.CTCLoss`.
    - `DurationBucketedBatchSampler(Sampler)`: Groups audio utterances by duration to eliminate padding waste, restricts each batch to a user-defined padded seconds budget (e.g. 400s), provides deterministic epoch shuffling (`set_epoch(epoch)`), and guarantees every valid example appears exactly once per epoch.
  - `tests/test_phase3.py`: 5 comprehensive acceptance tests covering dataset loading, collation shapes and zero padding, duration bucketed sampler coverage and budget checks, PyTorch DataLoader integration, sacred text policy exclusion, and duration/digit filtering.
- **What was verified**:
  - `pytest -v tests/test_phase3.py`: 5/5 passed in 3.99s.
  - Entire test suite `pytest -q`: 27/27 passed in 9.33s.
  - `ruff check .`: all checks passed cleanly.
- **What was NOT verified**:
  - Full-scale multi-gigabyte dataset downloading and conversion (designated as HUMAN STEP on Colab in Phase 6).
- **Decisions made / deviations from spec**:
  - Calibrated batch sampler duration budget dynamically based on available hardware; ensured hardware safety on 4 GB VRAM devices (RTX 3050).
- **Open issues and risks**:
  - None for Phase 3.
- **HUMAN STEPS now required**:
  - Run `colab_prepare_data.ipynb` in Colab with Hugging Face access tokens for full corpus downloads (scheduled in Phase 6).
- **Exact next step**:
  - Proceed to Phase 4 (The Model: Conformer-CTC).

---

## Phase 4: The Model (Conformer-CTC)
- **Date**: 2026-10-27
- **Status**: COMPLETED
- **What was built**:
  - `awaaz/model.py`:
    - `ConformerConfig`: Dataclass supporting presets (`tiny`, `small`, `base`) and YAML loading.
    - `Conv2dSubsampling`: Factor 4 (and optional factor 8) 2D Conv subsampling front-end with exact length reduction formula $T_{\text{out}} = \lfloor (T_{\text{in}} - 1) / 2 \rfloor + 1$ applied twice.
    - `RotaryEmbedding (RoPE)`: Relative positional encoding applied directly to Query and Key tensors in multi-head attention.
    - `build_chunk_attention_mask`: Dynamic-chunk mask generator parameterized by `chunk_size` and `left_chunks` with strict future blocking and diagonal safety preventing softmax NaNs on padded rows.
    - `RelMultiHeadAttention`: Multi-head attention with RoPE and scaling.
    - `FeedForwardModule`: Macaron-style FFN with SiLU and LayerNorm ($0.5\times$ residual weighting).
    - `CausalConvModule`: LayerNorm $\to$ Pointwise Conv $\to$ GLU $\to$ Causal Depthwise Conv (kernel 15, left-padding $K-1=14$) $\to$ LayerNorm $\to$ SiLU $\to$ Pointwise Conv $\to$ Dropout. Padded frames masked to zero before depthwise conv.
    - `ConformerBlock`: Full Macaron block: FFN(0.5) $\to$ SelfAttention $\to$ ConvModule $\to$ FFN(0.5) $\to$ LayerNorm.
    - `ConformerEncoder`: Stack of $N$ Conformer blocks.
    - `CTCHead`: Linear projection from $d_{\text{model}} \to 1024$ emitting `log_softmax` probabilities.
    - `ConformerCTC`: Full end-to-end model with `forward()` and `count_parameters()`.
  - `docs/DECISIONS.md`: Recorded ADR-008 documenting exact measured parameter counts:
    - `tiny`: 6,051,248 params (+0.85% from 6M target)
    - `small`: 18,663,008 params (-1.77% from 19M target)
    - `base`: 55,190,240 params (+0.35% from 55M target)
  - `tests/test_phase4.py`: 6 acceptance tests covering shapes and subsampling formula, padding invariance (< 1e-4 diff), **chunk-causality verification** (< 1e-5 diff), gradient flow on all parameters, parameter count checks, and attention mask non-NaN properties.
- **What was verified**:
  - `pytest -v tests/test_phase4.py`: 6/6 passed in 4.54s.
  - Entire test suite `pytest -q`: 33/33 passed in 7.89s.
  - `ruff check .`: all checks passed cleanly.
- **What was NOT verified**:
  - Convergence on full 500-hour training dataset (scheduled for Phase 5/6).
- **Decisions made / deviations from spec**:
  - Preserved self-diagonal for padded rows in dynamic chunk mask to prevent softmax NaNs without corrupting valid frame outputs.
- **Open issues and risks**:
  - None for Phase 4.
- **HUMAN STEPS now required**:
  - None for Phase 4.
- **Exact next step**:
  - Commit Phase 4 and push to `origin main`.
  - Proceed to Phase 5 (Training Loop).

---

## Phase 5: Training Loop
- **Date**: 2026-10-27
- **Status**: COMPLETED
- **What was built**:
  - `awaaz/train.py`:
    - Full training loop supporting YAML configuration (`python -m awaaz.train --config ...`).
    - AdamW optimizer ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$, weight decay $10^{-3}$) with gradient clipping at 5.0 and non-finite gradient skipping.
    - Linear warmup + Cosine decay schedule (`build_lr_scheduler`).
    - CTCLoss with blank index 0, zero_infinity=True, and float32 log-prob upcasting.
    - Dynamic-chunk parameter sampling (`sample_chunk_params`) with prob 0.6 chunked ({8, 16, 32} frames, {2, 4, 8, unlimited} left chunks) and prob 0.4 full context.
    - Atomic checkpointing (`save_checkpoint` to `last.pt` via temporary file and `os.replace`).
    - Periodic numbered checkpoints (`step_XXXXXXX.pt`) with rolling history retention (keeps newest $K=12$).
    - Automatic state resumption (`load_checkpoint` restoring model, optimizer, scaler, step, and epoch).
    - Config-derived planning printout (trainable parameters, dataset hours per language, batches per epoch, throughput formula).
    - Periodic evaluation hook computing validation WER and CER per language.
    - Structured CSV telemetry (`train_log.csv`) and graceful CUDA OOM recovery.
  - `docs/TRAINING.md`:
    - Hardware budgeting and VRAM allocation guide (CPU, RTX 3050 4GB, T4 16GB, L4 24GB, A100 40/80GB).
    - Budget hours to `total_steps` calculation formula (`total_steps = budget_hours * 3600 * steps_per_sec`).
    - Common failure symptom diagnosis table covering flat loss, inf/NaN loss, CTC garbage decode, language imbalance, low GPU utilization, and CUDA OOM.
  - `tests/test_phase5.py`:
    - 6 unit/integration tests: LR scheduler warmup/cosine, dynamic chunk distribution, atomic checkpoint save/resume, greedy CTC decode, fixture overfit convergence (>70% loss drop in 75 steps), and `docs/TRAINING.md` validation.
- **What was verified**:
  - `pytest -v tests/test_phase5.py`: 6/6 passed in 9.98s.
  - Full test suite `pytest -v tests/`: 39/39 passed in 14.99s.
  - `ruff check .`: all checks passed cleanly.
- **What was NOT verified**:
  - Multi-day 100,000-step training run on real cloud GPU (HUMAN STEP for Phase 6).
- **Decisions made / deviations from spec**:
  - Added helper `hours_per_language()` to `AwaazDataset` to power the pre-training dataset balance report.
- **Open issues and risks**:
  - None for Phase 5.
- **HUMAN STEPS now required**:
  - None for Phase 5.
- **Exact next step**:
  - Commit Phase 5 and push to `origin main`.
  - Proceed to Phase 6 (Colab Runner & Persistence).

---

## Phase Status Summary Table

| Phase | Title | Status | Completion Date |
|---|---|---|---|
| Phase 0 | Repo Bootstrap | COMPLETED | 2026-10-27 |
| Phase 1 | Text Normalization & Tokenizer | COMPLETED | 2026-10-27 |
| Phase 2 | Audio Features & Augmentation | COMPLETED | 2026-10-27 |
| Phase 3 | Data Pipeline | COMPLETED | 2026-10-27 |
| Phase 4 | The Model (Conformer-CTC) | COMPLETED | 2026-10-27 |
| Phase 5 | Training Loop | COMPLETED | 2026-10-27 |
| Phase 6 | Colab Runner & Persistence | QUEUED | - |
| Phase 7 | Evaluation & Decoding | QUEUED | - |
| Phase 8 | Streaming Inference | QUEUED | - |
| Phase 9 | Export, Quantization & Edge Budget | QUEUED | - |
| Phase 10 | Full Pipeline (VAD, Diarization, Enhance) | QUEUED | - |
| Phase 11 | Optional v2 Models (Enhancer & Separation) | QUEUED | - |
| Phase 12 | Demo, Docs, Model Card & Release | QUEUED | - |
| Phase 13 | Edge Applications | QUEUED | - |
