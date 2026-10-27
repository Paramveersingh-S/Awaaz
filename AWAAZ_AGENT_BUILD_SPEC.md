# AWAAZ — Agent Build Spec
**A phase-by-phase instruction document for a coding agent (Antigravity IDE) to build a small, open-source English + Punjabi speech-to-text system, plus speaker separation and audio smoothing, designed to later run on watches and other edge devices.**

> This file contains **instructions, not code.** The agent writes all code. Each phase is self-contained: goal, what to build, exact specs, how to verify, and what the human must do.
> "Awaaz" (ਆਵਾਜ਼) is a working name; rename freely.

---

## HOW TO USE THIS FILE (for the human)

1. Put this file in the repo as `docs/SPEC.md`.
2. Copy the **"Global Agent Rules"** section (§2) into your IDE's workspace rules / agent-instructions file (whatever Antigravity calls it) so it applies to every task.
3. Give the agent **one phase at a time**. Use this prompt each time:
   > "Read `docs/SPEC.md` §1–§3 and Phase N. First write a short plan and list any questions or risks. Wait for my OK. Then implement Phase N exactly as specified, run the acceptance checks, update `docs/PROGRESS.md`, and stop."
4. Review each phase against its **Definition of Done** before moving on. Don't skip phases; later ones depend on earlier ones.
5. Anything marked **HUMAN STEP** cannot be done by the agent (GPU training in Colab, accounts, tokens, recording audio). The agent must prepare everything so that step is a one-click run.

---

## 1. PROJECT CONTEXT (the agent must read this first)

### 1.1 Goal
Build **Awaaz**: a small automatic speech recognition (ASR) model, **trained from scratch by us**, that transcribes **English and Punjabi (Gurmukhi script)** — including sentences that mix both — and is small enough to run on edge devices (phones, watches, embedded Linux boards). Around the model, build a full project: voice activity detection, speaker separation/diarization ("who said what"), audio smoothing (denoising), a CLI, a demo, and an open-source release.

### 1.2 What "like Whisper" means here
Same end-user result (audio in → text out), **not** the same size or training scale. Whisper-scale training is out of reach. Awaaz is a 6M–55M-parameter streaming-capable model.

### 1.3 Hard constraints
| Constraint | Detail |
|---|---|
| Training compute | **Google Colab Pro** (GPU primary; TPU optional/advanced). Sessions disconnect → everything must be **resumable**. Colab compute units are limited → iterate small, run long jobs rarely |
| Development machine | **Assume a weak laptop (dual-core CPU, ~8 GB RAM, no GPU, slow disk).** All local tests must be tiny (CPU, seconds). **No local training runs and no large dataset downloads on the dev machine** |
| Open source | Code Apache-2.0. Every dataset and its license documented. Weights licensed only in a way the data licenses allow |
| Edge target | Final model must export to ONNX and int8, with a defined latency/size/RAM budget (§Phase 9) |
| Scripts | Gurmukhi only (not Shahmukhi). English + Gurmukhi in **one shared tokenizer** |
| Sacred text | Gurbani/religious recitation is handled by an explicit policy (§1.5) |

### 1.4 Fixed architecture decisions (do not change without asking the human)
- **Model**: streaming-capable **Conformer encoder + CTC head**, implemented in **PyTorch** by us (no importing a pretrained ASR model as the core).
- **Subsampling**: 2-D convolutional front, factor **4** (40 ms frames); factor 8 supported as an option for the tiniest preset.
- **Attention**: multi-head self-attention with **rotary positional embeddings (RoPE)**; no learned/absolute positional table.
- **Streaming**: **dynamic-chunk training** — the attention mask is randomly either full-context or *chunked* (a frame sees its own chunk plus a limited number of past chunks, never future chunks). The convolution module uses a **causal** depthwise convolution. One trained model serves both offline and streaming.
- **Conv module normalization**: LayerNorm (no BatchNorm) to avoid padding/batch-statistics problems and simplify export.
- **Tokens**: one **SentencePiece unigram** model, vocab **1024**, trained on a language-balanced English + Gurmukhi text corpus. Token id **0 is the CTC blank** (the pad piece); id 1 is unknown; no BOS/EOS.
- **Features**: 16 kHz mono; 80-bin log-mel; 25 ms window (400 samples), 10 ms hop (160), FFT 400, mel range 20–7600 Hz, log with a tiny floor; **global** mean/std normalization computed once from training data (never per-utterance — it breaks streaming and complicates on-device parity).
- **Decoding**: greedy CTC first; optional n-gram LM later.
- **Edge runtime**: ONNX (opset ≥ 17) → int8 via ONNX Runtime; others (Core ML, TFLite, ExecuTorch) later.

### 1.5 Sacred-text policy (implement, don't just document)
Public Punjabi audio contains kirtan, katha, and Gurbani recitation. A general model can mis-transcribe sacred text in subtle ways, and a plausible wrong transcript is worse than none. Therefore:
- Manifests carry an optional `domain` field. Clips identified as sacred/devotional are tagged `"sacred"`.
- **v1 excludes `sacred` clips from training, dev, and test** by default (a config flag controls it).
- No automatic correction, summarization, or rewriting of sacred text anywhere in the product.
- Model card and README state this limitation and that any future Gurbani support must be a separate, opt-in, human-reviewed path.

### 1.6 Model size presets (target; the agent prints exact counts)
| Preset | d_model | Blocks | Heads | FFN dim | ≈ Params | ≈ int8 size | Intended device |
|---|---|---|---|---|---|---|---|
| `tiny` | 144 | 12 | 4 | 576 | ~6M | ~6 MB | watches, low-end phones |
| `small` | 256 | 12 | 4 | 1024 | ~19M | ~19 MB | phones, embedded Linux |
| `base` | 384 | 16 | 6 | 1536 | ~55M | ~55 MB | phones/servers, best accuracy |

Conv kernel 15 for all. Dropout 0.1.

### 1.7 Repository layout (target)
```
awaaz/
├── README.md, LICENSE (Apache-2.0), DATA_LICENSES.md, pyproject.toml
├── AGENTS.md                       # global agent rules (copied from §2)
├── docs/  SPEC.md, PROGRESS.md, DECISIONS.md, MODEL_CARD.md
├── configs/  tiny.yaml, small.yaml, base.yaml, data_sources.yaml
├── awaaz/
│   ├── text.py  features.py  dataset.py  model.py  decode.py  metrics.py
│   ├── train.py  average.py  evaluate.py  stream.py  export.py
│   ├── pipeline/  vad.py  diarize.py  enhance.py  separate.py  merge.py
│   └── cli.py
├── scripts/  prepare_hf.py  merge_manifests.py  make_fixture.py  ...
├── notebooks/  colab_train.ipynb  colab_prepare_data.ipynb  colab_export.ipynb
├── tests/    (fast CPU tests; fixtures are tiny and synthetic)
├── app/      gradio demo
└── edge/     Android/Wear OS and Core ML integration notes + sample apps
```

---

## 2. GLOBAL AGENT RULES (paste into the IDE's agent rules)

1. **Plan first, then build.** For every task: short plan → wait for human OK → implement → verify → report.
2. **Never claim something works without running it.** Run the tests/commands in the phase's Acceptance Checks and paste the real output summary. If you can't run something (needs GPU/account), say so plainly and mark it HUMAN STEP.
3. **Never invent results.** No made-up WER numbers, dataset sizes, speeds, or citations. Unknown = "unknown, measure it."
4. **Weak dev machine.** No heavy downloads, no training, no big tensors locally. Use tiny synthetic fixtures (generated by a script, not committed as large files). Keep the whole local test suite under ~2 minutes on a dual-core CPU.
5. **Everything resumable and reproducible.** Fixed seeds, configs in YAML, no hard-coded paths (paths come from config/env), atomic file writes for checkpoints.
6. **Small, reviewable commits** — one logical change each, with a message that says what and why.
7. **Tests are mandatory** for model, features, text normalization, tokenizer round-trip, streaming causality, export parity. Do not delete or weaken a failing test to make it pass; fix the cause or ask.
8. **No unlicensed data.** Before adding any dataset, record its name, version, license, and access terms in `DATA_LICENSES.md`. If the license is unclear or non-commercial, stop and ask the human. Never scrape or download copyrighted media.
9. **No secrets in the repo.** Tokens (Hugging Face etc.) come from environment variables / Colab secrets. Never print them.
10. **Honor the sacred-text policy (§1.5)** in every data-touching component.
11. **Ask, don't guess,** when a spec is ambiguous or a dependency/API has changed. Prefer pinning versions and recording them in `docs/DECISIONS.md`.
12. **Keep `docs/PROGRESS.md` current**: after each phase, record what was built, what was verified, open issues, and the exact next step. This file is how work survives IDE session resets.
13. **Don't add features outside the current phase.** Note ideas in `docs/DECISIONS.md` instead.
14. **Prefer simple, readable code** over clever code; every public function gets a docstring stating shapes and units (seconds vs frames vs samples).
15. **Version-compatibility care.** PyTorch, torchaudio, `datasets`, and ONNX tooling change APIs often. Before using an API, check the installed version; read audio with `soundfile` (not `torchaudio.load`); if a dependency breaks, pin and document rather than hack around it.

---

## 3. PHASE MAP

| Phase | Name | Where it runs | Output |
|---|---|---|---|
| 0 | Repo bootstrap | local | skeleton, CI, docs, rules |
| 1 | Text normalization + tokenizer | local (tiny) → HUMAN STEP for real corpus | `text.py`, tokenizer model |
| 2 | Audio features + augmentation | local | `features.py`, CMVN tool |
| 3 | Data pipeline | local code; HUMAN STEP downloads on Colab | manifests, dataset, sampler |
| 4 | Model | local tests | `model.py` + causality tests |
| 5 | Training loop | local overfit on fixture | `train.py` |
| 6 | Colab runner + persistence | HUMAN STEP | notebooks |
| 7 | Evaluation + decoding | local + Colab | `evaluate.py`, reports |
| 8 | Streaming inference | local | `stream.py` |
| 9 | Export + quantization + edge budget | local + device | ONNX/int8, parity report |
| 10 | Full pipeline: VAD → ASR → diarization → enhancement → merge | local + Colab | `awaaz/pipeline/`, CLI |
| 11 | Separation + custom enhancer (v2) | Colab | optional models |
| 12 | Demo, docs, model card, release | local + HF | public repo/Space/weights |
| 13 | Edge apps | device | Wear OS / iOS integration |

---

# PHASE 0 — Repo bootstrap

**Goal:** a clean, testable skeleton so every later phase slots in.

**Build:**
- Create the layout in §1.7 with empty modules containing docstrings describing their future role.
- `pyproject.toml`: package `awaaz`, Python ≥ 3.10; core dependencies (torch, torchaudio, soundfile, sentencepiece, jiwer, pyyaml, huggingface_hub, numpy); optional extras `pipeline` (silero-vad, speechbrain, scikit-learn), `export` (onnx, onnxruntime), `dev` (pytest, ruff). Console script `awaaz` → `awaaz.cli:main`.
- `AGENTS.md` = §2 verbatim. `docs/SPEC.md` = this file. Create empty `docs/PROGRESS.md` and `docs/DECISIONS.md` with headings.
- `DATA_LICENSES.md` with a table (dataset, version, license, access terms, used for train/dev/test, date added).
- GitHub Actions workflow: install, lint, run the fast test suite on CPU.
- `scripts/make_fixture.py`: generates a **tiny synthetic ASR fixture** (a few dozen short 16 kHz FLAC clips of synthesized tones/noise with fake transcripts, in both English letters and Gurmukhi characters, plus manifest files). Used by every later test; nothing large committed.

**Acceptance checks:**
- `pip install -e .[dev]` succeeds; `pytest -q` runs (even if only placeholder tests); `awaaz --help` works.
- Running `scripts/make_fixture.py` produces the fixture in under 30 seconds.

**Definition of Done:** repo installs, CI is green, docs exist, fixture generator works.

---

# PHASE 1 — Text normalization and tokenizer

**Goal:** a single, deterministic text pipeline used identically for training, evaluation, and inference.

**Build `awaaz/text.py` with:**
1. **`normalize(text)`** — returns normalized text or "drop" (None). Rules, in order:
   - Unicode **NFC** normalization (note: NFC splits a few Gurmukhi letters like ਸ਼ ਖ਼ ਗ਼ ਜ਼ ਫ਼ ਲ਼ into base + nukta because they are composition-excluded; this is acceptable and must be applied *everywhere* the same way).
   - Lowercase (affects English only).
   - **Drop the utterance** if it contains any digit (ASCII 0–9 or Gurmukhi digits U+0A66–U+0A6F). Number verbalization is out of scope for v1.
   - Replace Gurmukhi danda/double danda (U+0964/U+0965) with a space.
   - Remove every character except: a–z, the Gurmukhi block U+0A00–U+0A7F, apostrophe, space.
   - Collapse whitespace; strip; return None if empty.
2. **`train_tokenizer(...)`** — builds the SentencePiece model: unigram, vocab 1024, character coverage 1.0, **pad id 0 (used as CTC blank), unk id 1, no BOS/EOS**, identity normalization (we normalize ourselves), shuffled sentences. The corpus is **balanced by language**: sample an equal number of lines (configurable, default up to 300k each) from English and Punjabi sources so English does not consume the vocabulary. Accepts extra text-only files (e.g., Punjabi Wikipedia or news text) tagged with a language.
3. A small CLI: `python -m awaaz.text train-tokenizer --config ...`.

**Tests (tiny, local):**
- `normalize` handles: mixed-case English, apostrophes, danda, digits (dropped), emojis/punctuation (stripped), empty results (dropped), Gurmukhi nukta letters (stable under repeated normalization — idempotent).
- Tokenizer round-trip: for every normalized fixture sentence, `decode(encode(x)) == x`. Id 0 is never produced by encoding real text.
- Tokenizer trained on the fixture text works end-to-end (this validates the code, not quality).

**HUMAN STEP (later, on Colab):** train the real tokenizer on the real corpora; save it to Drive and commit the `.model/.vocab` files to the repo (they are small).

**Pitfalls to flag in docs:** inconsistent normalization between train and eval silently ruins WER; Gurmukhi coverage must be 1.0; an unbalanced corpus produces a bad vocabulary.

**Definition of Done:** tests pass; `docs/DECISIONS.md` records the exact normalization rules.

---

# PHASE 2 — Audio features and augmentation

**Goal:** a feature pipeline that is deterministic, fast, and *re-implementable on-device*.

**Build `awaaz/features.py`:**
1. **Log-mel front end** (a torch module) with the exact parameters in §1.4, output shape (batch, frames, 80). Frames per utterance = samples // 160 + 1. Holds global mean/std as buffers; loads them from an `.npz`.
2. **CMVN tool**: computes global mean/std over a random subset (default 2000 utterances) of a manifest and saves `cmvn.npz`. Command-line runnable.
3. **SpecAugment** module: 2 frequency masks (width ≤ 27 bins) and 2 time masks (each up to 5% of the utterance length), applied only in training mode, only within valid (unpadded) frames; mask value is 0 (the mean after normalization).
4. **Waveform augmentations** (separate module, each independently switchable by config):
   - speed perturbation 0.9 / 1.0 / 1.1 (via resampling),
   - additive noise at random SNR 5–20 dB from a noise-file directory (optional; skipped gracefully if no directory configured),
   - room-impulse-response reverb (optional, same treatment),
   - random gain and mild band-limiting to imitate cheap microphones.

**Tests:**
- Output shapes and frame-count formula for several lengths (including very short clips).
- Normalization: with computed stats, features of the fixture have ≈ zero mean / unit variance per bin.
- SpecAugment never touches padded frames and is the identity in eval mode.
- Determinism with a fixed seed.

**Deliverable doc:** `docs/FEATURES.md` — a precise, language-neutral description of the feature algorithm (window type, padding mode, mel scale/norm, log floor, frame count rule), because the Android/iOS implementations must match it numerically.

**Definition of Done:** tests pass; `docs/FEATURES.md` complete.

---

# PHASE 3 — Data pipeline

**Goal:** turn open datasets into clean manifests and feed them efficiently to training.

### 3A. Manifest format (fixed)
JSON Lines, one object per clip: audio path, text, duration (seconds), language code (`en` or `pa`), optional `domain` (e.g. `sacred`), optional `source` (dataset name). All audio is **16 kHz, mono, FLAC**, duration 0.5–20 s.

### 3B. Dataset preparation script
Build `scripts/prepare_hf.py` (and a generic "prepare from local archive" variant) that:
- streams a Hugging Face audio dataset, converts each clip to 16 kHz mono FLAC, filters by duration, and appends to a manifest;
- requests **undecoded** audio and decodes with `soundfile` plus a resampler, to avoid torchaudio backend breakage;
- is **resumable** (skips already-converted clips) and tolerant of decode errors (logs and continues);
- records source and split per line;
- supports a `--max_items` flag so the agent can test on ~20 clips.
The agent must also create `configs/data_sources.yaml` listing candidate datasets with fields: name, HF id or URL, language, expected license, split names, text-column name, notes. **The agent must verify each entry's current availability and license by reading the dataset card, and write the result in `DATA_LICENSES.md`; entries it cannot verify stay marked "UNVERIFIED — human to confirm".**

Candidate sources (to verify, not to trust blindly):
- English: LibriSpeech, Common Voice (English; check its current distribution channel), optional larger sets (check licenses).
- Punjabi: FLEURS (Punjabi) — mainly dev/test; Common Voice (Punjabi); AI4Bharat Indic speech collections that include Punjabi (several are gated — human must accept terms and provide a token); permissively licensed pseudo-labeled audio (Phase 3E).
- Punjabi text-only (tokenizer + language model): Wikipedia and news/text corpora with clear licenses.
Note for the agent: newer `datasets` releases dropped script-based loaders. If a dataset fails to load, try a pinned older version or a raw-archive path, and record what worked.

### 3C. Normalization-aware filtering
When loading manifests for training, apply the Phase-1 normalizer; drop clips whose text normalizes to nothing/None; drop `sacred`-domain clips per §1.5; optionally drop clips whose token count exceeds what the encoder can emit (tokens must be fewer than encoder frames, with margin for repeated tokens).

### 3D. Dataset, collation, batching (`awaaz/dataset.py`)
- Dataset item: waveform (float32, 16 kHz), token ids, normalized text, language. Optional speed perturbation when training.
- Collate: pad waveforms to batch max; return waveform lengths; targets as one concatenated 1-D tensor plus per-utterance lengths (the CTC loss format); keep texts and languages for evaluation.
- **Duration-bucketed batch sampler**: sort by duration, fill each batch up to a maximum of *padded audio seconds* (config, e.g. 400 s), shuffle batch order each epoch with a seed + epoch; expose `set_epoch`.
- DataLoader settings suited to Colab: a few workers, pinned memory, persistent workers.

### 3E. Balancing and pseudo-labeling
- `scripts/merge_manifests.py`: combine manifests with a **repeat factor per source** and an optional hour cap for English; print resulting hours per language; target roughly 40–60% Punjabi *including repeats*, subject to having enough distinct Punjabi hours (report distinct vs repeated hours separately).
- Pseudo-labeling tool (optional now, important later): for unlabeled **permissively licensed** Punjabi audio, transcribe with a teacher model the human approves, keep only high-confidence or two-teacher-agreement clips, store them under a separate `source` tag, and **never** use them in dev/test. The agent must first measure candidate teachers on held-out real Punjabi data and report that number before using any teacher; no assumption that a general multilingual model is good at Punjabi.

### 3F. Data storage rules for Colab
- Archives live on Drive; at session start they are untarred to local disk; training reads local disk only. Provide a helper script that tars/untars a data directory with a checksum.

**Tests:**
- Fixture round-trip: manifest → dataset → batch → shapes/lengths consistent.
- Sampler: every example appears exactly once per epoch; no batch exceeds the padded-seconds budget; deterministic given seed/epoch.
- Dropping rules (digits, sacred, empty text, duration bounds) each covered.

**HUMAN STEP:** on Colab, run the preparation notebook to download/convert the real datasets (needs Hugging Face token for gated sets), build `train/dev_en/dev_pa/test` manifests, compute CMVN, train the real tokenizer. **Also** record ~100 test clips on the actual target device (watch/phone mic, realistic noise) with reference transcripts; this is the `test_device` set and is never trained on.

**Definition of Done:** tests pass; a Colab prep notebook exists and was dry-run with `--max_items`; `DATA_LICENSES.md` populated honestly.

---

# PHASE 4 — The model

**Goal:** implement Conformer-CTC to the §1.4 spec with correct streaming semantics.

**Build `awaaz/model.py` (a configuration object + modules):**
1. **Config** with fields for n_mels, d_model, layers, heads, FFN dim, conv kernel, dropout, subsample factor (4 or 8), vocab size, causal-conv flag. Presets from §1.6 loadable by name from YAML.
2. **Conv2d subsampling**: stacked stride-2 convolutions (count = log2 of factor), 32 channels, ReLU, then a linear projection to d_model. Also returns updated lengths using the exact rule output_len = (len − 1) // 2 + 1 per stride-2 stage.
3. **RoPE** applied to query/key inside attention (relative by construction, so sliding-window inference needs no position bookkeeping).
4. **Attention mask builder**: given valid lengths and optional `chunk` (in encoder frames) and `left_chunks` (−1 = unlimited): a frame may attend to valid keys in its own chunk and up to `left_chunks` previous chunks, never later chunks; `chunk=None` means full context. **Every query row must keep at least one allowed key** (e.g., always allow the diagonal) to avoid NaNs on padded positions.
5. **Conformer block** (macaron style): half-weight feed-forward → self-attention → convolution module → half-weight feed-forward → final LayerNorm, with residual connections and dropout.
6. **Convolution module**: LayerNorm → pointwise conv (2× width) → GLU → **causal** depthwise conv (kernel 15, left-padding only when causal flag set) → LayerNorm → SiLU → pointwise conv → dropout. Zero out padded frames before the depthwise conv.
7. **CTC head**: linear layer to vocab size; output log-probabilities and output lengths. Blank = id 0.
8. Helper to report exact parameter count.

**Tests (all CPU, tiny model sizes):**
- **Shapes/lengths**: for batches with different lengths, output lengths follow the subsampling rule; outputs finite.
- **Padding invariance**: the output for a clip is the same (within tolerance) whether it is alone or padded inside a larger batch.
- **Chunk-causality test (critical):** with a chunked mask and causal conv, changing input features from chunk k onward must not change encoder outputs for chunks before k (allow a margin of ~2 frames for the subsampling convolutions' small look-ahead). This test protects the whole streaming design — it must exist and pass.
- **Full-context equals chunk=None**: large chunk size behaves like full context.
- **Gradient check**: a backward pass on random data produces finite gradients for all parameters (no dead branches).
- **Parameter counts** for tiny/small/base are within ±15% of §1.6; record actual numbers in `docs/DECISIONS.md`.

**Definition of Done:** all tests pass on CPU in seconds; parameter counts recorded.

---

# PHASE 5 — Training loop

**Goal:** a resumable, mixed-precision, dynamic-chunk training script that survives Colab disconnects.

**Build `awaaz/train.py` (config-driven; command: `python -m awaaz.train --config configs/small.yaml`):**
- **Optimizer**: AdamW, betas (0.9, 0.98), eps 1e-9, weight decay 1e-3.
- **Schedule**: linear warmup (default 5000 steps) to peak LR (default 1e-3), then cosine decay to 1e-5 at `total_steps`.
- **Loss**: CTC with blank 0, `zero_infinity=True`, computed in float32 (cast log-probs up), summed and divided by batch size.
- **Precision**: bf16 autocast if the GPU supports it, otherwise fp16 with a gradient scaler; feature extraction stays fp32.
- **Gradient clipping** at 5.0; **skip the optimizer step if the gradient norm is non-finite**.
- **Dynamic chunk training**: each batch, with probability 0.6 use a chunked mask with chunk size sampled from {8, 16, 32} encoder frames and left-chunks from {2, 4, 8, unlimited}; otherwise full context.
- **SpecAugment + speed perturbation** on; waveform noise/reverb augmentation switchable.
- **Checkpointing**:
  - `last.pt` every N steps (default 500): model, optimizer, scaler, step, epoch, config — written **atomically** (write temp file, then rename).
  - Numbered model-only checkpoints every `eval_every` steps (default 2000); keep the newest K (default 12) for later averaging.
  - **Automatic resume** from `last.pt` if present, including restoring the epoch so the sampler shuffles consistently.
  - Optional mirroring of checkpoints to a Hugging Face model repo (failures must warn, never crash training).
- **Evaluation hook** every `eval_every` steps: greedy-decode each dev manifest *separately* (English, Punjabi) and log WER and CER per language.
- **Logging**: loss, learning rate, gradient norm, steps/sec, seconds of audio per sec, GPU memory; plain text plus a CSV/TensorBoard-compatible log in the run directory.
- **Config-derived planning printout** at start: parameter count, dataset hours per language, batches per epoch, and a note telling the human how to compute `total_steps` from measured steps/sec (total_steps = budget_hours × 3600 × steps_per_sec).
- Crash-safety: handle OOM by logging and skipping that batch once; if repeated, instruct the human to lower the batch-seconds setting.

**Tests/Checks (local, fixture-sized):**
- **Overfit test (mandatory):** a tiny model on ~30 fixture clips, dropout 0, no augmentation, a few hundred steps on CPU: loss must fall substantially and greedy decoding must reproduce the training transcripts. If it can't, the pipeline has a bug — find it before anything else.
- **Resume test:** run N steps, kill, restart; the step counter, LR, and loss continue seamlessly; the checkpoint file is never left corrupt if the process is killed mid-save.
- **Chunk sampling test:** both full-context and chunked batches occur over many steps.

**Common failure table to include in `docs/TRAINING.md`:**
| Symptom | First things to check |
|---|---|
| Loss stays flat from the start | LR too high/low, features not normalized (CMVN not loaded), frame-count rule mismatch |
| inf/NaN | targets longer than encoder output, silent/empty audio, fp16 overflow (use bf16 or keep zero_infinity) |
| Loss falls, decoded text is garbage | tokenizer/normalizer mismatch between train and eval |
| English fine, Punjabi stalls | Punjabi share too low/too little distinct data/unbalanced tokenizer corpus |
| GPU idle | data loading bottleneck: data not on local disk, too few workers, MP3 decoding |

**Definition of Done:** overfit, resume, and chunk-sampling checks pass locally; `docs/TRAINING.md` written.

---

# PHASE 6 — Colab runner and persistence

**Goal:** the human can start, stop, and resume the real training with a handful of clicks.

**Build three notebooks (thin: they call repo code, no logic of their own):**
1. `colab_prepare_data.ipynb` — mount Drive, read secrets (Hugging Face token) from Colab secrets, install the package, run the data preparation for each configured source, build manifests, tar to Drive, compute CMVN, train the tokenizer, print per-language hours (distinct vs repeated).
2. `colab_train.ipynb` — mount Drive, install the package, untar data to **local** disk, verify GPU type, auto-select a sensible `batch_seconds` for the detected GPU (T4/L4/A100 tiers; the human can override), print the planning summary, run training, and tell the human "re-run this cell to resume." Include a short measured-throughput cell that estimates `total_steps` for a stated compute budget.
3. `colab_export.ipynb` — average checkpoints, evaluate, export ONNX, quantize, run parity and WER checks, upload release artifacts.

**Rules:** no keep-alive hacks that violate Colab's terms; everything must be safe to stop at any time; outputs written to Drive, mirrored to the Hugging Face Hub when configured.

**TPU note (document in `docs/TPU.md`, do not implement unless the human asks):** PyTorch-on-TPU recompiles for each new tensor shape, so variable-length audio requires fixed-shape length buckets; CTC loss support is limited. The recommended TPU route is a separate **JAX/Flax port** of the same architecture with fixed-shape buckets. Until then, use GPUs; use Kaggle/Lightning free tiers as extra resumable GPU capacity via the same checkpoint scheme.

**HUMAN STEP:** run the notebooks; do a ~15-minute smoke run first (e.g. 500 steps on a small subset) and report steps/sec back so `total_steps` can be set.

**Definition of Done:** notebooks exist, were reviewed, and the human confirmed a smoke run resumed correctly after a deliberate stop.

---

# PHASE 7 — Evaluation and decoding

**Goal:** trustworthy numbers, reported per language and per condition.

**Build:**
1. **`awaaz/decode.py`**
   - Greedy CTC decoding: argmax per frame, collapse repeats, remove blank, detokenize.
   - **Word timestamps from CTC**: using the frame→time rule (frame duration = 10 ms × subsample factor), start a new word at each token whose piece begins with the SentencePiece word-boundary marker; word start = first emitting frame, end = last emitting frame + one frame. Documented as approximate (CTC spikes are often slightly late) but sufficient for speaker assignment.
   - Optional **beam search with an n-gram LM** behind a feature flag: build a pruned 3/4-gram word LM from normalized text (English + Punjabi) with KenLM; decode with a CTC beam-search library that supports it; tune LM weight and insertion bonus on **dev only**. The agent must read the decoder library's documentation on subword vocabularies before wiring it and note any adaptation needed.
2. **`awaaz/metrics.py`** — WER and CER, computed after the same normalization as training; handle empty references safely. Always report **both**; for Punjabi, treat CER as the primary number (spelling variation inflates WER).
3. **`awaaz/evaluate.py`** — given a checkpoint (or ONNX file), tokenizer, CMVN, and manifest(s), report per-language WER/CER for: **offline**, **streaming-mask chunk 16 / left 8 (≈0.64 s)**, **chunk 8 / left 4 (≈0.32 s)**. Also report real-time factor on CPU and the slowest/worst-WER utterances (top 20) with reference vs hypothesis to a text file for error analysis.
4. **`awaaz/average.py`** — average the weights of the last N numbered checkpoints (default 10) into a single checkpoint; verify by evaluating before/after.
5. **Reports**: `evaluate.py` writes a Markdown results table in the format below plus a JSON file, so the model card can embed it.

| Test set | Offline WER/CER | Stream 0.64 s | Stream 0.32 s | int8 offline |
|---|---|---|---|---|
| LibriSpeech test-clean / test-other | | | | |
| FLEURS English / Punjabi | | | | |
| Common Voice dev (en / pa) | | | | |
| Code-switch (Punjabi + English words) | | | | |
| **Device-recorded set** | | | | |

**Tests:** metrics on hand-computed examples; greedy decode on synthetic posteriors with known answers (repeats, blanks between repeats, leading/trailing blanks); timestamp extraction on synthetic posteriors; checkpoint averaging of two tiny checkpoints equals the arithmetic mean.

**HUMAN STEP:** run evaluation on real checkpoints in Colab and read the error-analysis file; decide the next data/augmentation priority from the failure patterns.

**Definition of Done:** tests pass; evaluation runs end-to-end on the fixture and produces the report files.

---

# PHASE 8 — Streaming inference

**Goal:** a recognizer that consumes live audio blocks and emits text incrementally, provably consistent with how the model was trained.

**Build `awaaz/stream.py`:**
- A `StreamingRecognizer` that accepts arbitrary-sized audio blocks (16 kHz mono float), buffers until a full **chunk** (default 16 encoder frames = 0.64 s) is available, and processes it.
- **v1 algorithm (rolling window):** keep a rolling audio buffer of at most (left_chunks + 1) chunks, always trimmed in whole-chunk multiples so chunk alignment is preserved; for each new chunk, recompute features and the encoder over the window using the same chunked mask used in training; take **only the newest chunk's frames**; apply CTC collapse using persistent state (the previous emitted token carries across chunks); append tokens to the running hypothesis. Document that the encoder produces one extra edge frame (from the feature front-end's centered padding), and show how the newest-chunk frame slice is computed.
- `reset()`, `accept(block)` → current text, `flush()` → pad the remainder to a full chunk and return the final text. `left_chunks` must be ≥ 0 for streaming.
- Optional **partial-result stability** feature: report which tokens are "final" (older than one chunk) vs "tentative".
- **v2 (later, only after v1 is verified):** cached streaming — per-layer attention key/value cache and convolution state — so cost per chunk is constant instead of proportional to the window length. Write the design in `docs/STREAMING.md` (what state each layer must carry, how RoPE offsets are handled, how ONNX inputs/outputs would carry the state) and implement only when the human asks.

**Tests:**
- **Equivalence test:** streaming output on a fixture clip equals the offline chunked-mask output (same tokens) within an agreed tolerance for the edge frames; document any systematic difference.
- Block-size independence: feeding the audio in 100 ms, 320 ms, or 1 s blocks yields the same text.
- `reset` fully clears state; `flush` handles empty and sub-chunk remainders.
- Latency accounting test: reports algorithmic latency = chunk duration + (if applicable) look-ahead frames.

**Definition of Done:** tests pass; `docs/STREAMING.md` explains v1 behavior, measured latency math, and the v2 design.

---

# PHASE 9 — Export, quantization, edge budget

**Goal:** a verified ONNX and int8 model with a documented size/latency/memory budget, and a plan for each device class.

**Build `awaaz/export.py` + `docs/EDGE.md`:**
1. **ONNX export** (opset ≥ 17) of the encoder + CTC head: inputs = features and lengths; outputs = log-probabilities and output lengths; dynamic batch and time axes. Export **two variants**: *offline* (full context) and *streaming-window* (fixed chunk/left-chunks baked in). Use the stable exporter path for the installed PyTorch version; record which exporter was used and why.
2. **Parity check:** compare ONNX Runtime outputs to PyTorch on random inputs of several lengths; report max absolute difference; fail the command if above a threshold (default 1e-3).
3. **Quantization:** int8 via ONNX Runtime dynamic quantization first; if accuracy drops too much, add static quantization with a calibration set drawn from training data, then (later) quantization-aware fine-tuning. **Never ship an int8 file without re-running the full WER/CER evaluation on it.**
4. **Budget report** (auto-generated): parameter count, file sizes (fp32/int8), estimated FLOPs/sec (≈ 2 × non-embedding params × encoder frames per second, plus attention), measured CPU real-time factor on the dev machine (noting it's a weak CPU, so it is a conservative upper bound), peak RAM during inference.
5. **Device matrix in `docs/EDGE.md`** (the agent fills in measured numbers only when measured; otherwise "not measured"):
   - Android / Wear OS: ONNX Runtime Mobile or a TFLite conversion; feature extraction must be re-implemented natively and matched to `docs/FEATURES.md`.
   - Apple Watch / iPhone: Core ML conversion; often best to run on the paired iPhone and relay text.
   - Embedded Linux (Raspberry Pi-class): ONNX Runtime on ARM.
   - Microcontrollers: **not viable** for open-vocabulary ASR at these sizes; document a separate tiny keyword-spotting model as the wake-up path if desired.
6. **Native feature-parity test kit**: generate reference WAV files and the exact expected feature matrices (saved as binary files) so mobile engineers can verify their feature extractor numerically (target: max absolute difference far below 0.01).
7. **Alternative-runtime note:** document, without implementing, that a k2/icefall Zipformer + sherpa-onnx path exists as a proven edge recipe to benchmark against.

**Tests:** export of a tiny model succeeds; parity under threshold for multiple lengths and both variants; quantized tiny model loads and runs; budget report generates.

**HUMAN STEP:** run the best checkpoint through `colab_export.ipynb`; copy the int8 file to a real phone/watch and measure latency/battery with the Phase 13 sample app.

**Definition of Done:** parity and quantization checks pass on the fixture model; `docs/EDGE.md` complete with all unmeasured fields clearly marked.

---

# PHASE 10 — Full pipeline: VAD → ASR → diarization → enhancement → transcript

**Goal:** an end-to-end tool that takes a recording and outputs a speaker-labelled, timestamped transcript, optionally with per-speaker cleaned audio.

**Where each stage runs (document this):**
| Stage | Watch | Phone / server |
|---|---|---|
| VAD | yes | yes |
| Awaaz ASR (streaming) | yes | yes |
| Denoising ("smoothing") | no (unless a tiny custom model is trained, Phase 11) | yes |
| Speaker embeddings + clustering | no | yes |
| Overlapped-speech separation | no | yes (v2) |

**Build `awaaz/pipeline/` (each stage behind a small interface so implementations are swappable):**
1. **`vad.py`** — speech segments from audio using a small pretrained VAD (e.g. Silero VAD; verify license and install on the current Python). Pad each segment by ~0.15 s; merge segments separated by gaps under ~0.4 s. Provide a splitter that cuts any segment longer than ~20 s into pieces — **at the quietest point near the cut**, not at fixed steps, so words aren't sliced (self-attention cost grows quadratically with length).
2. **ASR stage** — loads a checkpoint (or ONNX file), transcribes each VAD segment, adds segment start offsets to word times.
3. **`diarize.py`** — v1 method: cut VAD segments into ~1.5 s windows with ~0.75 s hop; compute speaker embeddings with a pretrained speaker model (e.g. ECAPA-TDNN family via SpeechBrain; verify license/version); L2-normalize; cluster with agglomerative clustering using cosine distance and a tunable distance threshold, or a fixed number of speakers when the user supplies it. Output speaker-labelled time windows. Make the threshold configurable and write a small **tuning script** that scores diarization error against labelled recordings so the human can tune on real data.
4. **`merge.py`** — assign each word the speaker whose window overlaps it most (fallback: nearest window by center); group consecutive same-speaker words (gap < ~1 s) into lines; output formats: plain text with timestamps, SRT/VTT subtitles, and JSON (words, times, speakers, confidences).
5. **`enhance.py`** — an `Enhancer` interface with a wrapper around a pretrained denoiser (e.g. DeepFilterNet; install may be version-sensitive, so make it optional and fail gracefully with a clear message). **Evaluation rule:** always measure WER with and without enhancement on dev data and report it. Denoisers can introduce artifacts that *hurt* ASR; the recommended default is noise-augmented ASR training with enhancement **off**, and enhancement used for listening quality (per-speaker "smoothed" audio export).
6. **Per-speaker track export** — after diarization, write one audio file per speaker (silence elsewhere) and optionally pass each through the enhancer.
7. **`awaaz/cli.py`** — `awaaz transcribe <audio> --ckpt ... --tokenizer ... --cmvn ... [--diarize] [--speakers N] [--enhance] [--format txt|srt|json] [--export-tracks DIR] [--device cpu|cuda]`; also `awaaz stream` (microphone or file simulating a live stream) and `awaaz evaluate`.

**Known limits to document honestly in the README:**
- Overlapped speech is not handled in v1 (the ASR was trained on single-speaker audio and diarization assigns one label per window). Plan: separation on overlapped regions (Phase 11) and later multi-talker training.
- Diarization quality depends heavily on microphone/room; the threshold needs tuning per use case.
- Word times from CTC are approximate.

**Tests (local, using the fixture and mocks):**
- Segment splitting never exceeds the max length and never produces empty segments.
- Merge logic on hand-made word/turn examples (overlaps, gaps, fallbacks, speaker switches mid-line).
- Output formatters (txt/SRT/VTT/JSON) produce valid files.
- Pipeline runs end-to-end on a synthetic two-"speaker" file (distinct tones/pitches) using a stub ASR and the real or mocked diarizer; the CLI exits cleanly.
- Optional-dependency absence (no DeepFilterNet/Silero/SpeechBrain) produces clear errors, not crashes.

**HUMAN STEP:** record two or three real 2-speaker conversations (English, Punjabi, and mixed) with ground-truth transcripts and speaker labels; use them to tune the diarization threshold and to compare enhancement on/off.

**Definition of Done:** CLI works end-to-end on fixtures; limits documented; tuning script exists.

---

# PHASE 11 — Optional v2 models: custom enhancer and speaker separation

**Only start after Phases 1–10 work and the human confirms.**

### 11A. Tiny custom speech enhancer (for on-device smoothing)
- **Goal:** a causal model of tens of thousands to a few hundred thousand parameters that denoises in real time on a phone and possibly a watch.
- **Data:** clean speech (our English + Punjabi corpora) mixed on the fly with noise (open noise sets such as MUSAN/DNS-type collections — verify licenses) and room impulse responses at SNR −5 to 20 dB; also mix in noise recorded from the watch's real environments.
- **Model family:** a lightweight causal time-frequency network of the GTCRN type (agent to read the paper/repo, verify its reported size, and decide whether to adapt it or write a simpler equivalent).
- **Loss:** compressed-spectrum magnitude/complex loss plus SI-SNR. **Success metric: downstream ASR WER**, not only PESQ/STOI.
- **Export** to ONNX/int8 and apply the same budget/parity process as Phase 9.

### 11B. Overlapped-speech separation
- **Detect overlap** (overlap detector or diarizer segmentation output).
- **Separate** only those regions with a pretrained separation model first (SepFormer/Conv-TasNet family; verify license), then ASR each output stream and merge by time.
- **Custom training (research-sized):** synthesize 2-speaker mixtures from our single-speaker data (random offsets, relative SNR ±5 dB, optional reverb), train with SI-SNR and permutation-invariant training; report SI-SNR improvement and ASR WER on separated streams.
- **Longer term:** multi-talker CTC with permutation-invariant training so overlapped speech is transcribed directly.

**Deliverables:** training configs, notebooks, evaluation scripts, and an A/B report (with vs without) using the Phase 7 reporting format.

---

# PHASE 12 — Demo, documentation, model card, release

**Goal:** a public, reproducible, honestly documented open-source release.

**Build/write:**
1. **Gradio demo** (`app/`) for a Hugging Face Space: audio upload and microphone input, toggle for speaker separation, language shown, transcript with timestamps/speakers, download as TXT/SRT/JSON. Model files are pulled from the Hub at startup. Explicit notice about limitations.
2. **README**: what it is, who it's for, quick start (install, transcribe a file, stream, run the demo), the results table, model sizes, supported languages/scripts (Gurmukhi only), limitations, roadmap, citation, license.
3. **`docs/MODEL_CARD.md`** (also pushed to the Hub): intended use; **out-of-scope use** (sacred-text transcription, legal/medical records, surveillance, identifying people); training data with licenses and the exclusion policy; evaluation tables including **failure cases** and per-language numbers; known biases (accents, age/gender, regional Punjabi dialects, code-switching, noise); hardware/latency/size numbers (only measured ones); how to reproduce; license; contact/issues.
4. **Reproducibility package:** exact configs, seeds, data manifest *recipes* (not the audio), package versions, commands to retrain, and the checkpoint-averaging recipe.
5. **Release artifacts on the Hugging Face Hub:** fp32 checkpoint, ONNX offline + streaming, int8 files, tokenizer, `cmvn.npz`, `config.yaml`, model card, evaluation JSON. Versioned tags (v0.1.0 …). Weights license chosen strictly from the data licenses (attribution for CC-BY; no NC data if allowing commercial use) and stated in the card.
6. **Repository hygiene:** CONTRIBUTING.md, CODE_OF_CONDUCT.md, issue templates (including a "Punjabi transcription error report" template asking for audio permission and the expected text), GitHub Actions for tests, a release checklist, and a security note for model files.
7. **Community data policy:** if accepting contributed recordings later, require explicit consent and a license, and respect the sacred-text policy.

**Tests/Checks:** fresh-environment install from the README commands works; demo launches locally against the fixture model; every number in the README/model card traces to a generated report file.

**Definition of Done:** a stranger can clone, install, transcribe a sample, and reproduce the evaluation table from the docs alone.

---

# PHASE 13 — Edge apps (instructions only; built after Phase 9 is proven)

**Goal:** prove the model runs in real time on real devices.

1. **Android / Wear OS sample:** Kotlin app that captures microphone audio at 16 kHz, runs a native implementation of the feature extractor (matching `docs/FEATURES.md`), feeds the streaming ONNX model through ONNX Runtime Mobile, applies greedy CTC with persistent state, and shows live text. Include a built-in **feature-parity self-test** that compares against the reference feature files from Phase 9. Measure and display: real-time factor, per-chunk latency, memory, and battery drain over a 10-minute session.
2. **iOS / watchOS sample:** Swift app using a Core ML conversion; same instrumentation. Document the practical option of running on the paired iPhone.
3. **Embedded Linux sample:** a minimal command-line streamer for ARM boards.
4. **Power strategy doc:** always-on VAD (tiny) wakes the ASR; ASR runs only during speech; batching/duty-cycling suggestions; thermal limits on watches.
5. **Decision gate:** if `tiny` int8 doesn't meet real-time and battery targets on the target watch, the agent proposes (and the human chooses between) subsample factor 8, fewer/narrower layers, distillation from a larger model, or offloading to the phone.

**HUMAN STEP:** build/run the sample apps on real hardware and report the measured numbers back into `docs/EDGE.md`.

---

# APPENDIX A — Roadmap after v1 (priority order)

1. Noise/reverb augmentation + fine-tuning on **device-recorded** audio (largest real-world gain).
2. Word n-gram language model (large Punjabi accuracy gain, cheap).
3. More distinct Punjabi hours (filtered pseudo-labeling from verified-good teachers).
4. Knowledge distillation from a larger bilingual teacher into `tiny`.
5. Cached streaming state (Phase 8 v2).
6. Quantization-aware training for int8.
7. RNN-T (transducer) head or CTC + attention rescoring for better streaming accuracy.
8. Per-segment language ID + script-aware post-processing.
9. Punctuation and casing restoration (separate tiny model).
10. Multi-talker (PIT-CTC) training.
11. JAX/Flax port for TPU-scale training.
12. Optional Shahmukhi (Arabic-script Punjabi) as a separate, clearly labeled project.

# APPENDIX B — Rules for what the agent must *not* do
- Do not train or evaluate on data that is not in a verified, licensed manifest.
- Do not use dev/test data (or pseudo-labeled data) for training or hyperparameter tuning beyond the dev split explicitly designated for it.
- Do not report numbers that are not produced by a script in the repo.
- Do not quietly change the architecture, vocab size, feature parameters, or normalization rules — those are contracts shared by training, evaluation, export, and the on-device code. If a change is truly needed, propose it in `docs/DECISIONS.md` and ask the human.
- Do not treat sacred text as ordinary training/eval data (§1.5).
- Do not hide failures: if a phase's acceptance check fails, say so, show the output, and propose the fix.

# APPENDIX C — Per-phase handoff template (agent fills this into `docs/PROGRESS.md` after each phase)
- **Phase / date**
- **What was built** (files, commands)
- **What was verified** (exact commands run, pass/fail summaries)
- **What was NOT verified** (and why)
- **Decisions made / deviations from the spec**
- **Open issues and risks**
- **HUMAN STEPS now required**
- **Exact next step**
