# AWAAZ — Agent Build Spec
**A phase-by-phase instruction document for a coding agent (Antigravity IDE) to build a small, open-source English + Punjabi speech-to-text system, plus speaker separation and audio smoothing, designed to later run on watches and other edge devices.**

> This file contains **instructions, not code.** The agent writes all code. Each phase is self-contained: goal, what to build, exact specs, how to verify, and what the human must do.
> "Awaaz" (ਆਵਾਜ਼) is an open-source bilingual ASR & voice intelligence system.

---

## HOW TO USE THIS SPECIFICATION

1. This file is the primary architecture document in `docs/SPEC.md`.
2. The **"Global Agent Rules"** section (§2) is copied verbatim into `AGENTS.md` and applies to every single task.
3. Every individual phase has been modularized into its own dedicated specification file under `docs/phases/`:
   - [Phase 0: Repo Bootstrap](phases/PHASE_00_REPO_BOOTSTRAP.md)
   - [Phase 1: Text Normalization and Tokenizer](phases/PHASE_01_TEXT_NORMALIZATION_TOKENIZER.md)
   - [Phase 2: Audio Features and Augmentation](phases/PHASE_02_AUDIO_FEATURES_AUGMENTATION.md)
   - [Phase 3: Data Pipeline](phases/PHASE_03_DATA_PIPELINE.md)
   - [Phase 4: The Model](phases/PHASE_04_MODEL.md)
   - [Phase 5: Training Loop](phases/PHASE_05_TRAINING_LOOP.md)
   - [Phase 6: Colab Runner and Persistence](phases/PHASE_06_COLAB_RUNNER_PERSISTENCE.md)
   - [Phase 7: Evaluation and Decoding](phases/PHASE_07_EVALUATION_DECODING.md)
   - [Phase 8: Streaming Inference](phases/PHASE_08_STREAMING_INFERENCE.md)
   - [Phase 9: Export, Quantization, and Edge Budget](phases/PHASE_09_EXPORT_QUANTIZATION_EDGE.md)
   - [Phase 10: Full Pipeline (VAD → ASR → Diarization → Enhancement → Merge)](phases/PHASE_10_FULL_PIPELINE.md)
   - [Phase 11: Optional v2 Models (Custom Enhancer and Speaker Separation)](phases/PHASE_11_OPTIONAL_V2_MODELS.md)
   - [Phase 12: Demo, Documentation, Model Card, and Release](phases/PHASE_12_DEMO_DOCS_RELEASE.md)
   - [Phase 13: Edge Applications](phases/PHASE_13_EDGE_APPS.md)
4. Work proceeds strictly one phase at a time. Review each phase against its **Definition of Done** before proceeding.
5. Anything marked **HUMAN STEP** cannot be executed by the local agent (e.g. multi-hour GPU training runs, physical device recording). The agent prepares fully automated, one-click artifacts for these steps.

---

## 1. PROJECT CONTEXT

### 1.1 Goal
Build **Awaaz**: a small automatic speech recognition (ASR) model, **trained from scratch by us**, that transcribes **English and Punjabi (Gurmukhi script)** — including sentences that mix both (code-switching) — and is small enough to run on edge devices (smartphones, smartwatches, embedded Linux boards). Around the model, build a complete production pipeline: voice activity detection (VAD), speaker separation/diarization ("who said what"), audio smoothing (denoising), a CLI, a web demo, and an open-source release.

### 1.2 What "like Whisper" means here
Same end-user result (audio in → text out), **not** the same parameter scale or training compute. Whisper-scale training is out of reach. Awaaz is a 6M–55M-parameter streaming-capable Conformer-CTC model.

### 1.3 Hard constraints
| Constraint | Detail |
|---|---|
| Training compute | **Google Colab Pro** (GPU primary; TPU optional/advanced). Sessions disconnect → everything must be **resumable**. Colab compute units are limited → iterate small, run long jobs rarely |
| Development machine | **Assume a weak laptop (dual-core CPU, ~8 GB RAM, no GPU, slow disk).** All local tests must be tiny (CPU, seconds). **No local training runs and no large dataset downloads on the dev machine** |
| Open source | Code Apache-2.0. Every dataset and its license documented in `DATA_LICENSES.md`. Weights licensed only in a way the data licenses allow |
| Edge target | Final model must export to ONNX and int8, with a defined latency/size/RAM budget (§Phase 9) |
| Scripts | Gurmukhi only (not Shahmukhi). English + Gurmukhi in **one shared SentencePiece tokenizer** |
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

---

## 2. GLOBAL AGENT RULES
*(See `AGENTS.md` for the official verbatim workspace rules)*

1. Plan first, then build.
2. Never claim something works without running it.
3. Never invent results.
4. Weak dev machine: CPU-only tests, fast synthetic fixtures.
5. Everything resumable and reproducible.
6. Small, reviewable commits with professional messages.
7. Tests are mandatory.
8. No unlicensed data.
9. No secrets in the repo.
10. Honor the sacred-text policy (§1.5).
11. Ask, don't guess.
12. Keep `docs/PROGRESS.md` current.
13. Don't add features outside the current phase.
14. Prefer simple, readable code with docstrings for shapes and units.
15. Version-compatibility care.

---

## 3. PHASE MAP SUMMARY

| Phase | Name | Focus | Output |
|---|---|---|---|
| [Phase 0](phases/PHASE_00_REPO_BOOTSTRAP.md) | Repo bootstrap | Local skeleton & tests | Skeleton, CI, packaging, synthetic fixture generator |
| [Phase 1](phases/PHASE_01_TEXT_NORMALIZATION_TOKENIZER.md) | Text normalization & tokenizer | Gurmukhi + English handling | `text.py`, 1024-vocab SentencePiece tokenizer |
| [Phase 2](phases/PHASE_02_AUDIO_FEATURES_AUGMENTATION.md) | Audio features & augmentation | 80-bin Mel, CMVN, SpecAugment | `features.py`, `docs/FEATURES.md` |
| [Phase 3](phases/PHASE_03_DATA_PIPELINE.md) | Data pipeline | JSONL manifests & duration bucketing | `dataset.py`, `scripts/prepare_hf.py` |
| [Phase 4](phases/PHASE_04_MODEL.md) | The model | Streaming Conformer-CTC with RoPE | `model.py`, causality verification tests |
| [Phase 5](phases/PHASE_05_TRAINING_LOOP.md) | Training loop | Dynamic-chunk resumable trainer | `train.py`, overfit verification |
| [Phase 6](phases/PHASE_06_COLAB_RUNNER_PERSISTENCE.md) | Colab runner & persistence | Cloud GPU execution notebooks | Colab notebooks, Drive sync |
| [Phase 7](phases/PHASE_07_EVALUATION_DECODING.md) | Evaluation & decoding | WER/CER metrics, CTC timestamps | `decode.py`, `evaluate.py`, `average.py` |
| [Phase 8](phases/PHASE_08_STREAMING_INFERENCE.md) | Streaming inference | Real-time incremental recognizer | `stream.py`, `docs/STREAMING.md` |
| [Phase 9](phases/PHASE_09_EXPORT_QUANTIZATION_EDGE.md) | Export, quantization & edge budget | ONNX, int8 quantization, parity | `export.py`, `docs/EDGE.md` |
| [Phase 10](phases/PHASE_10_FULL_PIPELINE.md) | Full pipeline | VAD, diarization, denoising, CLI | `awaaz/pipeline/`, `awaaz/cli.py` |
| [Phase 11](phases/PHASE_11_OPTIONAL_V2_MODELS.md) | Optional v2 models | Tiny custom enhancer & separation | Research models & benchmarks |
| [Phase 12](phases/PHASE_12_DEMO_DOCS_RELEASE.md) | Demo, docs & release | Hugging Face Spaces & Model Card | `app/app.py`, `docs/MODEL_CARD.md`, releases |
| [Phase 13](phases/PHASE_13_EDGE_APPS.md) | Edge applications | Mobile/Wear OS deployment | Native sample applications |

---

## 4. APPENDICES

### Appendix A — Roadmap After v1
1. Noise/reverb augmentation + fine-tuning on device-recorded audio.
2. Word n-gram language model for Gurmukhi accuracy boost.
3. Filtered pseudo-labeling with verified high-confidence teacher models.
4. Student-teacher knowledge distillation into `tiny` preset.
5. Cached streaming state (KV-cache and convolution FIFO).
6. Quantization-aware training (QAT).
7. Transducer (RNN-T) or CTC + attention rescoring.
8. Punctuation and capitalization restoration.
9. Multi-talker PIT-CTC training.
10. JAX/Flax TPU port.

### Appendix B — Negative Constraints (What Agent Must NOT Do)
- Do not train or evaluate on unlicensed or unverified data.
- Do not leak dev/test splits into training.
- Do not report synthetic or unmeasured performance numbers.
- Do not modify core architectural constants (vocab 1024, 80-bin mel, subsampling 4, LayerNorm) without explicit approval.
- Do not process sacred texts as ordinary speech data.
- Do not hide failing acceptance checks.

### Appendix C — Per-Phase Handoff Template
Each entry in `docs/PROGRESS.md` follows this exact structure:
- **Phase / date**
- **What was built** (files, commands)
- **What was verified** (exact commands run, pass/fail summaries)
- **What was NOT verified** (and why)
- **Decisions made / deviations from the spec**
- **Open issues and risks**
- **HUMAN STEPS now required**
- **Exact next step**
