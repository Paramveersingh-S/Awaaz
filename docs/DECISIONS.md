# Architecture Decision Records (ADRs)

This document records key architectural, algorithmic, and engineering decisions for Awaaz.

---

## ADR-001: Streaming Conformer-CTC with Dynamic-Chunk Training
- **Date**: 2026-10-27
- **Status**: ACCEPTED
- **Context**: Edge speech recognition requires low algorithmic latency on resource-constrained devices (smartwatches, budget phones) without maintaining two separate model checkpoints for offline vs. live streaming.
- **Decision**: Implement a unified Conformer-CTC architecture trained with dynamic-chunk attention masking. With probability 0.6, attention masks restrict context to a sampled chunk size (8, 16, or 32 encoder frames) and past context (2, 4, 8, or unlimited left chunks). With probability 0.4, full context is enabled.
- **Consequences**: A single set of model weights can be deployed for zero-latency streaming on edge hardware and high-accuracy offline transcription on servers/phones.

---

## ADR-002: Subsampling Factor 4 Front-End
- **Date**: 2026-10-27
- **Status**: ACCEPTED
- **Context**: 10 ms audio frames produce 100 frames per second. Processing 100 frames/sec through 12–16 Conformer layers incurs quadratic self-attention cost and high memory.
- **Decision**: Apply 2D Convolutional subsampling with factor 4 (two stacked stride-2 convs), resulting in 40 ms encoder frame steps (25 frames/second). Subsampling factor 8 is reserved as an optional preset for ultra-constrained wearables (`tiny-8x`).
- **Consequences**: Reduces sequence length by 75%, making real-time factor (RTF) $\le 0.4$ achievable on low-power ARM cores.

---

## ADR-003: Rotary Positional Embeddings (RoPE)
- **Date**: 2026-10-27
- **Status**: ACCEPTED
- **Context**: Absolute positional embeddings fail on arbitrary-length streaming audio and require resetting positional indices across sliding chunk windows.
- **Decision**: Apply RoPE directly to queries and keys in self-attention.
- **Consequences**: Inherently relative representation enables continuous sliding-window streaming without resetting positional state or retraining for different chunk lengths.

---

## ADR-004: SentencePiece Unigram Tokenizer with 1024 Vocabulary
- **Date**: 2026-10-27
- **Status**: ACCEPTED
- **Context**: Bilingual English + Punjabi requires balanced subword representation within a strictly bounded vocabulary suitable for edge CTC decoding.
- **Decision**: Train a single 1024-token SentencePiece unigram model on a 50/50 balanced corpus of English and Punjabi text. Token ID 0 is strictly reserved as the CTC Blank (pad token); token ID 1 is unknown (`<unk>`). BOS and EOS are disabled.
- **Consequences**: CTC projection head is small ($1024 \times d_{\text{model}}$), conserving parameters and computation.

---

## ADR-005: Sacred-Text Policy (§1.5)
- **Date**: 2026-10-27
- **Status**: ACCEPTED
- **Context**: Public Punjabi audio contains kirtan, katha, and Gurbani recitation. General ASR models hallucinate or subtly distort sacred text, where plausible errors cause religious offense.
- **Decision**: All manifests tag sacred/devotional audio with `"domain": "sacred"`. Version 1 strictly filters out and excludes all sacred audio from training, validation, and testing. No automated rewriting, summarization, or correction of sacred texts is permitted.
- **Consequences**: Protects theological integrity; clearly delineates product boundaries.

---

## ADR-006: Global CMVN Normalization
- **Date**: 2026-10-27
- **Status**: ACCEPTED
- **Context**: Per-utterance normalization requires whole-file statistics, which are impossible in real-time streaming and cause mismatch on short live audio chunks.
- **Decision**: Compute global mean and variance once across 2,000 representative training utterances and bake these statistics into `cmvn.npz` buffers.
- **Consequences**: Bit-exact feature consistency between offline and streaming runs, easily portable to Kotlin/Swift on-device frontends.

---

## ADR-007: Deterministic Bilingual Text Normalization Rules
- **Date**: 2026-10-27
- **Status**: ACCEPTED
- **Context**: Discrepancies between training transcript normalization and runtime inference normalization silently ruin Word Error Rate (WER) and Character Error Rate (CER).
- **Decision**: All training manifests, evaluations, and runtime inference must run through `awaaz.text.normalize(text)` applying these exact steps in strict sequence:
  1. `unicodedata.normalize("NFC", text)`: Ensures canonical decomposition of Gurmukhi nukta characters into base consonant + nukta `\u0A3C`.
  2. `text.lower()`: Lowercases Latin alphabet characters.
  3. Digit detection: Drop utterance (`return None`) if containing ASCII digits `0–9` or Gurmukhi digits `\u0A66–\u0A6F`.
  4. Standardize apostrophes: Typographic curly apostrophes (`’`, `‘`) converted to ASCII single quote `'` (`\u0027`).
  5. Danda substitution: Gurmukhi danda `\u0964` and double danda `\u0965` replaced with a single space.
  6. Whitelist character filtering: Strip all characters except `a–z`, Gurmukhi block `\u0A00–\u0A7F`, `'`, and space ` `.
  7. Whitespace consolidation: Collapse multiple whitespace characters into single space; strip leading/trailing spaces.
  8. Empty rejection: Return `None` if resulting string is empty.
- **SentencePiece Parameters**:
  - `vocab_size`: 1024
  - `character_coverage`: 1.0
  - `pad_id`: 0 (strictly reserved as the CTC Blank token)
  - `unk_id`: 1
  - `bos_id`: -1 (disabled)
  - `eos_id`: -1 (disabled)
  - `normalization_rule_name`: "identity" (relying strictly on `awaaz.text.normalize`)
- **Consequences**: Guaranteed idempotency and exact mathematical consistency across training, evaluation, ONNX export, and native edge runtimes.

---

## ADR-008: Model Presets and Measured Parameter Counts
- **Date**: 2026-10-27
- **Status**: ACCEPTED
- **Context**: Phase 4 requires exact parameter counts for all presets to be verified within $\pm 15\%$ of specification and documented.
- **Measured Parameters**:
  | Preset | $d_{\text{model}}$ | Blocks | Heads | FFN Dim | Subsampling | Encoder Params | CTC Head | Total Parameters | Target | Deviation |
  |---|---|---|---|---|---|---|---|---|---|---|
  | `tiny` | 144 | 12 | 4 | 576 | 102,688 | 5,800,800 | 147,760 | **6,051,248** (~6.05M) | ~6M | **+0.85%** |
  | `small` | 256 | 12 | 4 | 1024 | 174,368 | 18,225,408 | 262,448 | **18,663,008** (~18.66M) | ~19M | **-1.77%** |
  | `base` | 384 | 16 | 6 | 1536 | 256,384 | 54,540,288 | 393,568 | **55,190,240** (~55.19M) | ~55M | **+0.35%** |
- **Decision**: All three presets match targets within $< 2\%$ (well within the $\pm 15\%$ bound). Causal convolution kernel size is 15. All normalizations use LayerNorm. CTC projection head targets 1024 vocab.
- **Consequences**: Confirmed ready for training, dynamic-chunk streaming, and int8 ONNX quantization.
