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
