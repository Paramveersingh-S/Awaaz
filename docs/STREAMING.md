# Awaaz Streaming Inference Architecture (`docs/STREAMING.md`)

This document details the mathematical formulation, algorithmic latency accounting, and cache architectures for real-time streaming speech recognition in Awaaz.

---

## 1. Algorithmic Latency & Chunk Accounting

Streaming Awaaz models use dynamic-chunk Conformer-CTC with causal convolutions and left-constrained attention masks. The algorithmic latency is mathematically defined as:

$$\text{Algorithmic Latency} = \text{Chunk Duration} + \text{Acoustic Lookahead}$$

### 1.1 Timing Parameters
- **Audio Sample Rate**: 16,000 Hz ($16\text{ samples/ms}$).
- **STFT Hop Size**: 10 ms ($160\text{ samples}$).
- **Front-End Subsampling**: Factor 4 2D depthwise convolution ($40\text{ ms}$ per encoder frame, or $640\text{ audio samples/frame}$).
- **Acoustic Lookahead**: 0 ms. Conformer convolution modules use left-padding of size $(\text{kernel\_size} - 1)$ with right padding $= 0$, ensuring strictly zero right lookahead.

### 1.2 Latency by Chunk Configuration

| Preset | Chunk Size ($N_{\text{chunk}}$) | Left Chunks ($L$) | Window Duration | Algorithmic Latency | Use Cases |
|---|---|---|---|---|---|
| **Stream-0.64s** | 16 frames | 8 chunks | 5.76 s | **640 ms** | Meeting transcription, live subtitles |
| **Stream-0.32s** | 8 frames | 4 chunks | 1.60 s | **320 ms** | Voice assistants, interactive dialogue |
| **Stream-0.16s** | 4 frames | 4 chunks | 0.80 s | **160 ms** | Ultra-low latency voice search |
| **Offline** | Full sequence | $\infty$ | Full file | $T_{\text{audio}}$ | Batch transcription, archiving |

---

## 2. v1 Algorithm: Rolling-Window Streaming

The v1 streaming implementation in `awaaz/stream.py` uses an exact rolling-window algorithm:

```
[ Chunk t-2 ] [ Chunk t-1 ] [ Chunk t ]
<---------- Left Context ------------><-- Newest Chunk -->
                                      [ Sliced CTC Logits ]
```

### 2.1 Chunk Alignment & Whole-Chunk Trimming
- Audio arrives in arbitrary buffer sizes (e.g., 50 ms, 100 ms, 1000 ms).
- Samples are accumulated in an internal ring buffer.
- Processing triggers strictly when the buffer accumulates $\ge N_{\text{chunk}} \times 640\text{ samples}$ ($10,240\text{ samples}$ for 0.64s).
- The historical window is trimmed in **exact integer multiples of whole chunks** ($L \times N_{\text{chunk}}$ frames). This ensures the convolutional subsampling receptive field alignment remains 100% identical to the training stage.

### 2.2 Slicing the Newest Chunk
For an active window containing $K \le L + 1$ chunks:
- Total encoder frames emitted: $T_{\text{window}} = K \times N_{\text{chunk}}$.
- The recognizer extracts **strictly the newest chunk's frames**:
  $$\text{logits}_{\text{chunk}} = \text{logits}[-N_{\text{chunk}} : ]$$
- Frames prior to $-N_{\text{chunk}}$ served purely as acoustic and contextual history for the Conformer self-attention blocks.

### 2.3 Incremental CTC State Collapse
Standard CTC greedy decoding collapses consecutive identical tokens. In a streaming context, a single phoneme may span across the boundary of chunk $t$ and chunk $t+1$.
- `StreamingRecognizer` maintains persistent state `last_token_id`.
- For each frame $\tau$ in the newest chunk:
  $$\hat{c}_\tau = \operatorname{argmax}_v \log p(v \mid x_\tau)$$
  $$\text{if } \hat{c}_\tau \neq \text{last\_token\_id}:$$
  $$\quad \text{if } \hat{c}_\tau \neq \text{blank\_id}: \text{emit}(\hat{c}_\tau)$$
  $$\quad \text{last\_token\_id} = \hat{c}_\tau$$
- If a speaker holds a vowel across chunk boundaries (`... A | A ...`), the second `A` is correctly collapsed and not duplicated.

### 2.4 Partial Result Stability
- **Finalized Tokens**: Tokens emitted in chunks older than the current chunk are finalized and cannot be modified by future acoustic frames.
- **Tentative Tokens**: Tokens emitted in the newest chunk represent partial hypotheses and may receive word-boundary adjustments.

---

## 3. v2 Roadmap: Constant-Time Cached Streaming (KV-Cache)

While the v1 rolling-window algorithm is provably equivalent to training and incurs zero state discrepancy, its computational complexity scales with window length $O(L \cdot N_{\text{chunk}})$.

The upcoming v2 engine migrates to stateful cached execution with $O(1)$ constant compute per chunk:

### 3.1 Per-Layer Attention Key/Value Cache
- Each of the $M$ Conformer self-attention layers retains a persistent key and value buffer of shape:
  $$\mathbf{K}_{\text{cache}}, \mathbf{V}_{\text{cache}} \in \mathbb{R}^{B \times H \times (L \cdot N_{\text{chunk}}) \times d_k}$$
- In chunk $t$, query projections $\mathbf{Q}_t \in \mathbb{R}^{B \times H \times N_{\text{chunk}} \times d_k}$ attend to the concatenated key matrix $[\mathbf{K}_{\text{cache}}, \mathbf{K}_t]$.
- The cache FIFO drops the oldest $N_{\text{chunk}}$ keys and appends $\mathbf{K}_t$.

### 3.2 Rotary Positional Embedding (RoPE) Offsets
- In cached execution, $\mathbf{Q}_t$ has length $N_{\text{chunk}}$ while $\mathbf{K}$ has length $(L+1) N_{\text{chunk}}$.
- RoPE embeddings must apply relative position offsets:
  $$\Delta m = m_{\text{query}} - m_{\text{key}} + \text{offset}$$
  where $\text{offset} = t \times N_{\text{chunk}}$.

### 3.3 Causal Convolution FIFO State
- Each depthwise convolution block maintains a circular FIFO buffer of size $(\text{kernel\_size} - 1)$ frames ($14\text{ frames}$ for kernel 15).
- For each new chunk, the 14 historical frames serve as left padding, avoiding repeated convolution recomputations across chunk boundaries.

### 3.4 ONNX Stateful Graph Interface
When exported to ONNX, the v2 stateful model defines dynamic state inputs and outputs:
- **Inputs**: `audio_chunk`, `cache_k_0..11`, `cache_v_0..11`, `conv_state_0..11`, `step_offset`.
- **Outputs**: `chunk_logits`, `new_cache_k_0..11`, `new_cache_v_0..11`, `new_conv_state_0..11`.
