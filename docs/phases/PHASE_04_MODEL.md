# PHASE 4 — The Model

## 1. Goal
Implement a clean, modular Conformer-CTC speech recognition architecture from scratch in PyTorch, satisfying all edge-streaming and dynamic-chunk specifications defined in §1.4.

## 2. Technical Specifications (`awaaz/model.py`)
### 2.1 Presets Architecture Matrix (§1.6)
| Preset | $d_{\text{model}}$ | Blocks | Heads | FFN Dim | Kernel | $\approx$ Params | Intended Hardware |
|---|---|---|---|---|---|---|---|
| `tiny` | 144 | 12 | 4 | 576 | 15 | ~6M | Smartwatches, low-end wearables |
| `small` | 256 | 12 | 4 | 1024 | 15 | ~19M | Smartphones, embedded Linux boards |
| `base` | 384 | 16 | 6 | 1536 | 15 | ~55M | High-accuracy mobile / server nodes |

### 2.2 Subsampling Module (`Conv2dSubsampling`)
- 2D Convolution front-end with factor **4** (or 8 for tiny edge preset).
- Factor 4: Two stacked 2D convolutions with kernel size 3, stride 2, padding 1, 32 channels, ReLU activations.
- Linear projection from `32 * (n_mels // 4)` to $d_{\text{model}}$.
- Length reduction formula for stride 2:
  $$T_{\text{out}} = \lfloor (T_{\text{in}} - 1) / 2 \rfloor + 1$$

### 2.3 Rotary Positional Embeddings (RoPE)
- Applied to queries and keys in multi-head attention.
- Relative position formulation eliminates fixed length limits and allows continuous sliding-window streaming without resetting positional state.

### 2.4 Dynamic-Chunk Attention Masking
- Attention mask parameterized by `chunk` size (in encoder frames) and `left_chunks` (number of past chunks attended to; `-1` = unlimited).
- A frame at index $t$ in chunk $C$ may attend only to keys in chunks $[C - \text{left\_chunks}, C]$.
- **Zero Future Leaking**: Attention to frames $> t$ outside the current chunk boundary is strictly blocked.
- Padded position safety: Every query row maintains at least one valid key (e.g., self-diagonal) to prevent softmax NaNs.

### 2.5 Macaron Conformer Block
Each block follows the Macaron-style ordering:
1. Feed-Forward module with 0.5 residual weight
2. Multi-Head Self-Attention module with RoPE
3. Convolution module with 1.0 residual weight
4. Feed-Forward module with 0.5 residual weight
5. Final LayerNorm

### 2.6 Causal Convolution Module
- LayerNorm on input features.
- Pointwise Conv (expands channels to $2 \times d_{\text{model}}$) followed by Gated Linear Unit (GLU).
- **Causal Depthwise Convolution**: Kernel size 15. When `causal=True`, asymmetric left-padding of $K-1=14$ frames ensures zero look-ahead into future audio frames.
- LayerNorm (strictly no BatchNorm to avoid export and streaming state discrepancies).
- SiLU activation, Pointwise Conv projection back to $d_{\text{model}}$, and Dropout.
- Valid masking: masked frames zeroed out prior to depthwise convolution.

### 2.7 CTC Projection Head
- Linear projection from $d_{\text{model}}$ to `vocab_size` (1024).
- Log-softmax output with blank at token ID 0.

## 3. Mandatory Tests & Acceptance Checks
- **Shape Invariance**: Forward pass with varying batch sizes and sequence lengths produces exact expected output dimensions.
- **Padding Invariance**: The encoder output for an utterance alone matches within numerical precision ($< 10^{-5}$) its output when batched with longer padded utterances.
- **Chunk-Causality Verification (CRITICAL)**: With chunked mask and causal convolution, mutating input spectrogram frames from chunk $k$ onward produces **zero difference** in encoder outputs for chunks $< k$.
- **Parameter Count Check**: `tiny`, `small`, and `base` match specification within $\pm 15\%$. Exact counts recorded in `docs/DECISIONS.md`.
- **Gradient Check**: Backward pass on dummy batch produces finite, non-zero gradients across all parameters.

## 4. Definition of Done
All unit tests pass on CPU in under 10 seconds; causality is mathematically verified.
