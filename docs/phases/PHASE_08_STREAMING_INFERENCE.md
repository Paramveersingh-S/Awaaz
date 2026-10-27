# PHASE 8 — Streaming Inference

## 1. Goal
Implement an incremental streaming recognizer that processes continuous audio streams with low algorithmic latency, matching the mathematical constraints enforced during dynamic-chunk training.

## 2. Technical Specifications (`awaaz/stream.py`)
### 2.1 StreamingRecognizer Architecture
- Receives arbitrary-sized audio blocks (16 kHz mono float32).
- Maintains internal sample ring buffer.
- When sufficient audio arrives for a **chunk** (default 16 encoder frames $= 16 \times 40\text{ ms} = 0.64\text{ s} = 10,240\text{ samples}$), triggers forward inference.

### 2.2 Rolling-Window Processing (v1 Algorithm)
- Buffers up to $(\text{left\_chunks} + 1)$ chunks of audio.
- Trims history in exact whole-chunk increments to ensure frame alignments remain identical to training.
- Computes log-mel features and passes features to Conformer encoder with the exact chunked mask.
- Slices **only the newest chunk's output frames** ($T_{\text{chunk}}$ frames).
- Applies incremental CTC collapse using persistent emitter state:
  - Tracks `last_token_id`.
  - Discards consecutive identical non-blank tokens across chunk boundaries.
  - Emits confirmed tokens immediately to the running transcript.

### 2.3 Partial Result Stability
- Classifies emitted tokens into:
  - **Finalized**: Tokens older than 1 chunk (stable, cannot be mutated by future audio).
  - **Tentative**: Tokens in the current chunk (subject to boundary updates).

### 2.4 Interface Contract
```python
class StreamingRecognizer:
    def __init__(self, model, config, chunk_frames=16, left_chunks=8):
        ...
    def reset(self) -> None:
        """Clears all audio buffers and persistent decoding state."""
        ...
    def accept(self, audio_chunk: np.ndarray) -> str:
        """Consumes raw audio block and returns updated transcript."""
        ...
    def flush(self) -> str:
        """Pads remainder to a full chunk and returns complete final transcript."""
        ...
```

### 2.5 Cached Streaming Architecture (v2 Roadmap — `docs/STREAMING.md`)
- Detailed design document specifying per-layer key/value KV-cache, RoPE position offsets, and causal convolution FIFO state.
- Enables $O(1)$ constant compute cost per chunk instead of linear window cost.

## 3. Mandatory Tests & Acceptance Checks
- **Equivalence Test**: Streaming recognizer output on fixture audio matches offline chunked-mask forward pass with $\ge 99.9\%$ token equivalence.
- **Block-Size Independence**: Feeding identical audio in 100 ms, 320 ms, or 1000 ms chunks yields identical final transcripts.
- **State Integrity**: `reset()` returns the recognizer to pristine initial state.
- **Algorithmic Latency Test**: Verify reported latency $= \text{chunk\_duration} + \text{lookahead}$.

## 4. Definition of Done
Streaming tests pass on CPU; `docs/STREAMING.md` is complete with latency derivations and v2 KV-cache specifications.
