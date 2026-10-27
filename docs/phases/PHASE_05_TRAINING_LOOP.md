# PHASE 5 — Training Loop

## 1. Goal
A battle-tested, resumable, mixed-precision, dynamic-chunk training loop engineered to survive unexpected Google Colab disconnects and yield reproducible convergence.

## 2. Technical Specifications (`awaaz/train.py`)
### 2.1 Optimization & Schedule
- **Optimizer**: AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$, weight decay $= 10^{-3}$.
- **Learning Rate Schedule**: Linear warmup for 5,000 steps to peak LR ($1 \times 10^{-3}$), followed by cosine decay down to $1 \times 10^{-5}$ across `total_steps`.
- **Loss Function**: PyTorch `nn.CTCLoss(blank=0, zero_infinity=True)`. CTC loss computed in `float32` by upcasting log-probabilities. Normalized by batch size.
- **Precision**: `torch.cuda.amp.autocast(dtype=torch.bfloat16)` when supported (L4, A100), otherwise `float16` with `GradScaler`. Feature extraction remains in `float32`.
- **Gradient Clipping**: Max gradient norm 5.0. If gradient norm is non-finite (NaN/inf), skip the optimizer step and log warning.

### 2.2 Dynamic-Chunk Training
- For each batch, with probability `0.6`, sample a chunked attention configuration:
  - Chunk size sampled uniformly from $\{8, 16, 32\}$ encoder frames (approx. 0.32s, 0.64s, 1.28s).
  - Left chunks attended sampled from $\{2, 4, 8, \text{unlimited}\}$.
- With probability `0.4`, train with full context (`chunk=None`).

### 2.3 Robust Checkpointing
- **`last.pt`**: Written atomically (write to temporary file, then atomic rename `os.replace`) every 500 steps. Stores:
  `model_state_dict`, `optimizer_state_dict`, `scaler_state_dict`, `step`, `epoch`, `config`.
- **Periodic Checkpoints**: Saved every `eval_every` steps (default 2000) as `step_0002000.pt`. Retain newest $K=12$ checkpoints for post-training weight averaging (§Phase 7).
- **Auto-Resume**: If `last.pt` exists in run directory, automatically restore state and resume at `step + 1` with accurate epoch index and sampler state.
- **Hugging Face Hub Sync**: Optional non-blocking checkpoint push; network errors log a warning without interrupting training.

### 2.4 Monitoring & Telemetry
- Expose training diagnostics: Loss, learning rate, gradient norm, steps/second, audio throughput (seconds of audio processed per wall-clock second), and peak GPU VRAM.
- Plain-text logs and structured CSV telemetry saved to run directory.
- Graceful Out-Of-Memory (OOM) handling: catch CUDA OOM, free cache, skip failing batch, and alert user if repeated.

## 3. Mandatory Tests & Acceptance Checks
- **Overfit Test (Mandatory)**: Train a miniature Conformer model on ~30 synthetic fixture clips (dropout 0, no augmentation, CPU) for 200–300 steps. Loss must decrease substantially and greedy CTC decoding must achieve zero or near-zero error on the training text.
- **Checkpoint Resilience Test**: Run training for 20 steps, terminate process, verify atomic rename, restart; verify step counter, learning rate, and optimizer states match expectations.
- **Chunk Sampling Invariance**: Verify random seed yields deterministic distribution of chunk sizes and left chunks.
- Deliverable doc: `docs/TRAINING.md` containing common failure symptom diagnosis table.

## 4. Definition of Done
Overfit test passes on fixture; checkpoint resume verified locally; `docs/TRAINING.md` published.
