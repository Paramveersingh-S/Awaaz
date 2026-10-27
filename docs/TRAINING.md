# Awaaz Training Guide & Troubleshooting (`docs/TRAINING.md`)

This document serves as the operational guide and diagnostic manual for training Awaaz Conformer-CTC models across local development workstations, NVIDIA RTX GPUs, and Google Colab cloud runtimes.

---

## 1. Quickstart & Command Invocations

### 1.1 Local CPU / GPU Smoke Run
```bash
python -m awaaz.train \
  --config configs/tiny.yaml \
  --run_dir runs/local_test \
  --manifest_train data/fixtures/manifest.jsonl \
  --tokenizer artifacts/tokenizer/tokenizer.model \
  --max_steps 100
```

### 1.2 Full Cloud Run (T4 / L4 / A100 / RTX 3050)
```bash
python -m awaaz.train \
  --config configs/small.yaml \
  --run_dir runs/exp1_small \
  --manifest_train data/manifests/train.jsonl \
  --manifest_dev data/manifests/dev.jsonl \
  --tokenizer artifacts/tokenizer/tokenizer.model \
  --cmvn artifacts/cmvn.npz
```

### 1.3 Seamless Resumption after Preemption / Disconnect
Awaaz saves full training state to `last.pt` atomically every 500 steps. To resume after a session reset or crash:
```bash
python -m awaaz.train \
  --config configs/small.yaml \
  --run_dir runs/exp1_small \
  --resume
```

---

## 2. Hardware Budgeting & Compute Planning

### 2.1 Recommended `batch_seconds` by GPU Tier
Memory usage scales linearly with the total audio duration in a batch. Tune `batch_seconds` to stay within VRAM bounds without triggering CUDA OOM:

| Hardware Tier | VRAM | Recommended `batch_seconds` (Tiny) | Recommended `batch_seconds` (Small) | Recommended `batch_seconds` (Base) |
|---|---|---|---|---|
| **Local CPU** | System RAM | 120s | 80s | 40s |
| **NVIDIA RTX 3050** | 4 GB | 200s | 160s | 80s |
| **Google Colab T4** | 16 GB (GDDR6) | 480s | 360s | 200s |
| **Google Colab L4** | 24 GB (GDDR6) | 800s | 600s | 360s |
| **Google Colab A100** | 40 / 80 GB (HBM2) | 1600s | 1200s | 800s |

### 2.2 Computing `total_steps` for a Time Budget
To compute `total_steps` for a fixed allocation of compute time (e.g. an 8-hour Colab session):
1. Launch the training script for 100 steps to measure the steady-state `steps_per_sec` from `train_log.csv` or console logs.
2. Calculate target steps:
```
total_steps = budget_hours * 3600 * steps_per_sec
```
$$\text{total\_steps} = \text{budget\_hours} \times 3600 \times \text{steps\_per\_sec}$$
3. For example, on a Colab L4 running at 4.2 steps/sec over a 12-hour budget:
$$12 \times 3600 \times 4.2 \approx 181,440 \text{ steps}$$

---

## 3. Training Architecture & Dynamics

### 3.1 Optimizer & Learning Rate Schedule
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9}$, weight decay $= 10^{-3}$).
- **Schedule**: Linear warmup from 0 to peak LR ($1 \times 10^{-3}$) over 5,000 steps, followed by smooth cosine decay down to $1 \times 10^{-5}$ at `total_steps`.
- **Loss**: PyTorch `nn.CTCLoss(blank=0, zero_infinity=True)`. All log-probabilities are upcast to `float32` before computing loss to eliminate numerical underflow.

### 3.2 Dynamic-Chunk Attention Sampling
To produce a single checkpoint that excels at both offline transcription and low-latency streaming inference, each batch samples:
- **With probability 0.6**: A chunked attention mask with:
  - Chunk size sampled uniformly from $\{8, 16, 32\}$ encoder frames (approx. 0.32s, 0.64s, 1.28s).
  - Left chunks attended sampled from $\{2, 4, 8, \text{unlimited}\}$.
- **With probability 0.4**: Full context attention (`chunk_size=None, left_chunks=-1`).

### 3.3 Atomic Checkpoint Resilience
- **`last.pt`**: Writes to `last.pt.tmp` and performs an atomic filesystem replace (`os.replace`) every 500 steps. If Colab disconnects mid-save, the existing `last.pt` is completely intact.
- **Rolling Checkpoint History**: Checkpoints saved at periodic evaluation intervals (`step_0002000.pt`, etc.) retain a rolling window of the newest $K=12$ checkpoints for post-training weight averaging (§Phase 7).

---

## 4. Common Failure Symptom Diagnosis Table

| Symptom | Probable Cause | Diagnostic Check & Fix |
|---|---|---|
| **Loss stays flat from the start (~6.0 - 7.0)** | 1. Learning rate misconfigured.<br>2. Audio features unnormalized.<br>3. Frame count rule mismatch. | - Verify peak learning rate is $1 \times 10^{-3}$.<br>- Verify `cmvn.npz` is loaded in `frontend`. Without CMVN, log-mel features have mean $\sim -20$, causing gradient saturation.<br>- Verify encoder output length matches `samples // 160 + 1` divided by subsampling factor 4. |
| **Loss explodes to `inf` or `NaN`** | 1. Target sequence longer than encoder frames.<br>2. Empty/silent audio clips.<br>3. `fp16` underflow/overflow. | - Ensure `CTCLoss(zero_infinity=True)` is set.<br>- Ensure log-probabilities are cast to `float32` before CTC loss.<br>- Verify `AwaazDataset` length filter drops clips where `tokens >= duration * 25`.<br>- Check gradient clipping is active (`max_norm=5.0`). |
| **Loss falls smoothly, but decoded text is garbage** | 1. Tokenizer mismatch.<br>2. Normalization discrepancy.<br>3. Blank index confusion. | - Ensure the exact same `tokenizer.model` is used for training and greedy decoding.<br>- Verify blank token index is `0` (`tokenizer.blank_id`).<br>- Verify `normalize()` is applied consistently before encoding and evaluation. |
| **English WER improves, but Punjabi WER stalls** | 1. Imbalanced training data.<br>2. Tokenizer vocabulary starvation.<br>3. Low Punjabi acoustic variance. | - Check manifest statistics via `dataset.hours_per_language()`. Ensure Punjabi audio accounts for $\ge 40\%$ of training hours.<br>- Verify tokenizer was trained with balanced 50/50 English/Punjabi corpus. |
| **GPU utilization is low / GPU idle** | 1. IO bottleneck reading small files.<br>2. Audio decompression on CPU.<br>3. Slow drive access. | - Untar audio archive onto the local SSD (`/content/` on Colab), **never** stream directly from Google Drive.<br>- Ensure audio is pre-converted to 16 kHz 16-bit mono FLAC.<br>- Use `num_workers=4` or higher in Linux cloud environments. |
| **CUDA Out of Memory (OOM)** | 1. `batch_seconds` too large for GPU VRAM.<br>2. Unusually long audio outlier. | - Lower `batch_seconds` in config (e.g. from 400s to 250s).<br>- Verify `max_duration` in dataset is capped at 20.0 seconds.<br>- Training loop automatically catches CUDA OOM, purges cache, and skips batch. |

---

## 5. Ethical & Sacred-Text Verification
Per repository rule §1.5, manifests flag sacred/devotional clips with `"domain": "sacred"`. `awaaz/train.py` unconditionally respects `exclude_sacred: True`, dropping sacred clips during dataset loading. This ensures no devotional text is ever ingested by the neural network weights.
