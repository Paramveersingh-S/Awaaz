# Awaaz TPU Architecture & Scalability Roadmap (`docs/TPU.md`)

This document explains the technical challenges of running streaming Conformer-CTC speech recognition models on Google Cloud TPUs (Tensor Processing Units) using PyTorch, and details the architectural roadmap for migrating to JAX/Flax for distributed TPU pod scale-up.

---

## 1. Challenges of PyTorch Audio on TPUs (PyTorch/XLA)

While NVIDIA GPUs execute variable-length dynamic graphs efficiently via CUDA streams and CuDNN, Google TPUs rely on Ahead-Of-Time (AOT) and Just-In-Time (JIT) graph compilation via the XLA (Accelerated Linear Algebra) compiler. Training standard PyTorch speech pipelines on TPUs encounters several severe architectural bottlenecks:

### 1.1 Dynamic Audio Lengths & Graph Recompilation
- **The Problem**: Speech utterances naturally vary from 0.5 seconds to 20 seconds. In standard PyTorch, batches are padded to the maximum duration present in that specific batch.
- **The Impact on TPU**: Every unique `(batch_size, max_audio_samples)` and `(batch_size, max_tokens)` shape triggers a full XLA compilation cycle. A single compilation can take 30–90 seconds. With unbucketed or continuously variable batch lengths, the TPU pipeline spends over 80% of wall-clock time compiling graphs rather than executing matrix multiplications, resulting in near-zero TPU FLOPS utilization.

### 1.2 Dynamic Chunk Attention Masking
- **The Problem**: Awaaz dynamic chunk training samples chunk sizes uniformly from $\{8, 16, 32\}$ encoder frames with variable left contexts $\{2, 4, 8, \infty\}$ on each batch.
- **The Impact on TPU**: In PyTorch/XLA, conditional masking matrices and slice indexing with dynamic tensor bounds invalidate compiled kernels and force repeated recompilations of the self-attention and rotary embedding blocks.

### 1.3 CTC Loss Kernel Support & CPU Fallback
- **The Problem**: PyTorch `nn.CTCLoss` requires dynamic dynamic-programming lattice traversal (forward-backward variables).
- **The Impact on TPU**: As of PyTorch 2.x, native XLA lowering for `torch.nn.functional.ctc_loss` often exhibits edge-case fallbacks to host CPU memory when target lengths differ across batches or when non-finite gradient skipping is triggered. Moving batch tensors between TPU HBM and host CPU introduces severe PCIe latency stalls.

---

## 2. JAX/Flax Architectural Roadmap for TPU Clusters

To scale Awaaz training across Cloud TPU v4 and v5e pods (e.g. TPU v4-16 to v4-128 slices), the following architectural changes are designed for the upcoming `v2` migration:

### 2.1 Static Duration Bucketing & Fixed Shapes
Rather than dynamic batch padding:
1. Manifests are partitioned into $B=5$ strictly fixed audio duration buckets:
   - Bucket 1: 4.0 seconds ($64,000$ samples $\to 100$ encoder frames)
   - Bucket 2: 8.0 seconds ($128,000$ samples $\to 200$ encoder frames)
   - Bucket 3: 12.0 seconds ($192,000$ samples $\to 300$ encoder frames)
   - Bucket 4: 16.0 seconds ($256,000$ samples $\to 400$ encoder frames)
   - Bucket 5: 20.0 seconds ($320,000$ samples $\to 500$ encoder frames)
2. Every utterance within a bucket is padded with zeros to the exact bucket limit.
3. Each bucket is bound to a single compiled `@jax.jit` executable. The TPU compiles exactly 5 execution graphs during the warmup epoch and never recompiles again throughout the entire multi-day training run.

### 2.2 Static Dynamic-Chunk Attention Kernels
- Instead of generating variable-sized attention mask tensors on the host CPU, define static attention mask constants for the cartesian product of chunk sizes $\{8, 16, 32\}$ and left contexts $\{2, 4, 8, \text{all}\}$.
- Pass an integer `mask_id` as a static index into a precompiled 3D mask table, keeping tensor dimensions strictly static.

### 2.3 Native Optax CTC Loss
- Utilize `optax.ctc_loss_with_forward_and_state` which is natively lowered to vectorized XLA primitives and optimized for TPU matrix multiply units (MXUs) without host CPU round-trips.

### 2.4 Multi-Host Pod Parallelism with `shard_map`
- **Data Parallelism**: Scale batch seconds proportionally with pod size ($N=64$ TPU cores).
- **FSDP / Megatron Tensor Parallelism**: For larger architectures (Base 55M and future 300M parameter models), shard the projection layers across TPU tensor cores using `jax.experimental.shard_map` and `jax.sharding.NamedSharding`.

---

## 3. Recommended Compute Setup Today

For Awaaz v1:
- **Development & Unit Tests**: Local CPU workstation (Ryzen 7, fast fixtures in $< 15\text{s}$).
- **Prototyping & Mid-Scale Training**: NVIDIA RTX 3050 (4 GB) / RTX 4090 / Colab T4 / L4 (16–24 GB).
- **Production Full-Corpus Training**: Google Colab Pro / Pro+ NVIDIA A100 (40 GB / 80 GB) running PyTorch native mixed precision (`bfloat16`).
- **TPU Execution**: Reserved for Awaaz v2 JAX/Flax implementation.
