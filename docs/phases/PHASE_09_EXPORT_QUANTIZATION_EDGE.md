# PHASE 9 — Export, Quantization, and Edge Budget

## 1. Goal
Export trained Conformer models to production-grade ONNX formats, apply int8 quantization, and establish rigorous edge hardware budgets (latency, RAM, power) across target wearables and phones.

## 2. Technical Specifications (`awaaz/export.py` + `docs/EDGE.md`)
### 2.1 ONNX Export
- Exports using ONNX opset $\ge 17$.
- Inputs: `features` `(B, T, 80)` and `feature_lengths` `(B,)`.
- Outputs: `log_probs` `(B, T_out, 1024)` and `output_lengths` `(B,)`.
- Dynamic axes enabled for batch size ($B$) and time ($T$).
- Exports **two model variants**:
  1. `awaaz_offline.onnx`: Full-context attention.
  2. `awaaz_stream.onnx`: Fixed chunk and left-chunk attention parameters baked into graph.

### 2.2 Numerical Parity Verification
- Evaluates maximum absolute difference between PyTorch and ONNX Runtime:
  $$\Delta_{\max} = \max | y_{\text{PyTorch}} - y_{\text{ONNX}} |$$
- Threshold: $\Delta_{\max} \le 1 \times 10^{-3}$ across multiple sequence lengths (0.5s, 2.0s, 6.0s). Exporter fails hard if parity threshold is breached.

### 2.3 Int8 Quantization
- Dynamic int8 quantization via ONNX Runtime (`quantize_dynamic`).
- Target: $\approx 75\%$ reduction in storage footprint (e.g. ~19 MB for `small`, ~6 MB for `tiny`).
- Re-run full WER/CER benchmark on int8 model: degradation must not exceed 0.5% absolute CER.

### 2.4 Device Budget Matrix (`docs/EDGE.md`)
| Device Target | Chipset / Architecture | Runtime Engine | Size Budget | RAM Budget | Target RTF |
|---|---|---|---|---|---|
| Wear OS Watch | Snapdragon W5+ / ARM Cortex-A53 | ONNX Runtime Mobile / TFLite | $< 10\text{ MB}$ | $< 60\text{ MB}$ | $\le 0.4$ |
| Low-end Android | MediaTek Helio / Unisoc | ONNX Runtime Mobile | $< 25\text{ MB}$ | $< 120\text{ MB}$ | $\le 0.25$ |
| iPhone / Apple Watch | Apple Silicon A-series / S-series | Core ML | $< 25\text{ MB}$ | $< 120\text{ MB}$ | $\le 0.15$ |
| Raspberry Pi 4/5 | Broadcom BCM2711/BCM2712 | ONNX Runtime ARM64 | $< 60\text{ MB}$ | $< 200\text{ MB}$ | $\le 0.2$ |

### 2.5 Native Feature-Parity Verification Kit
- Script that exports reference audio WAV files and binary Float32 arrays of expected log-mel features.
- Enables mobile engineers building C++/Kotlin/Swift audio front-ends to verify mathematical equivalence to `docs/FEATURES.md`.

## 3. Mandatory Tests & Acceptance Checks
- Export of fixture model to ONNX runs cleanly.
- Parity check passes below $10^{-3}$ tolerance.
- Dynamic quantization produces runnable int8 model with verified inference execution.
- Budget report generator outputs exact parameter counts and estimated FLOPs.

## 4. Definition of Done
ONNX and int8 models run on CPU; parity report verified; `docs/EDGE.md` complete with target device limits.
