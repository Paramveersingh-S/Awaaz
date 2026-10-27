# Awaaz Edge Deployment & Device Hardware Budget (`docs/EDGE.md`)

This document defines the edge hardware budgets, computational limits, quantization benchmarks, and runtime execution targets for running Awaaz streaming models on embedded hardware, wearables, and smartphones.

---

## 1. Target Device Hardware Budget Matrix

The following table defines the hard resource limits for on-device Awaaz deployment. Measured figures are updated as physical benchmarks are collected on target hardware; unmeasured targets are explicitly labeled.

| Device Target | Chipset / Architecture | Runtime Engine | Int8 Model Budget | RAM Budget (Inference) | Target RTF (Real-Time Factor) | Measured Status |
|---|---|---|---|---|---|---|
| **Wear OS Smartwatch** | Qualcomm Snapdragon W5+ Gen 1 (4× Cortex-A53 @ 1.7 GHz) | ONNX Runtime Mobile / TFLite | $< 10\text{ MB}$ | $< 60\text{ MB}$ | $\le 0.40\text{x}$ | *Target (Unmeasured)* |
| **Low-End Android Phone** | MediaTek Helio G85 / Unisoc T606 (8-core ARM) | ONNX Runtime Mobile | $< 25\text{ MB}$ | $< 120\text{ MB}$ | $\le 0.25\text{x}$ | *Target (Unmeasured)* |
| **Mid / High-End Phone** | Snapdragon 8 Gen 2/3 / Dimensity 9200 | ONNX Runtime / QNN Execution Provider | $< 60\text{ MB}$ | $< 250\text{ MB}$ | $\le 0.10\text{x}$ | *Target (Unmeasured)* |
| **Apple Watch / iPhone** | Apple Silicon S-series / A-series (Bionic/Pro) | Apple Core ML (Neural Engine) | $< 25\text{ MB}$ | $< 120\text{ MB}$ | $\le 0.12\text{x}$ | *Target (Unmeasured)* |
| **Embedded Linux (SBC)** | Raspberry Pi 4 / 5 (Broadcom BCM2711 / BCM2712 ARM64) | ONNX Runtime ARM64 (NEON vectorized) | $< 60\text{ MB}$ | $< 200\text{ MB}$ | $\le 0.20\text{x}$ | *Target (Unmeasured)* |
| **Microcontrollers (MCU)** | ESP32-S3 / STM32H7 (Cortex-M4/M7, 512KB SRAM) | Not Viable for full Conformer ASR | N/A | N/A | N/A | **Infeasible** (Use tiny KWS wake-word) |

> **Note on Microcontrollers**: Full open-vocabulary speech recognition (Conformer-CTC at 6M–55M parameters) requires at minimum 20–60 MB RAM for activations and weights, which exceeds microcontroller SRAM limits. The recommended microcontroller architecture is a 200 KB Keyword-Spotting (KWS) model (e.g. TC-ResNet) that wakes an application processor or paired smartphone.

---

## 2. Model Footprint & Computational Complexity

Parameter counts and storage footprints across standard Awaaz presets:

| Model Preset | Total Parameters | Non-Embedding Params | FP32 ONNX Size | Int8 Quantized Size | Est. GFLOPs/sec (Streaming 25 fps) | Target Hardware Class |
|---|---|---|---|---|---|---|
| **`tiny`** | 6.05M | 5.92M | ~24.2 MB | **~6.1 MB** | ~0.30 GFLOPs/s | Smartwatches, Wear OS, Wearables |
| **`small`** | 18.66M | 18.40M | ~74.6 MB | **~18.7 MB** | ~0.92 GFLOPs/s | Smartphones, Raspberry Pi, SBCs |
| **`base`** | 55.19M | 54.67M | ~220.8 MB | **~55.2 MB** | ~2.73 GFLOPs/s | Tablets, Laptops, Edge Hubs, Servers |

$$\text{Estimated GFLOPs/sec} \approx 2 \times \text{Non-Embedding Params} \times 25\text{ encoder frames/sec} + \text{Self-Attention Overheads}$$

---

## 3. Quantization Protocol & Parity Policy

### 3.1 Dynamic Int8 Quantization
- Dynamic 8-bit integer quantization (`QInt8`) is applied via ONNX Runtime to all linear feed-forward, projection, and depthwise convolution weights.
- Activations are dynamically quantized to int8 during runtime inference.
- Expected compression: **$\approx 75\%$ reduction in binary file size**.

### 3.2 Parity & Validation Mandate
- **Numerical Parity**: Before quantization, FP32 ONNX outputs must match PyTorch outputs with max absolute error:
  $$\Delta_{\max} = \max | y_{\text{PyTorch}} - y_{\text{ONNX}} | < 1 \times 10^{-3}$$
- **Quantization Degradation Check**: Every quantized model (`.quant.onnx`) must be benchmarked on validation test sets. Degradation must satisfy:
  $$\Delta \text{CER}_{\text{int8}} \le +0.5\% \text{ absolute}$$
  If dynamic quantization degrades accuracy beyond this bound, static calibration or quantization-aware training (QAT) is triggered.

---

## 4. Native Feature-Parity Verification Kit

For edge deployment on iOS (Swift) and Android (Kotlin / C++ NDK), the feature extraction front-end (80-bin log-mel, 25ms Hamming, 10ms hop, global CMVN) must be implemented natively using mobile audio DSP APIs.

To verify mathematical parity against `awaaz/features.py`:
1. Use `python -m awaaz.export --test_kit` to generate:
   - `test_kit/reference_audio.wav`: Reference audio sample.
   - `test_kit/expected_features.npy`: Binary Float32 array of expected CMVN-normalized log-mel features.
2. The native mobile DSP pipeline must process `reference_audio.wav` and verify:
   $$\max | \mathbf{F}_{\text{native}} - \mathbf{F}_{\text{expected}} | < 0.01$$

---

## 5. Alternative Runtime Architecture Note

While Awaaz natively exports to ONNX Runtime and Core ML, an alternative proven edge production recipe is **k2 / icefall Zipformer + sherpa-onnx**.
- **Comparison**: Sherpa-onnx provides dedicated C/C++ embedding wrappers for Zipformer architectures with pre-allocated KV-caches.
- **Awaaz Strategy**: Awaaz uses standard Conformer-CTC with dynamic-chunk attention and RoPE. The ONNX Runtime path provides direct multi-platform portability (Windows, Linux, macOS, Android, iOS, WebAssembly) without requiring custom C++ runtime compilations.
