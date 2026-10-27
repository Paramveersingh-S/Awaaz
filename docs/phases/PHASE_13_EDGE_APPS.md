# PHASE 13 — Edge Applications

## 1. Goal
Demonstrate production-ready on-device speech recognition on physical hardware (Wear OS smartwatches, Android phones, iOS devices, and embedded Linux boards).

## 2. Technical Specifications
### 2.1 Android / Wear OS Sample (`edge/android/`)
- Native Kotlin / Jetpack Compose application.
- Real-time 16 kHz audio capture via Android AudioRecord.
- Native feature extractor implementation matching `docs/FEATURES.md`.
- Built-in **Feature-Parity Self-Test**: Automatically tests the native frontend against reference feature binary arrays from Phase 9.
- ONNX Runtime Mobile integration running quantized `awaaz_int8.onnx`.
- Persistent CTC state decoding with live streaming transcript display.
- Telemetry dashboard: Real-time factor (RTF), per-chunk latency (ms), RAM usage (MB), and battery discharge rate (% per hour).

### 2.2 iOS / watchOS Sample (`edge/apple/`)
- Native Swift / SwiftUI application.
- Audio capture via `AVAudioEngine`.
- Model converted to Apple Core ML format.
- Architecture option: Direct on-watch execution vs. low-latency relay to paired iPhone.

### 2.3 Embedded Linux (`edge/embedded/`)
- Python and C++ streaming clients for ARM64 (Raspberry Pi 4 / 5).
- ALSA / PulseAudio real-time capture.

### 2.4 Wearable Power Management Strategy
- Tiny always-on VAD running continuously at microwatt power consumption.
- Conformer ASR model woken up strictly upon speech trigger.
- Thermal management and duty cycling strategies for ultra-low power envelopes.

### 2.5 Decision Gate
If `tiny` int8 does not achieve real-time latency ($RTF \le 0.4$) on the target smartwatch, evaluate:
1. Subsampling factor 8 front-end.
2. Architecture slimming (fewer Conformer blocks).
3. Student-teacher knowledge distillation.
4. Offload processing to paired mobile device.

## 3. Human Step
Deploy sample apps to physical hardware; record real latency, memory, and battery consumption metrics; update `docs/EDGE.md`.

## 4. Definition of Done
Sample apps build cleanly; feature parity tests pass natively on device; telemetry reports documented.
