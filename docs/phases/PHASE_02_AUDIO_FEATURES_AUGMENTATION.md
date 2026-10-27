# PHASE 2 — Audio Features and Augmentation

## 1. Goal
A high-performance feature pipeline that is deterministic, computationally lightweight, and re-implementable with bit-exact consistency on mobile and wearable edge devices.

## 2. Technical Specifications
### 2.1 Log-Mel Frontend (`awaaz/features.py:LogMelSpectrogram`)
Torch module matching exact architectural constraints from §1.4:
- **Sample Rate**: 16,000 Hz mono
- **Window Length**: 25 ms (400 samples)
- **Hop Length**: 10 ms (160 samples)
- **FFT Size**: 400 points
- **Mel Filterbanks**: 80 triangular bins
- **Frequency Range**: 20 Hz – 7,600 Hz
- **Log Floor**: Natural logarithm with a tiny numerical stabilization floor (`1e-5` or `1e-7`)
- **Frame Calculation**:
  $$\text{frames} = \lfloor \text{samples} / 160 \rfloor + 1$$
- **Output Shape**: `(batch_size, time_frames, 80)`

### 2.2 Global CMVN (Cumulative Mean & Variance Normalization)
- Holds global mean and std vectors (shape `(80,)`) as buffers inside the module.
- Stats are computed once over training manifests (subset of ~2000 utterances) and stored in `cmvn.npz`.
- **Zero per-utterance normalization** (per-utterance CMVN breaks streaming causality and impairs on-device parity).
- Formula:
  $$\hat{X}_{b, t, f} = \frac{X_{b, t, f} - \mu_f}{\sigma_f + \epsilon}$$

### 2.3 SpecAugment (`awaaz/features.py:SpecAugment`)
Applied strictly in training mode, restricted to valid (unpadded) frames:
- **Frequency Masking**: 2 independent masks, each with width $F \le 27$ bins.
- **Time Masking**: 2 independent masks, each masking up to 5% of valid utterance length ($T_{mask} \le 0.05 \cdot T$).
- **Mask Fill Value**: `0.0` (which corresponds to the mean after global CMVN).
- **Eval Behavior**: Strict identity pass-through.

### 2.4 Waveform Augmentations (`awaaz/features.py:WaveformAugmenter`)
- **Speed Perturbation**: 0.9x, 1.0x, 1.1x via high-quality resampling (changes length and shifts pitch).
- **Additive Noise**: Random SNR between 5 dB and 20 dB from background noise files (gracefully skipped if noise dir not configured).
- **Room Impulse Response (RIR)**: Convolutive reverberation simulation.
- **Microphone Emulation**: Random gain adjustment (±6 dB) and mild band-limiting filter to simulate watch/phone microphone acoustics.

## 3. Mandatory Tests & Acceptance Checks
- Output shapes and frame-count formula validated against varied clip lengths (e.g. 0.5s, 1.23s, 5.0s).
- Normalization check: features normalized with computed CMVN exhibit approximately zero mean and unit variance per bin.
- SpecAugment zero-integrity: masked bins are exactly 0.0 and padded regions remain untouched.
- Determinism check: identical seed yields bitwise identical features and masks.
- Deliverable doc: `docs/FEATURES.md` detailing the language-neutral feature extraction algorithm for Android/iOS engineers.

## 4. Definition of Done
Unit tests pass on CPU; `docs/FEATURES.md` is complete and verified.
