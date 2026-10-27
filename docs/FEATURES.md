# Awaaz Feature Extraction Specification

**Language-Neutral Engineering Specification for Mobile and Embedded Implementations (Android / Kotlin / C++ and iOS / Swift / Core ML)**

This document details the exact mathematical parameters, windowing functions, padding schemes, filterbank matrices, and normalization formulas used by the Awaaz acoustic front-end. Any on-device feature extractor must replicate this pipeline bit-for-bit to ensure numerical parity with the trained Conformer-CTC model.

---

## 1. Global Audio Constants

| Parameter | Value | Notes |
|---|---|---|
| **Sampling Frequency ($f_s$)** | `16,000 Hz` | Mono single-channel 32-bit float audio |
| **Window Length ($N_{\text{win}}$)** | `25 ms` (400 samples) | $25 \times 10^{-3} \times 16000 = 400$ |
| **Hop Length ($N_{\text{hop}}$)** | `10 ms` (160 samples) | $10 \times 10^{-3} \times 16000 = 160$ |
| **FFT Size ($N_{\text{fft}}$)** | `400` points | Produces $N_{\text{freqs}} = 400 // 2 + 1 = 201$ frequency bins |
| **Mel Filterbanks ($M$)** | `80` triangular bins | Slaney-normalized area, HTK frequency warping |
| **Frequency Range** | `[20.0 Hz, 7600.0 Hz]` | Excludes sub-audible DC noise and high Nyquist edge |
| **Log Stabilization Floor** | `1.0e-5` | Natural logarithm floor $\ln(\max(E, 10^{-5}))$ |

---

## 2. Frame Count & Boundary Padding Formulation

For an input audio array of $N$ samples:

$$\text{Frames} = \left\lfloor \frac{N}{160} \right\rfloor + 1$$

### Boundary Padding Mode (Centered Reflection)
To center frame $0$ at sample $t = 0$, the signal is padded with reflection on both ends:
- Left padding length: $P = \lfloor N_{\text{win}} / 2 \rfloor = 200$ samples.
- Right padding length: $P = 200$ samples.
- **Reflection rule**: $x[-i] = x[i]$ for $i = 1, \dots, P$.
- Padded signal length: $N_{\text{padded}} = N + 400$ samples.
- Frame indexing: Frame $k$ spans samples from $k \times 160$ to $k \times 160 + 400$.

---

## 3. Windowing Function

The 400-sample periodic Hann window $w[n]$ is defined for $n = 0, 1, \dots, 399$:

$$w[n] = 0.5 - 0.5 \cos\left( \frac{2\pi n}{400} \right)$$

---

## 4. Short-Time Fourier Transform & Power Spectrum

For each frame $k$ of windowed audio $s_k[n] = x_{\text{padded}}[k \cdot 160 + n] \cdot w[n]$:

1. Compute Discrete Fourier Transform (DFT) of length 400:
   $$X_k[f] = \sum_{n=0}^{399} s_k[n] e^{-j 2\pi f n / 400}, \quad f = 0, 1, \dots, 200$$

2. Compute Power Spectrum:
   $$P_k[f] = |X_k[f]|^2 = \text{Re}(X_k[f])^2 + \text{Im}(X_k[f])^2$$

---

## 5. Mel Filterbank Matrix ($201 \times 80$)

1. **Hertz to Mel conversion (HTK scale)**:
   $$m(f) = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

2. **Mel to Hertz conversion**:
   $$f(m) = 700 \cdot \left(10^{m / 2595} - 1\right)$$

3. **Center frequencies**:
   Compute 82 equally spaced points in the mel domain between $m(20.0)$ and $m(7600.0)$:
   $$m_i = m(20) + i \cdot \frac{m(7600) - m(20)}{81}, \quad i = 0, 1, \dots, 81$$
   Convert $m_i$ back to linear frequencies $f_i$ and round to nearest FFT bin:
   $$b_i = \text{round}\left( \frac{400 \cdot f_i}{16000} \right)$$

4. **Triangular weighting with Slaney normalization**:
   For each filter $m \in [0, 79]$ and frequency bin $f \in [0, 200]$:
   $$H_m[f] = \begin{cases}
   \frac{f - b_m}{b_{m+1} - b_m}, & b_m \le f < b_{m+1} \\
   \frac{b_{m+2} - f}{b_{m+2} - b_{m+1}}, & b_{m+1} \le f \le b_{m+2} \\
   0, & \text{otherwise}
   \end{cases}$$
   Area normalization factor:
   $$\hat{H}_m[f] = H_m[f] \cdot \frac{2}{f_{m+2} - f_m}$$

5. **Mel Energy Calculation**:
   $$E_k[m] = \sum_{f=0}^{200} P_k[f] \cdot \hat{H}_m[f], \quad m = 0, 1, \dots, 79$$

6. **Log Compression**:
   $$L_k[m] = \ln\left( \max(E_k[m], 10^{-5}) \right)$$

---

## 6. Global Cumulative Mean & Variance Normalization (CMVN)

Awaaz strictly uses **global** normalization instead of per-utterance normalization to support instantaneous, chunk-by-chunk streaming inference without requiring look-ahead or whole-audio statistics.

Global statistics are stored in `cmvn.npz`:
- `mean`: Vector $\mu \in \mathbb{R}^{80}$
- `std`: Vector $\sigma \in \mathbb{R}^{80}$

Normalized frame feature vector $\hat{L}_k \in \mathbb{R}^{80}$:

$$\hat{L}_k[m] = \frac{L_k[m] - \mu[m]}{\sigma[m] + 10^{-5}}, \quad m = 0, 1, \dots, 79$$

---

## 7. Numerical Parity Acceptance Target

When implementing this pipeline natively on Android (Kotlin/C++) or iOS (Swift/Metal):
* Max Absolute Difference across all bins: $\max |\hat{L}_{\text{native}} - \hat{L}_{\text{reference}}| < 1.0 \times 10^{-3}$.
* Mean Absolute Difference: $< 1.0 \times 10^{-4}$.
