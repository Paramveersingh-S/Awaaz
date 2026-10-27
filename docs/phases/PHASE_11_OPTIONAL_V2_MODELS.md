# PHASE 11 — Optional v2 Models: Custom Enhancer and Speaker Separation

## 1. Goal
Design and prototype lightweight custom audio enhancement models tailored for wearable hardware, along with overlapped-speech separation for multi-talker conversations.

> **Condition**: This phase is started only after Phases 1–10 are verified and the human operator approves.

## 2. Technical Specifications
### 2.1 Custom On-Device Speech Enhancer (11A)
- **Goal**: Causal time-frequency denoising network with $< 500\text{k}$ parameters capable of running in real time on mobile/wearable processors.
- **Architecture**: Lightweight causal convolutional-recurrent model (e.g. GTCRN-style or Depthwise Separable Conv-GRU).
- **Training Strategy**:
  - Mix clean speech (our English + Punjabi corpora) with diverse noise sets (MUSAN, DNS Challenge) at SNRs from $-5\text{ dB}$ to $+20\text{ dB}$.
  - Incorporate room impulse response (RIR) convolutions.
  - Mix in real smartwatch acoustic background recordings.
- **Loss Function**: Multi-resolution compressed STFT magnitude and complex loss combined with SI-SNR.
- **Validation Metric**: Downstream ASR WER/CER improvement (not merely objective PESQ/STOI scores).

### 2.2 Overlapped-Speech Separation (11B)
- **Overlap Detection**: Identify regions containing multiple simultaneous speakers from diarization outputs.
- **Separation Architecture**:
  - Initial implementation: SepFormer or Conv-TasNet pretrained models.
  - Custom training: Synthesize 2-speaker overlapping mixtures with random temporal offsets and relative SNRs ($-5\text{ dB}$ to $+5\text{ dB}$).
  - Optimization with Permutation-Invariant Training (PIT) and SI-SNR loss.
- **Dual-Stream Decoding**: Feed separated output channels through Awaaz ASR and merge hypotheses chronologically.

## 3. Deliverables
- Model definition in `awaaz/pipeline/separate.py` and custom enhancer module.
- Training scripts and Colab notebooks for mixture synthesis.
- Comparative A/B evaluation report (with vs. without separation/enhancement).

## 4. Definition of Done
Prototype code passes unit tests; evaluation report documents downstream ASR impact honestly.
