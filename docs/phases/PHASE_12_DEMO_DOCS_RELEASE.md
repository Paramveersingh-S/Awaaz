# PHASE 12 — Demo, Documentation, Model Card, and Release

## 1. Goal
Publish an enterprise-grade open-source release with an interactive Gradio demo on Hugging Face Spaces, complete reproducibility packages, and a comprehensive model card.

## 2. Technical Specifications
### 2.1 Gradio Web Application (`app/app.py`)
- Deployed to Hugging Face Spaces.
- Features:
  - Audio file upload and live microphone recording.
  - Automatic language identification display.
  - Dynamic chunk streaming visualization.
  - Optional speaker diarization and audio smoothing toggle.
  - Export transcripts to TXT, SRT, and JSON.
  - Prominent notices regarding the Sacred-Text Policy and intended use.

### 2.2 Model Card (`docs/MODEL_CARD.md`)
Following Hugging Face and ML Commons standards:
1. **Model Summary**: Bilingual Conformer-CTC (6M to 55M parameters).
2. **Intended Use**: General conversation, transcription, voice search on edge devices.
3. **Out-of-Scope Use Cases**:
   - Sacred religious recitation / Gurbani transcription.
   - Medical transcription, legal proceedings, surveillance, or biometric identification.
4. **Training Data & Licensing**: Itemized breakdown of datasets and explicit exclusion of sacred domains.
5. **Evaluation Results**: Detailed WER and CER tables across English, Punjabi, and code-switched audio.
6. **Known Limitations & Biases**: Punjabi dialect variations (Malwai, Majhi, Doabi, Powadhi), background noise sensitivity, fast speech rates.
7. **License**: Apache-2.0.

### 2.3 Reproducibility & Release Packaging
- Git tagged releases (`v0.1.0`).
- Model checkpoints on Hugging Face Hub: `pytorch_model.bin`, `awaaz_offline.onnx`, `awaaz_stream.onnx`, `awaaz_int8.onnx`, `tokenizer.model`, `cmvn.npz`, `config.yaml`.
- Community governance files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and issue templates for transcription bug reports.

## 3. Mandatory Tests & Acceptance Checks
- Clean installation in a fresh Python virtual environment.
- Gradio demo launches locally without runtime warnings.
- All numbers cited in README and Model Card trace to generated report files.

## 4. Definition of Done
Open-source release is completely documented and easily cloneable by external developers.
