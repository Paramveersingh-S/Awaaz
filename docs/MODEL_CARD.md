# Model Card for Awaaz (ਆਵਾਜ਼)

## Model Details
- **Architecture**: Streaming Conformer-CTC with Rotary Positional Embeddings (RoPE)
- **Presets**:
  - `awaaz-tiny`: ~6M parameters (~6 MB int8) — targeted for smartwatches & low-end wearables
  - `awaaz-small`: ~19M parameters (~19 MB int8) — targeted for mobile smartphones & embedded Linux
  - `awaaz-base`: ~55M parameters (~55 MB int8) — targeted for server & high-accuracy offline nodes
- **Input**: 16,000 Hz single-channel mono audio
- **Feature Extraction**: 80-bin log-mel filterbanks, 25 ms window, 10 ms hop, global CMVN
- **Output**: English and Punjabi (Gurmukhi script) text
- **Tokenizer**: 1024-token SentencePiece unigram model
- **License**: Apache-2.0

## Intended Use
- **Primary Use**: Real-time on-device speech-to-text, voice commands, and conversational transcription in English, Punjabi, and code-switched bilingual utterances.
- **Target Environments**: Wear OS smartwatches, Android phones, iOS devices, and edge Linux boards (e.g. Raspberry Pi).

## Explicitly Out-of-Scope & Prohibited Uses
- **Sacred / Religious Text Transcription**: v1 models are strictly not trained on Gurbani, kirtan, or katha recitation per §1.5 Sacred-Text Policy. Awaaz must not be deployed for religious transcription where inaccuracies alter spiritual meaning.
- **High-Stakes Applications**: Legal testimony transcription, medical diagnostics, biometric voice identification, or surveillance.

## Training Data & Policy
- Candidate Corpora: Common Voice (EN/PA), FLEURS (EN/PA), LibriSpeech (EN), and AI4Bharat Indic speech collections.
- Sacred Exclusion: All audio marked `"domain": "sacred"` is filtered out prior to tokenizer creation and model training.
- Dataset audit and licensing terms are logged in `DATA_LICENSES.md`.

## Evaluation & Metrics
| Benchmark Split | Metric | Offline Model | Streaming (0.64s chunk) | int8 Quantized |
|---|---|---|---|---|
| English Clean | WER | *TBD (Colab run)* | *TBD (Colab run)* | *TBD (Colab run)* |
| Punjabi Clean | CER / WER | *TBD (Colab run)* | *TBD (Colab run)* | *TBD (Colab run)* |
| Code-Switched (EN+PA) | CER / WER | *TBD (Colab run)* | *TBD (Colab run)* | *TBD (Colab run)* |
| On-Device Mic (Watch/Phone) | CER / WER | *TBD (Phase 13)* | *TBD (Phase 13)* | *TBD (Phase 13)* |

*Note: In accordance with Rule 3 of AGENTS.md, all benchmark metrics remain unpopulated until measured through reproducible scripts.*

## Ethical Considerations & Limitations
- **Dialect Sensitivity**: Punjabi features distinct regional dialects (Majhi, Malwai, Doabi, Powadhi). Accuracy may fluctuate across accented colloquialisms.
- **Acoustic Noise**: Near-field watch microphones capture significant wind and wrist-movement noise; preprocessing via VAD and optional enhancement is advised.
