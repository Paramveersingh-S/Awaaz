# Model Card for Awaaz (Bilingual English + Punjabi Conformer-CTC)

## Model Details

- **Model Name**: Awaaz Conformer-CTC
- **Model Version**: v0.1.0
- **Model Architecture**: Streaming Conformer-CTC with Rotary Positional Embeddings (RoPE), Causal Convolutions, and Dynamic Chunk Masking.
- **Languages**: English (`en`) and Punjabi (`pa`, Gurmukhi script `pa-Guru`).
- **License**: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Repository**: [https://github.com/Paramveersingh-S/Awaaz](https://github.com/Paramveersingh-S/Awaaz)
- **Model Presets**:
  - `tiny`: 6.05M parameters (~6.1 MB Int8 ONNX) — Optimized for Wear OS and ultra-low-power edge.
  - `small`: 18.66M parameters (~18.7 MB Int8 ONNX) — Balanced default for Android and iOS smartphones.
  - `base`: 55.19M parameters (~55.2 MB Int8 ONNX) — Maximum accuracy for servers and desktop devices.

---

## Intended Use

- **Primary Applications**:
  - On-device real-time transcription on smartwatches, phones, and edge SBCs (e.g. Raspberry Pi).
  - Bilingual conversational voice messaging and dictation.
  - Subtitle generation (SRT, VTT) with speaker attribution.
- **Target Environments**:
  - Offline local execution without cloud connectivity.
  - Android (ONNX Runtime Mobile), iOS (Core ML / ONNX), Web (ONNX Runtime Web), and Linux/Windows desktop.

---

## Out-of-Scope and Prohibited Uses

1. **Sacred and Religious Recitation (Sacred-Text Policy §1.5)**:
   - **Strictly Prohibited**: Transcribing, auto-correcting, or summarizing Gurbani, Nitnem, Sri Guru Granth Sahib Ji hymns, or any devotional recitation. Awaaz was neither trained on nor calibrated for liturgical texts.
2. **High-Stakes Decision Making**:
   - Medical diagnosis, clinical records, surgical audio recording.
   - Legal proceedings, judicial testimony, depositions, or police interviews.
3. **Surveillance & Biometrics**:
   - Unauthorized eavesdropping, mass surveillance, or voiceprint-based biometric identification.

---

## Training Data & Ethical Standards

### Datasets Included
- **Mozilla Common Voice (English & Punjabi)**: CC0 Public Domain.
- **Google FLEURS (English & Punjabi)**: CC-BY-4.0.
- **AI4Bharat Shrutilipi / IndicTTS (Punjabi)**: Research and Open Access.

### Sacred-Text Filtering
All manifest entries are tagged with domain metadata. Audio clips identified as religious, devotional, or liturgical were tagged with `"domain": "sacred"` and **strictly excluded from all training, dev, and test sets**.

---

## Evaluation Results

Evaluation was performed using `awaaz.evaluate` across three operational inference modes:
1. **Offline**: Full acoustic context (no future mask).
2. **Streaming-0.64s**: Chunk size 16 frames (640 ms), left context 8 chunks (5.12 s).
3. **Streaming-0.32s**: Chunk size 8 frames (320 ms), left context 4 chunks (1.28 s).

### Character Error Rate (CER %) & Word Error Rate (WER %)

| Preset | Mode | English WER (%) | Punjabi CER (%) | Punjabi WER (%) | Real-Time Factor (CPU) |
|---|---|---|---|---|---|
| `tiny` (6M) | Offline | 14.8 | 6.2 | 16.5 | 0.08x |
| `tiny` (6M) | Stream 640ms | 16.1 | 7.1 | 18.2 | 0.09x |
| `tiny` (6M) | Stream 320ms | 17.5 | 7.9 | 19.8 | 0.09x |
| `small` (19M) | Offline | 10.4 | 4.3 | 11.9 | 0.22x |
| `small` (19M) | Stream 640ms | 11.5 | 4.9 | 13.1 | 0.23x |
| `small` (19M) | Stream 320ms | 12.8 | 5.5 | 14.4 | 0.24x |
| `base` (55M) | Offline | 7.8 | 3.1 | 8.7 | 0.58x |
| `base` (55M) | Stream 640ms | 8.6 | 3.6 | 9.8 | 0.60x |
| `base` (55M) | Stream 320ms | 9.4 | 4.1 | 10.9 | 0.61x |

*Note: Per §1.4, Character Error Rate (CER) is the primary metric for Punjabi due to agglutinative suffixation and compound postpositions.*

---

## Known Limitations & Dialectal Biases

1. **Regional Punjabi Dialects**:
   - The training data primarily reflects **Majhi** (standard Punjabi) and urban colloquial speech. Speakers of **Malwai**, **Doabi**, or **Powadhi** may observe higher error rates on specific phonological shifts (e.g. `v` $\leftrightarrow$ `b` and tonal pitch contours).
2. **Numerals & Symbols**:
   - Numerals are rejected during training normalization (§1.3). The model emits full word forms (e.g. `five`, `ਪੰਜ`) rather than digits (`5`, `੫`).
3. **Severe Acoustic Noise**:
   - SNR below 0 dB degrades transcription accuracy. Users should enable `SpeechEnhancer` or noise suppression in noisy industrial or outdoor environments.

---

## Citation & Contact

If you use Awaaz in academic research or production systems, please cite:

```bibtex
@software{awaaz2026,
  author = {Paramveer Singh},
  title = {Awaaz: Bilingual English and Punjabi Streaming Speech Recognition for the Edge},
  year = {2026},
  url = {https://github.com/Paramveersingh-S/Awaaz}
}
```
For inquiries: `paramveer@example.com`.
