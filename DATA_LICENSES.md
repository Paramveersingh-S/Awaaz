# DATA_LICENSES.md — Dataset Registry & Licensing Terms

This document logs all candidate and verified datasets used in the Awaaz project, their licensing terms, access conditions, and verification status in adherence to Rule 8 of `AGENTS.md`.

## Dataset Licensing Registry

| Dataset Name | Source / URL | Language | Expected License | Access Terms | Intended Usage | Verification Status | Date Added |
|---|---|---|---|---|---|---|---|
| **Common Voice (English)** | Mozilla / Hugging Face `mozilla-foundation/common_voice_16_1` | English (`en`) | CC0-1.0 | Open download | Train / Dev / Test | Verified (CC0-1.0) | 2026-10-27 |
| **Common Voice (Punjabi)** | Mozilla / Hugging Face `mozilla-foundation/common_voice_16_1` | Punjabi (`pa-IN`) | CC0-1.0 | Open download | Train / Dev / Test | Verified (CC0-1.0) | 2026-10-27 |
| **FLEURS (English)** | Google / Hugging Face `google/fleurs` | English (`en_us`) | CC-BY-4.0 | Open download (Attribution required) | Dev / Test Benchmark | Verified (CC-BY-4.0) | 2026-10-27 |
| **FLEURS (Punjabi)** | Google / Hugging Face `google/fleurs` | Punjabi (`pa_in`) | CC-BY-4.0 | Open download (Attribution required) | Dev / Test Benchmark | Verified (CC-BY-4.0) | 2026-10-27 |
| **LibriSpeech ASR** | OpenSLR / Hugging Face `librispeech_asr` | English (`en`) | CC-BY-4.0 | Open download (Public domain audiobooks) | Train (clean subset) | Verified (CC-BY-4.0) | 2026-10-27 |
| **IndicVoices (Punjabi)** | AI4Bharat / Hugging Face `ai4bharat/IndicVoices` | Punjabi (`pa`) | CC-BY-4.0 / Research | Gated repository (Human token required) | Train / Dev | UNVERIFIED — human to confirm | 2026-10-27 |
| **Punjabi Wikipedia (Text)** | Wikimedia Foundation dumps | Punjabi (`pa`) | CC-BY-SA-3.0 | Open download | Tokenizer & LM text | Verified (CC-BY-SA-3.0) | 2026-10-27 |
| **Synthetic Fixture** | Internal generator `scripts/make_fixture.py` | EN + PA | Apache-2.0 | Generated locally on CPU | Fast CPU tests | Verified (Self-generated) | 2026-10-27 |

## Licensing Compliance Rules
1. **Commercial Usability**: The Awaaz code is released under Apache-2.0. Model weights can only be distributed with commercial freedom if all underlying training datasets permit commercial derivative works. Datasets tagged Non-Commercial (NC) will produce research-only checkpoints.
2. **Attribution**: CC-BY-4.0 datasets (FLEURS, LibriSpeech) are acknowledged in model card releases and repo documentation.
3. **Sacred Text Exclusion**: In accordance with §1.5 of `docs/SPEC.md`, all audio tagged with `"domain": "sacred"` is filtered out from training and evaluation splits.
