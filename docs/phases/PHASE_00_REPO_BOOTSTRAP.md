# PHASE 0 — Repo Bootstrap

## 1. Goal
Establish a clean, robust, and fully testable skeleton for the Awaaz repository so every subsequent phase slots in without architectural friction or missing dependencies.

## 2. Directory Layout
```
awaaz/
├── README.md                           # Enterprise documentation & project showcase
├── LICENSE                             # Apache-2.0
├── DATA_LICENSES.md                    # Curated registry of dataset licenses & terms
├── pyproject.toml                      # Modern packaging & entry points
├── AGENTS.md                           # Global agent rules (verbatim from §2)
├── .gitignore                          # Clean repository hygiene
├── .github/
│   └── workflows/
│       └── ci.yml                      # Automated linting and fast CPU testing
├── docs/
│   ├── SPEC.md                         # Master specification index & context
│   ├── PROGRESS.md                     # Living development log & phase handoff notes
│   ├── DECISIONS.md                    # Architecture decision records (ADRs)
│   ├── MODEL_CARD.md                   # Enterprise model card & ethics disclosures
│   └── phases/                         # Modular phase documentation
│       ├── PHASE_00_REPO_BOOTSTRAP.md
│       ├── PHASE_01_TEXT_NORMALIZATION_TOKENIZER.md
│       ├── ...
│       └── PHASE_13_EDGE_APPS.md
├── configs/
│   ├── tiny.yaml                       # ~6M param watch preset
│   ├── small.yaml                      # ~19M param phone/embedded preset
│   ├── base.yaml                       # ~55M param high-accuracy preset
│   └── data_sources.yaml               # Candidate datasets & manifests registry
├── awaaz/
│   ├── __init__.py                     # Package init & version (__version__ = "0.1.0")
│   ├── text.py                         # Normalization & SentencePiece tokenizer
│   ├── features.py                     # 80-bin Log-Mel frontend, CMVN, SpecAugment
│   ├── dataset.py                      # JSONL manifest loader, duration bucketing
│   ├── model.py                        # Streaming Conformer-CTC with RoPE
│   ├── decode.py                       # Greedy CTC & word timestamp extractor
│   ├── metrics.py                      # Robust WER & CER computation
│   ├── train.py                        # Resumable dynamic-chunk training loop
│   ├── average.py                      # Model checkpoint averaging utility
│   ├── evaluate.py                     # Offline & streaming evaluation suite
│   ├── stream.py                       # Incremental chunk streaming recognizer
│   ├── export.py                       # ONNX export, int8 dynamic quantization & parity
│   ├── cli.py                          # Unified CLI entry point (`awaaz`)
│   └── pipeline/                       # Multi-stage production pipeline
│       ├── __init__.py
│       ├── vad.py                      # Voice activity detection & smart chunking
│       ├── diarize.py                  # Speaker embedding extraction & clustering
│       ├── enhance.py                  # Audio denoising & smoothing interface
│       ├── separate.py                 # Multi-speaker separation interface
│       └── merge.py                    # Alignment of words, timestamps & speakers
├── scripts/
│   ├── make_fixture.py                 # Fast CPU synthetic ASR fixture generator
│   ├── prepare_hf.py                   # Hugging Face audio streaming & conversion
│   └── merge_manifests.py              # Manifest merger with balancing & repeat factors
├── notebooks/
│   ├── colab_prepare_data.ipynb        # Data preparation & Drive persistence
│   ├── colab_train.ipynb               # Resumable training loop on Colab GPU
│   └── colab_export.ipynb              # Checkpoint averaging, ONNX export & upload
├── tests/
│   ├── __init__.py
│   ├── conftest.py                     # Pytest fixtures & synthetic data hooks
│   └── test_phase0.py                  # Repo skeleton, packaging & fixture tests
├── app/
│   └── app.py                          # Hugging Face Spaces Gradio demo
└── edge/
    ├── README.md                       # On-device deployment architecture
    ├── android/                        # Android / Wear OS native notes & reference
    └── apple/                          # Core ML & iOS / watchOS notes
```

## 3. Specifications to Build
1. **`pyproject.toml`**:
   - Package name: `awaaz`
   - Python: `>=3.10`
   - Dependencies: `torch`, `torchaudio`, `soundfile`, `sentencepiece`, `jiwer`, `pyyaml`, `huggingface_hub`, `numpy`
   - Optional extras:
     - `pipeline`: `silero-vad`, `speechbrain`, `scikit-learn`
     - `export`: `onnx`, `onnxruntime`
     - `dev`: `pytest`, `ruff`
   - Console script entrypoint: `awaaz = "awaaz.cli:main"`
2. **`AGENTS.md`**: Global agent instructions matching §2 of the build spec.
3. **`docs/PROGRESS.md`**: Structured progress tracking using the Appendix C handoff template.
4. **`docs/DECISIONS.md`**: Initial architecture decisions and constants.
5. **`DATA_LICENSES.md`**: Table tracking candidate datasets, licenses, attribution terms, and verification status.
6. **`scripts/make_fixture.py`**:
   - Synthesizes 16 kHz mono FLAC audio files (pure tones, chirps, white/pink noise).
   - Generates accompanying English and Gurmukhi transcripts (avoiding digits).
   - Produces clean JSON Lines manifest files for train and dev.
   - Executes in `< 30 seconds` on any dual-core CPU.
7. **CI Workflow (`.github/workflows/ci.yml`)**:
   - Sets up Python 3.11.
   - Installs dependencies.
   - Lints with `ruff`.
   - Runs `pytest` test suite on CPU.

## 4. Acceptance Checks
- `pip install -e .` or package structure validation succeeds.
- `python scripts/make_fixture.py --output_dir tests/fixtures` completes in `< 30s` and outputs valid 16 kHz FLAC files and JSONL manifests.
- `pytest -v tests/test_phase0.py` passes completely.
- `python -m awaaz.cli --help` exits cleanly with code 0.

## 5. Definition of Done
Repo installs cleanly, CI is configured, fixture generator works reliably, core modules have proper docstrings, and docs are initialized.
