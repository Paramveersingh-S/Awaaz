# PHASE 3 — Data Pipeline

## 1. Goal
Convert diverse open-source audio datasets into structured, verified JSON Lines manifests and deliver an efficient, streaming-friendly training pipeline.

## 2. Technical Specifications
### 2.1 Fixed Manifest Format (JSON Lines)
Each line is a self-contained JSON object:
```json
{
  "audio_path": "data/audio/train/clip_0001.flac",
  "text": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਜੀ",
  "duration": 4.52,
  "language": "pa",
  "domain": "general",
  "source": "common_voice_pa"
}
```
- Audio specifications: 16,000 Hz, mono, FLAC format, duration 0.5s – 20.0s.
- `domain`: optional tag (e.g. `"sacred"` for religious recitations/kirtan/katha).
- `source`: identifier of originating dataset.

### 2.2 Dataset Preparation Script (`scripts/prepare_hf.py`)
- Streams Hugging Face audio datasets.
- Requests undecoded raw audio chunks, decoding via `soundfile` + high-fidelity resampler (avoiding unstable torchaudio backends).
- Fully resumable: checks existing FLAC targets before decoding.
- Error tolerance: logs corrupted audio files and proceeds.
- Supports `--max_items N` for fast developer sanity checks.
- Produces balanced source tracking and split manifests (`train.jsonl`, `dev_en.jsonl`, `dev_pa.jsonl`, `test.jsonl`).

### 2.3 Sacred-Text Policy Enforcement (§1.5)
- Clips tagged with `"domain": "sacred"` are **strictly excluded** from training, validation, and testing sets by default.
- Controlled via `exclude_sacred: true` in dataset configurations.

### 2.4 Dataset & Sampler (`awaaz/dataset.py`)
- **`AwaazDataset`**: Loads waveform (float32, 16 kHz), applies Phase-1 text normalization, maps tokens using SentencePiece, filters out rejected/digit-bearing utterances.
- **`CollateFn`**: Pads audio waveforms to batch maximum; outputs 1D concatenated targets and target lengths (PyTorch `nn.CTCLoss` format).
- **`DurationBucketedBatchSampler`**:
  - Groups clips by audio duration to minimize padding waste.
  - Limits each batch by total *padded audio seconds* (e.g. 400 seconds).
  - Shuffles batches every epoch with `set_epoch(epoch)`.
  - Configured for Colab worker optimization (pinned memory, persistent workers).

### 2.5 Balancing & Merging (`scripts/merge_manifests.py`)
- Merges English and Punjabi manifests with controllable repetition factors.
- Cap English hours to target 40%–60% Punjabi representation in training mixtures.
- Generates transparent metrics distinguishing *distinct audio hours* from *effective training hours with repeats*.

## 3. Mandatory Tests & Acceptance Checks
- Manifest round-trip: JSONL -> Dataset -> Batch -> valid tensor dimensions and lengths.
- Sampler coverage: every item appears exactly once per epoch; batch padded duration does not exceed budget.
- Policy tests: verify that items with digits, sacred domains, empty text, or durations outside [0.5, 20.0]s are properly filtered out.

## 4. Human Step
Run `colab_prepare_data.ipynb` in Colab with Hugging Face access tokens. Download and prepare FLEURS, Common Voice, and Indic datasets. Tar prepared datasets to Google Drive.

## 5. Definition of Done
Unit tests pass on synthetic fixture manifests; Colab preparation notebook dry-runs cleanly with `--max_items 20`; `DATA_LICENSES.md` is populated.
