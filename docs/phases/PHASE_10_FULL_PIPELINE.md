# PHASE 10 — Full Pipeline: VAD → ASR → Diarization → Enhancement → Transcript

## 1. Goal
Build an enterprise multi-stage audio processing pipeline that takes raw meeting or conversational audio and generates clean, speaker-attributed, timestamped transcripts with per-speaker denoised audio tracks.

## 2. Technical Specifications (`awaaz/pipeline/` & `awaaz/cli.py`)
### 2.1 Multi-Stage Architecture
| Pipeline Stage | Module | Watch Capability | Phone / Server Capability |
|---|---|---|---|
| Voice Activity Detection (VAD) | `awaaz.pipeline.vad` | Yes | Yes |
| Conformer ASR (Streaming/Offline) | `awaaz.model` | Yes | Yes |
| Audio Enhancement / Denoising | `awaaz.pipeline.enhance` | No (v1) | Yes |
| Speaker Diarization & Clustering | `awaaz.pipeline.diarize` | No | Yes |
| Timestamp & Speaker Merge | `awaaz.pipeline.merge` | Yes | Yes |
| Multi-Talker Separation | `awaaz.pipeline.separate` | No | Yes (v2) |

### 2.2 Voice Activity Detection (`awaaz/pipeline/vad.py`)
- Interfaces with pretrained lightweight VAD (e.g. Silero VAD).
- Segments audio stream into speech regions.
- Adds 0.15s padding before and after each speech burst.
- Merges neighbouring speech segments separated by gaps $< 0.4$ seconds.
- **Acoustic-Aware Chunk Splitting**: Long segments ($> 20$s) are cut at the lowest energy (quietest) frame near the 15–20s mark, preventing splitting words in half.

### 2.3 Speaker Diarization (`awaaz/pipeline/diarize.py`)
- Windows VAD segments into 1.5s sub-segments with 0.75s stride.
- Extracts speaker embeddings using pretrained models (e.g. ECAPA-TDNN via SpeechBrain or PyAnnote interface).
- L2-normalizes embedding vectors.
- Clusters vectors using Agglomerative Hierarchical Clustering with cosine distance metric.
- Exposes user-controllable distance threshold and `--speakers N` override.
- Includes diarization error scoring script (`scripts/tune_diarization.py`).

### 2.4 Audio Enhancement & Smoothing (`awaaz/pipeline/enhance.py`)
- Abstract `Enhancer` class wrapping models like DeepFilterNet.
- Fails gracefully with informative warning if dependencies are missing.
- **Evaluation Rule**: Transcribe with and without enhancement; default ASR configuration keeps enhancement off during transcription to avoid synthetic artifacts.

### 2.5 Transcript Merging & Export (`awaaz/pipeline/merge.py`)
- Maps CTC word timestamps to the dominant overlapping speaker window.
- Formats:
  - Text (`.txt`): Timestamped speaker-attributed paragraphs.
  - Subtitles (`.srt`, `.vtt`): Standard subtitle formats.
  - Structured (`.json`): Complete word tokens, confidences, exact start/end frames, and speaker IDs.
- Per-speaker audio track export: Writes isolated `.wav` files per speaker with silence on other turns.

### 2.6 Unified CLI (`awaaz/cli.py`)
Command interface:
```bash
awaaz transcribe audio.wav --ckpt models/awaaz.pt --tokenizer models/tokenizer.model --cmvn models/cmvn.npz --diarize --enhance --format srt --export-tracks output_dir/
awaaz stream --device 0
awaaz evaluate --manifest manifests/test.jsonl
```

## 3. Mandatory Tests & Acceptance Checks
- Segment splitter never outputs segments exceeding maximum duration or zero-length chunks.
- Merge logic validates speaker assignment across synthetic word sequences and speaker transitions.
- Export formats (TXT, SRT, VTT, JSON) validate against structural parsers.
- End-to-end dry run on synthetic 2-speaker fixture file succeeds and outputs valid formats.

## 4. Definition of Done
CLI runs end-to-end on synthetic audio; formats are validated; limitations are documented honestly.
