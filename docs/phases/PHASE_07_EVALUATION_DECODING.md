# PHASE 7 — Evaluation and Decoding

## 1. Goal
Provide rigorous, auditable evaluation metrics and decoding mechanisms across offline and streaming configurations for both English and Punjabi.

## 2. Technical Specifications
### 2.1 Decoding Modules (`awaaz/decode.py`)
- **Greedy CTC Decoder**:
  - Takes log-probability matrix $(T, V)$.
  - Applies $\text{argmax}$ across vocab dimension for each frame.
  - Collapses consecutive duplicate token IDs.
  - Removes CTC blank token (ID 0).
  - Detokenizes remaining IDs using SentencePiece unigram model.
- **CTC Word Timestamp Extractor**:
  - Utilizes the frame-to-time mapping:
    $$\Delta t = 10\text{ ms} \times \text{subsample\_factor} = 40\text{ ms (for factor 4)}$$
  - Identifies word boundaries based on SentencePiece leading whitespace marker (` ` / `\u2581`).
  - Estimates word start time as the first non-blank emitting frame of the token.
  - Estimates word end time as the frame after the final emitting subword.
- **Optional N-gram LM Rescoring (Feature-flagged)**:
  - Integration interface for KenLM 3-gram/4-gram word language model.
  - Tunable LM weight $\alpha$ and word insertion bonus $\beta$.
  - Tuned strictly on `dev` splits, never on test splits.

### 2.2 Metrics Calculation (`awaaz/metrics.py`)
- Word Error Rate (WER) and Character Error Rate (CER) computed using JiWER on normalized reference and hypothesis texts.
- Robust handling of empty references.
- Primary metric policy:
  - **English**: WER is primary.
  - **Punjabi (Gurmukhi)**: **CER is primary**, because orthographic variants and split nuktas unfairly penalize WER without semantic error.

### 2.3 Evaluation Pipeline (`awaaz/evaluate.py`)
- Evaluates checkpoints or ONNX models on test manifests across 3 operational modes:
  1. **Offline**: Full-context attention.
  2. **Stream-0.64s**: Dynamic-chunk mask with chunk size 16 frames (0.64s), left chunks 8.
  3. **Stream-0.32s**: Dynamic-chunk mask with chunk size 8 frames (0.32s), left chunks 4.
- Generates markdown results table and JSON telemetry.
- Exports top-20 worst-WER utterances with reference vs. hypothesis diffs for error triage.

### 2.4 Checkpoint Averaging (`awaaz/average.py`)
- Computes parameter-wise arithmetic mean across the top $N$ (default 10) checkpoints.
- Evaluates averaged weights against individual checkpoints to confirm expected perplexity/WER gains.

## 3. Mandatory Tests & Acceptance Checks
- Unit tests on hand-crafted posteriors (repeats, blanks between repeats, leading/trailing blanks).
- Exact timestamp calculation verification on synthetic posteriors.
- Checkpoint averaging verification on dummy model states.
- End-to-end evaluation run against fixture dataset producing markdown report.

## 4. Definition of Done
Unit tests pass; evaluation runner produces structured reports; error-analysis export verified.
