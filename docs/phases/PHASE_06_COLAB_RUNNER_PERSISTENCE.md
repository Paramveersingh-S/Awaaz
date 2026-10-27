# PHASE 6 — Colab Runner and Persistence

## 1. Goal
Provide three streamlined, highly robust Jupyter notebooks that enable running full-scale GPU training on Google Colab Pro with zero friction and guaranteed persistence across disconnections.

## 2. Technical Specifications
All notebooks are thin runners that call packaged CLI commands from `awaaz` rather than embedding custom training logic.

### 2.1 Notebook 1: `notebooks/colab_prepare_data.ipynb`
1. Mount Google Drive (`/content/drive/MyDrive/awaaz`).
2. Read secrets (Hugging Face token) securely via `google.colab.userdata`.
3. Install package: `pip install -e .[pipeline,export]`.
4. Run dataset downloads for Common Voice, FLEURS, and Indic corpora using `scripts/prepare_hf.py`.
5. Execute `scripts/merge_manifests.py` to balance English and Punjabi splits.
6. Package prepared audio into tar archives on Google Drive with checksum verification.
7. Compute global `cmvn.npz` and train bilingual SentencePiece tokenizer.
8. Output transparent reporting of distinct hours vs. repeated hours.

### 2.2 Notebook 2: `notebooks/colab_train.ipynb`
1. Mount Google Drive and install package.
2. Untar audio dataset from Google Drive to fast **local NVMe disk** (`/content/data/`). Never train directly across Google Drive FUSE filesystem.
3. Detect GPU architecture (Tesla T4, L4, A100-SXM4) and dynamically suggest optimal `batch_seconds` (e.g. 180s on T4, 400s on L4, 800s on A100).
4. Run throughput benchmark (50 steps) to measure audio seconds per second and calculate required `total_steps` for target budget.
5. Launch `python -m awaaz.train --config configs/small.yaml --run_dir /content/drive/MyDrive/awaaz/runs/small_v1`.
6. Provide one-click resumption instructions ("re-run this cell to resume").

### 2.3 Notebook 3: `notebooks/colab_export.ipynb`
1. Average the top-10 checkpoint weights via `awaaz.average`.
2. Run validation suite and compute offline and streaming WER/CER on English and Punjabi test sets.
3. Export PyTorch model to ONNX (opset 17) for both offline and streaming modes.
4. Run ONNX Runtime int8 dynamic quantization and generate numerical parity report.
5. Push release weights, ONNX models, tokenizer, CMVN, and model card to Hugging Face Hub.

### 2.4 TPU Roadmap Architecture (`docs/TPU.md`)
- Document why standard PyTorch audio pipelines struggle on TPUs (XLA recompilations on variable audio lengths, limited CTC kernel support).
- Outline the JAX/Flax bucketed architecture roadmap for future TPU cluster scale-up.

## 3. Acceptance Checks
- Notebooks follow clean syntax and validate without import or execution errors on clean sessions.
- No keep-alive scripts or terms-of-service violations.
- Resumption verified by stopping training after 100 steps and re-executing.

## 4. Human Step
Open and execute notebooks in Google Colab Pro; run a 15-minute smoke test; report throughput numbers back.

## 5. Definition of Done
All three notebooks are committed, documented, and tested; `docs/TPU.md` is published.
