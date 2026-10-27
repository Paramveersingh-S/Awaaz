"""Unit tests for Phase 6 (Colab Runner & Persistence).

Validates:
1. Presence and syntax validity of the three Colab notebooks.
2. Mandatory operational cells in notebooks (Drive mount, local NVMe untar, resumption, export).
3. Existence and thoroughness of docs/TPU.md (recompilation, dynamic shapes, JAX roadmap).
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def notebooks_dir() -> Path:
    nb_dir = Path("notebooks")
    assert nb_dir.is_dir(), "notebooks/ directory must exist."
    return nb_dir


def test_colab_notebooks_exist_and_valid_json(notebooks_dir: Path):
    """Verifies all three notebooks exist and parse as valid nbformat 4 JSON."""
    expected_notebooks = [
        "colab_prepare_data.ipynb",
        "colab_train.ipynb",
        "colab_export.ipynb",
    ]

    for nb_name in expected_notebooks:
        nb_path = notebooks_dir / nb_name
        assert nb_path.is_file(), f"Notebook {nb_name} must exist."

        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "cells" in data, f"{nb_name} must contain 'cells'."
        assert data.get("nbformat") == 4, f"{nb_name} must be nbformat version 4."
        assert len(data["cells"]) > 0, f"{nb_name} must contain cells."


def test_colab_prepare_data_cells(notebooks_dir: Path):
    """Verifies colab_prepare_data.ipynb contains necessary pipeline stages."""
    nb_path = notebooks_dir / "colab_prepare_data.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_code = "".join(
        "".join(cell["source"]) for cell in data["cells"] if cell.get("cell_type") == "code"
    )

    # Required stages
    assert "drive.mount" in all_code
    assert "HF_TOKEN" in all_code
    assert "prepare_hf.py" in all_code
    assert "merge_manifests.py" in all_code
    assert "train_audio.tar" in all_code
    assert "train-tokenizer" in all_code
    assert "compute-cmvn" in all_code


def test_colab_train_cells(notebooks_dir: Path):
    """Verifies colab_train.ipynb contains local untar, GPU detection, benchmark, and resume."""
    nb_path = notebooks_dir / "colab_train.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_code = "".join(
        "".join(cell["source"]) for cell in data["cells"] if cell.get("cell_type") == "code"
    )

    # Required stages
    assert "drive.mount" in all_code
    assert "/content/data" in all_code  # Local NVMe disk
    assert "tar -xf" in all_code
    assert "torch.cuda.is_available()" in all_code
    assert "batch_seconds" in all_code
    assert "--resume" in all_code
    assert "awaaz.train" in all_code


def test_colab_export_cells(notebooks_dir: Path):
    """Verifies colab_export.ipynb contains averaging, evaluation, ONNX export, and quantization."""
    nb_path = notebooks_dir / "colab_export.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_code = "".join(
        "".join(cell["source"]) for cell in data["cells"] if cell.get("cell_type") == "code"
    )

    assert "drive.mount" in all_code
    assert "awaaz.average" in all_code
    assert "awaaz.evaluate" in all_code
    assert "awaaz.export" in all_code
    assert "--quantize" in all_code


def test_tpu_documentation():
    """Verifies docs/TPU.md exists and covers recompilation, dynamic shapes, and JAX roadmap."""
    tpu_doc = Path("docs/TPU.md")
    assert tpu_doc.is_file(), "docs/TPU.md must exist."

    content = tpu_doc.read_text(encoding="utf-8")
    assert "XLA" in content
    assert "recompilation" in content.lower()
    assert "dynamic chunk" in content.lower() or "mask" in content.lower()
    assert "ctc" in content.lower()
    assert "jax" in content.lower()
    assert "flax" in content.lower()
    assert "bucket" in content.lower()
