"""Phase 0 Acceptance Tests — Repo Bootstrap & Skeleton Verification."""

import json
from pathlib import Path

import pytest
import soundfile as sf
import yaml

import awaaz
from awaaz.cli import build_parser, main
from scripts.make_fixture import create_fixture


def test_package_metadata():
    """Verify package version and metadata."""
    assert hasattr(awaaz, "__version__")
    assert awaaz.__version__ == "0.1.0"


def test_cli_interface():
    """Verify CLI parser construction and help/version exit behavior."""
    parser = build_parser()
    assert parser.prog == "awaaz"

    # Main with --help should return code 0
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0

    # Running main without subcommands should return 0
    assert main([]) == 0


def test_required_docs_exist():
    """Verify AGENTS.md, SPEC.md, PROGRESS.md, DECISIONS.md, and DATA_LICENSES.md exist."""
    root = Path(__file__).resolve().parent.parent
    assert (root / "AGENTS.md").is_file()
    assert (root / "docs" / "SPEC.md").is_file()
    assert (root / "docs" / "PROGRESS.md").is_file()
    assert (root / "docs" / "DECISIONS.md").is_file()
    assert (root / "docs" / "MODEL_CARD.md").is_file()
    assert (root / "DATA_LICENSES.md").is_file()
    assert (root / "LICENSE").is_file()


def test_all_14_phase_files_exist():
    """Verify each phase has a modularized markdown file in docs/phases/."""
    root = Path(__file__).resolve().parent.parent
    phases_dir = root / "docs" / "phases"
    assert phases_dir.is_dir()

    expected_phases = [
        "PHASE_00_REPO_BOOTSTRAP.md",
        "PHASE_01_TEXT_NORMALIZATION_TOKENIZER.md",
        "PHASE_02_AUDIO_FEATURES_AUGMENTATION.md",
        "PHASE_03_DATA_PIPELINE.md",
        "PHASE_04_MODEL.md",
        "PHASE_05_TRAINING_LOOP.md",
        "PHASE_06_COLAB_RUNNER_PERSISTENCE.md",
        "PHASE_07_EVALUATION_DECODING.md",
        "PHASE_08_STREAMING_INFERENCE.md",
        "PHASE_09_EXPORT_QUANTIZATION_EDGE.md",
        "PHASE_10_FULL_PIPELINE.md",
        "PHASE_11_OPTIONAL_V2_MODELS.md",
        "PHASE_12_DEMO_DOCS_RELEASE.md",
        "PHASE_13_EDGE_APPS.md",
    ]

    for fname in expected_phases:
        phase_path = phases_dir / fname
        assert phase_path.is_file(), f"Missing phase specification: {fname}"
        content = phase_path.read_text(encoding="utf-8")
        assert len(content) > 100, f"Phase file {fname} is suspiciously short"


def test_config_files_valid_yaml():
    """Verify all YAML configs parse cleanly and contain expected model fields."""
    root = Path(__file__).resolve().parent.parent
    configs_dir = root / "configs"

    for preset in ["tiny.yaml", "small.yaml", "base.yaml"]:
        cfg_file = configs_dir / preset
        assert cfg_file.is_file(), f"Missing config: {preset}"
        with open(cfg_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "model" in data
        assert "frontend" in data
        assert "training" in data
        assert data["model"]["vocab_size"] == 1024
        assert data["model"]["n_mels"] == 80
        assert data["frontend"]["sample_rate"] == 16000

    data_sources = configs_dir / "data_sources.yaml"
    assert data_sources.is_file()
    with open(data_sources, "r", encoding="utf-8") as f:
        src_data = yaml.safe_load(f)
    assert "sources" in src_data


def test_fixture_generation_and_integrity(tmp_path):
    """Verify synthetic fixture generator runs in seconds and produces valid 16kHz audio."""
    train_m, dev_m = create_fixture(str(tmp_path), num_train=4, num_dev=2)

    assert Path(train_m).is_file()
    assert Path(dev_m).is_file()

    with open(train_m, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 4
    for item in lines:
        assert "audio_path" in item
        assert "text" in item
        assert "duration" in item
        assert "language" in item
        assert "domain" in item
        assert item["language"] in ["en", "pa"]

        # Verify audio file validity
        audio_p = Path(item["audio_path"])
        assert audio_p.is_file()
        info = sf.info(str(audio_p))
        assert info.samplerate == 16000
        assert info.channels == 1
        assert info.format == "FLAC"
        assert abs(info.duration - item["duration"]) < 0.05
