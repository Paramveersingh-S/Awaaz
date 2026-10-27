"""Unit and integration tests for Phase 12: Demo, Documentation, Model Card, and Release.

Validates the Gradio web application, language identification, transcription callback,
model card compliance, and community governance files.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from app.app import build_app, detect_language_balance, transcribe_audio_pipeline


def test_language_balance_detector():
    """Verifies heuristic character-level language balance detection."""
    assert "English" in detect_language_balance("hello world this is an english sentence")
    assert "Punjabi" in detect_language_balance("ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਜੀ ਕੀ ਹਾਲ ਹੈ")
    assert "Code-Switched" in detect_language_balance("hello ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ world")
    assert detect_language_balance("") == "Unknown"


def test_app_transcribe_audio_pipeline(tmp_path: Path):
    """Verifies core Gradio transcription callback generates valid TXT, SRT, and JSON."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 400 * t).astype(np.float32)

    audio_path = tmp_path / "demo_audio.wav"
    sf.write(str(audio_path), audio, sr)

    txt_out, srt_out, json_out, lang_info = transcribe_audio_pipeline(
        audio_file=str(audio_path),
        mode="Offline (Full Context)",
        enable_diarization=True,
        enable_enhancement=False,
    )

    assert len(txt_out) > 0
    assert "SPEAKER_" in txt_out

    assert len(srt_out) > 0
    assert "-->" in srt_out

    data = json.loads(json_out)
    assert "segments" in data
    assert "words" in data
    assert len(data["segments"]) >= 1

    assert len(lang_info) > 0


def test_gradio_app_construction():
    """Verifies Gradio web app constructs cleanly without errors."""
    app = build_app()
    assert app is not None
    assert hasattr(app, "launch")


def test_model_card_completeness():
    """Verifies docs/MODEL_CARD.md complies with ML Commons and project specifications."""
    card_path = Path("docs/MODEL_CARD.md")
    assert card_path.exists(), "docs/MODEL_CARD.md must exist"

    content = card_path.read_text(encoding="utf-8")
    assert "Model Details" in content
    assert "Intended Use" in content
    assert "Out-of-Scope" in content
    assert "Sacred-Text Policy" in content or "Sacred and Religious Recitation" in content
    assert "Evaluation Results" in content
    assert "Known Limitations & Dialectal Biases" in content
    assert "License" in content
    assert "Majhi" in content
    assert "Common Voice" in content
    assert "FLEURS" in content


def test_governance_and_issue_templates_exist():
    """Verifies that open-source community governance and issue templates are in place."""
    assert Path("CONTRIBUTING.md").exists()
    assert Path("CODE_OF_CONDUCT.md").exists()
    assert Path("SECURITY.md").exists()
    assert Path(".github/ISSUE_TEMPLATE/bug_report.md").exists()
    assert Path(".github/ISSUE_TEMPLATE/feature_request.md").exists()

    contributing = Path("CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "Sacred-Text Policy" in contributing
    assert "Weak dev machine" in contributing

    security = Path("SECURITY.md").read_text(encoding="utf-8")
    assert "Reporting a Vulnerability" in security
