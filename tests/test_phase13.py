"""Unit and integration tests for Phase 13: Edge Applications.

Verifies Android / Wear OS native Kotlin templates, iOS / watchOS Swift implementations,
embedded ARM64 Linux streaming clients, and edge deployment strategy documentation.
"""

from pathlib import Path

import numpy as np
import pytest

from awaaz.text import Tokenizer, train_tokenizer
from edge.embedded.rpi_stream import EmbeddedStreamingClient


@pytest.fixture
def synthetic_tokenizer(tmp_path: Path) -> Tokenizer:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world sat sri akal punjab awaaz speech audio", encoding="utf-8")
    prefix = str(tmp_path / "spm")
    train_tokenizer(output_prefix=prefix, pa_corpus=[str(corpus)], vocab_size=64, character_coverage=1.0)
    return Tokenizer(f"{prefix}.model")


def test_android_wear_os_source_files_exist():
    """Verifies that the Android / Wear OS native reference architecture is complete."""
    android_dir = Path("edge/android")
    assert android_dir.exists()

    manifest = android_dir / "AndroidManifest.xml"
    assert manifest.exists()
    manifest_text = manifest.read_text(encoding="utf-8")
    assert "android.permission.RECORD_AUDIO" in manifest_text
    assert "ai.awaaz.edge" in manifest_text

    gradle = android_dir / "build.gradle.kts"
    assert gradle.exists()
    gradle_text = gradle.read_text(encoding="utf-8")
    assert "com.microsoft.onnxruntime:onnxruntime-android" in gradle_text
    assert "androidx.compose" in gradle_text

    # Core Kotlin source files
    kt_files = [
        "FeatureExtractor.kt",
        "AudioRecordManager.kt",
        "AwaazStreamingEngine.kt",
        "TelemetryDashboard.kt",
        "MainActivity.kt",
    ]
    for kt_name in kt_files:
        kt_path = android_dir / kt_name
        assert kt_path.exists(), f"Missing Android Kotlin file: {kt_name}"
        kt_text = kt_path.read_text(encoding="utf-8")
        assert "package ai.awaaz.edge" in kt_text


def test_apple_ios_watchos_source_files_exist():
    """Verifies that the Apple iOS and watchOS Swift reference architecture is complete."""
    apple_dir = Path("edge/apple")
    assert apple_dir.exists()

    swift_files = [
        "AudioEngineManager.swift",
        "CoreMLStreamingRecognizer.swift",
        "AwaazWatchRelay.swift",
        "ContentView.swift",
    ]
    for swift_name in swift_files:
        swift_path = apple_dir / swift_name
        assert swift_path.exists(), f"Missing Apple Swift file: {swift_name}"
        swift_text = swift_path.read_text(encoding="utf-8")
        assert "import Foundation" in swift_text or "import SwiftUI" in swift_text


def test_embedded_linux_source_files_exist():
    """Verifies that the ARM64 embedded Linux and C++ streaming clients exist."""
    embedded_dir = Path("edge/embedded")
    assert embedded_dir.exists()

    py_client = embedded_dir / "rpi_stream.py"
    assert py_client.exists()

    cpp_client = embedded_dir / "stream_client.cpp"
    assert cpp_client.exists()
    cpp_text = cpp_client.read_text(encoding="utf-8")
    assert "Ort::Session" in cpp_text
    assert "process_chunk" in cpp_text


def test_embedded_streaming_client_execution(synthetic_tokenizer: Tokenizer, tmp_path: Path):
    """Verifies that EmbeddedStreamingClient processes synthetic chunks and produces telemetry."""
    tok_path = synthetic_tokenizer.model_path

    # Minimal dummy onnx path (client handles session gracefully if not found)
    dummy_onnx = str(tmp_path / "dummy.onnx")

    client = EmbeddedStreamingClient(
        onnx_model_path=dummy_onnx,
        tokenizer_path=tok_path,
        chunk_ms=640,
        left_chunks=4,
    )

    chunk = np.zeros(int(16000 * 0.64), dtype=np.float32)
    output_str = client.process_chunk(chunk)

    assert "RTF:" in output_str
    assert "Latency:" in output_str


def test_edge_deployment_documentation_completeness():
    """Verifies docs/EDGE_DEPLOYMENT.md covers power strategy and decision gate playbook."""
    doc_path = Path("docs/EDGE_DEPLOYMENT.md")
    assert doc_path.exists()
    content = doc_path.read_text(encoding="utf-8")

    assert "Wearable Power Budget" in content
    assert "Two-Stage Cascaded Power Management" in content
    assert "Decision Gate Playbook" in content
    assert "Subsampling Factor 8" in content
    assert "mermaid" in content
