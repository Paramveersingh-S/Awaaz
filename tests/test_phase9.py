"""Unit and integration tests for Phase 9 (Export, Quantization & Edge Budget).

Tests:
1. ONNX export for offline and streaming configurations.
2. Numerical parity verification between PyTorch and ONNX Runtime (< 1e-3 tolerance).
3. Dynamic int8 quantization and executable quantized session inference.
4. Edge hardware budget report generation (parameters, FLOPs, file footprints).
5. Native mobile feature-parity verification kit generation.
6. Verification of docs/EDGE.md documentation deliverable.
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import soundfile as sf

from awaaz.export import (
    export_feature_parity_test_kit,
    export_to_onnx,
    generate_budget_report,
    quantize_int8,
    verify_parity,
)
from awaaz.model import ConformerConfig, ConformerCTC


@pytest.fixture(scope="module")
def toy_model_for_export():
    """Generates a tiny Conformer model for fast CPU ONNX export tests."""
    cfg = ConformerConfig(
        vocab_size=32,
        n_mels=80,
        d_model=64,
        num_blocks=2,
        num_heads=2,
        ffn_dim=128,
        conv_kernel_size=15,
        dropout=0.0,
    )
    model = ConformerCTC(cfg)
    model.eval()
    return model


def test_onnx_export_offline_and_streaming(tmp_path, toy_model_for_export):
    """Verifies ONNX export produces valid opset 17 graphs for offline and streaming."""
    offline_p = tmp_path / "model_offline.onnx"
    stream_p = tmp_path / "model_stream.onnx"

    export_to_onnx(toy_model_for_export, str(offline_p), opset_version=18, streaming=False)
    assert offline_p.is_file()
    model_proto = onnx.load(str(offline_p))
    onnx.checker.check_model(model_proto)

    export_to_onnx(toy_model_for_export, str(stream_p), opset_version=18, streaming=True, chunk_size=16, left_chunks=4)
    assert stream_p.is_file()
    stream_proto = onnx.load(str(stream_p))
    onnx.checker.check_model(stream_proto)


def test_numerical_parity_check(tmp_path, toy_model_for_export):
    """Verifies ONNX Runtime and PyTorch outputs match below 1e-3 tolerance."""
    onnx_p = tmp_path / "parity_test.onnx"
    export_to_onnx(toy_model_for_export, str(onnx_p), opset_version=18, streaming=False)

    parity = verify_parity(toy_model_for_export, str(onnx_p), threshold=1e-3, streaming=False, test_lengths=[32, 64])
    assert parity["passed"] is True
    assert parity["max_abs_diff"] < 1e-3


def test_int8_quantization_and_inference(tmp_path, toy_model_for_export):
    """Verifies int8 dynamic quantization runs inference cleanly."""
    fp32_p = tmp_path / "fp32.onnx"
    int8_p = tmp_path / "int8.onnx"

    export_to_onnx(toy_model_for_export, str(fp32_p), opset_version=18, streaming=False)
    quantize_int8(str(fp32_p), str(int8_p))

    assert int8_p.is_file()
    assert int8_p.stat().st_size > 0

    # Verify int8 model executes in ONNX Runtime
    session = ort.InferenceSession(str(int8_p), providers=["CPUExecutionProvider"])
    dummy_feats = np.random.randn(1, 40, 80).astype(np.float32)
    dummy_lens = np.array([40], dtype=np.int64)

    outputs = session.run(None, {"features": dummy_feats, "feature_lengths": dummy_lens})
    assert len(outputs) == 2
    assert outputs[0].shape[0] == 1
    assert np.all(np.isfinite(outputs[0]))


def test_budget_report_generator(toy_model_for_export):
    """Verifies generation of computational and parameter budget metrics."""
    report = generate_budget_report(toy_model_for_export)

    assert "total_parameters" in report
    assert "non_embedding_parameters" in report
    assert "estimated_gflops_per_second" in report
    assert report["total_parameters"] > 0
    assert report["non_embedding_parameters"] <= report["total_parameters"]
    assert report["estimated_gflops_per_second"] > 0


def test_feature_parity_kit_generator(tmp_path):
    """Verifies export of reference audio and expected binary feature matrix."""
    kit_dir = tmp_path / "feature_kit"
    paths = export_feature_parity_test_kit(str(kit_dir))

    wav_p = Path(paths["wav"])
    npy_p = Path(paths["npy"])

    assert wav_p.is_file()
    assert npy_p.is_file()

    info = sf.info(str(wav_p))
    assert info.samplerate == 16000
    assert info.channels == 1

    features = np.load(str(npy_p))
    assert features.ndim == 2
    assert features.shape[1] == 80  # 80-bin mel
    assert np.all(np.isfinite(features))


def test_edge_documentation_exists():
    """Verifies docs/EDGE.md exists and contains target hardware matrix."""
    doc_path = Path("docs/EDGE.md")
    assert doc_path.is_file(), "docs/EDGE.md must exist."

    content = doc_path.read_text(encoding="utf-8")
    assert "Wear OS" in content
    assert "Raspberry Pi" in content
    assert "Microcontroller" in content or "MCU" in content
    assert "Zipformer" in content or "sherpa" in content
