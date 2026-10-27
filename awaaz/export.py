"""ONNX export, int8 dynamic quantization, and numerical parity testing for Awaaz (Phase 9).

Exports offline and streaming ONNX graphs, performs numerical parity checks,
applies int8 quantization, and generates edge budget reports and native feature test kits.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic

from awaaz.features import LogMelSpectrogram
from awaaz.model import ConformerConfig, ConformerCTC

logger = logging.getLogger("awaaz.export")


class OnnxExportWrapper(nn.Module):
    """Wrapper module that binds streaming chunk parameters for clean ONNX tracing."""

    def __init__(
        self,
        model: ConformerCTC,
        streaming: bool = False,
        chunk_size: Optional[int] = None,
        left_chunks: int = -1,
    ):
        super().__init__()
        self.model = model
        self.streaming = streaming
        self.chunk_size = chunk_size if streaming else None
        self.left_chunks = left_chunks if streaming else -1

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs forward pass with fixed chunk parameters.

        Args:
            features: (B, T, D) float32 mel spectrogram.
            feature_lengths: (B,) int64 sequence lengths.

        Returns:
            Tuple of (log_probs, output_lengths).
        """
        return self.model(
            features,
            feature_lengths,
            chunk_size=self.chunk_size,
            left_chunks=self.left_chunks,
        )


def export_to_onnx(
    model: ConformerCTC,
    output_path: str,
    opset_version: int = 18,
    streaming: bool = False,
    chunk_size: int = 16,
    left_chunks: int = 8,
) -> str:
    """Exports Conformer model to ONNX with dynamic batch and time dimensions.

    Args:
        model: ConformerCTC model instance.
        output_path: Destination path for the exported .onnx file.
        opset_version: Target ONNX opset version (default 17).
        streaming: If True, exports streaming-window model with baked chunk parameters.
        chunk_size: Encoder frames per chunk if streaming (default 16).
        left_chunks: Number of left context chunks if streaming (default 8).

    Returns:
        The output path string.
    """
    model.eval()
    wrapper = OnnxExportWrapper(
        model,
        streaming=streaming,
        chunk_size=chunk_size,
        left_chunks=left_chunks,
    )
    wrapper.eval()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    dummy_t = 80 if not streaming else (left_chunks + 1) * chunk_size
    dummy_features = torch.randn(1, dummy_t, model.config.n_mels, dtype=torch.float32)
    dummy_lengths = torch.tensor([dummy_t], dtype=torch.long)

    dynamic_axes = {
        "features": {0: "batch", 1: "time"},
        "feature_lengths": {0: "batch"},
        "log_probs": {0: "batch", 1: "time_out"},
        "output_lengths": {0: "batch"},
    }

    logger.info(
        "Exporting %s ONNX model (opset %d) to %s...",
        "streaming" if streaming else "offline",
        opset_version,
        out_p,
    )

    export_kwargs = {
        "input_names": ["features", "feature_lengths"],
        "output_names": ["log_probs", "output_lengths"],
        "dynamic_axes": dynamic_axes,
        "opset_version": opset_version,
        "do_constant_folding": True,
    }
    try:
        import inspect
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False
    except Exception:
        pass

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_features, dummy_lengths),
            str(out_p),
            **export_kwargs,
        )

    # Validate ONNX graph structure
    onnx_proto = onnx.load(str(out_p))
    onnx.checker.check_model(onnx_proto)
    logger.info("ONNX graph successfully verified and saved to %s", out_p)
    return str(out_p)


def verify_parity(
    pytorch_model: ConformerCTC,
    onnx_path: str,
    threshold: float = 1e-3,
    streaming: bool = False,
    chunk_size: int = 16,
    left_chunks: int = 8,
    test_lengths: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Verifies numerical parity between PyTorch and ONNX Runtime outputs.

    Args:
        pytorch_model: PyTorch model instance.
        onnx_path: Path to exported ONNX model.
        threshold: Maximum allowed absolute error (default 1e-3).
        streaming: Whether model is configured in streaming mode.
        chunk_size: Chunk size if streaming.
        left_chunks: Left context chunks if streaming.
        test_lengths: List of frame lengths to test.

    Returns:
        Dictionary containing max_abs_diff and parity status per length.
    """
    pytorch_model.eval()
    wrapper = OnnxExportWrapper(
        pytorch_model,
        streaming=streaming,
        chunk_size=chunk_size,
        left_chunks=left_chunks,
    )
    wrapper.eval()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    if test_lengths is None:
        if streaming:
            base = chunk_size * 2
            test_lengths = [base, base * 2, base * 3]
        else:
            test_lengths = [32, 64, 128]

    results: Dict[str, Any] = {"max_abs_diff": 0.0, "lengths_tested": {}}

    for t_len in test_lengths:
        torch.manual_seed(42 + t_len)
        features = torch.randn(1, t_len, pytorch_model.config.n_mels, dtype=torch.float32)
        lengths = torch.tensor([t_len], dtype=torch.long)

        # PyTorch execution
        with torch.no_grad():
            pt_log_probs, pt_out_lens = wrapper(features, lengths)
            pt_out = pt_log_probs.detach().cpu().numpy()

        # ONNX Runtime execution
        ort_inputs = {
            "features": features.numpy(),
            "feature_lengths": lengths.numpy(),
        }
        ort_outputs = session.run(["log_probs", "output_lengths"], ort_inputs)
        onnx_out = ort_outputs[0]

        abs_diff = float(np.max(np.abs(pt_out - onnx_out)))
        results["lengths_tested"][str(t_len)] = abs_diff
        results["max_abs_diff"] = max(results["max_abs_diff"], abs_diff)

        if abs_diff > threshold:
            raise ValueError(
                f"ONNX numerical parity check FAILED at length {t_len}! "
                f"Max diff: {abs_diff:.6e} exceeds threshold {threshold:.6e}"
            )

    results["passed"] = bool(results["max_abs_diff"] <= threshold)
    logger.info("Numerical parity check PASSED (Max Diff: %.6e <= %.6e)", results["max_abs_diff"], threshold)
    return results


def quantize_int8(onnx_path: str, output_path: str) -> str:
    """Applies dynamic int8 quantization using ONNX Runtime.

    Args:
        onnx_path: Path to input FP32 ONNX model.
        output_path: Target path for the int8 quantized model.

    Returns:
        Path to the quantized model.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Applying dynamic int8 quantization to %s...", onnx_path)
    quantize_dynamic(
        model_input=onnx_path,
        model_output=str(out_p),
        weight_type=QuantType.QInt8,
    )

    orig_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    quant_size_mb = os.path.getsize(str(out_p)) / (1024 * 1024)
    compression_pct = (1.0 - (quant_size_mb / orig_size_mb)) * 100.0

    logger.info(
        "Quantization complete: %.2f MB -> %.2f MB (%.1f%% reduction)",
        orig_size_mb,
        quant_size_mb,
        compression_pct,
    )
    return str(out_p)


def generate_budget_report(
    model: ConformerCTC,
    onnx_fp32_path: Optional[str] = None,
    onnx_int8_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates hardware budget metrics, parameter count, and FLOP estimates."""
    total_params = sum(p.numel() for p in model.parameters())
    # Non-embedding parameters: subtract CTC head final linear projection
    non_emb_params = total_params - (model.config.d_model * model.config.vocab_size)

    # 25 frames per second encoder rate (40ms hop)
    fps = 25.0
    est_gflops_per_sec = (2.0 * non_emb_params * fps) / 1e9

    fp32_size_mb = (os.path.getsize(onnx_fp32_path) / 1e6) if (onnx_fp32_path and Path(onnx_fp32_path).is_file()) else (total_params * 4.0 / 1e6)
    int8_size_mb = (os.path.getsize(onnx_int8_path) / 1e6) if (onnx_int8_path and Path(onnx_int8_path).is_file()) else (total_params * 1.0 / 1e6)

    return {
        "model_name": model.config.name,
        "total_parameters": total_params,
        "non_embedding_parameters": non_emb_params,
        "estimated_gflops_per_second": round(est_gflops_per_sec, 3),
        "fp32_size_mb": round(fp32_size_mb, 2),
        "int8_size_mb": round(int8_size_mb, 2),
    }


def export_feature_parity_test_kit(
    output_dir: str,
    frontend: Optional[LogMelSpectrogram] = None,
    num_samples: int = 1,
) -> Dict[str, str]:
    """Generates reference audio and binary expected log-mel features for native edge DSP validation."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if frontend is None:
        frontend = LogMelSpectrogram(sample_rate=16000, n_mels=80)
    frontend.eval()

    sr = 16000
    # Generate 1.0s reference tone
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = (0.2 * np.sin(2 * np.pi * 440.0 * t) + 0.1 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)

    wav_path = out_p / "reference_audio.wav"
    sf.write(str(wav_path), sig, sr)

    with torch.no_grad():
        waveform_t = torch.from_numpy(sig).unsqueeze(0)
        feats = frontend(waveform_t).squeeze(0).cpu().numpy()

    npy_path = out_p / "expected_features.npy"
    np.save(str(npy_path), feats)

    logger.info("Feature parity kit written: %s and %s", wav_path, npy_path)
    return {"wav": str(wav_path), "npy": str(npy_path)}


def main():
    parser = argparse.ArgumentParser(description="Export Awaaz Model to ONNX and int8")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt.")
    parser.add_argument("--output_dir", type=str, default="release/onnx", help="Directory for exported models.")
    parser.add_argument("--quantize", action="store_true", help="Generate int8 dynamically quantized model.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--test_kit", action="store_true", help="Generate mobile feature parity test kit.")

    args = parser.parse_args()

    cfg = ConformerConfig.from_yaml(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_state = ckpt.get("model", ckpt)

    model = ConformerCTC(cfg)
    model.load_state_dict(model_state, strict=False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export Offline ONNX
    offline_onnx = out_dir / "awaaz_offline.onnx"
    export_to_onnx(model, str(offline_onnx), opset_version=args.opset, streaming=False)
    verify_parity(model, str(offline_onnx), streaming=False)

    # 2. Export Streaming ONNX
    stream_onnx = out_dir / "awaaz_stream.onnx"
    export_to_onnx(model, str(stream_onnx), opset_version=args.opset, streaming=True)
    verify_parity(model, str(stream_onnx), streaming=True)

    # 3. Dynamic Quantization
    quant_offline = None
    if args.quantize:
        quant_offline = out_dir / "awaaz_offline.quant.onnx"
        quantize_int8(str(offline_onnx), str(quant_offline))

        quant_stream = out_dir / "awaaz_stream.quant.onnx"
        quantize_int8(str(stream_onnx), str(quant_stream))

    # 4. Budget report
    report = generate_budget_report(model, str(offline_onnx), str(quant_offline) if quant_offline else None)
    logger.info("Budget Report: %s", report)

    # 5. Native Feature Kit
    if args.test_kit:
        kit_dir = out_dir / "feature_kit"
        export_feature_parity_test_kit(str(kit_dir))


if __name__ == "__main__":
    main()
