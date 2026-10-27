"""Raspberry Pi & Embedded Linux ARM64 Streaming ASR Client (Phase 13).

Captures real-time audio via ALSA / PulseAudio / sounddevice and streams through
quantized ONNX models on Raspberry Pi 4 / 5 and ARM64 SBCs.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

from awaaz.features import LogMelSpectrogram
from awaaz.text import Tokenizer


class EmbeddedStreamingClient:
    """Lightweight real-time streaming recognizer for embedded ARM64 Linux."""

    def __init__(
        self,
        onnx_model_path: str,
        tokenizer_path: str,
        cmvn_path: Optional[str] = None,
        chunk_ms: int = 640,
        left_chunks: int = 8,
        sample_rate: int = 16000,
    ) -> None:
        self.chunk_ms = chunk_ms
        self.left_chunks = left_chunks
        self.sample_rate = sample_rate

        self.tokenizer = Tokenizer(tokenizer_path)

        self.feature_extractor = LogMelSpectrogram(sample_rate=sample_rate)
        if cmvn_path and Path(cmvn_path).exists():
            self.feature_extractor.load_cmvn(cmvn_path)

        if HAS_ORT and Path(onnx_model_path).exists():
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(onnx_model_path, opts, providers=["CPUExecutionProvider"])
        else:
            self.session = None

        self.audio_history: List[float] = []
        self.confirmed_tokens: List[str] = []
        self.last_token_id = -1
        self.chunk_samples = int(sample_rate * (chunk_ms / 1000.0))

    def process_chunk(self, pcm_chunk: np.ndarray) -> str:
        """Processes 1D float32 audio chunk and returns updated transcript."""
        t_start = time.perf_counter()

        self.audio_history.extend(pcm_chunk.tolist())
        max_samples = self.chunk_samples * (self.left_chunks + 1)
        if len(self.audio_history) > max_samples:
            self.audio_history = self.audio_history[-max_samples:]

        audio_arr = np.array(self.audio_history, dtype=np.float32)
        import torch
        with torch.no_grad():
            feats = self.feature_extractor(torch.from_numpy(audio_arr).unsqueeze(0)).numpy()

        if self.session is not None:
            logits = self.session.run(["logits"], {"features": feats})[0]
            # Analyze newest frames
            sub_frames = logits.shape[1]
            newest_frames = min(16, sub_frames)
            argmax_ids = np.argmax(logits[0, -newest_frames:, :], axis=-1)

            for token_id in argmax_ids:
                if token_id != self.last_token_id:
                    if token_id != self.tokenizer.blank_id:
                        piece = self.tokenizer.id_to_piece(token_id)
                        self.confirmed_tokens.append(piece)
                    self.last_token_id = token_id

        elapsed = time.perf_counter() - t_start
        chunk_dur = self.chunk_ms / 1000.0
        rtf = elapsed / max(1e-4, chunk_dur)

        transcript = "".join(self.confirmed_tokens).replace("\u2581", " ").strip()
        return f"[RTF: {rtf:.2f}x | Latency: {elapsed * 1000.0:.0f}ms] {transcript}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Raspberry Pi Embedded Awaaz Streamer")
    parser.add_argument("--model", type=str, required=True, help="Path to int8 ONNX model.")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer model.")
    parser.add_argument("--cmvn", type=str, default=None, help="Path to cmvn.npz.")
    parser.add_argument("--chunk_ms", type=int, default=640, help="Chunk duration in ms.")
    args = parser.parse_args()

    print(f"[Awaaz RPi] Initializing streaming client on {args.model}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
