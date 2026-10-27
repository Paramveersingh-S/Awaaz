"""Awaaz Command-Line Interface (CLI).

Provides unified commands for transcription, live streaming, model evaluation,
and dataset preparation.
"""

import argparse
import sys
from typing import List, Optional

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    """Builds the main command-line argument parser for Awaaz."""
    parser = argparse.ArgumentParser(
        prog="awaaz",
        description="Awaaz: Bilingual English + Punjabi Streaming Speech-to-Text & Diarization.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="awaaz 0.1.0",
        help="Show program version number and exit.",
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Transcribe subcommand
    transcribe_parser = subparsers.add_parser(
        "transcribe", help="Transcribe an audio file using an Awaaz checkpoint or ONNX model."
    )
    transcribe_parser.add_argument("audio", type=str, help="Path to input audio file (WAV/FLAC).")
    transcribe_parser.add_argument("--ckpt", type=str, required=False, help="Path to PyTorch or ONNX model checkpoint.")
    transcribe_parser.add_argument("--tokenizer", type=str, required=False, help="Path to SentencePiece tokenizer model.")
    transcribe_parser.add_argument("--cmvn", type=str, required=False, help="Path to CMVN stats (.npz).")
    transcribe_parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization.")
    transcribe_parser.add_argument("--speakers", type=int, default=None, help="Fixed number of speakers if known.")
    transcribe_parser.add_argument("--enhance", action="store_true", help="Enable speech denoising/smoothing.")
    transcribe_parser.add_argument(
        "--format", choices=["txt", "srt", "vtt", "json"], default="txt", help="Output transcript format."
    )
    transcribe_parser.add_argument("--export-tracks", type=str, default=None, help="Directory to save per-speaker audio tracks.")
    transcribe_parser.add_argument("--device", default="cpu", help="Compute device ('cpu' or 'cuda').")

    # Stream subcommand
    stream_parser = subparsers.add_parser("stream", help="Live streaming speech recognition from microphone or file.")
    stream_parser.add_argument("--ckpt", type=str, required=False, help="Path to model checkpoint.")
    stream_parser.add_argument("--chunk_ms", type=int, default=640, help="Streaming chunk size in milliseconds (default: 640ms).")
    stream_parser.add_argument("--device", default="cpu", help="Compute device.")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Compute WER/CER metrics across test manifests.")
    eval_parser.add_argument("--manifest", type=str, required=True, help="Path to JSONL test manifest.")
    eval_parser.add_argument("--ckpt", type=str, required=False, help="Path to model checkpoint.")
    eval_parser.add_argument("--tokenizer", type=str, required=False, help="Path to SentencePiece model.")
    eval_parser.add_argument("--cmvn", type=str, required=False, help="Path to CMVN stats file.")

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main execution entry point."""
    parser = build_parser()
    parsed_args = parser.parse_args(args=args)

    if not parsed_args.subcommand:
        parser.print_help()
        return 0

    if parsed_args.subcommand == "transcribe":
        print(f"[Awaaz] Transcribing audio: {parsed_args.audio} (Format: {parsed_args.format})")
        # Full transcription pipeline hooked up in Phase 10
        return 0
    elif parsed_args.subcommand == "stream":
        print(f"[Awaaz] Live streaming recognizer initialized (Chunk: {parsed_args.chunk_ms} ms)")
        # Streaming loop hooked up in Phase 8
        return 0
    elif parsed_args.subcommand == "evaluate":
        print(f"[Awaaz] Evaluating manifest: {parsed_args.manifest}")
        # Evaluation suite hooked up in Phase 7
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
