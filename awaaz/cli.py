"""Awaaz Command-Line Interface (CLI).

Provides unified commands for transcription, live streaming, model evaluation,
and dataset preparation.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from awaaz.decode import extract_word_timestamps
from awaaz.features import LogMelSpectrogram
from awaaz.pipeline.diarize import SpeakerDiarizer
from awaaz.pipeline.enhance import SpeechEnhancer
from awaaz.pipeline.merge import assign_speakers_to_words, export_speaker_tracks, format_transcript
from awaaz.pipeline.vad import VoiceActivityDetector
from awaaz.text import Tokenizer

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
    transcribe_parser.add_argument("--output", type=str, default=None, help="File path to save the output transcript.")
    transcribe_parser.add_argument("--device", default="cpu", help="Compute device ('cpu' or 'cuda').")

    # Stream subcommand
    stream_parser = subparsers.add_parser("stream", help="Live streaming speech recognition from audio chunks.")
    stream_parser.add_argument("--audio", type=str, default=None, help="Audio file to stream in real-time.")
    stream_parser.add_argument("--ckpt", type=str, required=False, help="Path to model checkpoint.")
    stream_parser.add_argument("--tokenizer", type=str, required=False, help="Path to SentencePiece model.")
    stream_parser.add_argument("--cmvn", type=str, required=False, help="Path to CMVN stats file.")
    stream_parser.add_argument("--chunk_ms", type=int, default=640, help="Streaming chunk size in milliseconds (default: 640ms).")
    stream_parser.add_argument("--device", default="cpu", help="Compute device.")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Compute WER/CER metrics across test manifests.")
    eval_parser.add_argument("--manifest", type=str, required=True, help="Path to JSONL test manifest.")
    eval_parser.add_argument("--ckpt", type=str, required=False, help="Path to model checkpoint.")
    eval_parser.add_argument("--tokenizer", type=str, required=False, help="Path to SentencePiece model.")
    eval_parser.add_argument("--cmvn", type=str, required=False, help="Path to CMVN stats file.")
    eval_parser.add_argument("--device", default="cpu", help="Compute device ('cpu' or 'cuda').")

    return parser


def load_audio_mono_16k(audio_path: str) -> Tuple[np.ndarray, int]:
    """Loads audio file and converts to mono 16kHz float32."""
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != 16000:
        # Simple resample
        num_target_samples = int(len(audio) * 16000 / sr)
        indices = np.linspace(0, len(audio) - 1, num_target_samples)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        sr = 16000

    return audio, sr


def run_transcribe_pipeline(args: argparse.Namespace) -> str:
    """Executes the full multi-stage transcription pipeline."""
    audio_path = args.audio
    audio, sr = load_audio_mono_16k(audio_path)

    # 1. Optional enhancement for listening/tracks
    processed_audio = audio
    if getattr(args, "enhance", False):
        enhancer = SpeechEnhancer()
        processed_audio = enhancer.enhance(audio, sample_rate=sr)

    # 2. VAD segmentation
    vad = VoiceActivityDetector(sample_rate=sr)
    speech_segments = vad.detect_speech(audio)

    # 3. Diarization
    speaker_intervals: List[Tuple[float, float, str]] = []
    if getattr(args, "diarize", False):
        diarizer = SpeakerDiarizer(sample_rate=sr)
        speaker_intervals = diarizer.diarize(
            audio, speech_segments, num_speakers=getattr(args, "speakers", None)
        )

        if getattr(args, "export_tracks", None):
            export_speaker_tracks(processed_audio, sr, speaker_intervals, args.export_tracks)

    # 4. ASR Inference and Word Timestamps
    words_with_times: List[Tuple[str, float, float]] = []

    if getattr(args, "ckpt", None) and getattr(args, "tokenizer", None):
        device = torch.device(getattr(args, "device", "cpu"))
        tokenizer = Tokenizer(args.tokenizer)

        # Load CMVN if available
        cmvn_mean, cmvn_std = None, None
        if getattr(args, "cmvn", None) and Path(args.cmvn).exists():
            cmvn_data = np.load(args.cmvn)
            cmvn_mean = cmvn_data.get("mean")
            cmvn_std = cmvn_data.get("std")

        feature_extractor = LogMelSpectrogram(sample_rate=sr, cmvn_mean=cmvn_mean, cmvn_std=cmvn_std)

        # Check if ONNX or PyTorch checkpoint
        ckpt_path = str(args.ckpt)
        if ckpt_path.endswith(".onnx"):
            import onnxruntime as ort
            sess = ort.InferenceSession(ckpt_path, providers=["CPUExecutionProvider"])
            feats = feature_extractor(torch.from_numpy(audio).unsqueeze(0)).squeeze(0).numpy()
            feats = np.expand_dims(feats, 0)
            logits = sess.run(["logits"], {"features": feats})[0]
            logits_tensor = torch.from_numpy(logits)
        else:
            from awaaz.model import ConformerCTC
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
            _ = ckpt.get("config", {})

            # Instantiate model
            model = ConformerCTC(vocab_size=tokenizer.vocab_size)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()

            feats = feature_extractor(torch.from_numpy(audio).unsqueeze(0)).to(device)
            feats_lens = torch.tensor([feats.size(1)], dtype=torch.long, device=device)
            with torch.no_grad():
                logits_tensor, _ = model(feats, feats_lens)

        words_with_times = extract_word_timestamps(
            logits=logits_tensor,
            tokenizer=tokenizer,
            sample_rate=sr,
            subsampling_factor=4,
            hop_ms=10.0,
        )
    else:
        # Fallback speech segment representation if no checkpoint supplied
        for i, (seg_s, seg_e) in enumerate(speech_segments):
            words_with_times.append((f"[speech_{i + 1}]", seg_s, seg_e))

    # 5. Merge words with speakers
    annotated_words = assign_speakers_to_words(words_with_times, speaker_intervals)

    # 6. Format transcript
    out_format = getattr(args, "format", "txt")
    transcript = format_transcript(annotated_words, output_format=out_format)

    if getattr(args, "output", None):
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(transcript)

    return transcript


def main(args: Optional[List[str]] = None) -> int:
    """Main execution entry point."""
    parser = build_parser()
    parsed_args = parser.parse_args(args=args)

    if not parsed_args.subcommand:
        parser.print_help()
        return 0

    if parsed_args.subcommand == "transcribe":
        transcript = run_transcribe_pipeline(parsed_args)
        if not getattr(parsed_args, "output", None):
            print(transcript)
        return 0

    elif parsed_args.subcommand == "stream":
        print(f"[Awaaz] Live streaming recognizer initialized (Chunk: {parsed_args.chunk_ms} ms)")
        if parsed_args.audio and parsed_args.ckpt and parsed_args.tokenizer:
            from awaaz.model import ConformerCTC
            from awaaz.stream import StreamingRecognizer
            tok = Tokenizer(parsed_args.tokenizer)
            ckpt = torch.load(parsed_args.ckpt, map_location="cpu")
            state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
            model = ConformerCTC(vocab_size=tok.vocab_size)
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            recognizer = StreamingRecognizer(model=model, tokenizer=tok, chunk_ms=parsed_args.chunk_ms)
            audio, sr = load_audio_mono_16k(parsed_args.audio)
            chunk_size = int(sr * (parsed_args.chunk_ms / 1000.0))
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                text, is_final = recognizer.accept(chunk)
                if text:
                    print(f"[{'FINAL' if is_final else 'PARTIAL'}] {text}")
            final_text = recognizer.flush()
            if final_text:
                print(f"[FLUSH] {final_text}")
        return 0

    elif parsed_args.subcommand == "evaluate":
        print(f"[Awaaz] Evaluating manifest: {parsed_args.manifest}")
        from awaaz.evaluate import evaluate_manifest
        if parsed_args.ckpt and parsed_args.tokenizer:
            from awaaz.model import ConformerCTC
            tok = Tokenizer(parsed_args.tokenizer)
            ckpt = torch.load(parsed_args.ckpt, map_location=parsed_args.device)
            state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
            model = ConformerCTC(vocab_size=tok.vocab_size)
            model.load_state_dict(state_dict, strict=False)
            model.to(parsed_args.device)
            model.eval()

            cmvn_m, cmvn_s = None, None
            if parsed_args.cmvn and Path(parsed_args.cmvn).exists():
                data = np.load(parsed_args.cmvn)
                cmvn_m, cmvn_s = data.get("mean"), data.get("std")

            fe = LogMelSpectrogram(cmvn_mean=cmvn_m, cmvn_std=cmvn_s)
            results = evaluate_manifest(
                model=model,
                feature_extractor=fe,
                tokenizer=tok,
                manifest_path=parsed_args.manifest,
                device=parsed_args.device,
            )
            print("[Awaaz Evaluation Results]:", results)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
