"""Alignment and formatting of word timestamps, speaker labels, and transcripts (Phase 10).

Merges CTC word emissions with diarization windows and formats into TXT, SRT, VTT, and JSON.
Also exports per-speaker isolated audio tracks.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import soundfile as sf


def _format_timestamp(seconds: float, separator: str = ".") -> str:
    """Formats seconds into HH:MM:SS,mmm or HH:MM:SS.mmm format."""
    total_seconds = max(0.0, float(seconds))
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = int(total_seconds % 60)
    millis = int(round((total_seconds - int(total_seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis -= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def assign_speakers_to_words(
    words_with_times: List[Union[Tuple[str, float, float], Dict[str, Any]]],
    speaker_intervals: List[Tuple[float, float, str]],
) -> List[Dict[str, Any]]:
    """Maps words to speakers based on maximum temporal overlap or nearest interval center.

    Args:
        words_with_times: List of (word, start_sec, end_sec) or dicts with 'word', 'start', 'end'.
        speaker_intervals: List of (start_sec, end_sec, speaker_id) intervals.

    Returns:
        List of dicts: [{'word': ..., 'start': ..., 'end': ..., 'speaker': ..., 'confidence': ...}]
    """
    if not words_with_times:
        return []

    # Standardize input to list of dicts
    words: List[Dict[str, Any]] = []
    for item in words_with_times:
        if isinstance(item, (tuple, list)):
            w, s, e = item[0], float(item[1]), float(item[2])
            conf = float(item[3]) if len(item) > 3 else 1.0
            words.append({"word": str(w), "start": round(s, 3), "end": round(e, 3), "confidence": conf})
        elif isinstance(item, dict):
            words.append({
                "word": str(item.get("word", "")),
                "start": round(float(item.get("start", 0.0)), 3),
                "end": round(float(item.get("end", 0.0)), 3),
                "confidence": float(item.get("confidence", 1.0)),
            })

    if not speaker_intervals:
        for w in words:
            w["speaker"] = "SPEAKER_00"
        return words

    for w in words:
        w_start = w["start"]
        w_end = w["end"]
        w_mid = (w_start + w_end) / 2.0

        best_spk = None
        max_overlap = 0.0

        # Check for temporal overlap
        for s_start, s_end, spk in speaker_intervals:
            overlap = max(0.0, min(w_end, s_end) - max(w_start, s_start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_spk = spk

        # If no overlap, assign to nearest interval center
        if best_spk is None:
            min_dist = float("inf")
            for s_start, s_end, spk in speaker_intervals:
                s_mid = (s_start + s_end) / 2.0
                dist = abs(w_mid - s_mid)
                if dist < min_dist:
                    min_dist = dist
                    best_spk = spk

        w["speaker"] = best_spk or "SPEAKER_00"

    return words


def group_words_into_turns(
    annotated_words: List[Dict[str, Any]],
    max_silence_gap: float = 1.5,
) -> List[Dict[str, Any]]:
    """Groups consecutive words by the same speaker into speech turns/paragraphs."""
    if not annotated_words:
        return []

    turns: List[Dict[str, Any]] = []
    current_words = [annotated_words[0]]
    current_spk = annotated_words[0]["speaker"]

    for w in annotated_words[1:]:
        spk = w["speaker"]
        gap = w["start"] - current_words[-1]["end"]

        if spk == current_spk and gap <= max_silence_gap:
            current_words.append(w)
        else:
            turns.append({
                "speaker": current_spk,
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"],
                "text": " ".join(item["word"] for item in current_words),
                "words": current_words,
            })
            current_words = [w]
            current_spk = spk

    if current_words:
        turns.append({
            "speaker": current_spk,
            "start": current_words[0]["start"],
            "end": current_words[-1]["end"],
            "text": " ".join(item["word"] for item in current_words),
            "words": current_words,
        })

    return turns


def format_transcript(
    annotated_words: List[Dict[str, Any]],
    output_format: str = "txt",
) -> str:
    """Formats annotated word sequences into plain text, SRT, VTT, or JSON.

    Args:
        annotated_words: Output from `assign_speakers_to_words`.
        output_format: One of 'txt', 'srt', 'vtt', 'json'.

    Returns:
        Formatted string representation.
    """
    if not annotated_words:
        if output_format.lower() == "json":
            return json.dumps({"segments": [], "words": []}, indent=2)
        elif output_format.lower() == "vtt":
            return "WEBVTT\n\n"
        return ""

    turns = group_words_into_turns(annotated_words)
    fmt = output_format.lower().strip()

    if fmt == "txt":
        lines = []
        for turn in turns:
            t_start = _format_timestamp(turn["start"], separator=".")
            t_end = _format_timestamp(turn["end"], separator=".")
            lines.append(f"[{t_start} --> {t_end}] {turn['speaker']}: {turn['text']}")
        return "\n".join(lines)

    elif fmt == "srt":
        blocks = []
        for idx, turn in enumerate(turns, start=1):
            t_start = _format_timestamp(turn["start"], separator=",")
            t_end = _format_timestamp(turn["end"], separator=",")
            blocks.append(f"{idx}\n{t_start} --> {t_end}\n[{turn['speaker']}] {turn['text']}\n")
        return "\n".join(blocks).strip() + "\n"

    elif fmt == "vtt":
        blocks = ["WEBVTT\n"]
        for turn in turns:
            t_start = _format_timestamp(turn["start"], separator=".")
            t_end = _format_timestamp(turn["end"], separator=".")
            blocks.append(f"{t_start} --> {t_end}\n<v {turn['speaker']}>{turn['text']}</v>\n")
        return "\n".join(blocks).strip() + "\n"

    elif fmt == "json":
        data = {
            "segments": turns,
            "words": annotated_words,
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    else:
        raise ValueError(f"Unsupported format '{output_format}'. Choose from 'txt', 'srt', 'vtt', 'json'.")


def export_speaker_tracks(
    audio: np.ndarray,
    sample_rate: int,
    speaker_intervals: List[Tuple[float, float, str]],
    output_dir: Union[str, Path],
) -> Dict[str, str]:
    """Writes isolated WAV files per speaker with silence during other speakers' turns.

    Args:
        audio: 1D numpy array of audio samples.
        sample_rate: Audio sampling rate in Hz.
        speaker_intervals: List of (start_sec, end_sec, speaker_id) intervals.
        output_dir: Directory where per-speaker WAV files will be saved.

    Returns:
        Mapping of {speaker_id: file_path}.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    unique_speakers = sorted(list(set(spk for _, _, spk in speaker_intervals)))
    track_paths = {}

    for spk in unique_speakers:
        isolated_audio = np.zeros_like(audio)
        for s_start, s_end, s_id in speaker_intervals:
            if s_id == spk:
                start_idx = max(0, int(s_start * sample_rate))
                end_idx = min(len(audio), int(s_end * sample_rate))
                isolated_audio[start_idx:end_idx] = audio[start_idx:end_idx]

        file_name = out_path / f"{spk}.wav"
        sf.write(str(file_name), isolated_audio, sample_rate)
        track_paths[spk] = str(file_name)

    return track_paths
