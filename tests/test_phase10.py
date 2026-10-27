"""Unit and integration tests for Phase 10: Multi-Stage Voice Pipeline.

Tests Voice Activity Detection, acoustic-aware chunk splitting, speaker diarization,
speech enhancement fallback, speaker-word alignment, export formats, and CLI dry run.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from awaaz.cli import build_parser, run_transcribe_pipeline
from awaaz.pipeline.diarize import SpeakerDiarizer
from awaaz.pipeline.enhance import SpeechEnhancer
from awaaz.pipeline.merge import (
    assign_speakers_to_words,
    export_speaker_tracks,
    format_transcript,
    group_words_into_turns,
)
from awaaz.pipeline.vad import VoiceActivityDetector, split_long_segments
from scripts.tune_diarization import compute_segment_overlap_accuracy


@pytest.fixture
def sample_rate() -> int:
    return 16000


@pytest.fixture
def synthetic_speech_audio(sample_rate: int) -> np.ndarray:
    """Creates a 5-second synthetic audio with two speech bursts separated by silence."""
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.zeros_like(t, dtype=np.float32)

    # Burst 1: 0.5s to 1.8s (frequency 300Hz modulated)
    mask1 = (t >= 0.5) & (t <= 1.8)
    audio[mask1] = 0.4 * np.sin(2 * np.pi * 300 * t[mask1])

    # Burst 2: 2.8s to 4.2s (frequency 600Hz modulated)
    mask2 = (t >= 2.8) & (t <= 4.2)
    audio[mask2] = 0.4 * np.sin(2 * np.pi * 600 * t[mask2])

    return audio


def test_segment_splitter_duration_and_non_empty(sample_rate: int):
    """Verifies that split_long_segments never exceeds max_duration and produces no zero-length chunks."""
    audio = np.random.randn(sample_rate * 50).astype(np.float32) * 0.1
    # Segment of 45 seconds (exceeding 20s)
    long_segments = [(2.0, 47.0)]

    splits = split_long_segments(long_segments, audio, sample_rate=sample_rate, max_duration=20.0, min_split_window=15.0)

    assert len(splits) >= 3, f"Expected at least 3 chunks for 45s, got {len(splits)}"
    for s_start, s_end in splits:
        dur = s_end - s_start
        assert dur > 0.0, f"Encountered zero or negative length chunk: ({s_start}, {s_end})"
        assert dur <= 20.001, f"Chunk duration {dur} exceeds max_duration 20.0s"

    # Verify coverage
    assert abs(splits[0][0] - 2.0) < 1e-3
    assert abs(splits[-1][1] - 47.0) < 1e-3
    for i in range(len(splits) - 1):
        assert abs(splits[i][1] - splits[i + 1][0]) < 1e-3, "Chunks must be contiguous"


def test_segment_splitter_quietest_point(sample_rate: int):
    """Verifies that long segment splitter cuts at the quietest local energy dip."""
    dur = 25.0
    audio = np.ones(int(dur * sample_rate), dtype=np.float32) * 0.5
    # Inject a dip of silence at exactly 17.5s - 18.0s
    dip_start = int(17.5 * sample_rate)
    dip_end = int(18.0 * sample_rate)
    audio[dip_start:dip_end] = 0.0

    splits = split_long_segments([(0.0, 25.0)], audio, sample_rate=sample_rate, max_duration=20.0, min_split_window=15.0)
    assert len(splits) == 2
    cut_sec = splits[0][1]
    # Cut point should be near 17.5 - 18.0s
    assert 17.0 <= cut_sec <= 18.5, f"Expected cut point near quiet dip (17.5-18.0), got {cut_sec}"


def test_vad_padding_and_merging(sample_rate: int):
    """Verifies VAD adds 0.15s padding and merges speech bursts separated by < 0.4s."""
    vad = VoiceActivityDetector(sample_rate=sample_rate, padding_sec=0.15, min_gap_sec=0.4)

    # 4s audio with two bursts separated by 0.2s:
    # Burst 1: 1.0s to 1.5s
    # Burst 2: 1.7s to 2.2s (gap = 0.2s < 0.4s => must merge)
    t = np.linspace(0, 4.0, int(sample_rate * 4.0), endpoint=False)
    audio = np.zeros_like(t, dtype=np.float32)
    audio[(t >= 1.0) & (t <= 1.5)] = 0.5 * np.sin(2 * np.pi * 400 * t[(t >= 1.0) & (t <= 1.5)])
    audio[(t >= 1.7) & (t <= 2.2)] = 0.5 * np.sin(2 * np.pi * 400 * t[(t >= 1.7) & (t <= 2.2)])

    segments = vad.detect_speech(audio)
    assert len(segments) == 1, f"Expected 1 merged segment, got {len(segments)}: {segments}"
    seg_start, seg_end = segments[0]
    # Start should be <= 1.0 - 0.15 = 0.85
    assert seg_start <= 0.86, f"Expected padded start <= 0.85s, got {seg_start}"
    # End should be >= 2.2 + 0.15 = 2.35
    assert seg_end >= 2.34, f"Expected padded end >= 2.35s, got {seg_end}"


def test_diarizer_synthetic_two_speakers(synthetic_speech_audio: np.ndarray, sample_rate: int):
    """Tests speaker diarization on two distinct frequency modulated speaker bursts."""
    speech_segments = [(0.5, 1.8), (2.8, 4.2)]
    diarizer = SpeakerDiarizer(sample_rate=sample_rate, threshold=0.5)

    intervals = diarizer.diarize(synthetic_speech_audio, speech_segments, num_speakers=2)
    assert len(intervals) >= 2

    speakers = set(spk for _, _, spk in intervals)
    assert len(speakers) == 2, f"Expected 2 distinct speakers, found {speakers}"

    # Accuracy against reference
    ref = [(0.5, 1.8, "SPEAKER_00"), (2.8, 4.2, "SPEAKER_01")]
    acc = compute_segment_overlap_accuracy(intervals, ref)
    assert acc > 0.80, f"Expected diarization overlap accuracy > 80%, got {acc * 100:.1f}%"


def test_speech_enhancer_fallback(synthetic_speech_audio: np.ndarray, sample_rate: int):
    """Verifies SpeechEnhancer operates smoothly without crashing on missing DeepFilterNet."""
    enhancer = SpeechEnhancer()
    cleaned = enhancer.enhance(synthetic_speech_audio, sample_rate=sample_rate)

    assert cleaned.shape == synthetic_speech_audio.shape
    assert cleaned.dtype == np.float32
    assert not np.isnan(cleaned).any()


def test_assign_speakers_to_words():
    """Verifies word-to-speaker temporal overlap mapping and nearest fallback."""
    words = [
        ("hello", 0.9, 1.3),
        ("world", 1.4, 1.7),
        ("punjab", 3.0, 3.5),
        ("awaaz", 4.5, 4.8),  # Slightly outside speaker interval
    ]
    speaker_intervals = [
        (0.8, 2.0, "SPEAKER_00"),
        (2.8, 4.0, "SPEAKER_01"),
    ]

    annotated = assign_speakers_to_words(words, speaker_intervals)
    assert len(annotated) == 4

    assert annotated[0]["speaker"] == "SPEAKER_00"
    assert annotated[1]["speaker"] == "SPEAKER_00"
    assert annotated[2]["speaker"] == "SPEAKER_01"
    # Word 4 is closest to SPEAKER_01 (4.0s) than SPEAKER_00 (2.0s)
    assert annotated[3]["speaker"] == "SPEAKER_01"


def test_group_words_into_turns():
    """Verifies that words with the same speaker within pause threshold are grouped."""
    annotated = [
        {"word": "sat", "start": 0.5, "end": 0.8, "speaker": "SPEAKER_00"},
        {"word": "sri", "start": 0.9, "end": 1.2, "speaker": "SPEAKER_00"},
        {"word": "akal", "start": 1.3, "end": 1.7, "speaker": "SPEAKER_00"},
        {"word": "hello", "start": 3.0, "end": 3.5, "speaker": "SPEAKER_01"},
    ]

    turns = group_words_into_turns(annotated, max_silence_gap=1.5)
    assert len(turns) == 2
    assert turns[0]["speaker"] == "SPEAKER_00"
    assert turns[0]["text"] == "sat sri akal"
    assert turns[0]["start"] == 0.5
    assert turns[0]["end"] == 1.7

    assert turns[1]["speaker"] == "SPEAKER_01"
    assert turns[1]["text"] == "hello"


def test_format_transcript_all_formats():
    """Verifies TXT, SRT, VTT, and JSON export formats adhere to standard syntax."""
    annotated = [
        {"word": "hello", "start": 1.0, "end": 1.5, "speaker": "SPEAKER_00"},
        {"word": "world", "start": 1.6, "end": 2.0, "speaker": "SPEAKER_00"},
        {"word": "namaste", "start": 3.2, "end": 3.8, "speaker": "SPEAKER_01"},
    ]

    # TXT
    txt = format_transcript(annotated, "txt")
    assert "[00:00:01.000 --> 00:00:02.000] SPEAKER_00: hello world" in txt
    assert "[00:00:03.200 --> 00:00:03.800] SPEAKER_01: namaste" in txt

    # SRT
    srt = format_transcript(annotated, "srt")
    assert "1\n00:00:01,000 --> 00:00:02,000\n[SPEAKER_00] hello world" in srt
    assert "2\n00:00:03,200 --> 00:00:03,800\n[SPEAKER_01] namaste" in srt

    # VTT
    vtt = format_transcript(annotated, "vtt")
    assert vtt.startswith("WEBVTT")
    assert "00:00:01.000 --> 00:00:02.000\n<v SPEAKER_00>hello world</v>" in vtt

    # JSON
    json_str = format_transcript(annotated, "json")
    data = json.loads(json_str)
    assert "segments" in data
    assert "words" in data
    assert len(data["segments"]) == 2
    assert data["segments"][0]["text"] == "hello world"
    assert data["segments"][1]["speaker"] == "SPEAKER_01"


def test_export_speaker_tracks(synthetic_speech_audio: np.ndarray, sample_rate: int, tmp_path: Path):
    """Verifies that isolated speaker audio tracks are written with silence on opposing turns."""
    speaker_intervals = [
        (0.5, 1.8, "SPEAKER_00"),
        (2.8, 4.2, "SPEAKER_01"),
    ]
    out_dir = tmp_path / "tracks"
    tracks = export_speaker_tracks(synthetic_speech_audio, sample_rate, speaker_intervals, out_dir)

    assert "SPEAKER_00" in tracks
    assert "SPEAKER_01" in tracks

    spk0_path = tracks["SPEAKER_00"]
    spk1_path = tracks["SPEAKER_01"]
    assert Path(spk0_path).exists()
    assert Path(spk1_path).exists()

    audio0, sr0 = sf.read(spk0_path)
    audio1, sr1 = sf.read(spk1_path)
    assert sr0 == sample_rate and sr1 == sample_rate

    # During speaker 1 turn (3.0s to 4.0s), speaker 0 track must be silent
    spk1_turn_samples = slice(int(3.0 * sample_rate), int(4.0 * sample_rate))
    assert np.all(audio0[spk1_turn_samples] == 0.0)

    # During speaker 0 turn (0.6s to 1.6s), speaker 1 track must be silent
    spk0_turn_samples = slice(int(0.6 * sample_rate), int(1.6 * sample_rate))
    assert np.all(audio1[spk0_turn_samples] == 0.0)


def test_cli_transcribe_end_to_end(synthetic_speech_audio: np.ndarray, sample_rate: int, tmp_path: Path):
    """Dry run of full CLI transcribe pipeline with diarization, enhancement, and format export."""
    audio_path = tmp_path / "test_dialogue.wav"
    sf.write(str(audio_path), synthetic_speech_audio, sample_rate)

    parser = build_parser()
    args = parser.parse_args([
        "transcribe",
        str(audio_path),
        "--diarize",
        "--speakers", "2",
        "--enhance",
        "--format", "json",
        "--export-tracks", str(tmp_path / "tracks_cli"),
        "--output", str(tmp_path / "transcript.json"),
    ])

    transcript = run_transcribe_pipeline(args)
    assert (tmp_path / "transcript.json").exists()

    data = json.loads(transcript)
    assert "segments" in data
    assert len(data["segments"]) >= 1

    tracks_dir = tmp_path / "tracks_cli"
    assert tracks_dir.exists()
    wavs = list(tracks_dir.glob("*.wav"))
    assert len(wavs) >= 1
