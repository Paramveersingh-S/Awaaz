"""Voice Activity Detection (VAD) and acoustic chunk splitting (Phase 10).

Detects speech bursts, pads segments, merges close gaps, and splits long segments
at the quietest local energy minima.
"""

from typing import List, Optional, Tuple

import numpy as np


def split_long_segments(
    segments: List[Tuple[float, float]],
    audio: np.ndarray,
    sample_rate: int = 16000,
    max_duration: float = 20.0,
    min_split_window: float = 15.0,
) -> List[Tuple[float, float]]:
    """Acoustic-aware segment splitter.

    Splits segments exceeding `max_duration` at the lowest-energy (quietest)
    point within the [min_split_window, max_duration] relative time window.
    Guarantees no segment exceeds `max_duration` and no segment has zero length.

    Args:
        segments: List of (start_sec, end_sec) intervals.
        audio: 1D numpy array of audio samples.
        sample_rate: Audio sampling rate in Hz.
        max_duration: Maximum allowed duration for any chunk (default: 20.0s).
        min_split_window: Earliest point within a long segment to search for quietest point (default: 15.0s).

    Returns:
        List of (start_sec, end_sec) intervals where each duration <= max_duration.
    """
    if not segments:
        return []

    split_results: List[Tuple[float, float]] = []
    hop_length = int(sample_rate * 0.010)  # 10ms hops
    win_length = int(sample_rate * 0.030)  # 30ms window

    for start_sec, end_sec in segments:
        if end_sec <= start_sec:
            continue

        remaining_start = start_sec
        while (end_sec - remaining_start) > max_duration:
            # We must split within [remaining_start + min_split_window, remaining_start + max_duration]
            earliest_split = remaining_start + min_split_window
            latest_split = remaining_start + max_duration

            # Guard against invalid window
            if earliest_split >= latest_split:
                earliest_split = remaining_start + (max_duration * 0.75)

            # Find quietest frame in search window
            search_start_sample = int(earliest_split * sample_rate)
            search_end_sample = int(latest_split * sample_rate)

            # Ensure bounds within audio
            search_start_sample = max(0, min(len(audio), search_start_sample))
            search_end_sample = max(search_start_sample, min(len(audio), search_end_sample))

            if search_end_sample - search_start_sample > win_length:
                search_region = audio[search_start_sample:search_end_sample]
                num_frames = (len(search_region) - win_length) // hop_length + 1
                if num_frames > 0:
                    energies = []
                    for i in range(num_frames):
                        frame = search_region[i * hop_length : i * hop_length + win_length]
                        energies.append(np.mean(frame**2))
                    min_idx = int(np.argmin(energies))
                    cut_sample = search_start_sample + min_idx * hop_length + (win_length // 2)
                    cut_sec = float(cut_sample) / sample_rate
                else:
                    cut_sec = (earliest_split + latest_split) / 2.0
            else:
                cut_sec = (earliest_split + latest_split) / 2.0

            # Clamp cut_sec strictly within (remaining_start, end_sec)
            cut_sec = max(remaining_start + 1.0, min(remaining_start + max_duration, cut_sec))
            split_results.append((round(remaining_start, 4), round(cut_sec, 4)))
            remaining_start = cut_sec

        if end_sec > remaining_start:
            split_results.append((round(remaining_start, 4), round(end_sec, 4)))

    return split_results


class VoiceActivityDetector:
    """Acoustic and Energy-based Voice Activity Detector (VAD).

    Segments audio stream into speech regions, adds padding, merges close gaps,
    and splits long segments at local energy minima.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        frame_ms: float = 30.0,
        hop_ms: float = 10.0,
        padding_sec: float = 0.15,
        min_gap_sec: float = 0.4,
        max_segment_sec: float = 20.0,
        min_split_window_sec: float = 15.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.padding_sec = padding_sec
        self.min_gap_sec = min_gap_sec
        self.max_segment_sec = max_segment_sec
        self.min_split_window_sec = min_split_window_sec

    def _detect_energy_speech(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Fast, robust energy-based speech detection without external dependencies."""
        if len(audio) == 0:
            return []

        frame_len = int(self.sample_rate * (self.frame_ms / 1000.0))
        hop_len = int(self.sample_rate * (self.hop_ms / 1000.0))

        if len(audio) < frame_len:
            # Audio is too short for a single frame; consider as speech if non-trivial energy
            rms = np.sqrt(np.mean(audio**2) + 1e-12)
            if rms > 0.01:
                return [(0.0, round(len(audio) / self.sample_rate, 4))]
            return []

        num_frames = (len(audio) - frame_len) // hop_len + 1
        rms = np.zeros(num_frames, dtype=np.float32)
        for i in range(num_frames):
            frame = audio[i * hop_len : i * hop_len + frame_len]
            rms[i] = np.sqrt(np.mean(frame**2) + 1e-12)

        # Dynamic threshold based on energy distribution
        p20 = np.percentile(rms, 20)
        p90 = np.percentile(rms, 90)
        dynamic_thresh = max(0.005, p20 + self.threshold * (p90 - p20))

        is_speech = rms >= dynamic_thresh

        # Extract contiguous speech runs
        raw_segments: List[Tuple[float, float]] = []
        in_speech = False
        start_frame = 0

        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                in_speech = True
                start_frame = i
            elif not speech and in_speech:
                in_speech = False
                start_sec = (start_frame * hop_len) / self.sample_rate
                end_sec = (i * hop_len + frame_len) / self.sample_rate
                raw_segments.append((start_sec, end_sec))

        if in_speech:
            start_sec = (start_frame * hop_len) / self.sample_rate
            end_sec = len(audio) / self.sample_rate
            raw_segments.append((start_sec, end_sec))

        return raw_segments

    def detect_speech(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> List[Tuple[float, float]]:
        """Returns list of (start_sec, end_sec) speech intervals.

        Applies:
          1. Speech detection (energy/acoustic)
          2. 0.15s padding before and after bursts
          3. Merging of gaps < 0.4s
          4. Acoustic-aware chunk splitting for segments > 20s
        """
        sr = sample_rate or self.sample_rate
        total_duration = len(audio) / float(sr)

        raw_segments = self._detect_energy_speech(audio)
        if not raw_segments:
            return []

        # 1. Add padding
        padded_segments: List[Tuple[float, float]] = []
        for start, end in raw_segments:
            padded_start = max(0.0, start - self.padding_sec)
            padded_end = min(total_duration, end + self.padding_sec)
            padded_segments.append((padded_start, padded_end))

        # 2. Merge gaps < min_gap_sec
        merged_segments: List[Tuple[float, float]] = []
        cur_start, cur_end = padded_segments[0]

        for nxt_start, nxt_end in padded_segments[1:]:
            if nxt_start - cur_end < self.min_gap_sec:
                # Merge
                cur_end = max(cur_end, nxt_end)
            else:
                merged_segments.append((round(cur_start, 4), round(cur_end, 4)))
                cur_start, cur_end = nxt_start, nxt_end

        merged_segments.append((round(cur_start, 4), round(cur_end, 4)))

        # 3. Acoustic-aware chunk splitting for segments > max_segment_sec
        final_segments = split_long_segments(
            merged_segments,
            audio,
            sample_rate=sr,
            max_duration=self.max_segment_sec,
            min_split_window=self.min_split_window_sec,
        )

        return final_segments
