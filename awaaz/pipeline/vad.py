"""Voice Activity Detection (VAD) and acoustic chunk splitting (Phase 10).

Detects speech bursts, pads segments, merges close gaps, and splits long segments
at the quietest local energy minima.
"""

from typing import List, Tuple

import numpy as np


class VoiceActivityDetector:
    """Wrapper for lightweight pretrained VAD (e.g. Silero VAD)."""

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        self.sample_rate = sample_rate
        self.threshold = threshold

    def detect_speech(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Returns list of (start_sec, end_sec) speech intervals."""
        raise NotImplementedError("VAD will be implemented in Phase 10.")
