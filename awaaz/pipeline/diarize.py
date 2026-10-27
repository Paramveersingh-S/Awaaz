"""Speaker diarization via acoustic embeddings and agglomerative clustering (Phase 10).

Extracts embeddings on sliding sub-windows, normalizes vectors, and clusters
speakers using cosine distance.
"""

from typing import List, Optional, Tuple

import numpy as np


class SpeakerDiarizer:
    """Computes speaker assignment intervals from audio segments."""

    def __init__(self, model_name: str = "speechbrain/spkrec-ecapa-voxceleb", threshold: float = 0.6):
        self.model_name = model_name
        self.threshold = threshold

    def diarize(
        self,
        audio: np.ndarray,
        speech_segments: List[Tuple[float, float]],
        num_speakers: Optional[int] = None,
    ) -> List[Tuple[float, float, str]]:
        """Returns list of (start_sec, end_sec, speaker_id) intervals."""
        raise NotImplementedError("Diarization will be implemented in Phase 10.")
