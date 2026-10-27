"""Multi-talker speech separation interface for overlapped audio (Phase 11).

Separates overlapping speech regions into isolated single-speaker channels.
"""

from typing import List

import numpy as np


class SpeechSeparator:
    """Overlapped audio separator wrapper."""

    def __init__(self, model_name: str = "speechbrain/sepformer-wsj02mix"):
        self.model_name = model_name

    def separate(self, audio: np.ndarray, num_sources: int = 2) -> List[np.ndarray]:
        """Separates mixture audio into isolated channel signals."""
        raise NotImplementedError("Separation will be implemented in Phase 11.")
