"""Speech enhancement and denoising interface (Phase 10 & 11).

Provides an abstract interface for audio denoising models (e.g. DeepFilterNet)
to generate smoothed per-speaker listening audio.
"""

import numpy as np


class SpeechEnhancer:
    """Audio smoothing and noise reduction wrapper."""

    def __init__(self, model_name: str = "DeepFilterNet3"):
        self.model_name = model_name

    def enhance(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Applies denoising filter to audio array."""
        raise NotImplementedError("Enhancer will be implemented in Phase 10.")
