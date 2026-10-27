"""Speech enhancement and audio denoising interface (Phase 10 & 11).

Provides an interface for audio denoising models (e.g. DeepFilterNet) to generate
smoothed per-speaker listening audio.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Optional DeepFilterNet import
try:
    import torch
    from df.enhance import enhance as df_enhance
    from df.enhance import init_df
    HAS_DEEPFILTER = True
except ImportError:
    HAS_DEEPFILTER = False


class SpeechEnhancer:
    """Audio smoothing and noise reduction wrapper.

    Applies neural denoising (via DeepFilterNet when available) or soft spectral
    gating fallback to clean raw audio tracks for human listening.

    Rule §2.4: Default ASR transcription keeps enhancement off during inference
    to prevent acoustic artifact distortion.
    """

    def __init__(self, model_name: str = "DeepFilterNet3") -> None:
        self.model_name = model_name
        self.model = None
        self.df_state = None
        self._init_attempted = False

    def _lazy_init(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        if HAS_DEEPFILTER:
            try:
                self.model, self.df_state, _ = init_df()
                logger.info("Loaded DeepFilterNet successfully.")
            except Exception as e:
                logger.warning("Failed to initialize DeepFilterNet (%s). Falling back to spectral gating.", e)
                self.model = None
        else:
            logger.info("DeepFilterNet is not installed. Using local spectral gating enhancer fallback.")

    def enhance(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Applies denoising filter to audio array.

        Args:
            audio: 1D numpy array of audio samples (float32).
            sample_rate: Audio sampling rate (default: 16000 Hz).

        Returns:
            Denoised 1D numpy array.
        """
        if len(audio) == 0:
            return audio

        self._lazy_init()

        if self.model is not None and HAS_DEEPFILTER:
            try:
                audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(torch.float32)
                enhanced_tensor = df_enhance(self.model, self.df_state, audio_tensor)
                return enhanced_tensor.squeeze(0).cpu().numpy()
            except Exception as e:
                logger.warning("DeepFilterNet enhancement error: %s. Using spectral gating fallback.", e)

        # Fallback: lightweight spectral subtraction / gating
        return self._spectral_gating_denoise(audio, sample_rate=sample_rate)

    def _spectral_gating_denoise(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        noise_floor_percentile: float = 15.0,
        gain_reduction_db: float = 12.0,
    ) -> np.ndarray:
        """Lightweight, non-causal spectral gating fallback for audio smoothing."""
        frame_len = int(sample_rate * 0.030)  # 30ms
        hop_len = int(sample_rate * 0.015)    # 15ms
        if len(audio) < frame_len:
            return audio

        # Compute STFT
        window = np.hanning(frame_len)
        num_frames = (len(audio) - frame_len) // hop_len + 1
        stft_matrix = []

        for i in range(num_frames):
            frame = audio[i * hop_len : i * hop_len + frame_len] * window
            stft_matrix.append(np.fft.rfft(frame))

        stft_matrix = np.array(stft_matrix)  # [num_frames, bins]
        mag = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)

        # Estimate stationary noise threshold per frequency bin
        noise_profile = np.percentile(mag, noise_floor_percentile, axis=0, keepdims=True)
        snr = mag / (noise_profile + 1e-8)

        # Soft Wiener-like mask
        mask = np.clip(1.0 - (1.0 / (snr + 1e-4)), 0.0, 1.0)
        # Apply maximum attenuation
        min_gain = 10.0 ** (-gain_reduction_db / 20.0)
        gain_mask = np.maximum(mask, min_gain)

        denoised_stft = (mag * gain_mask) * np.exp(1j * phase)

        # Overlap-add synthesis
        out = np.zeros(len(audio), dtype=np.float32)
        norm_window = np.zeros(len(audio), dtype=np.float32)

        for i in range(num_frames):
            rec_frame = np.fft.irfft(denoised_stft[i], n=frame_len) * window
            out[i * hop_len : i * hop_len + frame_len] += rec_frame
            norm_window[i * hop_len : i * hop_len + frame_len] += window**2

        nonzero = norm_window > 1e-8
        out[nonzero] /= norm_window[nonzero]

        return np.clip(out, -1.0, 1.0).astype(audio.dtype)
