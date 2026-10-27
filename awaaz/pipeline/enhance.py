"""Speech enhancement and audio denoising interface (Phase 10 & 11).

Provides an interface for audio denoising models (e.g. DeepFilterNet) as well as
a custom lightweight causal convolutional-recurrent enhancer (< 500k parameters)
tailored for wearable hardware.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Optional DeepFilterNet import
try:
    from df.enhance import enhance as df_enhance
    from df.enhance import init_df
    HAS_DEEPFILTER = True
except ImportError:
    HAS_DEEPFILTER = False


class CausalConv1d(nn.Module):
    """Causal Conv1d layer with strict left-only padding (no future frame leakage)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, **kwargs):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: [batch, channels, time]"""
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class LightweightEnhancer(nn.Module):
    """Lightweight causal convolutional-recurrent speech enhancer (< 500k parameters).

    Targeted for on-device and wearable processing. Uses causal 1D depthwise separable
    convolutions and a 2-layer causal GRU to predict a time-frequency gain mask.
    """

    def __init__(self, num_bins: int = 80, hidden_dim: int = 64, num_gru_layers: int = 2) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim

        # Front-end projection & causal depthwise separable conv
        self.in_proj = nn.Linear(num_bins, hidden_dim)
        self.causal_conv = CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim)
        self.pointwise_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()

        # Causal GRU core
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
        )

        # Output mask projector
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_bins),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

            x: Spectrogram or filterbank tensor of shape [batch, time, num_bins]

        Returns:
            enhanced_x: Masked spectrogram [batch, time, num_bins]
        """
        # Linear projection
        h = self.in_proj(x)  # [B, T, H]

        # Depthwise separable causal conv across time
        h_conv = h.transpose(1, 2)  # [B, H, T]
        h_conv = self.causal_conv(h_conv)
        h_conv = self.pointwise_conv(h_conv)
        h = h + h_conv.transpose(1, 2)  # Residual
        h = self.act(self.norm(h))

        # GRU recurrent temporal modeling
        out, _ = self.gru(h)

        # Predict spectral gain mask in [0, 1]
        gain_mask = self.out_proj(out)

        # Apply mask
        return x * gain_mask

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
        hop_len = int(sample_rate * 0.015)  # 15ms
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
