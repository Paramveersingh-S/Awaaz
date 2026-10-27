"""Audio feature extraction and augmentation module for Awaaz (Phase 2).

Implements 80-bin log-mel filterbanks, global Cumulative Mean and Variance
Normalization (CMVN), SpecAugment, and waveform augmentations.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.functional as F

logger = logging.getLogger("awaaz.features")


def create_mel_filterbanks(
    sample_rate: int = 16000,
    n_fft: int = 400,
    n_mels: int = 80,
    f_min: float = 20.0,
    f_max: float = 7600.0,
) -> torch.Tensor:
    """Generates standard triangular mel filterbank weights.

    Args:
        sample_rate: Audio sampling frequency in Hz.
        n_fft: FFT window size (determines n_freqs = n_fft // 2 + 1).
        n_mels: Number of output mel frequency channels.
        f_min: Lower bound frequency in Hz.
        f_max: Upper bound frequency in Hz.

    Returns:
        Tensor of shape (n_freqs, n_mels) with Slaney-style area normalization.
    """
    n_freqs = n_fft // 2 + 1
    fb = F.melscale_fbanks(
        n_freqs=n_freqs,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm="slaney",
        mel_scale="htk",
    )
    return fb


class LogMelSpectrogram(nn.Module):
    """Computes 80-bin Log-Mel filterbank energies with global CMVN buffers.

    Input shape: (batch, samples) at 16,000 Hz float32.
    Output shape: (batch, time_frames, 80).
    Frame count formula: samples // 160 + 1.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 20.0,
        f_max: float = 7600.0,
        log_floor: float = 1e-5,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.log_floor = log_floor

        # Periodic Hann window
        window = torch.hann_window(win_length, periodic=True)
        self.register_buffer("window", window)

        # Mel filterbank weights (201, 80)
        fb = create_mel_filterbanks(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self.register_buffer("filterbanks", fb)

        # Global CMVN statistics buffers (80,)
        self.register_buffer("cmvn_mean", torch.zeros(n_mels, dtype=torch.float32))
        self.register_buffer("cmvn_std", torch.ones(n_mels, dtype=torch.float32))
        self.cmvn_loaded = False

    def load_cmvn(self, cmvn_path: Union[str, Path]) -> None:
        """Loads global mean and std statistics from an .npz file."""
        data = np.load(str(cmvn_path))
        mean = torch.from_numpy(data["mean"]).float()
        std = torch.from_numpy(data["std"]).float()
        if mean.shape[0] != self.n_mels or std.shape[0] != self.n_mels:
            raise ValueError(f"CMVN dimensions ({mean.shape[0]}) mismatch n_mels ({self.n_mels})")
        self.cmvn_mean.copy_(mean)
        self.cmvn_std.copy_(std)
        self.cmvn_loaded = True
        logger.info("Loaded CMVN statistics from %s", cmvn_path)

    def save_cmvn(self, output_path: Union[str, Path]) -> None:
        """Saves current CMVN statistics to an .npz file."""
        np.savez(
            str(output_path),
            mean=self.cmvn_mean.detach().cpu().numpy(),
            std=self.cmvn_std.detach().cpu().numpy(),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Extracts log-mel filterbank features with global CMVN normalization.

        Args:
            waveforms: Float tensor of shape (batch, samples) or (samples,).

        Returns:
            Normalized log-mel tensor of shape (batch, time_frames, n_mels).
        """
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)

        # Compute Short-Time Fourier Transform with reflect padding
        stft = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )

        # Power spectrum: |STFT|^2 -> shape (batch, time, freq)
        power_spec = stft.abs().pow(2).transpose(1, 2)

        # Apply mel filterbanks: (batch, time, 201) @ (201, 80) -> (batch, time, 80)
        mel_energies = torch.matmul(power_spec, self.filterbanks)

        # Natural logarithm with stabilization floor
        log_mel = torch.log(torch.clamp(mel_energies, min=self.log_floor))

        # Apply Global CMVN normalization
        if self.cmvn_loaded:
            normalized = (log_mel - self.cmvn_mean) / (self.cmvn_std + 1e-5)
            return normalized

        return log_mel


def compute_cmvn(
    manifest_path: Union[str, Path],
    output_path: Union[str, Path],
    max_utterances: int = 2000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes global mean and standard deviation across audio in a manifest.

    Args:
        manifest_path: Path to JSONL manifest containing 'audio_path'.
        output_path: Output path for cmvn.npz.
        max_utterances: Number of utterances to sample (default 2000).
        seed: Random seed for shuffling.

    Returns:
        Tuple of (mean, std) numpy arrays of shape (80,).
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(seed)
    rng.shuffle(items)
    selected_items = items[:max_utterances]

    frontend = LogMelSpectrogram()
    all_frames: List[np.ndarray] = []

    for item in selected_items:
        audio_path = item.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            continue
        audio, sr = sf.read(audio_path, dtype="float32")
        if sr != 16000:
            import scipy.signal
            num_samples = int(len(audio) * 16000 / sr)
            audio = scipy.signal.resample(audio, num_samples)

        tensor_audio = torch.from_numpy(audio).float()
        with torch.no_grad():
            log_mel = frontend(tensor_audio)  # shape (1, time, 80)
        all_frames.append(log_mel.squeeze(0).cpu().numpy())

    if not all_frames:
        raise RuntimeError(f"No audio files could be loaded from {manifest_path}")

    concatenated = np.concatenate(all_frames, axis=0)  # shape (total_frames, 80)
    mean = np.mean(concatenated, axis=0).astype(np.float32)
    std = np.std(concatenated, axis=0).astype(np.float32)

    # Prevent division by zero
    std = np.maximum(std, 1e-5)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_p), mean=mean, std=std)
    logger.info("Saved global CMVN (mean and std) to %s (computed over %d utterances)", output_path, len(all_frames))
    return mean, std


class SpecAugment(nn.Module):
    """SpecAugment data augmentation module applied strictly in training mode.

    Applies 2 frequency masks (width <= 27) and 2 time masks (width <= 5% of valid length).
    Mask fill value is 0.0 (the mean after global CMVN).
    Strict identity pass-through in eval mode.
    """

    def __init__(
        self,
        freq_mask_max_width: int = 27,
        num_freq_masks: int = 2,
        time_mask_max_ratio: float = 0.05,
        num_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_mask_max_width = freq_mask_max_width
        self.num_freq_masks = num_freq_masks
        self.time_mask_max_ratio = time_mask_max_ratio
        self.num_time_masks = num_time_masks

    def forward(self, features: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Applies SpecAugment masking to batch of features.

        Args:
            features: Tensor of shape (batch, time, n_mels).
            lengths: Optional tensor of valid frame lengths (batch,).

        Returns:
            Masked feature tensor of shape (batch, time, n_mels).
        """
        if not self.training:
            return features

        augmented = features.clone()
        batch_size, time_dim, mel_dim = augmented.shape

        for b in range(batch_size):
            valid_time = int(lengths[b].item()) if lengths is not None else time_dim

            # 1. Frequency masking
            for _ in range(self.num_freq_masks):
                f_width = random.randint(0, min(self.freq_mask_max_width, mel_dim))
                if f_width > 0:
                    f_start = random.randint(0, mel_dim - f_width)
                    augmented[b, :valid_time, f_start : f_start + f_width] = 0.0

            # 2. Time masking (strictly up to 5% of valid length)
            max_time_mask = max(1, int(valid_time * self.time_mask_max_ratio))
            for _ in range(self.num_time_masks):
                t_width = random.randint(0, max_time_mask)
                if t_width > 0 and valid_time > t_width:
                    t_start = random.randint(0, valid_time - t_width)
                    augmented[b, t_start : t_start + t_width, :] = 0.0

        return augmented


class WaveformAugmenter:
    """Waveform data augmentations for Awaaz training.

    Supports speed perturbation (0.9x, 1.0x, 1.1x), additive noise,
    room impulse response (RIR) reverberation, and random gain.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        speed_perturb: bool = True,
        noise_dir: Optional[str] = None,
        rir_dir: Optional[str] = None,
        random_gain_db: float = 6.0,
    ):
        self.sample_rate = sample_rate
        self.speed_perturb = speed_perturb
        self.noise_dir = Path(noise_dir) if noise_dir else None
        self.rir_dir = Path(rir_dir) if rir_dir else None
        self.random_gain_db = random_gain_db

        self.noise_files = list(self.noise_dir.glob("*.flac")) + list(self.noise_dir.glob("*.wav")) if self.noise_dir and self.noise_dir.is_dir() else []
        self.rir_files = list(self.rir_dir.glob("*.flac")) + list(self.rir_dir.glob("*.wav")) if self.rir_dir and self.rir_dir.is_dir() else []

    def apply_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Perturbs speed by resampling."""
        if speed == 1.0 or abs(speed - 1.0) < 1e-3:
            return audio
        import scipy.signal
        num_target_samples = int(len(audio) / speed)
        return scipy.signal.resample(audio, num_target_samples).astype(np.float32)

    def apply_gain(self, audio: np.ndarray, gain_db: float) -> np.ndarray:
        """Applies random volume gain."""
        scale = 10.0 ** (gain_db / 20.0)
        return (audio * scale).astype(np.float32)

    def apply_noise(self, audio: np.ndarray, snr_db: float) -> np.ndarray:
        """Adds background noise at specified SNR."""
        if not self.noise_files:
            return audio

        noise_file = random.choice(self.noise_files)
        noise, sr = sf.read(str(noise_file), dtype="float32")
        if sr != self.sample_rate:
            import scipy.signal
            noise = scipy.signal.resample(noise, int(len(noise) * self.sample_rate / sr))

        # Loop or slice noise to match audio length
        if len(noise) < len(audio):
            repeats = int(np.ceil(len(audio) / len(noise)))
            noise = np.tile(noise, repeats)
        start = random.randint(0, len(noise) - len(audio))
        noise_slice = noise[start : start + len(audio)]

        # Calculate SNR and scale
        signal_power = np.mean(audio ** 2) + 1e-8
        noise_power = np.mean(noise_slice ** 2) + 1e-8
        target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        scaled_noise = noise_slice * np.sqrt(target_noise_power / noise_power)

        augmented = audio + scaled_noise
        return augmented.astype(np.float32)

    def augment(self, audio: np.ndarray) -> np.ndarray:
        """Applies configured waveform augmentations stochastically."""
        # 1. Speed perturbation (0.9, 1.0, 1.1)
        if self.speed_perturb:
            speed = random.choice([0.9, 1.0, 1.1])
            audio = self.apply_speed(audio, speed)

        # 2. Additive background noise (SNR 5 - 20 dB)
        if self.noise_files and random.random() < 0.5:
            snr = random.uniform(5.0, 20.0)
            audio = self.apply_noise(audio, snr)

        # 3. Random gain (± random_gain_db)
        if self.random_gain_db > 0:
            gain = random.uniform(-self.random_gain_db, self.random_gain_db)
            audio = self.apply_gain(audio, gain)

        # Clamp peak to avoid clipping
        peak = float(np.max(np.abs(audio)))
        if peak > 0.95:
            audio = audio * (0.95 / peak)

        return audio.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Awaaz Features & CMVN Utility")
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    cmvn_parser = subparsers.add_parser("compute-cmvn", help="Compute global CMVN from manifest.")
    cmvn_parser.add_argument("--manifest", type=str, required=True, help="Path to JSONL manifest.")
    cmvn_parser.add_argument("--output", type=str, default="cmvn.npz", help="Output path for cmvn.npz.")
    cmvn_parser.add_argument("--max_utterances", type=int, default=2000, help="Max utterances to sample.")

    args = parser.parse_args()

    if args.subcommand == "compute-cmvn":
        compute_cmvn(manifest_path=args.manifest, output_path=args.output, max_utterances=args.max_utterances)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
