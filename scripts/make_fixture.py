"""Synthetic ASR Fixture Generator for Awaaz (Phase 0).

Generates a tiny synthetic fixture (synthetic 16 kHz FLAC audio + JSONL manifests)
for fast, reproducible CPU testing without needing external downloads.
Runs in < 30 seconds.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000

# Sample sentences avoiding digits, adhering to Phase-1 normalization rules
ENGLISH_SENTENCES = [
    "hello world this is an awaaz speech recognition test",
    "streaming conformer on edge devices",
    "voice activity detection and speaker diarization",
    "open source bilingual automatic speech recognition",
    "we test audio features and rotary positional embeddings",
    "sound waves converted into log mel filterbanks",
    "continuous sliding window inference without latency",
    "compact neural network running on smartwatch hardware",
    "clean reproducible synthetic fixtures for fast cpu tests",
    "resumable training loop on cloud gpu clusters",
    "good morning everyone welcome to the meeting",
    "thank you for your collaboration and contribution",
]

PUNJABI_SENTENCES = [
    "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਦੋਸਤੋ ਕੀ ਹਾਲ ਹੈ",
    "ਆਵਾਜ਼ ਇਕ ਖੁੱਲਾ ਸਰੋਤ ਬੋਲੀ ਪਛਾਣ ਪ੍ਰੋਜੈਕਟ ਹੈ",
    "ਪੰਜਾਬੀ ਬੋਲੀ ਬਹੁਤ ਮਿੱਠੀ ਅਤੇ ਅਮੀਰ ਹੈ",
    "ਮਿਲ ਕੇ ਕੰਮ ਕਰਨਾ ਬਹੁਤ ਵਧੀਆ ਹੁੰਦਾ ਹੈ",
    "ਅਸੀਂ ਨਵੀਂ ਤਕਨਾਲੋਜੀ ਦੀ ਵਰਤੋਂ ਕਰ ਰਹੇ ਹਾਂ",
    "ਸਮਾਰਟ ਵਾਚ ਉੱਤੇ ਬੋਲੀ ਪਛਾਣ ਚਲਾਉਣਾ ਸਾਡਾ ਟੀਚਾ ਹੈ",
    "ਹਰ ਇਕ ਸ਼ਬਦ ਨੂੰ ਧਿਆਨ ਨਾਲ ਸਮਝੋ",
    "ਪੰਜਾਬ ਦੇ ਲੋਕ ਬਹੁਤ ਮਿਹਨਤੀ ਹਨ",
    "ਇਹ ਇਕ ਛੋਟਾ ਮਾਡਲ ਹੈ ਜੋ ਤੇਜ਼ ਚੱਲਦਾ ਹੈ",
    "ਸਾਰਿਆਂ ਦਾ ਧੰਨਵਾਦ ਅਤੇ ਨਮਸਤੇ ਜੀ",
    "ਸਾਡੀ ਮਾਂ ਬੋਲੀ ਸਾਡਾ ਮਾਣ ਹੈ",
    "ਸਵੇਰ ਦਾ ਸਮਾਂ ਬਹੁਤ ਸੁਹਾਵਣਾ ਹੁੰਦਾ ਹੈ",
]

CODE_SWITCH_SENTENCES = [
    "my name is simran ਅਤੇ ਮੈਂ ਚੰਡੀਗੜ੍ਹ ਤੋਂ ਹਾਂ",
    "the weather is very good ਅੱਜ ਬਹੁਤ ਵਧੀਆ ਦਿਨ ਹੈ",
    "welcome to the meeting ਸਾਰਿਆਂ ਦਾ ਸਵਾਗਤ ਹੈ",
    "please send the report ਜਲਦੀ ਭੇਜ ਦਿਓ",
]

SACRED_TEST_SENTENCES = [
    ("ਇਕ ਓਅੰਕਾਰ ਸਤਿਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ", "sacred"),
    ("ਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖਾਲਸਾ ਵਾਹਿਗੁਰੂ ਜੀ ਕੀ ਫਤਿਹ", "sacred"),
]


def generate_synthetic_audio(duration_sec: float, pitch_hz: float, seed: int) -> np.ndarray:
    """Generates synthetic audio signal with harmonic tones, decay envelope, and soft noise.

    Args:
        duration_sec: Length of audio in seconds.
        pitch_hz: Fundamental frequency of the tone in Hz.
        seed: Random seed for reproducibility.

    Returns:
        1D numpy array of float32 samples at 16,000 Hz.
    """
    rng = np.random.default_rng(seed)
    num_samples = int(duration_sec * SAMPLE_RATE)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)

    # Fundamental + 2 harmonics
    signal = 0.5 * np.sin(2 * np.pi * pitch_hz * t)
    signal += 0.25 * np.sin(2 * np.pi * (pitch_hz * 2) * t)
    signal += 0.12 * np.sin(2 * np.pi * (pitch_hz * 3) * t)

    # Smooth attack and decay envelope
    attack_samples = int(0.05 * SAMPLE_RATE)
    decay_samples = int(0.1 * SAMPLE_RATE)
    envelope = np.ones(num_samples, dtype=np.float32)

    if num_samples > attack_samples + decay_samples:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)

    signal = signal * envelope

    # Add gentle pink/gaussian background noise (-30 dB)
    noise = rng.normal(0, 0.01, size=num_samples).astype(np.float32)
    audio = signal + noise

    # Normalize to [-0.9, 0.9] peak amplitude
    max_val = np.max(np.abs(audio))
    if max_val > 1e-4:
        audio = 0.9 * (audio / max_val)

    return audio.astype(np.float32)


def create_fixture(output_dir: str, num_train: int = 24, num_dev: int = 8) -> Tuple[str, str]:
    """Creates a full synthetic dataset fixture with audio FLAC files and JSONL manifests.

    Args:
        output_dir: Directory where audio and manifests will be written.
        num_train: Number of training samples to generate.
        num_dev: Number of dev samples to generate.

    Returns:
        Tuple of (train_manifest_path, dev_manifest_path).
    """
    out_path = Path(output_dir)
    audio_dir = out_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    items = []

    # Compile pool of examples
    pool: List[Tuple[str, str, str]] = []  # (text, lang, domain)
    for txt in ENGLISH_SENTENCES:
        pool.append((txt, "en", "general"))
    for txt in PUNJABI_SENTENCES:
        pool.append((txt, "pa", "general"))
    for txt in CODE_SWITCH_SENTENCES:
        pool.append((txt, "pa", "general"))
    for txt, dom in SACRED_TEST_SENTENCES:
        pool.append((txt, "pa", dom))

    # Deterministic generation
    for i in range(num_train + num_dev):
        text, lang, domain = pool[i % len(pool)]
        duration = 1.0 + (i % 5) * 0.5  # Durations between 1.0s and 3.0s
        pitch = 180.0 + (i * 27) % 240  # Varied pitch 180 Hz - 420 Hz
        seed = 1000 + i

        audio = generate_synthetic_audio(duration, pitch, seed)
        filename = f"clip_{i:04d}_{lang}.flac"
        file_path = audio_dir / filename

        sf.write(str(file_path), audio, SAMPLE_RATE, format="FLAC")

        items.append({
            "audio_path": str(file_path.resolve()).replace("\\", "/"),
            "text": text,
            "duration": round(duration, 3),
            "language": lang,
            "domain": domain,
            "source": "synthetic_fixture",
        })

    train_items = items[:num_train]
    dev_items = items[num_train:num_train + num_dev]

    train_manifest = out_path / "train_manifest.jsonl"
    dev_manifest = out_path / "dev_manifest.jsonl"

    with open(train_manifest, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(dev_manifest, "w", encoding="utf-8") as f:
        for item in dev_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return str(train_manifest), str(dev_manifest)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ASR fixture.")
    parser.add_argument("--output_dir", type=str, default="tests/fixtures/synthetic", help="Output directory.")
    parser.add_argument("--train_count", type=int, default=24, help="Number of training clips.")
    parser.add_argument("--dev_count", type=int, default=8, help="Number of dev clips.")
    args = parser.parse_args()

    print(f"[make_fixture] Generating synthetic fixture in {args.output_dir}...")
    train_m, dev_m = create_fixture(args.output_dir, args.train_count, args.dev_count)
    print(f"[make_fixture] Successfully generated:\n - {train_m}\n - {dev_m}")


if __name__ == "__main__":
    main()
