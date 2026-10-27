"""Speaker diarization via acoustic embeddings and agglomerative clustering (Phase 10).

Windows VAD segments into 1.5s sub-windows with 0.75s stride, extracts L2-normalized
embeddings, and clusters speakers using cosine distance agglomerative clustering.
"""

from typing import List, Optional, Tuple

import numpy as np

# Optional scikit-learn import
try:
    from sklearn.cluster import AgglomerativeClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Optional SpeechBrain import
try:
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    HAS_SPEECHBRAIN = True
except ImportError:
    HAS_SPEECHBRAIN = False


class SpeakerDiarizer:
    """Computes speaker assignment intervals from audio segments."""

    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        threshold: float = 0.6,
        window_sec: float = 1.5,
        stride_sec: float = 0.75,
        sample_rate: int = 16000,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.sample_rate = sample_rate
        self._classifier = None

    def _get_speechbrain_model(self):
        """Lazy loader for SpeechBrain ECAPA-TDNN."""
        if self._classifier is None and HAS_SPEECHBRAIN:
            try:
                self._classifier = EncoderClassifier.from_hparams(
                    source=self.model_name,
                    run_opts={"device": "cpu"},
                )
            except Exception:
                self._classifier = None
        return self._classifier

    def _extract_builtin_embedding(self, sub_audio: np.ndarray) -> np.ndarray:
        """Extracts deterministic acoustic filterbank embedding with statistical pooling.

        Uses 40 filterbank energies across 25ms frames, computing mean, standard
        deviation, and delta features across time to produce a 120-dim speaker representation.
        """
        frame_len = int(self.sample_rate * 0.025)
        hop_len = int(self.sample_rate * 0.010)

        if len(sub_audio) < frame_len:
            pad = np.zeros(frame_len - len(sub_audio), dtype=sub_audio.dtype)
            sub_audio = np.concatenate([sub_audio, pad])

        num_frames = (len(sub_audio) - frame_len) // hop_len + 1
        num_filters = 40
        # Fast FFT-based filterbank simulation
        fft_size = 512
        frames = []
        for i in range(num_frames):
            frame = sub_audio[i * hop_len : i * hop_len + frame_len]
            # Hamming window
            window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(len(frame)) / (len(frame) - 1))
            spectrum = np.abs(np.fft.rfft(frame * window, n=fft_size))
            # Bin into 40 energy bands
            bin_size = len(spectrum) // num_filters
            band_energies = [
                np.log(np.mean(spectrum[b * bin_size : (b + 1) * bin_size] ** 2) + 1e-8)
                for b in range(num_filters)
            ]
            frames.append(band_energies)

        frames_arr = np.array(frames, dtype=np.float32)  # [num_frames, 40]
        mean_feat = np.mean(frames_arr, axis=0)  # [40]
        std_feat = np.std(frames_arr, axis=0)  # [40]
        if num_frames > 1:
            diff_feat = np.mean(np.diff(frames_arr, axis=0) ** 2, axis=0)  # [40]
        else:
            diff_feat = np.zeros(num_filters, dtype=np.float32)

        emb = np.concatenate([mean_feat, std_feat, diff_feat])  # [120]
        return emb

    def extract_embedding(self, sub_audio: np.ndarray) -> np.ndarray:
        """Extracts and L2-normalizes embedding for a sub-window."""
        classifier = self._get_speechbrain_model()
        if classifier is not None:
            try:
                wav = torch.tensor(sub_audio, dtype=torch.float32).unsqueeze(0)
                emb = classifier.encode_batch(wav).squeeze().detach().cpu().numpy()
            except Exception:
                emb = self._extract_builtin_embedding(sub_audio)
        else:
            emb = self._extract_builtin_embedding(sub_audio)

        norm = np.linalg.norm(emb)
        if norm > 1e-12:
            emb = emb / norm
        return emb

    def diarize(
        self,
        audio: np.ndarray,
        speech_segments: List[Tuple[float, float]],
        num_speakers: Optional[int] = None,
    ) -> List[Tuple[float, float, str]]:
        """Performs speaker diarization on detected speech segments.

        Args:
            audio: 1D numpy array of audio samples.
            speech_segments: List of (start_sec, end_sec) speech regions.
            num_speakers: Optional fixed speaker count override (--speakers N).

        Returns:
            List of (start_sec, end_sec, speaker_id) intervals.
        """
        if not speech_segments or len(audio) == 0:
            return []

        # 1. Generate sliding sub-windows (1.5s window, 0.75s stride)
        sub_windows: List[Tuple[float, float]] = []
        for start_sec, end_sec in speech_segments:
            seg_len = end_sec - start_sec
            if seg_len <= self.window_sec:
                sub_windows.append((start_sec, end_sec))
            else:
                cur = start_sec
                while cur + self.stride_sec <= end_sec:
                    w_end = min(end_sec, cur + self.window_sec)
                    sub_windows.append((round(cur, 4), round(w_end, 4)))
                    if w_end >= end_sec:
                        break
                    cur += self.stride_sec

        if not sub_windows:
            return []

        # 2. Extract embeddings
        embeddings = []
        for w_start, w_end in sub_windows:
            start_idx = int(w_start * self.sample_rate)
            end_idx = int(w_end * self.sample_rate)
            sub_audio = audio[start_idx:end_idx]
            emb = self.extract_embedding(sub_audio)
            embeddings.append(emb)

        embeddings_arr = np.array(embeddings, dtype=np.float32)  # [N, D]

        # 3. Cluster embeddings
        if len(sub_windows) == 1 or (num_speakers is not None and num_speakers == 1):
            labels = np.zeros(len(sub_windows), dtype=int)
        elif HAS_SKLEARN:
            # Cosine distance matrix: D = 1.0 - (X . X^T)
            similarity = np.clip(np.dot(embeddings_arr, embeddings_arr.T), -1.0, 1.0)
            dist_matrix = np.maximum(0.0, 1.0 - similarity)

            if num_speakers is not None:
                n_clusters = min(num_speakers, len(sub_windows))
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric="precomputed",
                    linkage="average",
                )
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=self.threshold,
                    metric="precomputed",
                    linkage="average",
                )
            labels = clustering.fit_predict(dist_matrix)
        else:
            # Fallback greedy cosine clustering
            labels = np.zeros(len(sub_windows), dtype=int)
            centers = [embeddings_arr[0]]
            for i in range(1, len(embeddings_arr)):
                sims = [np.dot(embeddings_arr[i], c) for c in centers]
                max_sim_idx = int(np.argmax(sims))
                max_sim = sims[max_sim_idx]
                if (1.0 - max_sim) > self.threshold and (num_speakers is None or len(centers) < num_speakers):
                    centers.append(embeddings_arr[i])
                    labels[i] = len(centers) - 1
                else:
                    labels[i] = max_sim_idx

        # 4. Group into continuous intervals per speaker
        speaker_intervals: List[Tuple[float, float, str]] = []
        cur_start, cur_end = sub_windows[0]
        cur_speaker = f"SPEAKER_{labels[0]:02d}"

        for i in range(1, len(sub_windows)):
            w_start, w_end = sub_windows[i]
            spk = f"SPEAKER_{labels[i]:02d}"

            if spk == cur_speaker and w_start <= cur_end + 0.1:
                cur_end = max(cur_end, w_end)
            else:
                speaker_intervals.append((round(cur_start, 4), round(cur_end, 4), cur_speaker))
                cur_start, cur_end = w_start, w_end
                cur_speaker = spk

        speaker_intervals.append((round(cur_start, 4), round(cur_end, 4), cur_speaker))
        return speaker_intervals
