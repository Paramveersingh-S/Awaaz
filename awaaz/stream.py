"""Incremental streaming speech recognizer for Awaaz (Phase 8).

Implements rolling-window chunk processing, incremental CTC state collapse,
partial-result stability, and algorithmic latency accounting.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from awaaz.features import LogMelSpectrogram
from awaaz.text import Tokenizer


class StreamingRecognizer:
    """Consumes arbitrary-sized 16 kHz audio blocks and emits live transcripts."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        frontend: Optional[LogMelSpectrogram] = None,
        chunk_frames: int = 16,
        left_chunks: int = 8,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initializes the streaming recognizer.

        Args:
            model: ConformerCTC model instance.
            tokenizer: Bilingual SentencePiece Tokenizer.
            frontend: Optional LogMelSpectrogram feature extractor.
            chunk_frames: Encoder frames per chunk (default 16 = 0.64s).
            left_chunks: Number of left context chunks (default 8).
            device: 'cpu' or 'cuda' (auto-detected if None).
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.model.eval()

        self.tokenizer = tokenizer
        if frontend is None:
            self.frontend = LogMelSpectrogram(sample_rate=16000, n_mels=80).to(self.device)
        else:
            self.frontend = frontend.to(self.device)
        self.frontend.eval()

        self.chunk_frames = chunk_frames
        self.left_chunks = left_chunks

        # Audio samples per encoder chunk: 4 subsample_factor * 160 hop_samples = 640 samples/frame
        self.samples_per_chunk = self.chunk_frames * 640
        self.max_window_samples = (self.left_chunks + 1) * self.samples_per_chunk

        # Latency accounting: chunk_frames * 40ms
        self.algorithmic_latency_sec = round((self.chunk_frames * 0.040), 3)

        self.reset()

    def reset(self) -> None:
        """Clears all audio buffers and persistent decoding state."""
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.history_audio = np.zeros(0, dtype=np.float32)
        self.tokens: List[int] = []
        self.last_token_id = self.tokenizer.blank_id
        self.finalized_tokens_count = 0

    def _process_chunk(self, chunk_audio: np.ndarray) -> None:
        """Appends chunk to history, runs encoder over window, and updates CTC state."""
        # 1. Update history audio (capped at max_window_samples in integer chunk multiples)
        if len(self.history_audio) == 0:
            self.history_audio = chunk_audio.copy()
        else:
            self.history_audio = np.concatenate([self.history_audio, chunk_audio])

        if len(self.history_audio) > self.max_window_samples:
            self.history_audio = self.history_audio[-self.max_window_samples:]

        # 2. Extract features and run model
        with torch.no_grad():
            waveform_t = torch.from_numpy(self.history_audio).unsqueeze(0).to(self.device)
            features = self.frontend(waveform_t)
            feat_lens = torch.tensor([features.shape[1]], dtype=torch.long, device=self.device)

            log_probs, out_lens = self.model(
                features,
                feat_lens,
                chunk_size=self.chunk_frames,
                left_chunks=self.left_chunks,
            )

            # 3. Slice strictly the newest chunk's logits
            chunk_logits = log_probs[0, -self.chunk_frames:, :]
            argmax_ids = chunk_logits.argmax(dim=-1).cpu().tolist()

            # 4. Incremental CTC Collapse across chunk boundary
            self.finalized_tokens_count = len(self.tokens)
            for tok in argmax_ids:
                if tok != self.last_token_id:
                    if tok != self.tokenizer.blank_id:
                        self.tokens.append(tok)
                    self.last_token_id = tok

    def accept(self, audio_block: Union[np.ndarray, torch.Tensor]) -> str:
        """Consumes a block of 16 kHz mono float32 audio and emits updated transcript.

        Args:
            audio_block: 1D numpy array or torch tensor of arbitrary length.

        Returns:
            The current decoded transcript string.
        """
        if torch.is_tensor(audio_block):
            audio_arr = audio_block.detach().cpu().numpy().astype(np.float32).flatten()
        else:
            audio_arr = np.asarray(audio_block, dtype=np.float32).flatten()

        if len(self.audio_buffer) == 0:
            self.audio_buffer = audio_arr
        else:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_arr])

        # Consume all full chunks available in the buffer
        while len(self.audio_buffer) >= self.samples_per_chunk:
            chunk = self.audio_buffer[:self.samples_per_chunk]
            self.audio_buffer = self.audio_buffer[self.samples_per_chunk:]
            self._process_chunk(chunk)

        return self.get_transcript()

    def flush(self) -> str:
        """Pads residual buffer to a full chunk and returns complete final transcript.

        Returns:
            Complete final transcribed string.
        """
        if len(self.audio_buffer) > 0:
            # Pad remainder with zeros to reach a full chunk boundary
            pad_needed = self.samples_per_chunk - len(self.audio_buffer)
            if pad_needed > 0:
                final_chunk = np.pad(self.audio_buffer, (0, pad_needed), mode="constant")
            else:
                final_chunk = self.audio_buffer[:self.samples_per_chunk]

            self.audio_buffer = np.zeros(0, dtype=np.float32)
            self._process_chunk(final_chunk)

        return self.get_transcript()

    def get_transcript(self) -> str:
        """Returns the full decoded text of all emitted tokens."""
        return self.tokenizer.decode(self.tokens).strip()

    def get_partial_stability(self) -> Dict[str, str]:
        """Separates hypothesis into confirmed (finalized) vs tentative words."""
        finalized = self.tokenizer.decode(self.tokens[:self.finalized_tokens_count]).strip()
        tentative = self.tokenizer.decode(self.tokens[self.finalized_tokens_count:]).strip()
        return {
            "finalized": finalized,
            "tentative": tentative,
        }
