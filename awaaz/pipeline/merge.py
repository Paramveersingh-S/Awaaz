"""Alignment and formatting of word timestamps, speaker labels, and transcripts (Phase 10).

Merges CTC word emissions with diarization windows and formats into TXT, SRT, VTT, and JSON.
"""

from typing import Any, Dict, List, Tuple


def assign_speakers_to_words(
    words_with_times: List[Tuple[str, float, float]],
    speaker_intervals: List[Tuple[float, float, str]],
) -> List[Dict[str, Any]]:
    """Maps words to speakers based on maximum temporal overlap or nearest interval center."""
    raise NotImplementedError("Word-speaker merging will be implemented in Phase 10.")


def format_transcript(
    annotated_words: List[Dict[str, Any]],
    output_format: str = "txt",
) -> str:
    """Formats annotated word sequences into plain text, SRT, VTT, or JSON."""
    raise NotImplementedError("Format export will be implemented in Phase 10.")
