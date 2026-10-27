"""Diarization threshold tuning and error scoring script (Phase 10).

Sweeps distance thresholds across a range (e.g. 0.3 to 0.9) to optimize
speaker clustering performance against reference intervals.
"""

import argparse
import sys
from typing import Dict, List, Tuple

import numpy as np

from awaaz.pipeline.diarize import SpeakerDiarizer


def compute_segment_overlap_accuracy(
    hypothesis: List[Tuple[float, float, str]],
    reference: List[Tuple[float, float, str]],
    time_step: float = 0.1,
) -> float:
    """Computes frame-level speaker attribution accuracy between hypothesis and reference."""
    if not reference or not hypothesis:
        return 0.0

    max_t = max(max(e for _, e, _ in reference), max(e for _, e, _ in hypothesis))
    times = np.arange(0.0, max_t, time_step)

    # Frame assignments
    ref_labels: Dict[float, str] = {}
    for s, e, spk in reference:
        for t in times:
            if s <= t <= e:
                ref_labels[round(t, 2)] = spk

    hyp_labels: Dict[float, str] = {}
    for s, e, spk in hypothesis:
        for t in times:
            if s <= t <= e:
                hyp_labels[round(t, 2)] = spk

    # Best-match speaker permutation mapping
    unique_ref = sorted(list(set(ref_labels.values())))
    unique_hyp = sorted(list(set(hyp_labels.values())))

    if not unique_ref or not unique_hyp:
        return 0.0

    # Build confusion matrix
    conf_matrix = np.zeros((len(unique_ref), len(unique_hyp)), dtype=int)
    for t_round, ref_spk in ref_labels.items():
        if t_round in hyp_labels:
            hyp_spk = hyp_labels[t_round]
            r_idx = unique_ref.index(ref_spk)
            h_idx = unique_hyp.index(hyp_spk)
            conf_matrix[r_idx, h_idx] += 1

    # Greedy assignment of hyp speaker to ref speaker
    correct_frames = 0
    assigned_hyp = set()
    for r_idx in range(len(unique_ref)):
        best_h = -1
        best_val = -1
        for h_idx in range(len(unique_hyp)):
            if h_idx not in assigned_hyp and conf_matrix[r_idx, h_idx] > best_val:
                best_val = conf_matrix[r_idx, h_idx]
                best_h = h_idx
        if best_h != -1:
            assigned_hyp.add(best_h)
            correct_frames += best_val

    total_ref_frames = len(ref_labels)
    return float(correct_frames) / max(1, total_ref_frames)


def sweep_thresholds(
    audio: np.ndarray,
    speech_segments: List[Tuple[float, float]],
    reference: List[Tuple[float, float, str]],
    threshold_min: float = 0.3,
    threshold_max: float = 0.9,
    step: float = 0.05,
) -> Dict[float, float]:
    """Sweeps distance thresholds and returns mapping of {threshold: accuracy}."""
    thresholds = np.arange(threshold_min, threshold_max + 1e-5, step)
    results = {}

    print(f"Sweeping thresholds from {threshold_min:.2f} to {threshold_max:.2f} (step={step:.2f})...")
    for th in thresholds:
        th = round(float(th), 3)
        diarizer = SpeakerDiarizer(threshold=th)
        hyp = diarizer.diarize(audio, speech_segments)
        acc = compute_segment_overlap_accuracy(hyp, reference)
        results[th] = acc
        print(f"  Threshold {th:0.2f} -> Accuracy: {acc * 100:0.1f}% (Found {len(set(s[2] for s in hyp))} speakers)")

    best_th = max(results, key=results.get)
    print(f"\nBest Threshold: {best_th:.2f} with Accuracy {results[best_th] * 100:.1f}%")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune diarization clustering threshold.")
    parser.add_argument("--audio", type=str, help="Path to evaluation audio file.")
    parser.add_argument("--min", type=float, default=0.3, help="Minimum threshold.")
    parser.add_argument("--max", type=float, default=0.9, help="Maximum threshold.")
    parser.add_argument("--step", type=float, default=0.05, help="Step size.")
    _ = parser.parse_args()

    print("[Awaaz] Diarization Tuning Tool ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
