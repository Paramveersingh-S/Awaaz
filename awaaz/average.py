"""Model checkpoint weight averaging utility for Awaaz (Phase 7).

Computes parameter-wise arithmetic mean across the top K checkpoints to boost accuracy
and generalizability without extra training compute.
"""

import argparse
import glob
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

logger = logging.getLogger("awaaz.average")


def average_checkpoints(
    checkpoint_paths: Union[str, List[str], List[Path]],
    output_path: Union[str, Path],
    num_checkpoints: Optional[int] = None,
) -> str:
    """Averages model parameters across provided checkpoints.

    Args:
        checkpoint_paths: Single glob pattern, or list of file paths.
        output_path: Target path to save the averaged checkpoint.
        num_checkpoints: If specified, uses only the latest N checkpoints.

    Returns:
        The output path string of the saved averaged checkpoint.
    """
    # 1. Resolve paths
    if isinstance(checkpoint_paths, (str, Path)):
        resolved = sorted(glob.glob(str(checkpoint_paths)))
    else:
        resolved = sorted([str(p) for p in checkpoint_paths])

    if not resolved:
        raise FileNotFoundError(f"No checkpoint files found matching: {checkpoint_paths}")

    if num_checkpoints is not None and len(resolved) > num_checkpoints:
        resolved = resolved[-num_checkpoints:]

    logger.info("Averaging %d checkpoints: %s", len(resolved), [Path(p).name for p in resolved])

    # 2. Accumulate weights
    accumulated_weights: Dict[str, torch.Tensor] = {}
    total_ckpts = len(resolved)
    base_meta: Dict = {}

    for i, ckpt_p in enumerate(resolved):
        ckpt = torch.load(ckpt_p, map_location="cpu")
        model_state = ckpt["model"] if "model" in ckpt else ckpt

        # Save metadata from newest checkpoint
        if i == total_ckpts - 1:
            base_meta = {k: v for k, v in ckpt.items() if k != "model"}

        for k, v in model_state.items():
            if not torch.is_tensor(v):
                continue
            if k not in accumulated_weights:
                if torch.is_floating_point(v):
                    accumulated_weights[k] = v.float().clone()
                else:
                    # Integer buffer (e.g. step or num_batches_tracked)
                    accumulated_weights[k] = v.clone()
            else:
                if torch.is_floating_point(v):
                    accumulated_weights[k] += v.float()
                else:
                    # Keep latest for non-floating point
                    accumulated_weights[k] = v.clone()

    # 3. Compute arithmetic mean for floating point tensors
    averaged_model: Dict[str, torch.Tensor] = {}
    for k, v in accumulated_weights.items():
        if torch.is_floating_point(v):
            averaged_model[k] = v / float(total_ckpts)
        else:
            averaged_model[k] = v

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    result_ckpt = {
        "model": averaged_model,
        "averaged_from": resolved,
        "num_checkpoints": total_ckpts,
        **base_meta,
    }

    # Save atomically
    temp_p = out_p.with_suffix(".tmp")
    torch.save(result_ckpt, temp_p)
    os.replace(str(temp_p), str(out_p))

    logger.info("Averaged checkpoint successfully written to %s", out_p)
    return str(out_p)


def main():
    parser = argparse.ArgumentParser(description="Average Awaaz Model Checkpoints")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint files or glob.")
    parser.add_argument("--output", type=str, required=True, help="Output destination for averaged weights.")
    parser.add_argument("--num_checkpoints", type=int, default=None, help="Maximum newest checkpoints to average.")

    args = parser.parse_args()

    # Handle wildcards if passed as a single string
    if len(args.checkpoints) == 1 and ("*" in args.checkpoints[0] or "?" in args.checkpoints[0]):
        ckpt_list = args.checkpoints[0]
    else:
        ckpt_list = args.checkpoints

    average_checkpoints(
        checkpoint_paths=ckpt_list,
        output_path=args.output,
        num_checkpoints=args.num_checkpoints,
    )


if __name__ == "__main__":
    main()
