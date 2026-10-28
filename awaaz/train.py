"""Resumable, mixed-precision, dynamic-chunk training loop for Awaaz (Phase 5).

Handles atomic checkpointing, warmup-cosine schedule, gradient clipping,
dynamic-chunk attention sampling, and multi-language evaluation.
"""

import argparse
import csv
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from awaaz.dataset import (
    AwaazDataset,
    DurationBucketedBatchSampler,
    ctc_collate_fn,
)
from awaaz.features import LogMelSpectrogram, SpecAugment
from awaaz.metrics import compute_wer_cer
from awaaz.model import ConformerConfig, ConformerCTC
from awaaz.text import Tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger("awaaz.train")


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 5000,
    total_steps: int = 100000,
    peak_lr: float = 1e-3,
    min_lr: float = 1e-5,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Builds a Linear Warmup + Cosine Decay learning rate scheduler."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed_lr = min_lr + (peak_lr - min_lr) * cosine_decay
        return decayed_lr / peak_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def sample_chunk_params(
    chunk_prob: float = 0.6,
    chunk_sizes: Optional[List[int]] = None,
    left_chunks: Optional[List[int]] = None,
) -> Tuple[Optional[int], int]:
    """Samples dynamic-chunk attention parameters for a training batch.

    With probability chunk_prob, samples chunk_size and left_chunks.
    Otherwise returns (None, -1) for full-context attention.
    """
    if chunk_sizes is None:
        chunk_sizes = [8, 16, 32]
    if left_chunks is None:
        left_chunks = [2, 4, 8, -1]

    if random.random() < chunk_prob:
        return random.choice(chunk_sizes), random.choice(left_chunks)
    return None, -1


def save_checkpoint(
    run_dir: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any],
    step: int,
    epoch: int,
    config: Dict[str, Any],
    is_periodic: bool = False,
    max_keep: int = 12,
) -> str:
    """Atomically saves checkpoint to last.pt and periodic step_XXXXXXX.pt."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "epoch": epoch,
        "config": config,
    }

    # 1. Atomic write for last.pt
    temp_last = run_path / "last.pt.tmp"
    final_last = run_path / "last.pt"
    torch.save(checkpoint, temp_last)
    os.replace(str(temp_last), str(final_last))

    # 2. Periodic numbered checkpoint (model only to conserve disk)
    if is_periodic:
        periodic_file = run_path / f"step_{step:07d}.pt"
        torch.save({"model": model.state_dict(), "step": step}, periodic_file)

        # Maintain rolling window of newest max_keep checkpoints
        existing = sorted(run_path.glob("step_*.pt"))
        if len(existing) > max_keep:
            for old_ckpt in existing[:-max_keep]:
                try:
                    old_ckpt.unlink()
                except OSError:
                    pass

    return str(final_last)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
) -> Tuple[int, int]:
    """Restores model, optimizer, and scaler states from checkpoint.

    Returns:
        Tuple of (resumed_step, resumed_epoch).
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])

    step = ckpt.get("step", 0)
    epoch = ckpt.get("epoch", 0)
    logger.info("Resumed checkpoint from %s at step %d, epoch %d", checkpoint_path, step, epoch)
    return step, epoch


def greedy_decode(
    log_probs: torch.Tensor,
    out_lengths: torch.Tensor,
    tokenizer: Tokenizer,
) -> List[str]:
    """Greedy CTC argmax decoding with repeat collapsing and blank removal."""
    argmax_ids = log_probs.argmax(dim=-1).cpu()
    batch_size = argmax_ids.shape[0]
    hypotheses: List[str] = []

    for b in range(batch_size):
        length = int(out_lengths[b].item())
        tokens = argmax_ids[b, :length].tolist()

        collapsed: List[int] = []
        prev = -1
        for t in tokens:
            if t != prev:
                if t != tokenizer.blank_id:
                    collapsed.append(t)
                prev = t
        hypotheses.append(tokenizer.decode(collapsed))

    return hypotheses


def evaluate(
    model: nn.Module,
    frontend: LogMelSpectrogram,
    val_loader: DataLoader,
    tokenizer: Tokenizer,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Evaluates WER and CER on validation dataset, separated by language."""
    model.eval()
    en_refs, en_hyps = [], []
    pa_refs, pa_hyps = [], []

    with torch.no_grad():
        for batch in val_loader:
            waveforms = batch["waveforms"].to(device)
            waveform_lens = batch["waveform_lengths"]
            features = frontend(waveforms)
            feat_lens = (waveform_lens // 160 + 1).to(device)

            log_probs, out_lens = model(features, feat_lens, chunk_size=None)
            hyps = greedy_decode(log_probs, out_lens, tokenizer)

            for ref, hyp, lang in zip(batch["texts"], hyps, batch["languages"]):
                if lang == "en":
                    en_refs.append(ref)
                    en_hyps.append(hyp)
                else:
                    pa_refs.append(ref)
                    pa_hyps.append(hyp)

    results = {
        "en": compute_wer_cer(en_refs, en_hyps) if en_refs else {"wer": 0.0, "cer": 0.0},
        "pa": compute_wer_cer(pa_refs, pa_hyps) if pa_refs else {"wer": 0.0, "cer": 0.0},
    }
    model.train()
    return results


def train(
    config_path: str,
    run_dir: str,
    resume: bool = False,
    manifest_train: Optional[str] = None,
    manifest_dev: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    cmvn_path: Optional[str] = None,
    max_steps: Optional[int] = None,
    log_every: Optional[int] = None,
    eval_every: Optional[int] = None,
    save_every: Optional[int] = None,
) -> None:
    """Executes the complete training loop."""
    run_p = Path(run_dir)
    run_p.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Device and precision setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda")
        else:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    logger.info("Awaaz Training Initialized on %s (Mixed-Precision: %s)", device, use_amp)

    # Optional Weights & Biases telemetry
    has_wandb = False
    if "WANDB_API_KEY" in os.environ:
        try:
            import wandb
            wandb.init(project="awaaz", name=run_p.name, config=cfg)
            has_wandb = True
            logger.info("Weights & Biases telemetry initialized successfully.")
        except Exception as e:
            logger.warning("Could not initialize Weights & Biases: %s", e)

    # 2. Tokenizer and Frontend setup
    tok_p = tokenizer_path or cfg.get("tokenizer_path")
    if not tok_p or not Path(tok_p).is_file():
        raise FileNotFoundError(f"Tokenizer not found at {tok_p}. Please provide a valid tokenizer.")
    tokenizer = Tokenizer(tok_p)

    frontend = LogMelSpectrogram(sample_rate=16000, n_mels=cfg.get("model", {}).get("n_mels", 80)).to(device)
    cmvn_p = cmvn_path or cfg.get("cmvn_path")
    if cmvn_p and Path(cmvn_p).is_file():
        frontend.load_cmvn(cmvn_p)
    frontend.eval()

    spec_aug = SpecAugment().to(device)
    spec_aug.train()

    # 3. Model setup
    model_cfg = ConformerConfig.from_yaml(config_path)
    model_cfg.vocab_size = tokenizer.vocab_size
    model = ConformerCTC(model_cfg).to(device)

    # 4. Optimizer and Scheduler setup
    train_cfg = cfg.get("training", {})
    peak_lr = float(train_cfg.get("peak_lr", 1e-3))
    min_lr = float(train_cfg.get("min_lr", 1e-5))
    warmup_steps = int(train_cfg.get("warmup_steps", 5000))
    total_steps = max_steps or int(train_cfg.get("total_steps", 100000))
    weight_decay = float(train_cfg.get("weight_decay", 1e-3))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer, warmup_steps, total_steps, peak_lr, min_lr)
    ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    # 5. Checkpoint resume
    step = 0
    start_epoch = 0
    last_ckpt = run_p / "last.pt"
    if resume and last_ckpt.is_file():
        step, start_epoch = load_checkpoint(last_ckpt, model, optimizer, scaler)
        scheduler.step(step)

    # 6. Dataset and DataLoader setup
    m_train = manifest_train or cfg.get("manifest_train")
    if not m_train:
        raise ValueError("Missing training manifest. Specify via config or --manifest_train.")

    train_ds = AwaazDataset(
        m_train,
        tokenizer=tokenizer,
        exclude_sacred=train_cfg.get("exclude_sacred", True),
        training=True,
        speed_perturb=train_cfg.get("speed_perturb", True),
    )
    batch_sec = float(train_cfg.get("batch_seconds", 400.0))
    sampler = DurationBucketedBatchSampler(train_ds, max_batch_seconds=batch_sec, shuffle=True)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=ctc_collate_fn,
        num_workers=0,  # 0 for safe Windows dev execution
    )

    # Optional Dev DataLoader
    dev_loader = None
    m_dev = manifest_dev or cfg.get("manifest_dev")
    if m_dev and Path(m_dev).is_file():
        dev_ds = AwaazDataset(
            m_dev,
            tokenizer=tokenizer,
            exclude_sacred=True,
            training=False,
            speed_perturb=False,
        )
        if len(dev_ds) > 0:
            dev_sampler = DurationBucketedBatchSampler(dev_ds, max_batch_seconds=batch_sec, shuffle=False)
            dev_loader = DataLoader(
                dev_ds,
                batch_sampler=dev_sampler,
                collate_fn=ctc_collate_fn,
                num_workers=0,
            )

    # Config-derived planning printout
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hours_per_lang = train_ds.hours_per_language()
    save_freq = save_every or int(train_cfg.get("save_every", 500))
    eval_freq = eval_every or int(train_cfg.get("eval_every", 2000))
    keep_ckpts = int(train_cfg.get("keep_checkpoints", 12))
    log_freq = log_every or int(train_cfg.get("log_every", 10))

    plan_lines = [
        "=" * 65,
        "AWAAZ TRAINING PLAN",
        f"  Model Parameters: {param_count / 1e6:.2f}M ({param_count} trainable)",
    ]
    for lang, hrs in sorted(hours_per_lang.items()):
        plan_lines.append(f"  Dataset Hours ({lang}): {hrs:.2f} hrs")
    plan_lines.extend([
        f"  Batches per Epoch: {len(sampler)}",
        f"  Total Steps: {total_steps:,}",
        f"  Logging Frequency: every {log_freq} steps",
        f"  Evaluation Frequency: every {eval_freq} steps",
        f"  Checkpoint Frequency: every {save_freq} steps to {run_p / 'last.pt'}",
        "=" * 65,
    ])
    for line in plan_lines:
        logger.info(line)
        print(line, flush=True)

    # CSV log file
    csv_file = run_p / "train_log.csv"
    csv_exists = csv_file.is_file()
    csv_fp = open(csv_file, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_fp)
    if not csv_exists:
        csv_writer.writerow(["step", "epoch", "loss", "lr", "grad_norm", "steps_per_sec", "audio_sec_per_sec"])
        csv_fp.flush()

    start_msg = f"Starting training loop from step {step} to {total_steps}..."
    logger.info(start_msg)
    print(start_msg, flush=True)
    model.train()

    epoch = start_epoch
    t_prev = time.time()

    while step < total_steps:
        sampler.set_epoch(epoch)
        epoch += 1

        for batch in train_loader:
            if step >= total_steps:
                break

            waveforms = batch["waveforms"].to(device)
            waveform_lens = batch["waveform_lengths"]
            targets = batch["targets"].to(device)
            target_lens = batch["target_lengths"].to(device)

            batch_audio_sec = float(waveform_lens.sum().item()) / 16000.0

            # 1. Feature extraction & SpecAugment
            with torch.no_grad():
                features = frontend(waveforms)
                feat_lens = (waveform_lens // 160 + 1).to(device)

            features = spec_aug(features, feat_lens)

            # 2. Dynamic-chunk parameter sampling
            chunk_size, left_chunks = sample_chunk_params(
                chunk_prob=train_cfg.get("chunk_prob", 0.6),
                chunk_sizes=train_cfg.get("chunk_sizes", [8, 16, 32]),
                left_chunks=train_cfg.get("left_chunks", [2, 4, 8, -1]),
            )

            # 3. Forward pass & CTC loss
            optimizer.zero_grad()
            try:
                if use_amp:
                    autocast_ctx = (
                        torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if hasattr(torch, "amp")
                        else torch.cuda.amp.autocast(dtype=torch.bfloat16)
                    )
                    with autocast_ctx:
                        log_probs, out_lens = model(
                            features, feat_lens, chunk_size=chunk_size, left_chunks=left_chunks
                        )
                        # Upcast to float32 for CTCLoss numerical stability
                        loss = ctc_loss(
                            log_probs.transpose(0, 1).float(),
                            targets,
                            out_lens,
                            target_lens,
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                    if torch.isfinite(grad_norm):
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logger.warning("Step %d: Non-finite gradient norm (%.2f). Skipping step.", step, grad_norm)
                        scaler.update()
                else:
                    log_probs, out_lens = model(
                        features, feat_lens, chunk_size=chunk_size, left_chunks=left_chunks
                    )
                    loss = ctc_loss(log_probs.transpose(0, 1), targets, out_lens, target_lens)
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                    if torch.isfinite(grad_norm):
                        optimizer.step()
                    else:
                        logger.warning("Step %d: Non-finite gradient norm. Skipping step.", step)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error("CUDA OOM at step %d! Skipping batch and freeing cache.", step)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                raise e

            scheduler.step()
            step += 1

            # Telemetry calculation
            t_now = time.time()
            dt = max(1e-4, t_now - t_prev)
            steps_per_sec = 1.0 / dt
            audio_sec_per_sec = batch_audio_sec / dt
            t_prev = t_now

            lr_now = scheduler.get_last_lr()[0]
            loss_val = float(loss.item())
            norm_val = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)

            csv_writer.writerow([step, epoch, f"{loss_val:.4f}", f"{lr_now:.6e}", f"{norm_val:.3f}", f"{steps_per_sec:.2f}", f"{audio_sec_per_sec:.1f}"])
            csv_fp.flush()

            if step == 1 or step % log_freq == 0 or step == total_steps:
                progress_pct = (step / total_steps) * 100.0
                msg = (
                    f"[{progress_pct:5.1f}%] Step {step}/{total_steps} | Epoch {epoch} | "
                    f"Loss: {loss_val:.4f} | LR: {lr_now:.2e} | Grad: {norm_val:.2f} | {audio_sec_per_sec:.1f} aud-s/s"
                )
                logger.info(msg)
                print(msg, flush=True)
                if has_wandb:
                    try:
                        import wandb
                        wandb.log({
                            "train/loss": loss_val,
                            "train/lr": lr_now,
                            "train/grad_norm": norm_val,
                            "train/audio_sec_per_sec": audio_sec_per_sec,
                            "step": step,
                        })
                    except Exception:
                        pass

            # Checkpoint save & Evaluation
            if step % save_freq == 0 or step == total_steps:
                is_periodic = (step % eval_freq == 0) or (step == total_steps)
                save_checkpoint(
                    run_dir, model, optimizer, scaler, step, epoch, cfg,
                    is_periodic=is_periodic, max_keep=keep_ckpts
                )
                if is_periodic and dev_loader is not None:
                    eval_metrics = evaluate(model, frontend, dev_loader, tokenizer, device)
                    val_msg = (
                        f"Step {step} Validation: EN WER={eval_metrics['en']['wer'] * 100:.2f}% "
                        f"CER={eval_metrics['en']['cer'] * 100:.2f}% | PA "
                        f"WER={eval_metrics['pa']['wer'] * 100:.2f}% CER={eval_metrics['pa']['cer'] * 100:.2f}%"
                    )
                    logger.info(val_msg)
                    print(val_msg, flush=True)
                    if has_wandb:
                        try:
                            import wandb
                            wandb.log({
                                "val/en_wer": eval_metrics["en"]["wer"],
                                "val/en_cer": eval_metrics["en"]["cer"],
                                "val/pa_wer": eval_metrics["pa"]["wer"],
                                "val/pa_cer": eval_metrics["pa"]["cer"],
                                "step": step,
                            })
                        except Exception:
                            pass

    csv_fp.close()
    if has_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
    logger.info("Training finished cleanly at step %d.", step)


def main():
    parser = argparse.ArgumentParser(description="Train Awaaz Conformer-CTC Model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training config.")
    parser.add_argument("--run_dir", type=str, default="runs/exp1", help="Output directory for checkpoints and logs.")
    parser.add_argument("--resume", action="store_true", help="Auto-resume from last.pt if present.")
    parser.add_argument("--manifest_train", type=str, default=None, help="Training JSONL manifest override.")
    parser.add_argument("--manifest_dev", type=str, default=None, help="Validation JSONL manifest override.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer .model path override.")
    parser.add_argument("--cmvn", type=str, default=None, help="CMVN .npz file override.")
    parser.add_argument("--max_steps", type=int, default=None, help="Override total steps (useful for smoke tests).")
    parser.add_argument("--log_every", type=int, default=10, help="Print progress every N steps (default: 10).")
    parser.add_argument("--eval_every", type=int, default=None, help="Evaluation and checkpoint interval in steps.")
    parser.add_argument("--save_every", type=int, default=None, help="Save last.pt interval in steps.")

    args = parser.parse_args()

    train(
        config_path=args.config,
        run_dir=args.run_dir,
        resume=args.resume,
        manifest_train=args.manifest_train,
        manifest_dev=args.manifest_dev,
        tokenizer_path=args.tokenizer,
        cmvn_path=args.cmvn,
        max_steps=args.max_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
