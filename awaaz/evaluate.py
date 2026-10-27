"""Evaluation runner for Awaaz (Phase 7).

Evaluates checkpoints across:
1. Offline (full context)
2. Stream-0.64s (chunk 16, left 8)
3. Stream-0.32s (chunk 8, left 4)

Outputs:
- Markdown results table (`results.md`)
- Machine-readable JSON summary (`results.json`)
- Top-20 worst utterances error triage (`error_analysis.txt`)
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from awaaz.dataset import AwaazDataset, ctc_collate_fn
from awaaz.decode import ctc_greedy_decode
from awaaz.features import LogMelSpectrogram
from awaaz.metrics import compute_wer_cer, format_error_report
from awaaz.model import ConformerConfig, ConformerCTC
from awaaz.text import Tokenizer

logger = logging.getLogger("awaaz.evaluate")


def evaluate_manifest(
    manifest_path: str,
    checkpoint_path: str,
    tokenizer_path: str,
    cmvn_path: Optional[str] = None,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs full evaluation across offline and streaming chunk modes.

    Args:
        manifest_path: Path to evaluation JSONL manifest.
        checkpoint_path: Path to PyTorch model checkpoint (.pt).
        tokenizer_path: Path to SentencePiece tokenizer (.model).
        cmvn_path: Optional path to CMVN .npz file.
        config_path: Optional path to model config YAML. If omitted, attempts
            to read config embedded in checkpoint.
        output_dir: Optional directory to save results.md and results.json.
        device: 'cpu' or 'cuda' (auto-detected if None).

    Returns:
        Structured dictionary with metrics for all evaluation modes.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Initializing Evaluation on %s...", dev)

    # 1. Load Tokenizer
    tokenizer = Tokenizer(tokenizer_path)

    # 2. Load Checkpoint and Model
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_state = ckpt.get("model", ckpt)

    if config_path and Path(config_path).is_file():
        cfg = ConformerConfig.from_yaml(config_path)
    elif "config" in ckpt and isinstance(ckpt["config"], dict):
        raw_cfg = ckpt["config"]
        m_cfg = raw_cfg.get("model", raw_cfg)
        cfg = ConformerConfig(
            vocab_size=tokenizer.vocab_size,
            n_mels=m_cfg.get("n_mels", 80),
            d_model=m_cfg.get("d_model", 256),
            num_blocks=m_cfg.get("num_blocks", 12),
            num_heads=m_cfg.get("num_heads", 4),
            ffn_dim=m_cfg.get("ffn_dim", 1024),
            conv_kernel_size=m_cfg.get("conv_kernel_size", 15),
            dropout=0.0,
        )
    else:
        # Default small config
        cfg = ConformerConfig.from_preset("small")

    cfg.vocab_size = tokenizer.vocab_size
    cfg.dropout = 0.0
    model = ConformerCTC(cfg).to(dev)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    # 3. Frontend setup
    frontend = LogMelSpectrogram(sample_rate=16000, n_mels=cfg.n_mels).to(dev)
    if cmvn_path and Path(cmvn_path).is_file():
        frontend.load_cmvn(cmvn_path)
    frontend.eval()

    # 4. Dataset loader
    dataset = AwaazDataset(
        manifest_path,
        tokenizer=tokenizer,
        exclude_sacred=True,
        training=False,
        speed_perturb=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,  # Single-utterance evaluation for accurate latency & audio timing
        shuffle=False,
        collate_fn=ctc_collate_fn,
        num_workers=0,
    )

    modes = [
        {"name": "offline", "chunk_size": None, "left_chunks": -1, "label": "Offline (Full Context)"},
        {"name": "stream_0.64s", "chunk_size": 16, "left_chunks": 8, "label": "Stream 0.64s (Chunk 16, Left 8)"},
        {"name": "stream_0.32s", "chunk_size": 8, "left_chunks": 4, "label": "Stream 0.32s (Chunk 8, Left 4)"},
    ]

    all_results: Dict[str, Any] = {
        "manifest": manifest_path,
        "checkpoint": checkpoint_path,
        "total_clips": len(dataset),
        "modes": {},
    }

    worst_errors_by_mode: Dict[str, List[Dict[str, Any]]] = {}

    for mode in modes:
        mode_name = mode["name"]
        chunk_size = mode["chunk_size"]
        left_chunks = mode["left_chunks"]

        logger.info("Evaluating mode: %s...", mode["label"])

        refs_all: List[str] = []
        hyps_all: List[str] = []
        langs_all: List[str] = []
        audio_paths_all: List[str] = []

        total_audio_sec = 0.0
        total_wall_sec = 0.0

        with torch.no_grad():
            for batch in loader:
                waveforms = batch["waveforms"].to(dev)
                lens = batch["waveform_lengths"]
                ref_text = batch["texts"][0]
                lang = batch["languages"][0]
                audio_sec = float(lens[0].item()) / 16000.0

                total_audio_sec += audio_sec

                t0 = time.perf_counter()
                features = frontend(waveforms)
                feat_lens = (lens // 160 + 1).to(dev)

                log_probs, out_lens = model(
                    features,
                    feat_lens,
                    chunk_size=chunk_size,
                    left_chunks=left_chunks,
                )
                hyp_list = ctc_greedy_decode(log_probs, out_lens, tokenizer)
                t1 = time.perf_counter()

                total_wall_sec += (t1 - t0)

                hyp_text = hyp_list[0] if hyp_list else ""
                refs_all.append(ref_text)
                hyps_all.append(hyp_text)
                langs_all.append(lang)
                audio_paths_all.append(batch.get("audio_path", [""])[0] if "audio_path" in batch else "")

        # Compute Metrics
        rtf = total_wall_sec / max(1e-4, total_audio_sec)
        combined_metrics = compute_wer_cer(refs_all, hyps_all)

        # Split metrics by language
        en_refs = [r for r, lang in zip(refs_all, langs_all) if lang == "en"]
        en_hyps = [h for h, lang in zip(hyps_all, langs_all) if lang == "en"]
        en_metrics = compute_wer_cer(en_refs, en_hyps) if en_refs else {"wer": 0.0, "cer": 0.0}

        pa_refs = [r for r, lang in zip(refs_all, langs_all) if lang == "pa"]
        pa_hyps = [h for h, lang in zip(hyps_all, langs_all) if lang == "pa"]
        pa_metrics = compute_wer_cer(pa_refs, pa_hyps) if pa_refs else {"wer": 0.0, "cer": 0.0}

        all_results["modes"][mode_name] = {
            "label": mode["label"],
            "rtf": round(rtf, 4),
            "combined": combined_metrics,
            "en": en_metrics,
            "pa": pa_metrics,
            "total_audio_seconds": round(total_audio_sec, 2),
            "total_compute_seconds": round(total_wall_sec, 2),
        }

        # Store worst utterances for triage
        worst_errors = format_error_report(refs_all, hyps_all, audio_paths_all, langs_all, top_k=20)
        worst_errors_by_mode[mode_name] = worst_errors

    # 5. Output Report Generation
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        # JSON Summary
        with open(out_p / "results.json", "w", encoding="utf-8") as f_json:
            json.dump(all_results, f_json, indent=2, ensure_ascii=False)

        # Markdown Report Table
        md_lines = [
            "# Awaaz Model Evaluation Report",
            "",
            f"**Manifest**: `{manifest_path}`  ",
            f"**Checkpoint**: `{checkpoint_path}`  ",
            f"**Total Utterances**: {len(dataset)}  ",
            "",
            "### Metric Summary Table",
            "",
            "| Evaluation Condition | EN WER | EN CER | PA WER | PA CER (Primary) | Combined WER | Real-Time Factor (RTF) |",
            "|---|---|---|---|---|---|---|",
        ]

        for m_name, m_data in all_results["modes"].items():
            en_w = f"{m_data['en']['wer'] * 100:.2f}%"
            en_c = f"{m_data['en']['cer'] * 100:.2f}%"
            pa_w = f"{m_data['pa']['wer'] * 100:.2f}%"
            pa_c = f"{m_data['pa']['cer'] * 100:.2f}%"
            comb_w = f"{m_data['combined']['wer'] * 100:.2f}%"
            rtf_str = f"{m_data['rtf']:.3f}x"
            label = m_data["label"]
            md_lines.append(f"| **{label}** | {en_w} | {en_c} | {pa_w} | **{pa_c}** | {comb_w} | {rtf_str} |")

        md_lines.extend([
            "",
            "> **Note on Metrics**: Per §1.4, Character Error Rate (CER) is the primary benchmark metric for Punjabi (Gurmukhi) to prevent orthographic and nukta spelling variations from artificially inflating error rates.",
            "",
        ])

        with open(out_p / "results.md", "w", encoding="utf-8") as f_md:
            f_md.write("\n".join(md_lines) + "\n")

        # Top-20 Worst Utterances
        with open(out_p / "error_analysis.txt", "w", encoding="utf-8") as f_err:
            f_err.write("=== TOP-20 WORST UTTERANCES (OFFLINE MODE) ===\n\n")
            for item in worst_errors_by_mode.get("offline", []):
                f_err.write(f"Index: {item['index']} | Lang: {item['language']} | WER: {item['wer']*100:.1f}% | CER: {item['cer']*100:.1f}%\n")
                f_err.write(f"  REF: {item['reference']}\n")
                f_err.write(f"  HYP: {item['hypothesis']}\n\n")

        logger.info("Evaluation reports written to: %s", out_p)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Awaaz Model Checkpoint")
    parser.add_argument("--manifest", type=str, required=True, help="Path to evaluation JSONL manifest.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PyTorch checkpoint.")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer .model file.")
    parser.add_argument("--cmvn", type=str, default=None, help="Path to CMVN .npz file.")
    parser.add_argument("--config", type=str, default=None, help="Path to model config YAML.")
    parser.add_argument("--output_dir", type=str, default="eval_output", help="Directory to save report files.")
    parser.add_argument("--device", type=str, default=None, help="'cpu' or 'cuda'.")

    args = parser.parse_args()

    evaluate_manifest(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        cmvn_path=args.cmvn,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
