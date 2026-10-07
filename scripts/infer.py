#!/usr/bin/env python3
"""Run TTM inference from processed transcription/diarization and face-crop files."""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egottm.data import load_dialogues
from egottm.images import load_face_window
from egottm.memory_bank import MemoryBank
from egottm.metrics import evaluate_prediction_rows, format_evaluation
from egottm.model import load_ttm_model
from egottm.runtime_memory_bank import RuntimeMemoryBank
from egottm.visual import load_lam_model, zero_visual_features


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="final/transcripts/val.csv")
    parser.add_argument(
        "--memory-bank",
        default="final/weights/memory.pt",
        help="Precomputed .pt Memory Bank",
    )
    parser.add_argument(
        "--runtime-memory-bank",
        action="store_true",
        help="Build a temporary Memory Bank lazily during this inference run",
    )
    parser.add_argument("--output", default="artifacts/predictions.csv")
    parser.add_argument("--tokenizer", default="external/roberta.large")
    parser.add_argument("--checkpoint", default="final/weights/ttm.pth")
    parser.add_argument(
        "--encoder-checkpoint",
        default="final/weights/roberta.pth",
        help="Optional separately fine-tuned RoBERTa checkpoint",
    )
    parser.add_argument("--face-crops", default="external/face_crops")
    parser.add_argument("--lam-checkpoint", default="final/weights/lam.pth")
    parser.add_argument("--legacy-root", default="external")
    parser.add_argument(
        "--context-size",
        type=int,
        default=None,
        help="Maximum history length; omit to use the complete dialogue history",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-visual", action="store_true", help="Use zero visual features for a text-only smoke demo")
    parser.add_argument(
        "--collapse-non-camera",
        action="store_true",
        help="Collapse all non-camera person IDs to speaker embedding 1 for cluster ablations",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold used for accuracy",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runtime_memory_bank and args.memory_bank != "final/weights/memory.pt":
        raise SystemExit("Choose either --memory-bank or --runtime-memory-bank, not both.")
    if not args.runtime_memory_bank and not args.memory_bank:
        raise SystemExit("Provide --memory-bank or pass --runtime-memory-bank.")
    if not args.no_visual and (not args.face_crops or not args.lam_checkpoint):
        raise SystemExit("Provide --face-crops and --lam-checkpoint, or pass --no-visual for a text-only demo.")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = load_ttm_model(args.tokenizer, args.checkpoint, device, encoder_checkpoint=args.encoder_checkpoint)
    tokenizer = None
    if args.runtime_memory_bank:
        from transformers import RobertaTokenizer

        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer)
        bank = RuntimeMemoryBank()
    else:
        bank = MemoryBank.load(args.memory_bank)
    lam_model = None if args.no_visual else load_lam_model(args.legacy_root, args.lam_checkpoint, device)
    rows = []
    runtime_cache_started = time.perf_counter()
    with torch.inference_mode():
        for dialogue in load_dialogues(args.csv, collapse_non_camera=args.collapse_non_camera):
            if args.runtime_memory_bank:
                bank.ensure_dialogue(dialogue, model.encoder, tokenizer, device)
            for target in dialogue.utterances:
                if target.speaker_id == 0:
                    continue
                context = bank.context(dialogue.uid, target.round_id, args.context_size)
                embeddings = context.embeddings.unsqueeze(0).to(device)
                lengths = torch.tensor([embeddings.size(1)], dtype=torch.long, device=device)
                if args.no_visual:
                    lam_features, lam_mask = zero_visual_features(1, 11, device)
                else:
                    images, lam_mask = load_face_window(
                        args.face_crops,
                        target.uid,
                        target.segment_id,
                        target.start_frame,
                        target.end_frame,
                        device,
                    )
                    lam_features, _ = lam_model(images, middle=True)
                logits = model.forward_from_memory(embeddings, lengths, lam_features, lam_mask)
                score = torch.softmax(logits.float(), dim=-1)[0, 1].item()
                rows.append(
                    {
                        "uid": target.uid,
                        "segment_id": target.segment_id,
                        "gt_id": target.gt_id,
                        "round_id": target.round_id,
                        "start_frame": target.start_frame,
                        "end_frame": target.end_frame,
                        "speaker_id": target.speaker_id,
                        "score": score,
                        "label": "" if target.label is None else target.label,
                    }
                )
    with open(args.output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else ["uid", "score"])
        writer.writeheader()
        writer.writerows(rows)
    if args.runtime_memory_bank:
        print(
            "[infer] runtime Memory Bank cached {} dialogues in {:.3f}s".format(
                bank.cached_dialogues, time.perf_counter() - runtime_cache_started
            )
        )
    print("[infer] wrote {} predictions to {}".format(len(rows), args.output))
    metrics = evaluate_prediction_rows(rows, threshold=args.threshold)

    if metrics is None:
        print("[infer] metrics skipped: ground-truth labels are unavailable.")
    else:
        print(format_evaluation(metrics, threshold=args.threshold))


if __name__ == "__main__":
    main()
