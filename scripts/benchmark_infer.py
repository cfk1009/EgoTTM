#!/usr/bin/env python3
"""Compare refactored inference with and without the projected-text Memory Bank."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egottm.data import load_dialogues
from egottm.images import load_face_window
from egottm.memory_bank import MemoryBank
from egottm.metrics import evaluate_binary_scores
from egottm.model import load_ttm_model
from egottm.visual import load_lam_model


def validate_prediction_alignment(labels, scores):
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")


def make_summary_row(
    arm,
    elapsed_seconds,
    sample_count,
    map_value,
    accuracy,
    total_params,
    trainable_params,
    gflops,
    context_utterances=0,
):
    numeric = {
        "elapsed_seconds": elapsed_seconds,
        "sample_count": sample_count,
        "map": map_value,
        "accuracy": accuracy,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "gflops": gflops,
        "context_utterances": context_utterances,
    }
    if any(value < 0 for value in numeric.values()):
        raise ValueError("benchmark metrics cannot be negative")
    return {"arm": arm, **numeric}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--memory-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--face-crops", required=True)
    parser.add_argument("--lam-checkpoint", required=True)
    parser.add_argument("--legacy-root", default="/home/cai/cfk/EgoT2/HHI")
    parser.add_argument("--context-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _count_parameters(module):
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return total, trainable


def _tokenized_dialogue(dialogue, tokenizer, device):
    encoded = tokenizer(
        [utterance.text for utterance in dialogue.utterances],
        truncation=True,
        max_length=512,
        # Match build_memory_bank.py exactly. Fixed-length padding produces
        # numerically different RoBERTa outputs and would confound the arm.
        padding=True,
        return_tensors="pt",
    )
    return (
        encoded["input_ids"].unsqueeze(0).to(device),
        encoded["attention_mask"].unsqueeze(0).to(device),
        torch.tensor([[item.speaker_id for item in dialogue.utterances]], dtype=torch.long, device=device),
        torch.tensor([[item.round_id for item in dialogue.utterances]], dtype=torch.long, device=device),
    )


def _context_bounds(round_id, context_size):
    end = int(round_id) + 1
    start = 0 if context_size is None else max(0, end - context_size)
    return start, end


def _visual_features(target, face_crops, lam_model, device):
    images, lam_mask = load_face_window(
        face_crops,
        target.uid,
        target.segment_id,
        target.start_frame,
        target.end_frame,
        device,
    )
    lam_features, _ = lam_model(images, middle=True)
    return lam_features, lam_mask


def _evaluate(labels, scores):
    validate_prediction_alignment(labels, scores)
    metrics = evaluate_binary_scores(
        labels,
        scores,
        threshold=0.5,
    )
    return metrics.map, metrics.accuracy


def _run_memory_bank(dialogues, bank, model, lam_model, face_crops, device, context_size):
    rows, labels, scores = [], [], []
    _sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for dialogue in dialogues:
            for target in dialogue.utterances:
                if target.speaker_id == 0:
                    continue
                context = bank.context(dialogue.uid, target.round_id, context_size)
                embeddings = context.embeddings.unsqueeze(0).to(device)
                lengths = torch.tensor([embeddings.size(1)], dtype=torch.long, device=device)
                lam_features, lam_mask = _visual_features(target, face_crops, lam_model, device)
                logits = model.forward_from_memory(embeddings, lengths, lam_features, lam_mask)
                score = torch.softmax(logits.float(), dim=-1)[0, 1].item()
                rows.append({"uid": target.uid, "segment_id": target.segment_id, "gt_id": target.gt_id, "round_id": target.round_id, "start_frame": target.start_frame, "end_frame": target.end_frame, "score": score, "label": target.label})
                labels.append(target.label)
                scores.append(score)
    _sync(device)
    return rows, labels, scores, time.perf_counter() - started


def _run_without_memory_bank(dialogues, tokenizer, model, lam_model, face_crops, device, context_size):
    rows, labels, scores = [], [], []
    _sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for dialogue in dialogues:
            input_ids, attention, speakers, positions = _tokenized_dialogue(dialogue, tokenizer, device)
            for target in dialogue.utterances:
                if target.speaker_id == 0:
                    continue
                start, end = _context_bounds(target.round_id, context_size)
                embeddings = model.encoder(
                    input_ids[:, start:end],
                    attention[:, start:end],
                    speakers[:, start:end],
                    positions[:, start:end],
                )
                lengths = torch.tensor([end - start], dtype=torch.long, device=device)
                lam_features, lam_mask = _visual_features(target, face_crops, lam_model, device)
                logits = model.forward_from_memory(embeddings, lengths, lam_features, lam_mask)
                score = torch.softmax(logits.float(), dim=-1)[0, 1].item()
                rows.append({"uid": target.uid, "segment_id": target.segment_id, "gt_id": target.gt_id, "round_id": target.round_id, "start_frame": target.start_frame, "end_frame": target.end_frame, "score": score, "label": target.label})
                labels.append(target.label)
                scores.append(score)
    _sync(device)
    return rows, labels, scores, time.perf_counter() - started


class _MemoryForward(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, embeddings, lengths, lam_features, lam_mask):
        return self.model.forward_from_memory(embeddings, lengths, lam_features, lam_mask)


def _profile_gflops(model, lam_model, sample, device):
    try:
        from thop import profile

        embeddings, lengths, input_ids, attention, speakers, positions, lam_images, lam_features, lam_mask = sample
        lam_macs, _ = profile(lam_model, inputs=(lam_images, True), verbose=False)
        memory_macs, _ = profile(
            _MemoryForward(model),
            inputs=(embeddings, lengths, lam_features, lam_mask),
            verbose=False,
        )
        no_bank_macs, _ = profile(
            model,
            inputs=(input_ids, attention, speakers, positions, lam_features, lengths, lam_mask),
            verbose=False,
        )
        return {
            "lam_gflops": float(lam_macs * 2 / 1e9),
            "memory_bank_gflops": float((lam_macs + memory_macs) * 2 / 1e9),
            "no_memory_bank_gflops": float((lam_macs + no_bank_macs) * 2 / 1e9),
            "profile_context_utterances": int(embeddings.size(1)),
        }
    except Exception as error:
        return {
            "lam_gflops": None,
            "memory_bank_gflops": None,
            "no_memory_bank_gflops": None,
        "profile_context_utterances": int(sample[0].size(1)),
            "profile_error": str(error),
        }


def _write_predictions(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["uid", "segment_id", "gt_id", "round_id", "start_frame", "end_frame", "score", "label"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_root = Path(args.legacy_root)

    from transformers import RobertaTokenizer

    load_started = time.perf_counter()
    dialogues = load_dialogues(args.csv)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer)
    model = load_ttm_model(args.tokenizer, args.checkpoint, device, encoder_checkpoint=args.encoder_checkpoint)
    bank = MemoryBank.load(args.memory_bank)
    lam_model = load_lam_model(args.legacy_root, args.lam_checkpoint, device)
    load_seconds = time.perf_counter() - load_started

    ttm_total, ttm_trainable = _count_parameters(model)
    lam_total, lam_trainable = _count_parameters(lam_model)
    total_params = ttm_total + lam_total
    trainable_params = ttm_trainable + lam_trainable

    representative = max(
        (target for dialogue in dialogues for target in dialogue.utterances if target.speaker_id != 0),
        key=lambda target: target.round_id,
    )
    representative_dialogue = next(dialogue for dialogue in dialogues if dialogue.uid == representative.uid)
    input_ids, attention, speakers, positions = _tokenized_dialogue(representative_dialogue, tokenizer, device)
    start, end = _context_bounds(representative.round_id, args.context_size)
    input_ids = input_ids[:, start:end]
    attention = attention[:, start:end]
    speakers = speakers[:, start:end]
    positions = positions[:, start:end]
    with torch.inference_mode():
        embeddings = model.encoder(input_ids, attention, speakers, positions)
        lam_images, lam_mask = load_face_window(
            args.face_crops,
            representative.uid,
            representative.segment_id,
            representative.start_frame,
            representative.end_frame,
            device,
        )
        lam_features, _ = lam_model(lam_images, middle=True)
    lengths = torch.tensor([embeddings.size(1)], dtype=torch.long, device=device)
    complexity = _profile_gflops(
        model,
        lam_model,
        (embeddings, lengths, input_ids, attention, speakers, positions, lam_images, lam_features, lam_mask),
        device,
    )
    # THOP may toggle wrapped modules while profiling. Restore inference mode
    # before either arm so Dropout/BatchNorm cannot affect the benchmark.
    model.eval()
    lam_model.eval()

    memory_rows, memory_labels, memory_scores, memory_seconds = _run_memory_bank(
        dialogues, bank, model, lam_model, args.face_crops, device, args.context_size
    )
    no_bank_rows, no_bank_labels, no_bank_scores, no_bank_seconds = _run_without_memory_bank(
        dialogues, tokenizer, model, lam_model, args.face_crops, device, args.context_size
    )
    memory_map, memory_accuracy = _evaluate(
        memory_labels,
        memory_scores,
    )
    no_bank_map, no_bank_accuracy = _evaluate(
        no_bank_labels,
        no_bank_scores,
    )

    _write_predictions(output_dir / "pred_mb.csv", memory_rows)
    _write_predictions(output_dir / "pred_nomb.csv", no_bank_rows)
    rows = [
        make_summary_row("memory_bank", memory_seconds, len(memory_scores), memory_map, memory_accuracy, total_params, trainable_params, complexity["memory_bank_gflops"] or 0.0, complexity["profile_context_utterances"]),
        make_summary_row("no_memory_bank", no_bank_seconds, len(no_bank_scores), no_bank_map, no_bank_accuracy, total_params, trainable_params, complexity["no_memory_bank_gflops"] or 0.0, complexity["profile_context_utterances"]),
    ]
    with (output_dir / "benchmark_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "load_seconds": load_seconds,
        "csv": str(args.csv),
        "memory_bank": str(args.memory_bank),
        "checkpoint": str(args.checkpoint),
        "encoder_checkpoint": str(args.encoder_checkpoint),
        "lam_checkpoint": str(args.lam_checkpoint),
        "context_size": args.context_size,
        "parameter_breakdown": {"ttm_total": ttm_total, "lam_total": lam_total, "ttm_trainable": ttm_trainable, "lam_trainable": lam_trainable},
        "complexity": complexity,
        "rows": rows,
    }
    (output_dir / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
