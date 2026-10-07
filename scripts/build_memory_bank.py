#!/usr/bin/env python3
"""Precompute projected speaker-aware text embeddings from processed CSV data."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egottm.data import load_dialogues
from egottm.memory_bank import MemoryBank
from egottm.model import load_ttm_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Processed CSV with transcription and person_id columns")
    parser.add_argument("--output", required=True, help="Output .pt Memory Bank")
    parser.add_argument("--tokenizer", required=True, help="Local RoBERTa directory")
    parser.add_argument("--checkpoint", required=True, help="TTM checkpoint containing encoder weights")
    parser.add_argument("--encoder-checkpoint", default=None, help="Optional separately fine-tuned RoBERTa checkpoint")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-rounds", type=int, default=500)
    parser.add_argument(
        "--collapse-non-camera",
        action="store_true",
        help="Collapse all non-camera person IDs to speaker embedding 1 for cluster ablations",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from transformers import RobertaTokenizer

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dialogues = load_dialogues(args.csv, collapse_non_camera=args.collapse_non_camera)
    model = load_ttm_model(args.tokenizer, args.checkpoint, device, encoder_checkpoint=args.encoder_checkpoint)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer)
    records = {}
    with torch.inference_mode():
        for dialogue in dialogues:
            texts = [utterance.text for utterance in dialogue.utterances]
            encoded = tokenizer(
                texts,
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].unsqueeze(0).to(device)
            attention = encoded["attention_mask"].unsqueeze(0).to(device)
            speaker_ids = torch.tensor([[u.speaker_id for u in dialogue.utterances]], device=device)
            round_ids = torch.tensor([[u.round_id for u in dialogue.utterances]], device=device)
            embeddings = model.encoder(input_ids, attention, speaker_ids, round_ids)[0].cpu()
            records[dialogue.uid] = [
                {
                    "round_id": utterance.round_id,
                    "speaker_id": utterance.speaker_id,
                    "embedding": embeddings[index],
                }
                for index, utterance in enumerate(dialogue.utterances)
            ]
            print("[memory-bank] {}: {} utterances".format(dialogue.uid, len(texts)))
    MemoryBank(
        records,
        metadata={
            "embedding_dim": model.encoder.hidden_dim,
            "csv": str(args.csv),
            "collapse_non_camera": args.collapse_non_camera,
        },
    ).save(args.output)
    print("[memory-bank] saved to {}".format(args.output))


if __name__ == "__main__":
    main()
