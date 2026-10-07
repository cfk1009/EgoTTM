"""Persistent conversational embeddings used by the inference demo."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import torch


@dataclass
class ContextBatch:
    """One chronological context window for a target utterance."""

    embeddings: torch.Tensor
    speaker_ids: List[int]
    round_ids: List[int]


class MemoryBank:
    """Stores projected speaker-aware utterance embeddings by video clip.

    The bank contains the output of the text encoder's projection layer.  The
    context Transformer is intentionally not cached because it depends on the
    target utterance and the selected context window.
    """

    FORMAT_VERSION = 1

    def __init__(self, records: Mapping[str, Iterable[Mapping]], metadata: Optional[dict] = None):
        self.metadata = dict(metadata or {})
        self.records: Dict[str, List[dict]] = {}
        for uid, items in records.items():
            normalized = []
            for item in items:
                normalized.append(
                    {
                        "round_id": int(item["round_id"]),
                        "speaker_id": int(item["speaker_id"]),
                        "embedding": torch.as_tensor(item["embedding"]).detach().cpu(),
                    }
                )
            self.records[str(uid)] = sorted(normalized, key=lambda x: x["round_id"])

    def context(self, uid: str, end_round: int, context_size: Optional[int] = None) -> ContextBatch:
        if context_size is not None and context_size < 1:
            raise ValueError("context_size must be at least 1")
        if str(uid) not in self.records:
            raise KeyError("No memory-bank entries for uid={!r}".format(uid))
        candidates = [item for item in self.records[str(uid)] if item["round_id"] <= int(end_round)]
        if not candidates:
            raise KeyError("No memory-bank entry at or before round {} for uid={!r}".format(end_round, uid))
        selected = candidates if context_size is None else candidates[-context_size:]
        return ContextBatch(
            embeddings=torch.stack([item["embedding"] for item in selected]),
            speaker_ids=[item["speaker_id"] for item in selected],
            round_ids=[item["round_id"] for item in selected],
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": self.FORMAT_VERSION,
                "metadata": self.metadata,
                "records": self.records,
            },
            str(path),
        )

    @classmethod
    def load(cls, path, map_location="cpu"):
        payload = torch.load(str(path), map_location=map_location)
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("Unsupported memory-bank format: {}".format(payload.get("format_version")))
        return cls(payload["records"], metadata=payload.get("metadata"))
