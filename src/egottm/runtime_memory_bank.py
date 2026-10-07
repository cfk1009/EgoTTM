"""Per-run, lazy Memory Bank for inference-time text embedding reuse."""

import torch

from .data import Dialogue
from .memory_bank import MemoryBank


class RuntimeMemoryBank(MemoryBank):
    """Compute each dialogue's projected text embeddings once per process.

    Unlike :class:`MemoryBank`, this cache is not loaded from or written to a
    ``.pt`` file.  A dialogue is tokenized and encoded on its first request,
    then its projected embeddings remain in CPU memory for later targets in
    the same inference run.
    """

    def __init__(self):
        super().__init__({}, metadata={"mode": "runtime"})

    @property
    def cached_dialogues(self) -> int:
        return len(self.records)

    def ensure_dialogue(
        self,
        dialogue: Dialogue,
        encoder,
        tokenizer,
        device: torch.device,
        max_length: int = 512,
    ) -> bool:
        """Populate one dialogue if absent; return whether computation occurred."""

        uid = str(dialogue.uid)
        if uid in self.records:
            return False
        encoded = tokenizer(
            [utterance.text for utterance in dialogue.utterances],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].unsqueeze(0).to(device)
        attention = encoded["attention_mask"].unsqueeze(0).to(device)
        speaker_ids = torch.tensor(
            [[utterance.speaker_id for utterance in dialogue.utterances]],
            dtype=torch.long,
            device=device,
        )
        round_ids = torch.tensor(
            [[utterance.round_id for utterance in dialogue.utterances]],
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            embeddings = encoder(input_ids, attention, speaker_ids, round_ids)[0].detach().cpu()
        self.records[uid] = [
            {
                "round_id": utterance.round_id,
                "speaker_id": utterance.speaker_id,
                "embedding": embeddings[index],
            }
            for index, utterance in enumerate(dialogue.utterances)
        ]
        return True
