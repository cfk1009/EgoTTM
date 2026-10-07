"""Readers for already-prepared EgoTTM transcription/diarization files."""

from dataclasses import dataclass
from typing import List, Optional


UNKNOWN_SPEAKER_ID = 29


def normalize_speaker_id(person_id, max_speakers: int = 30, collapse_non_camera: bool = False) -> int:
    """Map prepared diarization IDs to the speaker-embedding vocabulary.

    The diarization result is already present in ``person_id``.  This function
    deliberately does not run a diarizer.  The full checkpoint was trained
    with the original non-camera ``person_id`` values, while some cluster
    ablations intentionally collapse them to one identity.  The latter is
    opt-in so the default matches the full checkpoint.
    """

    person_id = int(person_id)
    if person_id == -1:
        return UNKNOWN_SPEAKER_ID
    if person_id == 0:
        return 0
    if collapse_non_camera:
        return 1
    if 0 < person_id < max_speakers - 1:
        return person_id
    return UNKNOWN_SPEAKER_ID


@dataclass(frozen=True)
class Utterance:
    uid: str
    text: str
    speaker_id: int
    round_id: int
    gt_id: int
    start_frame: int
    end_frame: int
    label: Optional[int]
    segment_id: str = ""


@dataclass(frozen=True)
class Dialogue:
    uid: str
    utterances: List[Utterance]


def load_dialogues(csv_path: str, max_speakers: int = 30, collapse_non_camera: bool = False) -> List[Dialogue]:
    """Load the processed CSV containing transcription and diarization fields."""

    import pandas as pd

    frame = pd.read_csv(csv_path)
    required = {"uid", "start_frame", "end_frame", "person_id", "gt_id", "transcription"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("CSV is missing required columns: {}".format(", ".join(missing)))

    dialogues = []
    for uid, group in frame.groupby("uid", sort=False):
        utterances = []
        for round_id, (_, row) in enumerate(group.iterrows()):
            text = str(row["transcription"])
            if text == "nan":
                text = ""
            label = None if "label" not in row or pd.isna(row["label"]) else int(row["label"])
            raw_segment_id = row.get("segment_id", round_id)
            segment_id = str(round_id if pd.isna(raw_segment_id) else raw_segment_id).strip()
            if not segment_id or segment_id.lower() == "nan":
                segment_id = str(round_id)
            utterances.append(
                Utterance(
                    uid=str(uid),
                    text=text,
                    speaker_id=normalize_speaker_id(row["person_id"], max_speakers, collapse_non_camera),
                    round_id=round_id,
                    gt_id=int(row["gt_id"]),
                    start_frame=int(row["start_frame"]),
                    end_frame=int(row["end_frame"]),
                    label=label,
                    segment_id=segment_id,
                )
            )
        dialogues.append(Dialogue(uid=str(uid), utterances=utterances))
    return dialogues
