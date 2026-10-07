# Local assets

This directory contains short, stable names for the final inference inputs and
weights used by the demo. On the server they are symbolic links to the original
EgoT2 data/checkpoint locations. In the downloaded local bundle, materialized
files are under `final/` and `external/`; see `LOCAL_RUN.md`.

The validation CSV is the prepared transcription-plus-diarization file. It
contains `transcription`, `person_id`, `segment_id`, `gt_id`, frame boundaries,
and labels. `gt_id` is retained as a reference; inference looks up face crops by
`uid` and `segment_id`. The inference demo does not run Whisper or diarization.

The prepared face-crop subset uses this layout:

```text
external/face_crops/{uid}/{segment_id}/{frame}_*.jpg
```

Only frames sampled by the validation inference window are included.

Generated Memory Banks, prediction CSVs, and benchmark summaries belong in the
ignored `artifacts/` directory.
