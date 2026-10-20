"""DoTA (Detection of Traffic Anomaly) annotation importer.

Input: the per-clip JSON files of ``dataset/DoTA_annotations.zip`` from
https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly (format checked against all 4,677 files).

What the files state, and how it is converted:
- Frames were extracted with ffmpeg ``fps=10`` (``video2frames.py``); clip frame ``i`` is 10 fps frame
  ``video_start - 1 + i`` of the YouTube video. Clip time is ``frame_id / 10`` s.
- ``anomaly_start``/``anomaly_end`` are clip frame indices. Per-frame labels show ``anomaly_end`` is
  exclusive in most files and inclusive in some, so the end time is uncertain by one frame (0.1 s).
- ``accident_name`` is a DoTA anomaly category and ``ego_involve`` whether the camera vehicle takes part.
  Neither is one of our event types; ``review_for`` only says which types a reviewer should look for.
- ``objects`` occur only on anomaly frames: they are the anomaly participants (``obj_track_id``), not
  every road user, so they cannot score detection.

No events are produced. A reviewer turns a source label into an event.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from evaluation.annotations import (BoxAt, ClipAnnotation, Conditions, EventType, Light, Provenance,
                                    SourceLabel, Split, dump_annotations)
from evaluation.importers import registry_licence

DATASET_ID = "dota"
FPS = 10.0
_KEYS = {"video_name", "channel", "num_frames", "ignore", "ego_involve", "night", "anomaly_start",
         "anomaly_end", "video_start", "video_end", "accident_id", "accident_name", "labels"}
_OBJECT_KEYS = {"obj_track_id", "bbox", "category", "category ID", "trunc"}

# Where a reviewer should look, by DoTA category. Categories naming another road user can hide a
# collision; a vehicle ahead (moving, waiting, starting or stopping) can be approached; a vehicle moving
# laterally in the same direction can cut in. Out-of-control and unknown give no hint.
_REVIEW_EGO = {
    "moving_ahead_or_waiting": (EventType.COLLISION, EventType.APPROACH),
    "start_stop_or_stationary": (EventType.COLLISION, EventType.APPROACH),
    "lateral": (EventType.COLLISION, EventType.CUT_IN),
    "oncoming": (EventType.COLLISION,),
    "turning": (EventType.COLLISION,),
    "pedestrian": (EventType.COLLISION,),
    "obstacle": (EventType.COLLISION,),
}
_REVIEW_OTHER = {k: (EventType.COLLISION,) for k in _REVIEW_EGO}


def _require(cond: bool, where: str, msg: str) -> None:
    if not cond:
        raise ValueError(f"{where}: {msg}")


def convert(raw: dict, where: str, split: Split = Split.DEV) -> ClipAnnotation:
    """One DoTA annotation file (parsed JSON) to a clip annotation."""
    _require(set(raw) == _KEYS, where, f"unexpected keys {sorted(set(raw) ^ _KEYS)}")
    labels = raw["labels"]
    frame_ids = [lab["frame_id"] for lab in labels]
    _require(frame_ids == list(range(len(labels))), where, "frame_id must run 0..n-1")
    s, e = raw["anomaly_start"], raw["anomaly_end"]
    _require(0 <= s <= e <= len(labels), where, f"anomaly frames {s}..{e} outside 0..{len(labels)}")
    vid = raw["video_name"]

    boxes, participants = [], []
    for lab in labels:
        for o in lab["objects"]:
            _require(set(o) == _OBJECT_KEYS, where, f"frame {lab['frame_id']}: unexpected object keys")
            pid = str(o["obj_track_id"])
            if pid not in participants:
                participants.append(pid)
            boxes.append(BoxAt(t=lab["frame_id"] / FPS, xyxy=tuple(o["bbox"]), label=o["category"],
                               participant_id=pid))

    name = raw["accident_name"]
    review = (_REVIEW_EGO if raw["ego_involve"] else _REVIEW_OTHER).get(name, ())
    notes = []
    if raw["num_frames"] != len(labels):
        notes.append(f"num_frames {raw['num_frames']} differs from {len(labels)} labelled frames")
    if raw["ignore"]:
        notes.append("marked ignore by DoTA")
    source = SourceLabel(
        label_id=f"dota:{vid}", dataset_id=DATASET_ID, record=f"annotations/{vid}.json",
        label=name, fields={k: raw[k] for k in sorted(_KEYS - {"labels"})},
        t_start=s / FPS, t_end=e / FPS, participant_ids=tuple(participants), boxes=tuple(boxes),
        review_for=review, note="; ".join(notes) or None,
    )
    return ClipAnnotation(
        clip_id=f"dota-{vid}", source="dota_youtube_compilation", licence=registry_licence(DATASET_ID),
        split=split, duration_s=len(labels) / FPS, fps_nominal=FPS, variable_frame_rate=False,
        conditions=Conditions(light=Light.NIGHT if raw["night"] else None),
        source_labels=(source,),
        provenance=Provenance(annotator="DoTA authors", method="imported:dota", created=date.today().isoformat()),
    )


def import_dir(annotation_dir: Path, split: Split = Split.DEV, names: set[str] | None = None) -> list[ClipAnnotation]:
    clips = []
    for path in sorted(Path(annotation_dir).glob("*.json")):
        if names is not None and path.stem not in names:
            continue
        clips.append(convert(json.loads(path.read_text()), str(path), split))
    return clips


def main() -> None:
    p = argparse.ArgumentParser(description="Convert DoTA annotation files to Spectra clip annotations.")
    p.add_argument("annotation_dir", type=Path, help="the extracted annotations/ directory")
    p.add_argument("--out", type=Path, required=True, help="JSON Lines output, under evaluation/restricted/")
    p.add_argument("--split", type=Split, default=Split.DEV)
    p.add_argument("--names", help="comma-separated video_name values to import (default: all)")
    args = p.parse_args()
    clips = import_dir(args.annotation_dir, args.split, set(args.names.split(",")) if args.names else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump_annotations(clips, args.out)
    print(f"{len(clips)} clips written to {args.out}")


if __name__ == "__main__":
    main()
