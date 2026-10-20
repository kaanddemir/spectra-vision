"""PREVENTION (UAH) lane-change annotation importer.

Input, per RECORD/DRIVE (from ``processed_data.zip``):
- ``detection_camera1/lane_changes.txt``: seven integers per line, ``id_l id_d type f0 fe ff blk``, as the
  official toolkit parses it (github.com/RbnIzqGnl/PREVENTION-Toolkit, ``read_lane_change_events``);
- ``logs/log_camera1.txt``: ``cloud frame t0 t1`` per camera frame (toolkit ``read_camera_log``).

Semantics, checked against the 852 lines of all 11 released drives (2026-10-17):
- ``type``: the toolkit names 1 cut-in, 2 cut-out, 3 left lane change, 4 right lane change, 5 hazardous,
  6 pedestrian. The released files contain only 3 (381), 4 (468), 5 (2) and 6 (1); **no drive has a
  cut-in label**. A lane change by a surrounding vehicle does not say whether it ends in the camera's lane.
- ``f0``, ``fe``, ``ff``: start, the frame where the vehicle's rear middle is between the lanes, and end.
  ``-1`` marks an unknown frame; many lines have ``f0 = -1`` with ``fe = ff``, so only the end is known.
- ``id_l`` and ``id_d``: either may be ``-1``. On RECORD4/DRIVE3, where ``labels.txt`` is empty, ``id_l`` is
  always -1, which suggests ``id_l`` refers to ``labels.txt`` and ``id_d`` to ``detections_tracked.txt``
  (unconfirmed). Participants are therefore namespaced ``l:<id>`` and ``d:<id>``.
- ``blk``: 0 or 1; presumably the blinker, undocumented. Kept verbatim.
- Camera log: frame numbers run 0..n-1 and ``t0`` advances by about 91,000 per frame. Over RECORD4/DRIVE3
  the span is 430.7e6 for a drive the homepage lists as 7 min, so the unit is taken as microseconds
  (about 10.98 fps). Times are ``(t0 - t0 of frame 0) / 1e6`` s.
- Only the camera vehicle's surroundings are annotated; its own lane changes are not in the file.

Every line is kept as a source label. A line with inconsistent frames is kept without times rather than
rejected. Only type 1 would become a cut-in event, and only as source-derived.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

from evaluation.annotations import (ClipAnnotation, Event, EventType, Provenance, SourceLabel, Split,
                                    dump_annotations)
from evaluation.importers import registry_licence

DATASET_ID = "prevention"
TYPES = {1: "cut_in", 2: "cut_out", 3: "left_lane_change", 4: "right_lane_change", 5: "hazardous",
         6: "pedestrian"}
# A surrounding vehicle's lane change may or may not end in the camera's lane; the file does not say.
_REVIEW = {1: (EventType.CUT_IN,), 3: (EventType.CUT_IN,), 4: (EventType.CUT_IN,)}
_FIELDS = ("id_l", "id_d", "type", "f0", "fe", "ff", "blk")
LOG_TIME_UNIT_S = 1e-6


def _ints(line: str, n: int, where: str, names: str) -> list[int]:
    parts = line.split()
    try:
        values = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"{where}: expected integers, got {line!r}") from None
    if len(values) != n:
        raise ValueError(f"{where}: expected {n} fields ({names}), got {len(values)}")
    return values


def parse(text: str, where: str) -> list[dict]:
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip():
            rows.append(dict(zip(_FIELDS, _ints(line, 7, f"{where}:{n}", " ".join(_FIELDS))), line=n))
    return rows


def parse_camera_log(text: str, where: str = "log_camera1.txt") -> list[float]:
    """Frame times in seconds from frame 0, indexed by frame number."""
    t0s = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        _cloud, frame, t0, _t1 = _ints(line, 4, f"{where}:{n}", "cloud frame t0 t1")
        if frame != len(t0s):
            raise ValueError(f"{where}:{n}: frame {frame} out of sequence, expected {len(t0s)}")
        if t0s and t0 <= t0s[-1]:
            raise ValueError(f"{where}:{n}: timestamps must increase")
        t0s.append(t0)
    if len(t0s) < 2:
        raise ValueError(f"{where}: needs at least two frames")
    return [(t - t0s[0]) * LOG_TIME_UNIT_S for t in t0s]


def _time(frame: int, times: Sequence[float]) -> Optional[float]:
    return times[frame] if 0 <= frame < len(times) else None


def convert(text: str, drive: str, frame_times: Sequence[float], split: Split = Split.DEV,
            record: str = "detection_camera1/lane_changes.txt") -> ClipAnnotation:
    """``drive`` names the recording, e.g. ``RECORD4/DRIVE3``; ``frame_times`` comes from the camera log."""
    where = f"{drive}/{record}"
    step = (frame_times[-1] - frame_times[0]) / (len(frame_times) - 1)
    labels, events = [], []
    for row in parse(text, where):
        code, f0, fe, ff = row["type"], row["f0"], row["fe"], row["ff"]
        label_id = f"prevention:{drive}:{row['line']}"
        notes = [] if code in TYPES else [f"type {code} is not defined by the toolkit"]
        known = [f for f in (f0, fe, ff) if f != -1]
        consistent = known == sorted(known) and all(_time(f, frame_times) is not None for f in known)
        if not consistent:
            notes.append("frames out of order or beyond the camera log; times not converted")
        elif f0 == -1:
            notes.append("start frame unknown (f0 = -1)")
        t_start = _time(f0, frame_times) if consistent and f0 != -1 else None
        t_end = _time(ff, frame_times) if consistent and ff != -1 else None
        participants = tuple(f"{p}:{row[k]}" for p, k in (("l", "id_l"), ("d", "id_d")) if row[k] != -1)
        labels.append(SourceLabel(
            label_id=label_id, dataset_id=DATASET_ID, record=f"{where}:{row['line']}",
            label=TYPES.get(code, f"type_{code}"), fields={k: row[k] for k in _FIELDS},
            t_start=t_start, t_end=t_end, participant_ids=participants,
            review_for=_REVIEW.get(code, ()) if consistent and known else (), note="; ".join(notes) or None,
        ))
        if code == 1 and consistent and known:
            events.append(Event(
                event_id=f"src-{row['line']}", type=EventType.CUT_IN, t_start=t_start,
                t_peak=_time(fe, frame_times) if fe != -1 else None, t_end=t_end,
                participant_ids=participants, source_refs=(label_id,),
                note="PREVENTION cut-in label; not reviewed against our definition",
            ))
    return ClipAnnotation(
        clip_id=f"prevention-{drive.replace('/', '-').lower()}", source="prevention_uah",
        licence=registry_licence(DATASET_ID), split=split, duration_s=frame_times[-1] + step,
        fps_nominal=1 / step, variable_frame_rate=False, frame_timestamps_s=tuple(frame_times),
        events=tuple(events), source_labels=tuple(labels),
        provenance=Provenance(annotator="PREVENTION authors", method="imported:prevention",
                              created=date.today().isoformat()),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Convert one PREVENTION drive's lane changes to a clip annotation.")
    p.add_argument("processed_dir", type=Path, help="the drive's extracted processed_data/ directory")
    p.add_argument("--drive", required=True, help="recording name, e.g. RECORD4/DRIVE3")
    p.add_argument("--out", type=Path, required=True, help="JSON Lines output, under evaluation/restricted/")
    p.add_argument("--split", type=Split, default=Split.DEV)
    args = p.parse_args()
    times = parse_camera_log((args.processed_dir / "logs" / "log_camera1.txt").read_text())
    clip = convert((args.processed_dir / "detection_camera1" / "lane_changes.txt").read_text(), args.drive,
                   times, args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump_annotations([clip], args.out)
    print(f"{len(clip.source_labels)} labels, {len(clip.events)} source-derived cut-ins, "
          f"{clip.duration_s:.1f} s at {clip.fps_nominal:.2f} fps written to {args.out}")


if __name__ == "__main__":
    main()
