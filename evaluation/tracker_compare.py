"""Re-track identical cached engine detections with different trackers and compare label-free indicators.

The detections come from an engine cache entry (for example one written by ``detector_compare``). Frames
are decoded again with ``VideoSource.frames`` on the same timestamp grid, so motion compensation sees the
images the engine would pass, and each tracker goes through the engine's ``Tracker`` protocol.

    python -m evaluation.tracker_compare SPEC.json --out RESULTS.json

SPEC: {"cache": "dir", "interval_s": 0.1, "windows": [{"id", "video", "start", "end", "assumed_fps", "key"}],
       "trackers": {"botsort": {"class": "spectra.engine.tracking:BoTSortTracker", "params": {}},
                    "mcbyte": {"class": "evaluation.tracker_compare:McByteTracker", "params": {}}}}

``probable_fragments`` and ``jumps`` follow ``evaluation.tracking.track_quality``. They are indicators of
fragmentation and identity swaps, not counts of real errors: there are no identity labels.
"""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

from evaluation.detector_compare import VEHICLES, load_frames
from evaluation.hardware import _peak_rss_mb, default_memory_budget_mb, run_isolated
from spectra.engine.tracking import BoTSortTracker


class McByteTracker(BoTSortTracker):
    """The engine BoT-SORT adapter with ``trackers.McByteTracker`` (no masks) as the association."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.info = self.info.model_copy(update={"name": "mcbyte"})

    def load(self) -> None:
        import trackers
        p = self.params
        original = trackers.BoTSORTTracker
        trackers.BoTSORTTracker = lambda **kw: trackers.McByteTracker(enable_mask_manager=False, **kw)
        try:
            super().load()
        finally:
            trackers.BoTSORTTracker = original
        if p["camera_motion"] and self._tracker.cmc is None:
            raise RuntimeError("McByte did not enable motion compensation")


def indicators(records: list[tuple[float, list]], min_score: float = 0.5, relink_s: float = 1.5) -> dict:
    """Continuity indicators from (t, tracks) pairs; vehicles with score >= ``min_score`` only."""
    spans: dict[int, list] = {}
    for t, tracks in records:
        for o in tracks:
            if o.score >= min_score and o.category.value in VEHICLES:
                spans.setdefault(o.track_id, []).append((t, o))
    minutes = (records[-1][0] - records[0][0]) / 60 if len(records) > 1 else 0.0
    jumps = 0
    for obs in spans.values():
        for (ta, a), (tb, b) in zip(obs, obs[1:]):
            ratio = (b.box.y2 - b.box.y1) / (a.box.y2 - a.box.y1)
            jumps += int(tb - ta <= 0.5 and (ratio > 2 or ratio < 0.5))

    def iou(a, b):
        ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
        iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
        inter = ix * iy
        return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter)

    ends = sorted((obs[-1] for obs in spans.values()), key=lambda x: x[0])
    begins = sorted((obs[0] for obs in spans.values()), key=lambda x: x[0])
    fragments, used = 0, set()
    for te, last in ends:
        for j, (tb, first) in enumerate(begins):
            if j not in used and te < tb <= te + relink_s and first.category == last.category \
                    and iou(last.box, first.box) >= 0.3:
                fragments += 1
                used.add(j)
                break
    durations = [obs[-1][0] - obs[0][0] for obs in spans.values()]
    per_min = lambda n: round(n / minutes, 1) if minutes else None
    return {"tracks": len(spans), "tracks_per_min": per_min(len(spans)),
            "median_track_s": round(float(np.median(durations)), 2) if durations else None,
            "tracks_ge_3s": int(sum(d >= 3 for d in durations)),
            "probable_fragments_per_min": per_min(fragments), "jumps_per_min": per_min(jumps)}


def _child(q, tracker_spec: dict, window: dict, cache_root: str, interval_s: float, budget_mb: float):
    try:
        from spectra.engine.video import VideoSource
        records = {round(f.t, 6): f for f in load_frames(cache_root, window["key"])}
        module, _, cls = tracker_spec["class"].partition(":")
        tracker = getattr(importlib.import_module(module), cls)(**tracker_spec.get("params", {}))
        t0 = time.perf_counter()
        tracker.load()
        load_s = time.perf_counter() - t0
        source = VideoSource(window["video"], assumed_fps=window.get("assumed_fps"))
        out, calls, eligible, tracked = [], [], 0, 0
        for frame in source.frames(interval_s, window["start"], window["end"]):
            rec = records.get(round(frame.t, 6))
            if rec is None:
                raise ValueError(f"no cached detections at t={frame.t}")
            dets = [d for d in rec.detections or () if not d.ego_vehicle]
            t0 = time.perf_counter()
            tracks = tracker.update(frame, dets)
            calls.append(time.perf_counter() - t0)
            out.append((frame.t, tracks))
            eligible += sum(d.score >= 0.5 and d.category.value in VEHICLES for d in dets)
            tracked += sum(o.score >= 0.5 and o.category.value in VEHICLES for o in tracks)
            if _peak_rss_mb() > budget_mb:
                raise MemoryError("peak RSS exceeded budget")
        tracker.close()
        if len(out) != len(records):
            raise ValueError(f"replayed {len(out)} frames, cache has {len(records)}")
        q.put({"ok": True, "frames": len(out), "load_s": round(load_s, 3),
               "update_median_ms": round(statistics.median(calls) * 1e3, 2),
               "update_p90_ms": round(sorted(calls)[int(0.9 * (len(calls) - 1))] * 1e3, 2),
               "tracked_fraction": round(tracked / eligible, 3) if eligible else None,
               "peak_rss_mb": round(_peak_rss_mb()), **indicators(out)})
    except Exception as exc:
        q.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("spec", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=600)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; results are kept, choose a new name")
    spec = json.loads(a.spec.read_text())
    budget = default_memory_budget_mb()
    report = {"spec": spec, "runs": {}}
    for name, tr in spec["trackers"].items():
        for w in spec["windows"]:
            r = run_isolated(_child, (tr, w, spec["cache"], spec.get("interval_s", 0.1), budget), a.timeout)
            report["runs"].setdefault(name, {})[w["id"]] = r
            print(name, w["id"], json.dumps(r), flush=True)
            if not r.get("ok"):
                break
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
