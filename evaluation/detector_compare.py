"""Run detectors through the engine's ``AnalysisRun`` on short windows and compare cost and output.

Each (detector, window) runs in its own spawned process, sequentially, under the engine's ``strict``
memory policy and the evaluation RSS budget. Detection only: no tracker, segmenter or analysers, so
the per-call timings are the detector's. Perception goes to an engine cache whose key covers the
detector's checkpoint sha256 and parameters, so outputs of different models are never confused.

    python -m evaluation.detector_compare SPEC.json --out RESULTS.json

SPEC: {"cache": "dir", "interval_s": 0.1,
       "detectors": {"name": {"class": "spectra.engine.detection:DFineOnnxDetector",
                              "model_path": "...", "sha256": "...", "params": {...}}},
       "windows": [{"id": "...", "video": "...", "start": 0, "end": 30, "assumed_fps": null}]}

Agreement between two detectors (``agreement``) is descriptive: neither output is ground truth.
"""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

from evaluation.hardware import _check_pressure, _cpu_s, _peak_rss_mb, default_memory_budget_mb, probe, run_isolated

HEIGHT_BINS = ((0, 30), (30, 80), (80, 1e9))  # box height in source pixels
VEHICLES = ("car", "truck", "bus", "motorcycle")


def _child(q, spec: dict, window: dict, cache_root: str, interval_s: float, budget_mb: float):
    try:
        from spectra.engine.cache import PerceptionCache
        from spectra.engine.contracts import RunConfig
        from spectra.engine.resources import ResourceGuard
        from spectra.engine.runner import AnalysisRun
        base_rss = _peak_rss_mb()
        module, _, cls = spec["class"].partition(":")
        det = getattr(importlib.import_module(module), cls)(spec["model_path"], spec["sha256"], **spec.get("params", {}))
        calls: list[float] = []
        load_s: list[float] = []
        detect, load = det.detect, det.load

        def timed_load():
            t0 = time.perf_counter()
            load()
            load_s.append(time.perf_counter() - t0)

        def timed_detect(frame):
            t0 = time.perf_counter()
            out = detect(frame)
            calls.append(time.perf_counter() - t0)
            if _peak_rss_mb() > budget_mb:
                raise MemoryError(f"peak RSS exceeded budget {budget_mb:.0f} MB")
            _check_pressure(1)
            return out

        det.load, det.detect = timed_load, timed_detect
        config = RunConfig(interval_s=interval_s, start_s=window["start"], end_s=window["end"],
                           assumed_fps=window.get("assumed_fps"))
        cpu0, wall0 = _cpu_s(), time.perf_counter()
        outcome = AnalysisRun(window["video"], config, detector=det, cache=PerceptionCache(cache_root),
                              guard=ResourceGuard(policy="strict")).run()
        wall = time.perf_counter() - wall0
        if outcome.result is None:
            raise RuntimeError(f"run {outcome.state.value}: {outcome.error}")
        frames = outcome.result.perception.frames
        warm = sorted(calls[1:])
        span = window["end"] - window["start"]
        q.put({"ok": True, "cache_hit": outcome.cache_hit, "perception_key": outcome.result.perception_key,
               "frames": len(frames), "video_s": span,
               "load_s": round(load_s[0], 3) if load_s else None,
               "first_detect_ms": round(calls[0] * 1e3, 1) if calls else None,
               "warm_median_ms": round(statistics.median(warm) * 1e3, 1) if warm else None,
               "warm_p90_ms": round(warm[int(0.9 * (len(warm) - 1))] * 1e3, 1) if warm else None,
               "run_wall_s": round(wall, 2), "s_per_video_s": round(wall / span, 3),
               "cpu_cores": round((_cpu_s() - cpu0) / wall, 2),
               "baseline_rss_mb": round(base_rss), "peak_rss_mb": round(_peak_rss_mb())})
    except Exception as exc:
        q.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def load_frames(cache_root: str, key: str) -> list:
    from spectra.engine.cache import PerceptionCache
    result = PerceptionCache(cache_root).lookup(key).result
    if result is None:
        raise ValueError(f"cache entry {key} not found or invalid")
    return list(result.frames)


def _boxes(frame, min_score: float, categories=VEHICLES) -> np.ndarray:
    rows = [[d.box.x1, d.box.y1, d.box.x2, d.box.y2, d.score] for d in frame.detections or ()
            if d.score >= min_score and d.category.value in categories]
    return np.array(rows, float).reshape(-1, 5)


def describe(frames: list, min_score: float = 0.5) -> dict:
    """Output statistics at a score threshold, by category and vehicle box height. Not accuracy."""
    cats: dict[str, int] = {}
    heights = [0] * len(HEIGHT_BINS)
    for f in frames:
        for d in f.detections or ():
            if d.score < min_score:
                continue
            cats[d.category.value] = cats.get(d.category.value, 0) + 1
            if d.category.value in VEHICLES:
                h = d.box.y2 - d.box.y1
                heights[next(i for i, (lo, hi) in enumerate(HEIGHT_BINS) if lo <= h < hi)] += 1
    n = max(len(frames), 1)
    return {"min_score": min_score, "per_frame": round(sum(cats.values()) / n, 2),
            "by_category": dict(sorted(cats.items())),
            "vehicles_by_height_px": {f"{lo}-{hi if hi < 1e9 else 'inf'}": c for (lo, hi), c in zip(HEIGHT_BINS, heights)}}


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def agreement(frames_a: list, frames_b: list, min_score: float = 0.5, iou: float = 0.5) -> dict:
    """Vehicle boxes found by both, by only A or only B (greedy one-to-one at IoU >= ``iou``), by height.

    Frames are paired by timestamp. Only-A boxes are not B's misses: neither model is ground truth.
    """
    by_t = {round(f.t, 6): f for f in frames_b}
    keys = ("both", "only_a", "only_b")
    out = {k: [0] * len(HEIGHT_BINS) for k in keys}
    ious, worst = [], []
    paired = 0
    for fa in frames_a:
        fb = by_t.get(round(fa.t, 6))
        if fb is None:
            continue
        paired += 1
        a, b = _boxes(fa, min_score), _boxes(fb, min_score)
        used_a, used_b = set(), set()
        if len(a) and len(b):
            m = _iou(a, b)
            for i, j in sorted(np.argwhere(m >= iou), key=lambda ij: -m[ij[0], ij[1]]):
                if i not in used_a and j not in used_b:
                    used_a.add(int(i)), used_b.add(int(j)), ious.append(float(m[i, j]))
        diff = 0
        for name, boxes, used in (("only_a", a, used_a), ("only_b", b, used_b)):
            for k, r in enumerate(boxes):
                h = r[3] - r[1]
                slot = next(i for i, (lo, hi) in enumerate(HEIGHT_BINS) if lo <= h < hi)
                if k in used:
                    if name == "only_a":
                        out["both"][slot] += 1
                else:
                    out[name][slot] += 1
                    diff += 1
        worst.append((diff, round(fa.t, 3)))
    label = [f"{lo}-{hi if hi < 1e9 else 'inf'}" for lo, hi in HEIGHT_BINS]
    return {"paired_frames": paired, "min_score": min_score, "iou": iou,
            **{k: dict(zip(label, v)) for k, v in out.items()},
            "matched_iou_median": round(float(np.median(ious)), 3) if ious else None,
            "most_disagreeing_t": [t for _, t in sorted(worst, reverse=True)[:5]]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("spec", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=900)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; results are kept, choose a new name")
    spec = json.loads(a.spec.read_text())
    budget = default_memory_budget_mb()
    report = {"spec": spec, "budget_mb": round(budget),
              "hardware": {k: probe().get(k) for k in ("cpu", "memory_gb", "os_release")}, "runs": {}}
    for name, det in spec["detectors"].items():
        for w in spec["windows"]:
            r = run_isolated(_child, (det, w, spec["cache"], spec.get("interval_s", 0.1), budget), a.timeout)
            if r.get("ok"):
                r["output"] = describe(load_frames(spec["cache"], r["perception_key"]))
            report["runs"].setdefault(name, {})[w["id"]] = r
            print(name, w["id"], json.dumps({k: r.get(k) for k in ("ok", "error", "s_per_video_s", "warm_median_ms",
                                                                   "peak_rss_mb", "load_s")}), flush=True)
            if not r.get("ok"):
                break
    names = list(spec["detectors"])
    base = names[0]
    report["agreement_vs_" + base] = {
        other: {w["id"]: agreement(load_frames(spec["cache"], report["runs"][base][w["id"]]["perception_key"]),
                                   load_frames(spec["cache"], report["runs"][other][w["id"]]["perception_key"]))
                for w in spec["windows"]
                if report["runs"][base].get(w["id"], {}).get("ok") and report["runs"][other].get(w["id"], {}).get("ok")}
        for other in names[1:]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
