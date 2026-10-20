"""Bounded, isolated detector runtime benchmark over frames from one video.

Each configuration runs in its own spawned process (see ``hardware.run_isolated``),
sequentially, with timeout, RSS budget and system memory-pressure checks.

    python -m evaluation.detector_bench VIDEO --adapter evaluation.detectors.dfine:DFine \
        --configs '{"input_px": 640}' '{"input_px": 480}' --frames 30 --warmup 3
"""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time

from evaluation.hardware import _check_pressure, _cpu_s, _peak_rss_mb, default_memory_budget_mb, probe, run_isolated


def _read_frames(video: str, n: int, stride: int):
    import cv2
    cap = cv2.VideoCapture(video)
    frames, i = [], 0
    while len(frames) < n and cap.grab():
        if i % stride == 0:
            ok, f = cap.retrieve()
            if ok:
                frames.append(f)
        i += 1
    cap.release()
    if not frames:
        raise OSError(f"no frames read from {video}")
    return frames


def _child(q, adapter, params, video, n_frames, stride, warmup, budget_mb):
    try:
        base = _peak_rss_mb()
        frames = _read_frames(video, n_frames + warmup, stride)
        module, _, cls = adapter.partition(":")
        det = getattr(importlib.import_module(module), cls)(**params)
        t0 = time.perf_counter()
        det.load()
        load_s = time.perf_counter() - t0
        rss_loaded = _peak_rss_mb()
        t0 = time.perf_counter()
        det.predict(frames[0])
        first_s = time.perf_counter() - t0
        for f in frames[1:warmup]:
            det.predict(f)
        times, counts = [], []
        cpu0, wall0 = _cpu_s(), time.perf_counter()
        for f in frames[warmup:]:
            t0 = time.perf_counter()
            out = det.predict(f)
            times.append(time.perf_counter() - t0)
            counts.append(len(out))
            if _peak_rss_mb() > budget_mb:
                raise MemoryError(f"peak RSS exceeded budget {budget_mb:.0f} MB")
            _check_pressure(1)
        wall = time.perf_counter() - wall0
        times.sort()
        q.put({"ok": True, "detector": det.name, "params": params, "frames_timed": len(times),
               "frame_size": list(frames[0].shape[1::-1]), "load_s": round(load_s, 2),
               "first_inference_s": round(first_s, 3), "warm_median_ms": round(statistics.median(times) * 1e3, 1),
               "warm_p90_ms": round(times[int(0.9 * (len(times) - 1))] * 1e3, 1),
               "throughput_fps": round(len(times) / wall, 2),
               "cpu_utilisation_cores": round((_cpu_s() - cpu0) / wall, 2),
               "baseline_rss_mb": round(base), "rss_after_load_mb": round(rss_loaded),
               "peak_rss_mb": round(_peak_rss_mb()), "mean_detections": round(statistics.mean(counts), 2)})
    except Exception as exc:
        q.put({"ok": False, "params": params, "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("video")
    p.add_argument("--adapter", required=True)
    p.add_argument("--configs", nargs="+", default=["{}"])
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--timeout", type=float, default=300)
    a = p.parse_args()
    budget = default_memory_budget_mb()
    results = []
    for c in a.configs:
        r = run_isolated(_child, (a.adapter, json.loads(c), a.video, a.frames, a.stride, a.warmup, budget), a.timeout)
        results.append(r)
        print(json.dumps(r), flush=True)
        if not r.get("ok"):
            break  # stop the sequence on the first failure or pressure abort
    hw = {k: probe().get(k) for k in ("cpu", "memory_gb", "os_release", "memory_pressure_level", "swap_used_mb")}
    print(json.dumps({"hardware": hw, "budget_mb": round(budget)}))


if __name__ == "__main__":
    main()
