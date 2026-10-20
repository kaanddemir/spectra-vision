"""Build a perception cache from a real video: detection, ByteTrack association and global motion.

One detector per isolated child process; frames are streamed and sampled by presentation time.

    evaluation/.venv/bin/python -m evaluation.build_cache VIDEO OUT_DIR --adapter evaluation.detectors.dfine:DFine \
        --params '{"input_px": 640}' --analysis-fps 10 --max-seconds 150 [--assume-fps 20]
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np

from evaluation.hardware import _check_pressure, _peak_rss_mb, default_memory_budget_mb, run_isolated

from spectra.engine.video import file_sha256

ROAD_USERS = (0, 1, 2, 3, 5, 7)  # person, bicycle, car, motorcycle, bus, truck (zero-based COCO)
MOTION_WIDTH = 640


def detector_model_info(det, params):
    """Hash the adapter's actual local checkpoint before loading; unknown paths stay unknown."""
    from evaluation.perception_cache import ModelInfo

    weights = getattr(det, "weights", None)
    digest = file_sha256(Path(weights)) if weights is not None else None
    return ModelInfo(det.name, json.dumps(params, sort_keys=True), sha256=digest)


def global_motion(prev_gray, gray):
    """Similarity transform between consecutive analysed frames from sparse LK tracks.

    Returns (tx, ty, rotation_rad, log_scale, inlier_ratio, photometric_change); NaN when unreliable.
    """
    import cv2
    photometric = float(np.mean(cv2.absdiff(prev_gray, gray)))
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01, minDistance=8)
    if pts is None or len(pts) < 12:
        return (np.nan,) * 4 + (0.0, photometric)
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
    ok = status.ravel() == 1
    if ok.sum() < 12:
        return (np.nan,) * 4 + (0.0, photometric)
    m, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if m is None:
        return (np.nan,) * 4 + (0.0, photometric)
    scale = float(np.hypot(m[0, 0], m[1, 0]))
    return (float(m[0, 2]), float(m[1, 2]), float(np.arctan2(m[1, 0], m[0, 0])), float(np.log(scale)),
            float(inl.sum() / len(pts)), photometric)


def _child(q, video, out_dir, adapter, params, analysis_fps, max_seconds, assume_fps, budget_mb):
    try:
        import cv2
        from supervision import Detections
        from supervision.tracker.byte_tracker.core import ByteTrack
        from evaluation.perception_cache import ModelInfo, PerceptionCache

        module, _, cls = adapter.partition(":")
        det = getattr(importlib.import_module(module), cls)(score_threshold=0.1, **params)
        model_info = detector_model_info(det, params)
        t0 = time.perf_counter()
        det.load()
        load_s = time.perf_counter() - t0
        tracker = ByteTrack(frame_rate=analysis_fps, lost_track_buffer=int(2 * analysis_fps))
        cap = cv2.VideoCapture(video)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        ts, rows, motion = [], [], []
        prev_gray, next_t, n_read = None, 0.0, 0
        stage = {"decode": 0.0, "detect": 0.0, "track": 0.0, "motion": 0.0}
        wall0 = time.perf_counter()
        while True:
            a = time.perf_counter()
            if not cap.grab():
                break
            t = n_read / assume_fps if assume_fps else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            n_read += 1
            if t > max_seconds:
                break
            if t + 1e-6 < next_t:
                stage["decode"] += time.perf_counter() - a
                continue
            ok, frame = cap.retrieve()
            stage["decode"] += time.perf_counter() - a
            if not ok:
                break
            next_t = t + 1.0 / analysis_fps
            idx = len(ts)
            ts.append(t)
            a = time.perf_counter()
            d = det.predict(frame)
            stage["detect"] += time.perf_counter() - a
            d = d[np.isin(d[:, 5].astype(int), ROAD_USERS)] if len(d) else d.reshape(0, 6)
            a = time.perf_counter()
            tids = np.full(len(d), -1.0)
            if len(d):
                sd = Detections(xyxy=d[:, :4].astype(np.float32), confidence=d[:, 4].astype(np.float32),
                                class_id=d[:, 5].astype(int))
                tr = tracker.update_with_detections(sd)
                for box, tid in zip(tr.xyxy, tr.tracker_id):
                    j = int(np.argmin(np.abs(d[:, :4] - box).sum(1)))
                    tids[j] = tid
            else:
                tracker.update_with_detections(Detections.empty())
            stage["track"] += time.perf_counter() - a
            for r, tid in zip(d, tids):
                rows.append((idx, *r[:4], r[4], r[5], tid))
            a = time.perf_counter()
            small = cv2.cvtColor(cv2.resize(frame, (MOTION_WIDTH, round(size[1] * MOTION_WIDTH / size[0]))),
                                 cv2.COLOR_BGR2GRAY)
            motion.append(global_motion(prev_gray, small) if prev_gray is not None else (0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
            prev_gray = small
            stage["motion"] += time.perf_counter() - a
            if _peak_rss_mb() > budget_mb:
                raise MemoryError("peak RSS above budget")
            if idx % 20 == 0:
                _check_pressure(1)
        cap.release()
        wall = time.perf_counter() - wall0
        dets = np.array(rows, dtype=np.float64).reshape(-1, 8)
        dets[:, 1:5:2] = np.clip(dets[:, 1:5:2], 0, size[0])
        dets[:, 2:5:2] = np.clip(dets[:, 2:5:2], 0, size[1])
        dets = dets[(dets[:, 3] > dets[:, 1]) & (dets[:, 4] > dets[:, 2])]
        cache = PerceptionCache(
            media_sha256=file_sha256(Path(video)), frame_size=size, timestamps=np.array(ts), detections=dets,
            motion=np.array(motion), models=(model_info, ModelInfo("bytetrack-supervision", "0.30.9")),
            config={"adapter": adapter, "params": params, "analysis_fps": analysis_fps, "max_seconds": max_seconds,
                    "timestamps": f"assumed {assume_fps} fps" if assume_fps else "container pts",
                    "score_threshold": 0.1, "classes": list(ROAD_USERS)},
            class_names={0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"})
        cache.save(Path(out_dir))
        video_s = ts[-1] - ts[0] if len(ts) > 1 else 0.0
        q.put({"ok": True, "video": Path(video).name, "detector": det.name, "frames_read": n_read,
               "frames_analysed": len(ts), "video_seconds": round(video_s, 1), "load_s": round(load_s, 2),
               "processing_s": round(wall, 1), "realtime_factor": round(video_s / wall, 2) if wall else None,
               "stage_s": {k: round(v, 1) for k, v in stage.items()}, "detections": len(dets),
               "tracks": len(cache.track_ids()), "peak_rss_mb": round(_peak_rss_mb())})
    except Exception as exc:
        q.put({"ok": False, "video": Path(video).name, "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("video")
    p.add_argument("out_dir")
    p.add_argument("--adapter", required=True)
    p.add_argument("--params", default="{}")
    p.add_argument("--analysis-fps", type=float, default=10.0)
    p.add_argument("--max-seconds", type=float, default=150.0)
    p.add_argument("--assume-fps", type=float, default=0.0, help="for raw streams without timestamps")
    p.add_argument("--timeout", type=float, default=900)
    a = p.parse_args()
    r = run_isolated(_child, (a.video, a.out_dir, a.adapter, json.loads(a.params), a.analysis_fps, a.max_seconds,
                              a.assume_fps, default_memory_budget_mb()), a.timeout)
    print(json.dumps(r))


if __name__ == "__main__":
    main()
