"""Offline re-tracking of cached detections, for controlled tracker comparisons.

Every configuration consumes the identical cached detections (scores down to 0.1), so differences
come from association alone. Requires the evaluation environment (``supervision``).

Camera motion compensation follows BoT-SORT (Aharon et al., 2022): before association, each track's
Kalman state is moved by the frame-to-frame similarity transform estimated from background motion,
which is already stored in the cache's motion features.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass

import numpy as np

from evaluation.perception_cache import MOTION_COLUMNS, PerceptionCache

MOTION_WIDTH = 640  # width at which cached motion features were estimated
VEHICLES = (2, 3, 5, 7)


@dataclass(frozen=True)
class TrackerConfig:
    name: str
    lost_seconds: float = 0.6
    activation: float = 0.25
    matching: float = 0.8
    min_consecutive: int = 1
    cmc: bool = False
    agnostic_nms: float | None = None  # IoU above which overlapping vehicle boxes of any class are merged


def agnostic_nms(det: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Drop lower-score vehicle boxes that overlap a higher-score vehicle box in the same frame.

    DETR-style detectors skip NMS and can emit one vehicle twice under different classes (car and truck);
    a class-agnostic association step then fragments the track.
    """
    keep = np.ones(len(det), bool)
    veh = np.isin(det[:, 6], VEHICLES)
    for f in np.unique(det[:, 0]):
        idx = np.where((det[:, 0] == f) & veh)[0]
        idx = idx[np.argsort(-det[idx, 5])]
        for a_pos, a in enumerate(idx):
            if not keep[a]:
                continue
            rest = idx[a_pos + 1:]
            rest = rest[keep[rest]]
            if not len(rest):
                break
            ix = np.clip(np.minimum(det[a, 3], det[rest, 3]) - np.maximum(det[a, 1], det[rest, 1]), 0, None)
            iy = np.clip(np.minimum(det[a, 4], det[rest, 4]) - np.maximum(det[a, 2], det[rest, 2]), 0, None)
            inter = ix * iy
            area = lambda r: (r[..., 3] - r[..., 1]) * (r[..., 4] - r[..., 2])
            iou = inter / (area(det[a]) + area(det[rest]) - inter)
            keep[rest[iou > iou_threshold]] = False
    return det[keep]


def _affine(cache: PerceptionCache, i: int) -> np.ndarray | None:
    """2x3 similarity mapping frame i-1 to frame i in full-resolution pixels; None when unknown."""
    tx, ty, rot, log_s = (cache.motion[i, MOTION_COLUMNS.index(c)] for c in ("tx", "ty", "rotation", "scale"))
    if not np.all(np.isfinite([tx, ty, rot, log_s])):
        return None
    k = cache.frame_size[0] / MOTION_WIDTH
    s = float(np.exp(log_s))
    return np.array([[s * np.cos(rot), -s * np.sin(rot), tx * k], [s * np.sin(rot), s * np.cos(rot), ty * k]])


def _compensate(tracker, m: np.ndarray) -> None:
    a, t = m[:, :2], m[:, 2]
    s = float(np.sqrt(abs(np.linalg.det(a))))
    for tr in (*tracker.tracked_tracks, *tracker.lost_tracks):
        if tr.mean is None:
            continue
        mean = tr.mean.astype(np.float64)
        mean[0:2] = a @ mean[0:2] + t       # centre x, y
        mean[4:6] = a @ mean[4:6]           # centre velocity
        mean[3] *= s                        # height (aspect ratio unchanged)
        mean[7] *= s
        tr.mean = mean.astype(tr.mean.dtype)


def retrack(cache: PerceptionCache, cfg: TrackerConfig, fps: float = 10.0) -> tuple[PerceptionCache, dict]:
    """Return a cache copy with track ids from ``cfg`` and runtime statistics."""
    from supervision import Detections
    from supervision.tracker.byte_tracker.core import ByteTrack

    # supervision measures the lost buffer in 30 fps frames: max_time_lost = int(frame_rate / 30 * buffer)
    tracker = ByteTrack(track_activation_threshold=cfg.activation, minimum_matching_threshold=cfg.matching,
                        frame_rate=fps, lost_track_buffer=int(round(cfg.lost_seconds * 30)),
                        minimum_consecutive_frames=cfg.min_consecutive)
    det = cache.detections.copy()
    if cfg.agnostic_nms is not None:
        det = agnostic_nms(det, cfg.agnostic_nms)
    det[:, 7] = -1
    n = len(cache.timestamps)
    order = np.argsort(det[:, 0], kind="stable")
    det = det[order]
    starts = np.searchsorted(det[:, 0], np.arange(n + 1))
    t0 = time.perf_counter()
    for i in range(n):
        if cfg.cmc and i > 0 and cache.motion is not None:
            m = _affine(cache, i)
            if m is not None:
                _compensate(tracker, m)
        rows = det[starts[i]:starts[i + 1]]
        if len(rows):
            sd = Detections(xyxy=rows[:, 1:5].astype(np.float32), confidence=rows[:, 5].astype(np.float32),
                            class_id=rows[:, 6].astype(int))
            out = tracker.update_with_detections(sd)
            for box, tid in zip(out.xyxy, out.tracker_id):  # returned boxes are the input boxes
                det[starts[i] + int(np.argmin(np.abs(rows[:, 1:5] - box).sum(1))), 7] = tid
        else:
            tracker.update_with_detections(Detections.empty())
    elapsed = time.perf_counter() - t0
    return dataclasses.replace(cache, detections=det), {"tracker_s": round(elapsed, 3),
                                                        "ms_per_frame": round(1000 * elapsed / n, 3)}


def track_quality(cache: PerceptionCache, min_score: float = 0.5, relink_s: float = 1.5) -> dict:
    """Ground-truth-free continuity indicators for vehicle tracks.

    ``probable_fragments``: a track ends and another of the same class starts within ``relink_s``
    with overlapping boxes (IoU >= 0.3 between the last and first box): likely one vehicle split in two.
    ``jumps``: consecutive observations of one track whose box height changes by more than 2x within
    0.5 s: likely identity swaps between vehicles of different apparent size.
    """
    d = cache.detections
    d = d[(d[:, 5] >= min_score) & np.isin(d[:, 6], VEHICLES)]
    t = cache.timestamps
    minutes = (t[-1] - t[0]) / 60 if len(t) > 1 else 0.0
    tracked = d[d[:, 7] >= 0]
    spans = []
    for tid in np.unique(tracked[:, 7]):
        r = tracked[tracked[:, 7] == tid]
        r = r[np.argsort(r[:, 0])]
        spans.append((tid, r))
    durations = np.array([t[int(r[-1, 0])] - t[int(r[0, 0])] for _, r in spans]) if spans else np.zeros(0)
    jumps = 0
    for _, r in spans:
        h = r[:, 4] - r[:, 2]
        ratio = h[1:] / h[:-1]
        close = np.diff(t[r[:, 0].astype(int)]) <= 0.5
        jumps += int(np.sum(close & ((ratio > 2) | (ratio < 0.5))))

    def iou(a, b):
        ix = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        iy = max(0.0, min(a[4], b[4]) - max(a[2], b[2]))
        inter = ix * iy
        return inter / ((a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter)

    ends = sorted(((t[int(r[-1, 0])], r[-1]) for _, r in spans), key=lambda x: x[0])
    begins = sorted(((t[int(r[0, 0])], r[0]) for _, r in spans), key=lambda x: x[0])
    fragments = 0
    used: set[int] = set()
    for te, last in ends:
        for j, (tb, first) in enumerate(begins):
            if j in used or not (te < tb <= te + relink_s) or first[6] != last[6]:
                continue
            if iou(last, first) >= 0.3:
                fragments += 1
                used.add(j)
                break
    return {
        "minutes": round(minutes, 2), "vehicle_detections": len(d),
        "tracked_fraction": round(len(tracked) / len(d), 3) if len(d) else None,
        "tracks": len(spans), "tracks_per_min": round(len(spans) / minutes, 1) if minutes else None,
        "median_track_s": round(float(np.median(durations)), 2) if len(durations) else None,
        "tracks_ge_3s": int(np.sum(durations >= 3)),
        "probable_fragments_per_min": round(fragments / minutes, 1) if minutes else None,
        "jumps_per_min": round(jumps / minutes, 1) if minutes else None,
    }


def lead_vehicle_switches(cache: PerceptionCache, t_from: float, t_to: float, min_score: float = 0.5) -> dict:
    """Track ids of the largest vehicle centred in the middle third of the image, and how often the id changes."""
    d = cache.detections
    w = cache.frame_size[0]
    ids = []
    for i in np.where((cache.timestamps >= t_from) & (cache.timestamps <= t_to))[0]:
        cx = (d[:, 1] + d[:, 3]) / 2
        r = d[(d[:, 0] == i) & (d[:, 5] >= min_score) & np.isin(d[:, 6], VEHICLES) & (np.abs(cx - w / 2) < w / 6)]
        if len(r):
            ids.append(int(r[np.argmax((r[:, 3] - r[:, 1]) * (r[:, 4] - r[:, 2])), 7]))
    return {"ids": list(dict.fromkeys(ids)), "switches": sum(a != b for a, b in zip(ids, ids[1:])),
            "untracked_frames": sum(i < 0 for i in ids), "frames": len(ids)}


def _frames_at(video: str, timestamps: np.ndarray, assume_fps: float = 0.0):
    """Yield decoded frames whose presentation time equals each cached timestamp, in order."""
    import cv2
    cap = cv2.VideoCapture(video)
    n = 0
    k = 0
    while k < len(timestamps) and cap.grab():
        t = n / assume_fps if assume_fps else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        n += 1
        if abs(t - timestamps[k]) < 1e-6:
            ok, f = cap.retrieve()
            yield f if ok else None
            k += 1
    cap.release()


def retrack_trackers(cache: PerceptionCache, kind: str, lost_seconds: float = 2.0, nms: float | None = 0.7,
                     use_timestamps: bool = True, video: str | None = None, assume_fps: float = 0.0,
                     **params) -> tuple[PerceptionCache, dict]:
    """Re-track with Roboflow ``trackers`` (Apache-2.0): ``ocsort``, ``botsort`` or ``bytetrack``.

    BoT-SORT's camera motion compensation reads the decoded frames, so ``video`` is required for it.
    """
    from supervision import Detections
    import trackers

    cls = {"ocsort": trackers.OCSORTTracker, "botsort": trackers.BoTSORTTracker,
           "bytetrack": trackers.ByteTrackTracker}[kind]
    fps = float(1 / np.median(np.diff(cache.timestamps)))
    tracker = cls(lost_track_buffer=int(round(lost_seconds * 30)), frame_rate=fps, **params)
    det = cache.detections.copy()
    if nms is not None:
        det = agnostic_nms(det, nms)
    det[:, 7] = -1
    det = det[np.argsort(det[:, 0], kind="stable")]
    n = len(cache.timestamps)
    starts = np.searchsorted(det[:, 0], np.arange(n + 1))
    frames = _frames_at(video, cache.timestamps, assume_fps) if (kind == "botsort" and video) else None
    decode_s, t0 = 0.0, time.perf_counter()
    for i in range(n):
        frame = None
        if frames is not None:
            a = time.perf_counter()
            frame = next(frames, None)
            decode_s += time.perf_counter() - a
        rows = det[starts[i]:starts[i + 1]]
        sd = (Detections(xyxy=rows[:, 1:5].astype(np.float32), confidence=rows[:, 5].astype(np.float32),
                         class_id=rows[:, 6].astype(int)) if len(rows) else Detections.empty())
        kw = {"timestamp": float(cache.timestamps[i])} if use_timestamps else {}
        if kind == "botsort":
            kw["frame"] = frame
        out = tracker.update(sd, **kw)
        if len(rows) and out.tracker_id is not None:
            for box, tid in zip(out.xyxy, out.tracker_id):
                if tid >= 0:
                    det[starts[i] + int(np.argmin(np.abs(rows[:, 1:5] - box).sum(1))), 7] = tid
    elapsed = time.perf_counter() - t0 - decode_s
    return dataclasses.replace(cache, detections=det), {"tracker_s": round(elapsed, 3),
                                                        "ms_per_frame": round(1000 * elapsed / n, 3),
                                                        "decode_s": round(decode_s, 2)}
