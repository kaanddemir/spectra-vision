"""Render frame-accurate comparison images of cached detections for human review.

    python -m evaluation.review_frames VIDEO CACHE_DIR [CACHE_DIR ...] --time 105.933 --out frame.jpg [--assume-fps 20]

Frames are decoded forward to the requested presentation time, because seeking is not frame-accurate on
variable-frame-rate or AV1 sources. Each cache is drawn as one panel (scores >= 0.5, label: track id,
class id, score).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluation.perception_cache import PerceptionCache

COLOURS = ((0, 255, 0), (0, 128, 255), (255, 0, 255))


def frame_at(video: str, t: float, assume_fps: float = 0.0):
    import cv2
    cap = cv2.VideoCapture(video)
    n, frame = 0, None
    while cap.grab():
        now = n / assume_fps if assume_fps else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        n += 1
        if now >= t - 1e-3:
            ok, frame = cap.retrieve()
            break
    cap.release()
    if frame is None:
        raise ValueError(f"no frame at {t} s in {video}")
    return frame


def panel(frame, cache: PerceptionCache, t: float, label: str, colour, min_score: float = 0.5):
    import cv2
    i = int(np.argmin(np.abs(cache.timestamps - t)))
    img = frame.copy()
    d = cache.detections
    for r in d[(d[:, 0] == i) & (d[:, 5] >= min_score)]:
        x1, y1, x2, y2 = map(int, r[1:5])
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(img, f"{int(r[7])} c{int(r[6])} {r[5]:.2f}", (x1, max(12, y1 - 4)), 0, 0.5, colour, 1)
    cv2.putText(img, f"{label} t={cache.timestamps[i]:.2f}", (10, 30), 0, 1, (0, 0, 255), 2)
    return img


def main() -> None:
    import cv2
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("video")
    p.add_argument("caches", nargs="+", type=Path)
    p.add_argument("--time", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--assume-fps", type=float, default=0.0)
    a = p.parse_args()
    frame = frame_at(a.video, a.time, a.assume_fps)
    panels = [panel(frame, PerceptionCache.load(c), a.time, c.parent.name, COLOURS[k % 3])
              for k, c in enumerate(a.caches)]
    img = np.vstack(panels)
    scale = 900 / img.shape[1]
    cv2.imwrite(a.out, cv2.resize(img, None, fx=scale, fy=scale), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
