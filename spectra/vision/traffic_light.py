"""Traffic-light colour state from a YOLO traffic-light bbox (vision-only).

YOLO localises the light; this module classifies the lit lamp colour by HSV.
It is an **advisory** signal, not a decision-maker: distant lights are tiny,
and "which light applies to my lane" is ambiguous from a single forward camera.
Callers should surface it as information, never gate collision logic on it.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detection import Detection

# Minimum lit-pixel fraction inside the bbox for a confident colour call.
_MIN_LIT_FRAC = 0.04
_S_MIN = 80
_V_MIN = 80


def _masks(hsv: np.ndarray) -> dict[str, int]:
    def count(lo, hi) -> int:
        return int(cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)).astype(bool).sum())

    red = count((0, _S_MIN, _V_MIN), (10, 255, 255)) + count((170, _S_MIN, _V_MIN), (180, 255, 255))
    yellow = count((15, _S_MIN, _V_MIN), (35, 255, 255))
    green = count((40, _S_MIN, _V_MIN), (90, 255, 255))
    return {"red": red, "yellow": yellow, "green": green}


def classify_light_state(
    frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[str, float]:
    """Return ``(state, confidence)`` for one light bbox.

    ``state`` is ``"red"|"yellow"|"green"|"unknown"``; ``confidence`` in
    ``[0, 1]`` is how dominant the winning colour is over the other two lit
    colours (``0.0`` when the call is ``"unknown"``).
    """

    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w_img, int(x1)))
    x2 = max(0, min(w_img, int(x2)))
    y1 = max(0, min(h_img, int(y1)))
    y2 = max(0, min(h_img, int(y2)))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return "unknown", 0.0

    roi = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    counts = _masks(hsv)
    total = roi.shape[0] * roi.shape[1]
    color, lit = max(counts.items(), key=lambda kv: kv[1])
    if total == 0 or (lit / total) < _MIN_LIT_FRAC:
        return "unknown", 0.0
    # Dominance of the winning colour over all lit colour pixels, attenuated by
    # how much of the bbox is actually lit (small/faint lights read less sure).
    colour_total = sum(counts.values())
    dominance = lit / colour_total if colour_total > 0 else 0.0
    lit_frac = min(lit / total, 1.0)
    confidence = float(min(1.0, dominance * (0.5 + 0.5 * min(lit_frac / _MIN_LIT_FRAC, 1.0))))
    return color, confidence


def frame_light_state(frame_bgr: np.ndarray, lights: list[Detection]) -> tuple[str, float]:
    """Aggregate to one advisory ``(state, confidence)`` for the frame.

    Picks the largest (nearest / most relevant) light bbox and classifies it.
    Returns ``("none", 0.0)`` when no lights are present.
    """

    if not lights:
        return "none", 0.0
    nearest = max(
        lights,
        key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
    )
    return classify_light_state(frame_bgr, nearest.bbox)
