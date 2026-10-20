"""Brake-light / deceleration cue for lead vehicles (vision-only, no model).

A lit brake-lamp pair shows up as bright, saturated red in the lower-left and
lower-right of a rear-facing vehicle bbox — and it lights up *before* the gap
visibly closes, so it is an early forward-collision cue that none of the TTC
signals (expansion / flow / depth) capture.

This is a heuristic corroborating signal, not a detector: red paintwork, sunset
glare and tail-lights (vs brake-lights) can confuse it. Callers should treat a
high score as "likely braking" and combine it with lane membership / closing,
never act on it alone.
"""

from __future__ import annotations

import cv2
import numpy as np

# HSV gates for a *lit* red lamp: red hue wraps around 0/180, with high
# saturation and high value (brightness) so illuminated lamps win over dark red
# bodywork.
_S_MIN = 90
_V_MIN = 150

# A few percent of symmetric bright-red already indicates a lit pair.
_PAIR_FULL_SCALE = 0.12
# Lamps are *localised* bright spots; bodywork fills the band. Coverage at/below
# _LOCALITY_LO is fully lamp-like; at/above _LOCALITY_HI it is treated as a red
# panel and suppressed to zero.
_LOCALITY_LO = 0.35
_LOCALITY_HI = 0.70


def _red_mask(hsv: np.ndarray) -> np.ndarray:
    lower1 = np.array([0, _S_MIN, _V_MIN], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, _S_MIN, _V_MIN], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)


def brake_score(frame_bgr: np.ndarray | None, bbox: tuple[int, int, int, int]) -> float:
    """Return a [0, 1] confidence that the vehicle in ``bbox`` is braking.

    Looks for a bright symmetric red pair in the lower band of the bbox.
    Returns 0.0 for missing input, tiny boxes, or ambiguous coverage.
    """

    if frame_bgr is None:
        return 0.0

    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w_img, int(x1)))
    x2 = max(0, min(w_img, int(x2)))
    y1 = max(0, min(h_img, int(y1)))
    y2 = max(0, min(h_img, int(y2)))
    width = x2 - x1
    height = y2 - y1
    if width < 12 or height < 12:
        return 0.0

    # Rear lamp band: lower ~55% of the bbox.
    band_top = y1 + int(0.45 * height)
    roi = frame_bgr[band_top:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = _red_mask(hsv) > 0
    cols = mask.shape[1]
    third = max(1, cols // 3)

    left_frac = float(mask[:, :third].mean())
    right_frac = float(mask[:, -third:].mean())
    overall = float(mask.mean())

    # A lit pair lights both corners, so the symmetric minimum is the signal.
    pair = float(np.clip(min(left_frac, right_frac) / _PAIR_FULL_SCALE, 0.0, 1.0))
    # Localisation: full-band red coverage is bodywork, not lamps -> suppress.
    locality = float(
        np.clip(1.0 - (overall - _LOCALITY_LO) / (_LOCALITY_HI - _LOCALITY_LO), 0.0, 1.0)
    )
    return round(pair * locality, 3)
