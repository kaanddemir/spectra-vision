"""Road-relative geometry helpers: vanishing point, lane position, EMA smoothing.

The risk pipeline uses these helpers so every spatial decision (corridor
membership, lateral velocity, crossing prediction) is expressed in lane-relative
units rather than raw pixel coordinates. That keeps the same thresholds usable
across cameras with different focal lengths and across road geometry that
changes shape (curves, hills).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .road_roi import RoadROI


# When the vanishing point fit is unreliable we use a fixed image-
# anchored point (slightly above center). Picked empirically — works for
# forward-mounted dashcam footage even before any lane detection succeeds.
_DEFAULT_VP_Y_FRACTION = 0.55

# EMA on the VP: rise faster than we fall so quick scene transitions
# (turns, lane changes) catch up, but per-frame jitter from Hough fits is
# absorbed.
_VP_EMA_RISE = 0.40
_VP_EMA_FALL = 0.15


@dataclass(frozen=True)
class LaneFrame:
    """All road-derived values needed by per-object risk for one frame."""

    vanishing_point: tuple[float, float]
    left_line: tuple[int, int, int, int] | None
    right_line: tuple[int, int, int, int] | None
    left_x_at_bottom: float
    right_x_at_bottom: float
    lane_width_at_bottom: float
    lane_center_x_at_bottom: float
    confidence: float
    detected: bool
    width: int
    height: int


def line_x_at_y(line: tuple[int, int, int, int], y: float) -> float:
    x1, y1, x2, y2 = line
    if y2 == y1:
        return float((x1 + x2) / 2.0)
    t = (y - y1) / float(y2 - y1)
    return float(x1 + ((x2 - x1) * t))


def compute_vanishing_point(road_roi: RoadROI, width: int, height: int) -> tuple[float, float]:
    """Intersection of the two lane lines. Falls back to image center-upper."""

    if road_roi.left_line is None or road_roi.right_line is None:
        return (float(width - 1) / 2.0, float(height) * _DEFAULT_VP_Y_FRACTION)

    lx1, ly1, lx2, ly2 = road_roi.left_line
    rx1, ry1, rx2, ry2 = road_roi.right_line

    # Solve [a1*x + b1*y = c1; a2*x + b2*y = c2] form for the two lines.
    # Each line "(x1,y1)-(x2,y2)" has normal coefficients (dy, -dx, dy*x1 - dx*y1).
    a1 = float(ly2 - ly1)
    b1 = float(lx1 - lx2)
    c1 = a1 * lx1 + b1 * ly1
    a2 = float(ry2 - ry1)
    b2 = float(rx1 - rx2)
    c2 = a2 * rx1 + b2 * ry1

    det = (a1 * b2) - (a2 * b1)
    if abs(det) < 1e-3:
        return (float(width - 1) / 2.0, float(height) * _DEFAULT_VP_Y_FRACTION)

    vx = (b2 * c1 - b1 * c2) / det
    vy = (a1 * c2 - a2 * c1) / det

    # Clip outside-frame VPs to a sane range so a runaway fit cannot push the
    # focus of expansion off-screen and break the corridor cone.
    vx = float(np.clip(vx, -0.5 * width, 1.5 * width))
    vy = float(np.clip(vy, 0.0, height * 0.9))
    return (vx, vy)


def build_lane_frame(
    road_roi: RoadROI,
    *,
    width: int,
    height: int,
    smoothed_vp: tuple[float, float] | None = None,
) -> LaneFrame:
    """Bundle the per-frame road-relative values used downstream."""

    vp = smoothed_vp if smoothed_vp is not None else compute_vanishing_point(road_roi, width, height)

    if road_roi.left_line is not None and road_roi.right_line is not None:
        left_x = line_x_at_y(road_roi.left_line, height - 1)
        right_x = line_x_at_y(road_roi.right_line, height - 1)
    else:
        left_x = width * 0.05
        right_x = width * 0.95

    if right_x - left_x < width * 0.10:
        # Pathological detection: collapse to a centered default so lane
        # offset math does not divide by ~0.
        left_x = width * 0.30
        right_x = width * 0.70

    lane_width = float(right_x - left_x)
    lane_center = float((left_x + right_x) / 2.0)

    return LaneFrame(
        vanishing_point=vp,
        left_line=road_roi.left_line,
        right_line=road_roi.right_line,
        left_x_at_bottom=float(left_x),
        right_x_at_bottom=float(right_x),
        lane_width_at_bottom=lane_width,
        lane_center_x_at_bottom=lane_center,
        confidence=float(road_roi.confidence),
        detected=bool(road_roi.detected),
        width=int(width),
        height=int(height),
    )


def lane_edges_at_y(lane: LaneFrame, y: float) -> tuple[float, float]:
    """Return lane left/right x coordinates at a specific image y."""

    y_clamped = float(np.clip(y, 0.0, max(0, lane.height - 1)))
    if lane.left_line is not None and lane.right_line is not None:
        left_x = line_x_at_y(lane.left_line, y_clamped)
        right_x = line_x_at_y(lane.right_line, y_clamped)
    else:
        left_x = lane.left_x_at_bottom
        right_x = lane.right_x_at_bottom

    if right_x < left_x:
        left_x, right_x = right_x, left_x

    min_width = max(1.0, lane.width * 0.08)
    if right_x - left_x < min_width:
        center_x = (left_x + right_x) / 2.0
        left_x = center_x - (min_width / 2.0)
        right_x = center_x + (min_width / 2.0)

    return float(left_x), float(right_x)


def lane_center_width_at_y(lane: LaneFrame, y: float) -> tuple[float, float]:
    left_x, right_x = lane_edges_at_y(lane, y)
    return float((left_x + right_x) / 2.0), float(right_x - left_x)


def lane_position(bbox: tuple[int, int, int, int], lane: LaneFrame) -> float:
    """Signed offset of the bbox bottom center from the ego lane center.

    Returned in half-lane units: 0.0 = ego lane center, ±1.0 = at the lane
    boundary, ±2.0 = one full lane away. We use the bbox **bottom** because
    it is the contact point with the road plane — the top of a tall vehicle
    can sit anywhere on screen even when the wheels are clearly in our lane.
    """

    x1, _, x2, y2 = bbox
    bottom_cx = (x1 + x2) / 2.0
    lane_center, lane_width = lane_center_width_at_y(lane, y2)
    half_width = max(1.0, lane_width / 2.0)
    return float((bottom_cx - lane_center) / half_width)


def lane_corridor_relevance(
    bbox: tuple[int, int, int, int],
    lane: LaneFrame,
) -> float:
    """How much of the bbox bottom edge sits inside the ego lane.

    Combines two cues: |lane_pos| (Gaussian falloff at lane boundary) and the
    bbox vertical position (further-away objects matter slightly less because
    the path can still be steered). Returned in [0, 1].
    """

    pos = lane_position(bbox, lane)
    proximity = float(np.exp(-(pos * pos) / 0.50))  # ~0.6 at boundary, ~0.13 at adjacent lane

    _, y1, _, y2 = bbox
    bottom_y = float(y2)
    vertical_weight = float(np.clip((bottom_y - 0.30 * lane.height) / max(0.70 * lane.height, 1.0), 0.0, 1.0))

    return float(np.clip((0.20 + (0.80 * proximity)) * (0.40 + (0.60 * vertical_weight)), 0.0, 1.0))


class VanishingPointSmoother:
    """EMA on the vanishing point so frame-frame Hough jitter is absorbed."""

    def __init__(
        self,
        *,
        rise_alpha: float = _VP_EMA_RISE,
        fall_alpha: float = _VP_EMA_FALL,
    ) -> None:
        self.rise_alpha = float(rise_alpha)
        self.fall_alpha = float(fall_alpha)
        self._state: tuple[float, float] | None = None

    def update(self, raw_vp: tuple[float, float], confidence: float) -> tuple[float, float]:
        if self._state is None:
            self._state = raw_vp
            return raw_vp

        # Lower-confidence frames pull the EMA less so a single misfit does
        # not yank the focus of expansion across the frame.
        gain = float(np.clip(confidence, 0.0, 1.0))
        prev_x, prev_y = self._state
        raw_x, raw_y = raw_vp

        def _step(prev: float, raw: float) -> float:
            alpha = self.rise_alpha if abs(raw - prev) > abs(prev) * 0.05 else self.fall_alpha
            alpha *= 0.4 + (0.6 * gain)
            return float((alpha * raw) + ((1.0 - alpha) * prev))

        smoothed = (_step(prev_x, raw_x), _step(prev_y, raw_y))
        self._state = smoothed
        return smoothed

    def reset(self) -> None:
        self._state = None
