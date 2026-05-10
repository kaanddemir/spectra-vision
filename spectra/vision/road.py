"""Road/lane ROI estimation for lane-relative risk scoring."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


Line = tuple[int, int, int, int]


@dataclass(frozen=True)
class RoadROI:
    mask: np.ndarray
    polygon: np.ndarray
    left_line: Line | None
    right_line: Line | None
    confidence: float
    detected: bool


def default_road_roi(shape: tuple[int, int] | tuple[int, int, int]) -> RoadROI:
    """Return the fixed perspective ROI used as a stable default."""

    height, width = shape[:2]
    polygon = np.array(
        [
            [int(width * 0.42), int(height * 0.60)],
            [int(width * 0.58), int(height * 0.60)],
            [int(width * 0.95), height - 1],
            [int(width * 0.05), height - 1],
        ],
        dtype=np.int32,
    )
    mask = _polygon_mask((height, width), polygon)
    return RoadROI(
        mask=mask,
        polygon=polygon,
        left_line=(
            int(width * 0.42),
            int(height * 0.60),
            int(width * 0.10),
            height - 1,
        ),
        right_line=(
            int(width * 0.58),
            int(height * 0.60),
            int(width * 0.90),
            height - 1,
        ),
        confidence=0.25,
        detected=False,
    )


def _polygon_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    return mask.astype(bool)


_DEFAULT_VP_Y_FRACTION = 0.55


@dataclass(frozen=True)
class LaneFrame:
    """All road-derived values needed by per-object risk for one frame."""

    vanishing_point: tuple[float, float]
    left_line: Line | None
    right_line: Line | None
    left_x_at_bottom: float
    right_x_at_bottom: float
    lane_width_at_bottom: float
    lane_center_x_at_bottom: float
    confidence: float
    detected: bool
    width: int
    height: int


def line_x_at_y(line: Line, y: float) -> float:
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
    """Signed offset of the bbox bottom center from the ego lane center."""

    x1, _, x2, y2 = bbox
    bottom_cx = (x1 + x2) / 2.0
    lane_center, lane_width = lane_center_width_at_y(lane, y2)
    half_width = max(1.0, lane_width / 2.0)
    return float((bottom_cx - lane_center) / half_width)


def lane_corridor_relevance(
    bbox: tuple[int, int, int, int],
    lane: LaneFrame,
) -> float:
    """How much of the bbox bottom edge sits inside the ego lane."""

    pos = lane_position(bbox, lane)
    proximity = float(np.exp(-(pos * pos) / 0.50))

    _, y1, _, y2 = bbox
    bottom_y = float(y2)
    vertical_weight = float(np.clip((bottom_y - 0.30 * lane.height) / max(0.70 * lane.height, 1.0), 0.0, 1.0))

    return float(np.clip((0.20 + (0.80 * proximity)) * (0.40 + (0.60 * vertical_weight)), 0.0, 1.0))


def _fit_polyline_to_y_range(
    points: np.ndarray,
    y_top: int,
    y_bottom: int,
    width: int,
) -> Line | None:
    """Fit x = a*y + b to a polyline and return endpoints at y_top / y_bottom."""

    if points.shape[0] < 2:
        return None
    xs = points[:, 0].astype(np.float32)
    ys = points[:, 1].astype(np.float32)
    if float(np.max(ys) - np.min(ys)) < 8.0:
        return None
    slope, intercept = np.polyfit(ys, xs, 1)
    x_top = int(np.clip((slope * y_top) + intercept, 0, width - 1))
    x_bottom = int(np.clip((slope * y_bottom) + intercept, 0, width - 1))
    return (x_top, y_top, x_bottom, y_bottom)


def estimate_road_roi_from_lanes(
    lanes: list[np.ndarray],
    *,
    width: int,
    height: int,
) -> RoadROI:
    """Build a RoadROI from UFLDv2 lane polylines.

    The input is a 4-element list [left-left, left, right, right-right] of
    (N, 2) ``(x, y)`` polylines. Empty arrays signal a missing lane. The ego
    corridor is defined strictly by the inner pair (indices 1 and 2).

    We never promote an outer lane to ego — doing so traces the road edges
    instead of the ego lane, which is worse than dropping to the cached or
    default ROI for that frame. When the inner pair is incomplete this function
    returns a non-detected ROI so the caller's fallback chain
    (cached_road_roi -> default) takes over.
    """

    y_top = int(height * 0.58)
    y_bottom = height - 1

    left = lanes[1] if len(lanes) > 1 and lanes[1].size else np.empty((0, 2), dtype=np.float32)
    right = lanes[2] if len(lanes) > 2 and lanes[2].size else np.empty((0, 2), dtype=np.float32)

    left_line = _fit_polyline_to_y_range(left, y_top, y_bottom, width) if left.size else None
    right_line = _fit_polyline_to_y_range(right, y_top, y_bottom, width) if right.size else None

    detected_count = int(left_line is not None) + int(right_line is not None)
    if detected_count == 0:
        return default_road_roi((height, width))

    if left_line is None or right_line is None:
        return default_road_roi((height, width))

    lx_top, ly_top, lx_bottom, ly_bottom = left_line
    rx_top, ry_top, rx_bottom, ry_bottom = right_line
    # Sanity-check at the bottom only. UFLD often detects lanes confidently
    # only in the lower portion of the frame; polyfit extrapolation up to
    # y_top can cross the lines even when the underlying data is good. The
    # bottom separation is what actually matters for ego-lane assignment.
    if rx_bottom - lx_bottom < width * 0.12:
        return default_road_roi((height, width))
    # If extrapolated tops cross, snap them to the bottom slope so the
    # polygon stays well-formed without throwing the detection away.
    if rx_top - lx_top < width * 0.02:
        midpoint_top = (lx_top + rx_top) // 2
        margin = max(int(width * 0.01), 4)
        lx_top = max(0, midpoint_top - margin)
        rx_top = min(width - 1, midpoint_top + margin)
        left_line = (lx_top, ly_top, lx_bottom, ly_bottom)
        right_line = (rx_top, ry_top, rx_bottom, ry_bottom)

    polygon = np.array(
        [
            [lx_top, ly_top],
            [rx_top, ry_top],
            [rx_bottom, ry_bottom],
            [lx_bottom, ly_bottom],
        ],
        dtype=np.int32,
    )
    mask = _polygon_mask((height, width), polygon)
    # Both inner lines from UFLDv2 are treated as a strong signal.
    confidence = 0.75 if detected_count == 1 else 0.92
    return RoadROI(
        mask=mask,
        polygon=polygon,
        left_line=left_line,
        right_line=right_line,
        confidence=confidence,
        detected=True,
    )


class LaneKalman:
    """Constant-velocity Kalman filter on the four lane endpoint x-positions.

    State vector: ``[lx_top, lx_bot, rx_top, rx_bot, vlx_top, vlx_bot, vrx_top, vrx_bot]``.
    Smoothing happens in image-x space at fixed y-anchors (top/bottom of the
    detection band), so it is independent of whether the source is UFLDv2 or
    the fixed default. The same smoother absorbs UFLDv2 row-anchor flicker.
    """

    _STATE_DIM = 8
    _MEAS_DIM = 4

    def __init__(self, *, process_var: float = 4.0, measurement_var_high_conf: float = 6.0) -> None:
        self._process_var = float(process_var)
        self._meas_var_hi = float(measurement_var_high_conf)
        self._initialized = False
        self._x = np.zeros((self._STATE_DIM, 1), dtype=np.float32)
        self._P = np.eye(self._STATE_DIM, dtype=np.float32) * 100.0

        F = np.eye(self._STATE_DIM, dtype=np.float32)
        for i in range(4):
            F[i, i + 4] = 1.0  # x += v
        self._F = F

        H = np.zeros((self._MEAS_DIM, self._STATE_DIM), dtype=np.float32)
        for i in range(4):
            H[i, i] = 1.0
        self._H = H

        Q = np.eye(self._STATE_DIM, dtype=np.float32) * (self._process_var * 0.25)
        for i in range(4, 8):
            Q[i, i] = self._process_var
        self._Q = Q

    def update(
        self,
        left_line: Line | None,
        right_line: Line | None,
        confidence: float,
    ) -> tuple[Line | None, Line | None]:
        """Smooth lane endpoints. Returns the filtered (left, right) lines.

        When both lines are missing we predict forward (no measurement
        update) so the corridor coasts through brief detection gaps.
        """

        if left_line is None and right_line is None:
            if not self._initialized:
                return None, None
            self._predict()
            return self._emit(left_line, right_line)

        # Anchor y values come from whichever UFLDv2 line is present; endpoints
        # are pinned at the same y_top / y_bottom so this is consistent.
        ref_line = left_line if left_line is not None else right_line
        assert ref_line is not None
        y_top = float(ref_line[1])
        y_bottom = float(ref_line[3])

        meas = np.zeros((self._MEAS_DIM, 1), dtype=np.float32)
        meas_mask = np.zeros(self._MEAS_DIM, dtype=bool)
        if left_line is not None:
            meas[0, 0] = float(left_line[0])
            meas[1, 0] = float(left_line[2])
            meas_mask[0] = True
            meas_mask[1] = True
        if right_line is not None:
            meas[2, 0] = float(right_line[0])
            meas[3, 0] = float(right_line[2])
            meas_mask[2] = True
            meas_mask[3] = True

        if not self._initialized:
            for i in range(4):
                if meas_mask[i]:
                    self._x[i, 0] = meas[i, 0]
            self._initialized = True
            return self._emit(left_line, right_line, y_top=y_top, y_bottom=y_bottom)

        self._predict()

        # Per-component Kalman update — confidence raises measurement
        # variance so low-confidence frames bend the state less.
        gain = float(np.clip(confidence, 0.05, 1.0))
        meas_var = self._meas_var_hi / max(gain, 0.1)
        for i in range(self._MEAS_DIM):
            if not meas_mask[i]:
                continue
            H_row = self._H[i : i + 1]
            S = float(H_row @ self._P @ H_row.T) + meas_var
            K = (self._P @ H_row.T) / S
            innovation = float(meas[i, 0] - (H_row @ self._x))
            self._x = self._x + (K * innovation)
            self._P = (np.eye(self._STATE_DIM, dtype=np.float32) - (K @ H_row)) @ self._P

        return self._emit(left_line, right_line, y_top=y_top, y_bottom=y_bottom)

    def _predict(self) -> None:
        self._x = self._F @ self._x
        self._P = (self._F @ self._P @ self._F.T) + self._Q

    def _emit(
        self,
        left_line: Line | None,
        right_line: Line | None,
        *,
        y_top: float | None = None,
        y_bottom: float | None = None,
    ) -> tuple[Line | None, Line | None]:
        # Reuse last-known anchors when a side is missing in the current frame.
        if y_top is None or y_bottom is None:
            ref = left_line or right_line
            if ref is not None:
                y_top = float(ref[1])
                y_bottom = float(ref[3])
            else:
                return None, None

        left_smoothed: Line = (
            int(round(float(self._x[0, 0]))),
            int(round(y_top)),
            int(round(float(self._x[1, 0]))),
            int(round(y_bottom)),
        )
        right_smoothed: Line = (
            int(round(float(self._x[2, 0]))),
            int(round(y_top)),
            int(round(float(self._x[3, 0]))),
            int(round(y_bottom)),
        )
        return left_smoothed, right_smoothed

    def reset(self) -> None:
        self._initialized = False
        self._x.fill(0.0)
        self._P = np.eye(self._STATE_DIM, dtype=np.float32) * 100.0


def apply_lane_kalman(roi: RoadROI, smoother: LaneKalman) -> RoadROI:
    """Run the Kalman smoother on a RoadROI and return a smoothed copy."""

    left_smoothed, right_smoothed = smoother.update(
        roi.left_line,
        roi.right_line,
        roi.confidence,
    )

    if left_smoothed is None or right_smoothed is None:
        return roi

    height, width = roi.mask.shape[:2]
    polygon = np.array(
        [
            [left_smoothed[0], left_smoothed[1]],
            [right_smoothed[0], right_smoothed[1]],
            [right_smoothed[2], right_smoothed[3]],
            [left_smoothed[2], left_smoothed[3]],
        ],
        dtype=np.int32,
    )
    mask = _polygon_mask((height, width), polygon)
    return RoadROI(
        mask=mask,
        polygon=polygon,
        left_line=left_smoothed,
        right_line=right_smoothed,
        confidence=roi.confidence,
        detected=roi.detected,
    )
