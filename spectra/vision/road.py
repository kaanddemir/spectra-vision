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


def estimate_road_roi(frame_bgr: np.ndarray) -> RoadROI:
    """Estimate a lane-bounded road ROI with a fixed default."""

    height, width = frame_bgr.shape[:2]
    default = default_road_roi(frame_bgr.shape)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 160)

    search_mask = np.zeros((height, width), dtype=np.uint8)
    search_poly = np.array(
        [
            [int(width * 0.03), height - 1],
            [int(width * 0.39), int(height * 0.50)],
            [int(width * 0.61), int(height * 0.50)],
            [int(width * 0.97), height - 1],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(search_mask, [search_poly], 255)
    edges = cv2.bitwise_and(edges, search_mask)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(18, width // 35),
        minLineLength=max(24, width // 12),
        maxLineGap=max(12, width // 40),
    )
    if lines is None:
        return default

    left_points: list[tuple[int, int]] = []
    right_points: list[tuple[int, int]] = []
    for item in lines[:, 0]:
        x1, y1, x2, y2 = (int(v) for v in item)
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 2:
            continue
        slope = dy / float(dx)
        if abs(slope) < 0.35 or abs(slope) > 4.0:
            continue
        if max(y1, y2) < int(height * 0.56):
            continue
        center_x = (x1 + x2) / 2.0
        if slope < 0 and center_x < width * 0.58:
            left_points.extend([(x1, y1), (x2, y2)])
        elif slope > 0 and center_x > width * 0.42:
            right_points.extend([(x1, y1), (x2, y2)])

    y_top = int(height * 0.58)
    y_bottom = height - 1
    left_line = _fit_line(left_points, y_top, y_bottom, width)
    right_line = _fit_line(right_points, y_top, y_bottom, width)

    detected_count = int(left_line is not None) + int(right_line is not None)
    if detected_count == 0:
        return default

    left_line = left_line or default.left_line
    right_line = right_line or default.right_line
    if left_line is None or right_line is None:
        return default

    lx_top, ly_top, lx_bottom, ly_bottom = left_line
    rx_top, ry_top, rx_bottom, ry_bottom = right_line
    if rx_top - lx_top < width * 0.06 or rx_bottom - lx_bottom < width * 0.18:
        return default

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
    confidence = 0.55 if detected_count == 1 else 0.85
    return RoadROI(
        mask=mask,
        polygon=polygon,
        left_line=left_line,
        right_line=right_line,
        confidence=confidence,
        detected=True,
    )


def _fit_line(
    points: list[tuple[int, int]],
    y_top: int,
    y_bottom: int,
    width: int,
) -> Line | None:
    if len(points) < 4:
        return None

    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)
    if float(np.max(ys) - np.min(ys)) < 8.0:
        return None

    slope, intercept = np.polyfit(ys, xs, 1)
    x_top = int(np.clip((slope * y_top) + intercept, 0, width - 1))
    x_bottom = int(np.clip((slope * y_bottom) + intercept, 0, width - 1))
    return (x_top, y_top, x_bottom, y_bottom)


def _polygon_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    return mask.astype(bool)


_DEFAULT_VP_Y_FRACTION = 0.55
_VP_EMA_RISE = 0.40
_VP_EMA_FALL = 0.15


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
