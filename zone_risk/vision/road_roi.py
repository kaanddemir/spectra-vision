"""Road/lane ROI estimation for zone-based risk scoring."""

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


def fallback_road_roi(shape: tuple[int, int] | tuple[int, int, int]) -> RoadROI:
    """Return the fixed perspective ROI used as a stable fallback."""

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
    """Estimate a lane-bounded road ROI with a fixed fallback."""

    height, width = frame_bgr.shape[:2]
    fallback = fallback_road_roi(frame_bgr.shape)
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
        return fallback

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
        return fallback

    left_line = left_line or fallback.left_line
    right_line = right_line or fallback.right_line
    if left_line is None or right_line is None:
        return fallback

    lx_top, ly_top, lx_bottom, ly_bottom = left_line
    rx_top, ry_top, rx_bottom, ry_bottom = right_line
    if rx_top - lx_top < width * 0.06 or rx_bottom - lx_bottom < width * 0.18:
        return fallback

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
