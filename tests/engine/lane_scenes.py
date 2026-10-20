"""Synthetic comma10k-style class masks rendered from a pinhole camera over a flat road with known lanes.

For geometry tests only: they check the arithmetic against ground truth, not real-world accuracy.
"""
from __future__ import annotations

import numpy as np

from spectra.engine.lanes import EGO, MARKING, ROAD, UNDRIVABLE

W, H = 512, 384


def render(camera_x: float = 0.0, lane_width: float = 3.6, height: float = 1.3, yaw_deg: float = 0.0,
           pitch_deg: float = 0.0, roll_deg: float = 0.0, focal: float = 300.0, curvature: float = 0.0,
           boundaries: int = 4, dashed: bool = True, hood_rows: int = 40, phase: float = 0.0,
           principal_x: float = W / 2, missing: tuple[int, ...] = ()) -> np.ndarray:
    """Lane boundaries at k * lane_width (k = -boundaries/2 .. boundaries/2 - 1 relative to the road
    centre line through 0 ... ) seen from a camera ``camera_x`` metres right of a boundary-centred lane.
    The road bends as X += curvature * Z**2 / 2. Returns a uint8 class mask."""
    mask = np.full((H, W), UNDRIVABLE, np.uint8)
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
    edges = [(-boundaries // 2 + k + 0.5) * lane_width for k in range(boundaries + 1)]
    z = np.linspace(2.0, 120.0, 6000)
    road_lo, road_hi = edges[0], edges[-1]

    def project(X, Z):
        # world: x right, y down, z forward; camera at (camera_x, -height, 0), yawed right by yaw
        xc, yc, zc = X - camera_x, np.full_like(Z, height), Z
        x1 = np.cos(yaw) * xc - np.sin(yaw) * zc
        z1 = np.sin(yaw) * xc + np.cos(yaw) * zc
        y2 = np.cos(pitch) * yc - np.sin(pitch) * z1
        z2 = np.sin(pitch) * yc + np.cos(pitch) * z1
        u, v = focal * x1 / z2, focal * y2 / z2
        ur = np.cos(roll) * u - np.sin(roll) * v
        vr = np.sin(roll) * u + np.cos(roll) * v
        return ur + principal_x, vr + H * 0.45, z2

    bend = curvature * z ** 2 / 2
    for X0 in (road_lo, road_hi):
        pass
    xl, yl, _ = project(road_lo + bend, z)
    xr, yr, _ = project(road_hi + bend, z)
    import cv2
    poly = np.vstack([np.c_[xl, yl], np.c_[xr, yr][::-1]]).astype(np.int32)
    cv2.fillPoly(mask, [poly], ROAD)
    for k, X0 in enumerate(edges):
        if k in missing:
            continue
        x, y, _ = project(X0 + bend, z)
        on = ((z + phase) % 12.0 < 4.0) if dashed and 0 < k < len(edges) - 1 else np.ones_like(z, bool)
        width_px = np.clip(0.15 * focal / z, 1, 6)
        for xi, yi, wi, o in zip(x, y, width_px, on):
            if o and 0 <= yi < H:
                cv2.circle(mask, (int(round(xi)), int(round(yi))), int(round(wi / 2)), MARKING, -1)
    if hood_rows:
        mask[H - hood_rows:] = EGO
    return mask


def truth_offset(camera_x: float, lane_width: float = 3.6, boundaries: int = 4) -> float:
    """Camera position in lane widths relative to the centre of the lane it occupies, + right."""
    edges = [(-boundaries // 2 + k + 0.5) * lane_width for k in range(boundaries + 1)]
    left = max(e for e in edges if e <= camera_x)
    return (camera_x - (left + lane_width / 2)) / lane_width
