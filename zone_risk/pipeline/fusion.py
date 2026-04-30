"""Fuse depth and optical flow into zone-based risk events."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterable

from ..vision.depth_estimator import DepthResult
from ..vision.optical_flow import FlowResult
from ..vision.road_roi import RoadROI
from .risk_calculator import RiskEvent, calculate_region_risk, select_primary_event


def compute_quick_risk(flow: FlowResult, width: int, height: int) -> float:
    """Estimate risk using smooth spatial importance and proximity."""
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    center_x = width / 2.0
    
    # 1. Smooth Horizontal Importance (Bell Curve)
    # Highest at center, dropping to ~0.4 at far edges
    sigma_x = width / 3.0
    horiz_weight = np.exp(-((x_coords - center_x) ** 2) / (2 * (sigma_x ** 2)))
    horiz_weight = 0.4 + 0.6 * horiz_weight # Range [0.4, 1.0]
    
    # 2. Smooth Collision Cone (Soft Edges)
    # Linear width from horizon (y=0) to bottom (y=height)
    alpha = (height - y_coords) / float(height)
    cone_half_width = ((0.05 * width) + (0.20 * width) * (1.0 - alpha))
    
    # Distance from center normalized by cone width
    dist_from_center = np.abs(x_coords - center_x)
    # Smooth step: 1.0 inside, falling off to 0.3 outside
    cone_weight = np.exp(-(dist_from_center ** 2) / (2 * (cone_half_width ** 2)))
    cone_weight = 0.3 + 0.7 * cone_weight # Range [0.3, 1.0]
    
    # 3. Combine spatial weights
    spatial_priority = horiz_weight * cone_weight
    
    # 4. Apply to motion signal
    motion_signal = 0.65 * flow.magnitude_norm + 0.35 * flow.divergence_norm
    weighted_motion = motion_signal * spatial_priority
    
    # 5. Proximity Factor (Vertical position)
    prox_map = y_coords / float(height) # 1.0 at bottom, 0.0 at top
    
    # 6. Combined Heuristic Risk Map
    # 0.7 Motion + 0.2 Spatial + 0.1 Proximity
    # (Simplified into a single importance-weighted motion check)
    final_risk_map = weighted_motion * (0.9 + 0.1 * prox_map)
    
    return float(np.percentile(final_risk_map, 90))


@dataclass(frozen=True)
class ZoneRegion:
    bbox: tuple[int, int, int, int]
    label: str
    id: int


def fuse_frame_risk(
    *,
    frame_index: int,
    timestamp_sec: float,
    depth: DepthResult,
    flow: FlowResult,
    road_roi: RoadROI | None = None,
) -> tuple[RiskEvent, list[RiskEvent]]:
    height, width = depth.near_map.shape
    regions = list(zone_regions(width, height))

    events = [
        calculate_region_risk(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            bbox=region.bbox,
            object_type=region.label,
            near_map=depth.near_map,
            magnitude_norm=flow.magnitude_norm,
            divergence_norm=flow.divergence_norm,
            flow=flow.flow,
            object_id=region.id,
            roi_mask=None if road_roi is None else road_roi.mask,
        )
        for region in regions
    ]

    return select_primary_event(events), events


def zone_regions(width: int, height: int) -> Iterable[ZoneRegion]:
    # Left 25%, Center 50%, Right 25%
    w25 = int(width * 0.25)
    w75 = int(width * 0.75)
    
    yield ZoneRegion((0, 0, w25, height), "left zone", 101)
    yield ZoneRegion((w25, 0, w75, height), "center zone", 102)
    yield ZoneRegion((w75, 0, width, height), "right zone", 103)
