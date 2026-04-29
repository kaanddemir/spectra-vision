"""Fuse depth and optical flow into zone-based risk events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..vision.depth_estimator import DepthResult
from ..vision.optical_flow import FlowResult
from .risk_calculator import RiskEvent, calculate_region_risk, select_primary_event


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
        )
        for region in regions
    ]

    return select_primary_event(events), events


def zone_regions(width: int, height: int) -> Iterable[ZoneRegion]:
    third = width // 3
    yield ZoneRegion((0, 0, third, height), "left zone", 101)
    yield ZoneRegion((third, 0, 2 * third, height), "center zone", 102)
    yield ZoneRegion((2 * third, 0, width, height), "right zone", 103)
