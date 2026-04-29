"""Fuse depth, optical flow, and object detections into risk events."""

from __future__ import annotations

from .depth_estimator import DepthResult
from .optical_flow import FlowResult
from .risk_calculator import RiskEvent, calculate_region_risk, select_primary_event
from .vehicle_detector import Detection, zone_detections


def fuse_frame_risk(
    *,
    frame_index: int,
    timestamp_sec: float,
    depth: DepthResult,
    flow: FlowResult,
    detections: list[Detection],
) -> tuple[RiskEvent, list[RiskEvent]]:
    height, width = depth.near_map.shape
    regions = detections if detections else list(zone_detections(width, height))

    events = [
        calculate_region_risk(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            bbox=region.bbox,
            object_type=region.label,
            detector_confidence=region.confidence,
            near_map=depth.near_map,
            magnitude_norm=flow.magnitude_norm,
            divergence_norm=flow.divergence_norm,
            flow=flow.flow,
            object_id=region.id,
        )
        for region in regions
    ]

    return select_primary_event(events), events

