"""Fuse YOLO detections + tracker history + depth/flow into risk events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..vision.depth_estimator import DepthResult
from ..vision.optical_flow import FlowResult
from ..vision.road_geometry import LaneFrame
from .risk_calculator import (
    DepthDeltaSmoother,
    ExpansionSmoother,
    RiskEvent,
    calculate_track_risk,
    expansion_rate_from_track,
    make_safe_event,
)
from .tracker import Track


@dataclass(frozen=True)
class SpatialFields:
    depth: DepthResult
    flow: FlowResult
    lane: LaneFrame
    flow_dt_sec: float
    depth_is_fresh: bool


def compute_quick_risk(flow: FlowResult, width: int, height: int) -> float:
    """Frame-level motion risk used to decide when to recompute depth.

    This is intentionally cheap — it does not classify state by itself, just
    flags frames where the scene is busy enough that a fresh depth estimate
    is worth the inference cost.
    """

    motion_signal = (0.65 * flow.magnitude_norm) + (0.35 * flow.divergence_norm)
    if motion_signal.size == 0:
        return 0.0
    return float(np.percentile(motion_signal, 90))


def build_object_events(
    *,
    frame_index: int,
    timestamp_sec: float,
    tracks: list[Track],
    fields: SpatialFields,
    expansion_smoother: ExpansionSmoother,
    depth_smoother: DepthDeltaSmoother,
) -> tuple[RiskEvent, list[RiskEvent]]:
    """Build per-object risk events plus a primary event for the frame.

    Returns the highest-scoring event and the full list. When no tracks
    survive, a synthetic SAFE event is returned so downstream consumers can
    still render a frame.
    """

    if not tracks:
        expansion_smoother.forget(set())
        depth_smoother.forget(set())
        safe = make_safe_event(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
        )
        return safe, [safe]

    events: list[RiskEvent] = []
    active_ids: set[int] = set()
    for track in tracks:
        active_ids.add(track.track_id)
        raw_rate = expansion_rate_from_track(track)
        expansion_rate = expansion_smoother.update(track.track_id, raw_rate)
        event = calculate_track_risk(
            track=track,
            near_map=fields.depth.near_map,
            flow=fields.flow.flow,
            magnitude_norm=fields.flow.magnitude_norm,
            lane=fields.lane,
            expansion_rate=expansion_rate,
            depth_history=depth_smoother.state,
            flow_dt_sec=fields.flow_dt_sec,
            depth_is_fresh=fields.depth_is_fresh,
        )
        events.append(event)

    expansion_smoother.forget(active_ids)
    depth_smoother.forget(active_ids)

    primary = max(events, key=lambda e: (
        {"SAFE": 0.0, "CAUTION": 1.0, "DANGER": 2.0}.get(e.state, 0.0)
        + (0.0 if e.ttc_sec is None else max(0.0, 3.0 - e.ttc_sec) / 3.0)
        + e.closing_speed
        + (0.5 * e.crossing_risk)
    ))
    return primary, events
