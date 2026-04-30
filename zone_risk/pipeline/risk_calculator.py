"""TTC, direction, and risk-state calculation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class RiskEvent:
    frame_index: int
    timestamp_sec: float
    state: str
    ttc_sec: float | None
    direction: str
    zone: str
    object_type: str
    confidence: float
    near_score: float
    velocity_magnitude: float
    closing_speed: float
    bbox: tuple[int, int, int, int] | None
    reason: str
    object_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def zone_from_bbox(bbox: tuple[int, int, int, int], width: int) -> str:
    x1, _, x2, _ = bbox
    center_x = (x1 + x2) / 2.0
    if center_x < width / 3.0:
        return "left"
    if center_x > (2.0 * width) / 3.0:
        return "right"
    return "center"


def direction_from_flow(flow_x_mean: float) -> str:
    if abs(flow_x_mean) < 0.015:
        return "center"
    return "left" if flow_x_mean < 0.0 else "right"


def compute_ttc(near_score: float, closing_speed: float) -> float | None:
    """Compute pseudo-TTC from normalized nearness and normalized closing speed.

    The depth map used here is a nearness map: larger values mean closer. TTC
    therefore uses the inverse as a distance proxy.
    """

    if closing_speed <= 1e-3:
        return None
    distance_proxy = max(0.0, 1.0 - near_score)
    return round(float(distance_proxy / closing_speed), 2)


def classify_state(near_score: float, closing_speed: float, ttc_sec: float | None) -> str:
    if ttc_sec is not None and near_score >= 0.35 and ttc_sec < 1.0:
        return "DANGER"
    if ttc_sec is not None and near_score >= 0.25 and ttc_sec < 3.0:
        return "CAUTION"
    if near_score >= 0.72 and closing_speed >= 0.10:
        return "CAUTION"
    return "SAFE"


def score_event(event: RiskEvent) -> float:
    state_weight = {"SAFE": 0.0, "CAUTION": 1.0, "DANGER": 2.0}.get(event.state, 0.0)
    ttc_weight = 0.0 if event.ttc_sec is None else max(0.0, 3.0 - event.ttc_sec) / 3.0
    return state_weight + ttc_weight + event.near_score + event.closing_speed


def calculate_region_risk(
    *,
    frame_index: int,
    timestamp_sec: float,
    bbox: tuple[int, int, int, int],
    object_type: str,
    near_map: np.ndarray,
    magnitude_norm: np.ndarray,
    divergence_norm: np.ndarray,
    flow: np.ndarray,
    object_id: int | None = None,
    roi_mask: np.ndarray | None = None,
) -> RiskEvent:
    height, width = near_map.shape
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    safe_bbox = (x1, y1, x2, y2)

    if x2 <= x1 or y2 <= y1:
        near_score = 0.0
        velocity_magnitude = 0.0
        divergence = 0.0
        flow_x_mean = 0.0
    else:
        near_crop = near_map[y1:y2, x1:x2]
        velocity_crop = magnitude_norm[y1:y2, x1:x2]
        divergence_crop = divergence_norm[y1:y2, x1:x2]
        flow_x_crop = flow[y1:y2, x1:x2, 0]
        if roi_mask is not None:
            mask_crop = roi_mask[y1:y2, x1:x2]
            if mask_crop.shape == near_crop.shape:
                mask_crop = mask_crop.astype(bool)
            else:
                mask_crop = np.ones_like(near_crop, dtype=bool)
        else:
            mask_crop = np.ones_like(near_crop, dtype=bool)

        valid_pixels = int(np.count_nonzero(mask_crop))
        min_valid_pixels = max(16, int(mask_crop.size * 0.01))
        if valid_pixels < min_valid_pixels:
            near_score = 0.0
            velocity_magnitude = 0.0
            divergence = 0.0
            flow_x_mean = 0.0
        else:
            near_score = float(np.percentile(near_crop[mask_crop], 80))
            velocity_magnitude = float(np.percentile(velocity_crop[mask_crop], 80))
            divergence = float(np.percentile(divergence_crop[mask_crop], 80))
            flow_x_mean = float(np.mean(flow_x_crop[mask_crop]))

    closing_speed = float(np.clip((0.65 * velocity_magnitude) + (0.35 * divergence), 0.0, 1.0))
    ttc_sec = compute_ttc(near_score, closing_speed)
    state = classify_state(near_score, closing_speed, ttc_sec)
    direction = direction_from_flow(flow_x_mean)
    zone = zone_from_bbox(safe_bbox, width)
    confidence = float(np.clip((0.6 * near_score) + (0.4 * closing_speed), 0.0, 1.0))

    if state == "DANGER":
        reason = "near object with strong closing motion"
    elif state == "CAUTION":
        reason = "object may be approaching"
    else:
        reason = "no immediate closing risk"

    return RiskEvent(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        state=state,
        ttc_sec=ttc_sec,
        direction=direction,
        zone=zone,
        object_type=object_type,
        confidence=round(confidence, 3),
        near_score=round(near_score, 3),
        velocity_magnitude=round(velocity_magnitude, 3),
        closing_speed=round(closing_speed, 3),
        bbox=safe_bbox,
        reason=reason,
        object_id=object_id,
    )


class StateStabilizer:
    """Smoothes risk state transitions using consecutive frame counts (hysteresis)."""
    def __init__(self, upgrade_frames: int = 3, downgrade_frames: int = 5):
        self.current_state = "SAFE"
        self.pending_state = "SAFE"
        self.counter = 0
        self.upgrade_frames = upgrade_frames
        self.downgrade_frames = downgrade_frames

    def process(self, raw_state: str) -> str:
        if raw_state == self.current_state:
            self.pending_state = raw_state
            self.counter = 0
            return self.current_state

        if raw_state != self.pending_state:
            self.pending_state = raw_state
            self.counter = 1
        else:
            self.counter += 1

        # Determine if we should transition
        r_curr = self._rank(self.current_state)
        r_pend = self._rank(self.pending_state)
        
        required = self.upgrade_frames if r_pend > r_curr else self.downgrade_frames
        
        if self.counter >= required:
            self.current_state = self.pending_state
            self.counter = 0
            
        return self.current_state

    def _rank(self, state: str) -> int:
        return {"SAFE": 0, "CAUTION": 1, "DANGER": 2}.get(state, 0)


def select_primary_event(events: list[RiskEvent]) -> RiskEvent:
    if not events:
        raise ValueError("At least one risk event is required.")
    return max(events, key=score_event)
