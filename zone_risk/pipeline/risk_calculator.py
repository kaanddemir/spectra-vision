"""TTC, direction, and risk-state calculation."""

from __future__ import annotations

from dataclasses import dataclass, replace
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


_TTC_MIN_CLOSING_SPEED = 0.08
_TTC_MIN_NEAR_SCORE = 0.20
_TTC_DISTANCE_PROXY_FLOOR = 0.18
_TTC_MAX_REPORTED_SEC = 9.9


def compute_ttc(near_score: float, closing_speed: float) -> float | None:
    """Compute pseudo-TTC from normalized nearness and normalized closing speed.

    The depth map used here is a nearness map: larger values mean closer. TTC
    therefore uses the inverse as a distance proxy. Returns ``None`` when
    evidence is too weak to trust — keeps the displayed TTC consistent with
    the classified state instead of producing phantom values like 30 s during
    momentary flow drop-outs.
    """

    if closing_speed < _TTC_MIN_CLOSING_SPEED:
        return None
    if near_score < _TTC_MIN_NEAR_SCORE:
        return None
    # Relative monocular depth cannot prove a literal zero-distance impact.
    # Keep a small floor so the UI never reports a misleading 0.0s TTC from a
    # saturated nearness pixel.
    distance_proxy = max(_TTC_DISTANCE_PROXY_FLOOR, 1.0 - near_score)
    ttc = float(distance_proxy / closing_speed)
    if ttc > _TTC_MAX_REPORTED_SEC:
        return None
    return round(ttc, 2)


def classify_state(near_score: float, closing_speed: float, ttc_sec: float | None) -> str:
    # When TTC is reported, it is already gated by compute_ttc's evidence
    # checks, so it is sufficient on its own to drive the classification.
    if ttc_sec is not None:
        if ttc_sec < 1.0:
            return "DANGER"
        if ttc_sec < 3.0:
            return "CAUTION"
    if near_score >= 0.72 and closing_speed >= 0.10:
        return "CAUTION"
    return "SAFE"


_STATE_SCORE = {"SAFE": 0.0, "CAUTION": 1.0, "DANGER": 2.0}
_MAX_CLOSING_FLOW_PX = 25.0
_LOCAL_MOTION_CONTRAST_GAIN = 2.8
_SIDE_LANE_IMMEDIATE_RELEVANCE = 0.55
_METRIC_EMA_RISE_ALPHA = 0.55
_METRIC_EMA_FALL_ALPHA = 0.30


def score_raw(state: str, ttc_sec: float | None, near_score: float, closing_speed: float) -> float:
    ttc_weight = 0.0 if ttc_sec is None else max(0.0, 3.0 - ttc_sec) / 3.0
    return _STATE_SCORE.get(state, 0.0) + ttc_weight + near_score + closing_speed


def score_event(event: RiskEvent) -> float:
    return score_raw(event.state, event.ttc_sec, event.near_score, event.closing_speed)


def _risk_reason(state: str) -> str:
    if state == "DANGER":
        return "near object with strong closing motion"
    if state == "CAUTION":
        return "object may be approaching"
    return "no immediate closing risk"


def _ema(prev: float, value: float, rise_alpha: float, fall_alpha: float) -> float:
    alpha = rise_alpha if value > prev else fall_alpha
    return float((alpha * value) + ((1.0 - alpha) * prev))


class MetricEmaSmoother:
    """Smooth per-zone motion metrics before TTC/state classification."""

    def __init__(
        self,
        *,
        rise_alpha: float = _METRIC_EMA_RISE_ALPHA,
        fall_alpha: float = _METRIC_EMA_FALL_ALPHA,
    ) -> None:
        self.rise_alpha = rise_alpha
        self.fall_alpha = fall_alpha
        self._metrics: dict[Any, tuple[float, float]] = {}

    def smooth_event(self, event: RiskEvent) -> RiskEvent:
        key = event.object_id if event.object_id is not None else (event.zone, event.object_type)
        previous = self._metrics.get(key)
        if previous is None:
            near_score = float(event.near_score)
            closing_speed = float(event.closing_speed)
        else:
            near_score = _ema(previous[0], float(event.near_score), self.rise_alpha, self.fall_alpha)
            closing_speed = _ema(previous[1], float(event.closing_speed), self.rise_alpha, self.fall_alpha)

        near_score = float(np.clip(near_score, 0.0, 1.0))
        closing_speed = float(np.clip(closing_speed, 0.0, 1.0))
        self._metrics[key] = (near_score, closing_speed)

        ttc_sec = compute_ttc(near_score, closing_speed)
        state = classify_state(near_score, closing_speed, ttc_sec)
        if (
            event.state == "DANGER"
            and state != "DANGER"
            and near_score >= 0.30
            and closing_speed >= 0.35
        ):
            state = "DANGER"
            ttc_sec = min(ttc_sec if ttc_sec is not None else 1.0, 0.9)
        elif event.state == "CAUTION" and state == "SAFE" and near_score >= 0.25 and closing_speed >= 0.25:
            state = "CAUTION"

        confidence = float(np.clip((0.6 * near_score) + (0.4 * closing_speed), 0.0, 1.0))
        return replace(
            event,
            state=state,
            ttc_sec=ttc_sec,
            confidence=round(confidence, 3),
            near_score=round(near_score, 3),
            closing_speed=round(closing_speed, 3),
            reason=_risk_reason(state),
        )

    def smooth_events(self, events: list[RiskEvent]) -> list[RiskEvent]:
        return [self.smooth_event(event) for event in events]


def is_imminent_danger(event: RiskEvent) -> bool:
    """Whether an event is urgent enough to bypass upgrade hysteresis."""

    return (
        event.state == "DANGER"
        and event.ttc_sec is not None
        and event.ttc_sec <= 1.0
    )


def is_clear_safe_event(event: RiskEvent) -> bool:
    """Whether current evidence is safe enough to clear a held alert state."""

    safe_ttc = event.ttc_sec is None or event.ttc_sec >= 3.0
    return (
        event.state == "SAFE"
        and safe_ttc
        and event.near_score < 0.45
        and event.closing_speed < 0.25
    )


def stabilized_event_state(stabilizer: "StateStabilizer", event: RiskEvent) -> str:
    """Apply hysteresis without showing stale severe states with safe metrics."""

    if is_imminent_danger(event):
        stabilizer.current_state = "DANGER"
        stabilizer.pending_state = "DANGER"
        stabilizer.counter = 0
        return "DANGER"

    if event.ttc_sec is not None and event.ttc_sec < 3.0 and event.state == "CAUTION":
        stabilizer.current_state = "CAUTION"
        stabilizer.pending_state = "CAUTION"
        stabilizer.counter = 0
        return "CAUTION"

    if is_clear_safe_event(event):
        stabilizer.current_state = "SAFE"
        stabilizer.pending_state = "SAFE"
        stabilizer.counter = 0
        return "SAFE"

    stabilized = stabilizer.process(event.state)
    if _STATE_SCORE.get(stabilized, 0.0) > _STATE_SCORE.get(event.state, 0.0):
        return event.state
    return stabilized


def _approach_motion_crop(
    flow: np.ndarray,
    divergence_norm: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Estimate object approach from radial expansion, not raw flow magnitude."""

    height, width = divergence_norm.shape
    x1, y1, x2, y2 = bbox
    if flow.shape[:2] != (height, width):
        return np.zeros((max(0, y2 - y1), max(0, x2 - x1)), dtype=np.float32)

    y_coords, x_coords = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    focus_x = (width - 1) / 2.0
    focus_y = height * 0.55
    radial_x = x_coords - focus_x
    radial_y = y_coords - focus_y
    radial_norm = np.sqrt((radial_x * radial_x) + (radial_y * radial_y))
    radial_norm = np.maximum(radial_norm, 1.0)

    flow_crop = flow[y1:y2, x1:x2]
    divergence_crop = divergence_norm[y1:y2, x1:x2]
    outward_flow = ((flow_crop[..., 0] * radial_x) + (flow_crop[..., 1] * radial_y)) / radial_norm
    outward_norm = np.clip(outward_flow / _MAX_CLOSING_FLOW_PX, 0.0, 1.0)

    # A lateral pan can have large outward components on one side of the frame.
    # Require positive divergence too, so sideways road flow is not treated as
    # an approaching object.
    approach = np.sqrt(outward_norm * np.clip(divergence_crop, 0.0, 1.0))

    alpha = (height - y_coords) / float(max(height, 1))
    cone_half_width = (0.06 * width) + (0.26 * width) * (1.0 - alpha)
    dist_from_center = np.abs(x_coords - focus_x)
    cone_weight = np.exp(-(dist_from_center * dist_from_center) / (2.0 * cone_half_width * cone_half_width))
    cone_weight = 0.20 + (0.80 * cone_weight)

    return np.clip(approach * cone_weight, 0.0, 1.0).astype(np.float32)


def _collision_relevance_crop(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Return how strongly pixels belong to the forward collision corridor."""

    height, width = shape
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return np.zeros((max(0, y2 - y1), max(0, x2 - x1)), dtype=np.float32)

    y_coords, x_coords = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    focus_x = (width - 1) / 2.0
    alpha = (height - y_coords) / float(max(height, 1))
    cone_half_width = (0.05 * width) + (0.24 * width) * (1.0 - alpha)
    cone_half_width = np.maximum(cone_half_width, 1.0)
    dist_from_center = np.abs(x_coords - focus_x)
    cone = np.exp(-(dist_from_center * dist_from_center) / (2.0 * cone_half_width * cone_half_width))

    lower_frame_weight = np.clip((y_coords - (0.38 * height)) / max(0.62 * height, 1.0), 0.0, 1.0)
    relevance = (0.12 + (0.88 * cone)) * (0.35 + (0.65 * lower_frame_weight))
    return np.clip(relevance, 0.0, 1.0).astype(np.float32)


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
        approach_motion = 0.0
        local_motion_contrast = 0.0
        collision_relevance = 0.0
        flow_x_mean = 0.0
    else:
        near_crop = near_map[y1:y2, x1:x2]
        velocity_crop = magnitude_norm[y1:y2, x1:x2]
        divergence_crop = divergence_norm[y1:y2, x1:x2]
        approach_crop = _approach_motion_crop(flow, divergence_norm, safe_bbox)
        relevance_crop = _collision_relevance_crop((height, width), safe_bbox)
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
            approach_motion = 0.0
            local_motion_contrast = 0.0
            collision_relevance = 0.0
            flow_x_mean = 0.0
        else:
            valid_near = near_crop[mask_crop]
            valid_velocity = velocity_crop[mask_crop]
            valid_divergence = divergence_crop[mask_crop]
            valid_approach = approach_crop[mask_crop]
            valid_relevance = relevance_crop[mask_crop]

            velocity_p50 = float(np.percentile(valid_velocity, 50))
            velocity_p92 = float(np.percentile(valid_velocity, 92))

            near_score = float(np.percentile(valid_near, 80))
            velocity_magnitude = float(np.percentile(valid_velocity, 80))
            divergence = float(np.percentile(valid_divergence, 80))
            approach_motion = float(np.percentile(valid_approach, 85))
            collision_relevance = float(np.percentile(valid_relevance, 85))
            local_motion_contrast = float(
                np.clip(
                    (velocity_p92 - velocity_p50) * _LOCAL_MOTION_CONTRAST_GAIN,
                    0.0,
                    1.0,
                )
            )
            motion_mask = valid_velocity >= max(velocity_p92, velocity_p50 + 0.08)
            if local_motion_contrast >= 0.12 and int(np.count_nonzero(motion_mask)) >= min_valid_pixels:
                motion_near_score = float(np.percentile(valid_near[motion_mask], 80))
                near_score = max(near_score, 0.90 * motion_near_score)
                collision_relevance = float(np.percentile(valid_relevance[motion_mask], 80))
            flow_x_mean = float(np.mean(flow_x_crop[mask_crop]))

    near_motion_gate = float(np.clip((near_score - 0.24) / 0.26, 0.0, 1.0))
    impact_motion = local_motion_contrast * near_motion_gate
    closing_speed = float(
        np.clip(
            (0.62 * approach_motion)
            + (0.18 * divergence)
            + (0.20 * impact_motion),
            0.0,
            1.0,
        )
    )
    relevance_gate = 0.45 + (0.55 * collision_relevance)
    closing_speed = float(np.clip(closing_speed * relevance_gate, 0.0, 1.0))
    if impact_motion >= 0.25 and near_score >= 0.28:
        closing_speed = max(
            closing_speed,
            float(
                np.clip(
                    ((0.48 * impact_motion) + (0.18 * velocity_magnitude)) * relevance_gate,
                    0.0,
                    1.0,
                )
            ),
        )
    ttc_sec = compute_ttc(near_score, closing_speed)
    if (
        ttc_sec is not None
        and ttc_sec < 1.0
        and collision_relevance < _SIDE_LANE_IMMEDIATE_RELEVANCE
    ):
        ttc_sec = 1.0
    state = classify_state(near_score, closing_speed, ttc_sec)
    if (
        state != "DANGER"
        and near_score >= 0.30
        and impact_motion >= 0.35
        and closing_speed >= 0.35
        and collision_relevance >= _SIDE_LANE_IMMEDIATE_RELEVANCE
    ):
        state = "DANGER"
        ttc_sec = min(ttc_sec if ttc_sec is not None else 1.0, 0.9)
    elif state == "SAFE" and near_score >= 0.25 and impact_motion >= 0.25 and closing_speed >= 0.25:
        state = "CAUTION"
    direction = direction_from_flow(flow_x_mean)
    zone = zone_from_bbox(safe_bbox, width)
    confidence = float(np.clip((0.6 * near_score) + (0.4 * closing_speed), 0.0, 1.0))

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
        reason=_risk_reason(state),
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
