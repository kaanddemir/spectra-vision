"""FastAPI-facing spatial-awareness risk analysis adapter."""

from __future__ import annotations

import base64
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np

from .risk import (
    _ETA_SURFACE_MIN_CROSSING,
    ConfidenceSmoother,
    DepthDeltaSmoother,
    ExpansionSmoother,
    RiskEvent,
    RiskSensitivity,
    SpatialFields,
    StateStabilizer,
    TtcImminenceSmoother,
    build_object_events,
    compute_quick_risk,
    resolve_sensitivity,
    score_event,
    stabilized_event_state,
    ttc_score,
)
from ..vision.depth import DepthResult, estimate_frame_depth
from ..vision.detection import get_detector, is_yolo_available
from ..vision.lanenet import UFLDv2ONNX, get_lanenet_model, is_lanenet_available
from ..vision.models import get_depth_model, is_depth_available
from ..vision.motion import compute_velocity
from ..vision.preprocessing import preprocess_frame
from ..vision.traffic_light import frame_light_state
from ..vision.road import (
    LaneFrame,
    LaneKalman,
    RoadROI,
    apply_lane_kalman,
    build_lane_frame,
    default_road_roi,
    estimate_road_roi_from_lanes,
    filter_relevant_detections,
)
from .tracking import IoUTracker
from .overlay import _lane_corridor, annotate_frame



@dataclass(frozen=True)
class VideoFrame:
    frame_index: int
    timestamp_sec: float
    bgr: np.ndarray


_PERFORMANCE_STAGES = ("preprocess", "lane", "flow", "depth", "yolo")


def _empty_performance_stats() -> dict[str, dict[str, list[float]]]:
    return {stage: {"active": [], "frame": []} for stage in _PERFORMANCE_STAGES}


def _record_stage(
    stats: dict[str, dict[str, list[float]]],
    stage: str,
    elapsed_sec: float,
    *,
    active: bool = True,
) -> None:
    elapsed_ms = max(0.0, elapsed_sec * 1000.0)
    stats[stage]["frame"].append(elapsed_ms)
    if active:
        stats[stage]["active"].append(elapsed_ms)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((percentile / 100.0) * len(ordered)) - 1))
    return ordered[idx]


def _sample_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "avg_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "count": len(values),
        "avg_ms": sum(values) / len(values),
        "p95_ms": _percentile(values, 95.0),
        "max_ms": max(values),
    }


def _build_performance_summary(
    stats: dict[str, dict[str, list[float]]],
    *,
    processed_frames: int,
    elapsed_sec: float,
    lane_every: int,
    flow_every: int,
    depth_every: int,
    adaptive_depth: bool,
    detect_every: int,
    depth_refresh: dict[str, int] | None = None,
) -> dict[str, Any]:
    stages: dict[str, dict[str, Any]] = {}
    for stage in _PERFORMANCE_STAGES:
        active = _sample_summary(stats[stage]["active"])
        frame = _sample_summary(stats[stage]["frame"])
        stages[stage] = {"active": active, "frame": frame}

    bottleneck_stage = max(
        _PERFORMANCE_STAGES,
        key=lambda stage: float(stages[stage]["frame"]["avg_ms"] or 0.0),
    )
    refresh = depth_refresh or {
        "runs": 0,
        "skips": processed_frames,
        "initial_runs": 0,
        "periodic_runs": 0,
        "motion_triggered_runs": 0,
        "cooldown_frames": max(3, depth_every // 2),
    }
    refresh.setdefault("cooldown_frames", max(3, depth_every // 2))
    refresh["effective_interval_frames"] = (
        processed_frames / refresh["runs"] if refresh["runs"] > 0 else None
    )

    return {
        "processed_frames": processed_frames,
        "elapsed_sec": elapsed_sec,
        "effective_fps": processed_frames / elapsed_sec if elapsed_sec > 0.0 else 0.0,
        "bottleneck": {
            "stage": bottleneck_stage,
            "active_avg_ms": stages[bottleneck_stage]["active"]["avg_ms"],
            "frame_avg_ms": stages[bottleneck_stage]["frame"]["avg_ms"],
        },
        "sampling": {
            "lane_every": lane_every,
            "flow_every": flow_every,
            "depth_every": depth_every,
            "adaptive_depth": adaptive_depth,
            "detect_every": detect_every,
        },
        "depth_refresh": refresh,
        "stages": stages,
    }


def _fmt_ms(value: float | int | None) -> str:
    return "-" if value is None else f"{float(value):.0f}"


def _fmt_frames(value: float | int | None) -> str:
    return "-" if value is None else f"{float(value):.1f}"


def _format_performance_summary(summary: dict[str, Any]) -> list[str]:
    bottleneck = summary["bottleneck"]
    sampling = summary["sampling"]
    depth_refresh = summary["depth_refresh"]
    lines = [
        "SUMMARY",
        (
            f"Processed: {summary['processed_frames']} frames in {summary['elapsed_sec']:.2f}s "
            f"| Effective FPS: {summary['effective_fps']:.1f}"
        ),
        (
            f"Bottleneck: {bottleneck['stage']} "
            f"active_avg={_fmt_ms(bottleneck['active_avg_ms'])}ms "
            f"frame_avg={_fmt_ms(bottleneck['frame_avg_ms'])}ms"
        ),
        "Stage active avg/p95/max | frame avg:",
    ]
    for stage in _PERFORMANCE_STAGES:
        stage_stats = summary["stages"][stage]
        active = stage_stats["active"]
        frame = stage_stats["frame"]
        lines.append(
            f"  {stage:<10} "
            f"{_fmt_ms(active['avg_ms'])}/{_fmt_ms(active['p95_ms'])}/{_fmt_ms(active['max_ms'])}ms "
            f"| frame_avg={_fmt_ms(frame['avg_ms'])}ms"
        )
    lines.append(
        "Depth refresh: "
        f"runs={depth_refresh['runs']} "
        f"skips={depth_refresh['skips']} "
        f"initial={depth_refresh['initial_runs']} "
        f"periodic={depth_refresh['periodic_runs']} "
        f"motion={depth_refresh['motion_triggered_runs']} "
        f"cooldown={depth_refresh['cooldown_frames']} "
        f"effective_interval={_fmt_frames(depth_refresh['effective_interval_frames'])}f"
    )
    lines.append(
        "Sampling: "
        f"lane_every={sampling['lane_every']} "
        f"depth_every={sampling['depth_every']} "
        f"adaptive_depth={'on' if sampling['adaptive_depth'] else 'off'} "
        f"detect_every={sampling['detect_every']} "
        f"flow_every={sampling['flow_every']}"
    )
    return lines


class VideoLoader:
    """Small wrapper around OpenCV video capture."""

    def __init__(
        self,
        source: str | Path,
        max_frames: int | None = None,
        start_sec: float = 0.0,
        end_sec: float | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> None:
        self.source = str(source)
        self.max_frames = max_frames
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.requested_start_frame = max(0, int(start_frame or 0))
        self.requested_end_frame = int(end_frame) if end_frame is not None else None
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        self.start_frame = self.requested_start_frame
        if self.start_frame <= 0 and self.start_sec > 0 and self.fps > 0:
            self.start_frame = int(self.start_sec * self.fps)
        if self.frame_count > 0:
            self.start_frame = min(self.start_frame, max(0, self.frame_count - 1))
        if self.start_frame > 0:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.end_frame = self.requested_end_frame
        if self.end_frame is not None:
            self.end_frame = max(self.start_frame, self.end_frame)
            if self.frame_count > 0:
                self.end_frame = min(self.end_frame, max(0, self.frame_count - 1))

    def frames(self) -> Iterator[VideoFrame]:
        frame_index = self.start_frame
        try:
            while self.max_frames is None or (frame_index - self.start_frame) < self.max_frames:
                if self.end_frame is not None and frame_index > self.end_frame:
                    break
                if self.end_sec and self.end_sec > 0:
                    current_sec = frame_index / self.fps if self.fps > 0 else frame_index
                    if current_sec > self.end_sec:
                        break

                ok, frame = self.capture.read()
                if not ok:
                    break

                timestamp_sec = frame_index / self.fps if self.fps > 0.0 else float(frame_index)
                yield VideoFrame(frame_index=frame_index, timestamp_sec=timestamp_sec, bgr=frame)
                frame_index += 1
        finally:
            self.close()

    def close(self) -> None:
        self.capture.release()


def _ensure_required_models() -> None:
    """Verify the hard-required vision backends are loadable.

    Depth, YOLO and UFLDv2 are all required. Hough lane detection was
    removed after benchmarking showed it traces the road's outer edges
    instead of the ego corridor on real dashcam footage — a misleading
    fallback is worse than a clear startup failure. Optical flow is
    classical (DIS) so it always works.
    """

    if not is_depth_available():
        raise RuntimeError(
            "Depth Anything metric ONNX model missing at "
            "models/depth_anything_v2_metric_vkitti_vits.onnx"
        )
    if not is_yolo_available():
        raise RuntimeError(
            "YOLOv8 detector unavailable. Install Ultralytics "
            "(`pip install -r requirements.txt`) and ensure "
            "models/yolov8n.pt exists."
        )
    if not is_lanenet_available():
        raise RuntimeError(
            "UFLDv2 ONNX model missing at models/ufld_v2_culane_r18.onnx. "
            "See spectra/vision/lanenet.py for export instructions."
        )
    # Eagerly load each backend so any post-file-check failure (corrupt
    # model, ONNX runtime mismatch, CoreML init crash) surfaces here
    # instead of mid-video.
    get_depth_model()
    get_lanenet_model()
    get_detector()


def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


_ETA_HORIZON_SEC = 10.0
_ETA_LOW_CONFIDENCE = 0.12
_MIN_CLOSING_FOR_DISPLAY_MPS = 0.30


# A near, in-corridor object: a close vehicle in (or one lane off) the ego path.
# Used to (a) gate the detector's near-band recovery pass off while a close lead
# is healthily tracked, and (b) decide when a CAUTION/DANGER primary is worth
# remembering as a "strong" threat for the dropout safety net.
_NEAR_THREAT_DISTANCE_M = 14.0
_NEAR_THREAT_LANE_POS = 1.3
# How long the banner is held at its prior band after a near threat drops out of
# the active set, so a brief detector miss cannot flip the frame to SAFE.
_STRONG_PRIMARY_HOLD_SEC = 1.0

# Geometry escape hatch: a big vehicle filling the lower frame is effectively
# ahead of us even when its computed ``lane_position`` snaps to the ±1.5 clamp
# (a tailgated lead car straddling the bottom-center reads far-left/right when
# the corridor geometry is uncertain). A box this wide and this bottom-anchored
# is a close lead regardless of lane bucket; a genuine side car is a narrow box
# near the edge and does not pass the width test.
_NEAR_THREAT_BOX_BOTTOM_FRAC = 0.90
_NEAR_THREAT_BOX_WIDTH_FRAC = 0.33


def _box_fills_lower_frame(event: RiskEvent, frame_shape: tuple[int, int] | None) -> bool:
    """Whether the bbox is wide and bottom-anchored (a big lead filling our path)."""

    if frame_shape is None or event.bbox is None:
        return False
    frame_h, frame_w = frame_shape
    if frame_h <= 0 or frame_w <= 0:
        return False
    x1, _y1, x2, y2 = event.bbox
    bottom_frac = float(y2) / float(frame_h)
    width_frac = float(x2 - x1) / float(frame_w)
    return bottom_frac >= _NEAR_THREAT_BOX_BOTTOM_FRAC and width_frac >= _NEAR_THREAT_BOX_WIDTH_FRAC


def _is_near_in_corridor(
    event: RiskEvent, frame_shape: tuple[int, int] | None = None
) -> bool:
    """Whether ``event`` is a close vehicle in or beside the ego corridor."""

    close = (
        event.distance_m <= _NEAR_THREAT_DISTANCE_M
        if event.distance_m is not None
        # No metric depth this frame → trust the normalized nearness instead.
        else event.proximity_score >= 0.85
    )
    if not close:
        return False
    if abs(event.lane_position) <= _NEAR_THREAT_LANE_POS:
        return True
    # Off-corridor by lane_position, but a wide bottom-anchored box is a close
    # lead straddling our path whose position estimate snapped to the clamp.
    return _box_fills_lower_frame(event, frame_shape)


# A small closing-speed deadband separates "approaching" from "receding" so
# depth jitter around zero doesn't flip the classification frame to frame.
_APPROACH_CLOSING_MPS = 0.3


def _is_approaching(event: RiskEvent) -> bool:
    """Whether ``event`` is closing on the ego vehicle (has a real collision TTC)."""

    if event.closing_mps is not None and event.closing_mps > _APPROACH_CLOSING_MPS:
        return True
    return event.collision_ttc_sec is not None


def _is_receding(event: RiskEvent) -> bool:
    """Whether ``event`` is pulling away — no longer a forward collision threat."""

    return event.closing_mps is not None and event.closing_mps < -_APPROACH_CLOSING_MPS


def _display_lane_position(lane_position: float) -> float:
    return round(float(np.clip(lane_position, -1.5, 1.5)), 3)


def _component(event: RiskEvent, name: str) -> Any | None:
    for component in event.ttc_components:
        if component.name == name:
            return component
    return None


def _eta_metric(event: RiskEvent) -> dict[str, Any]:
    """Collision TTC: the fused physical reading plus the per-cue breakdown.

    ``collision_ttc_sec`` + ``display`` reproduce the previous gated display logic
    (depth + Kalman only). ``sources`` exposes each TTC cue's own estimate and
    confidence — data that already lives on ``event.ttc_components`` but was
    never serialized before.
    """

    depth_component = _component(event, "depth")
    distance_m = event.distance_m
    closing_mps = event.closing_mps
    display = "—"
    sec: float | None = None

    if event.corridor_score < _ETA_SURFACE_MIN_CROSSING:
        # Off-corridor object we are passing (not approaching): its depth TTC is
        # a real relative closing time but not a forward-collision TTC. Withhold
        # it so a correctly-SAFE passing vehicle does not show "0.4s". The risk
        # score/state are unaffected (computed upstream from the physical TTC).
        display = "—"
    elif distance_m is None or closing_mps is None:
        display = "—"
    elif float(closing_mps) <= _MIN_CLOSING_FOR_DISPLAY_MPS:
        display = "—"
    else:
        display_eta_candidate = float(distance_m) / max(float(closing_mps), 1e-6)
        depth_confidence = float(getattr(depth_component, "confidence", 0.0) or 0.0)
        if display_eta_candidate > _ETA_HORIZON_SEC:
            display = f">{_ETA_HORIZON_SEC:.0f}s"
        elif depth_confidence < _ETA_LOW_CONFIDENCE:
            display = "—"
        else:
            sec = event.collision_ttc_sec if event.collision_ttc_sec is not None else display_eta_candidate
            display = f"{float(sec):.1f}s"

    sources: dict[str, Any] = {}
    for name in ("depth", "flow", "expansion"):
        component = _component(event, name)
        if component is None:
            continue
        value = getattr(component, "value", None)
        sources[name] = {
            "ttc_sec": None if value is None else round(float(value), 2),
            "confidence": _unit_score(getattr(component, "confidence", 0.0)),
        }

    eta: dict[str, Any] = {
        "collision_ttc_sec": None if sec is None else round(float(sec), 2),
        "display": display,
        "ttc_agreement": _unit_score(event.ttc_agreement),
        "sources": sources,
    }
    return eta


def _motion_metric(event: RiskEvent) -> dict[str, Any]:
    """Physical kinematics plus the raw image-space motion cues.

    Folds in the old ``evidence.flow`` diagnostics (bbox expansion + radial
    flow) so all per-object motion lives under one key.
    """

    motion: dict[str, Any] = {
        "expansion_rate": round(float(np.clip(event.expansion_rate, 0.0, 1.0)), 3),
        "radial_flow_score": round(float(np.clip(event.radial_flow_score, 0.0, 1.0)), 3),
    }
    if event.distance_m is not None:
        motion["distance_m"] = round(float(event.distance_m), 2)
    if event.closing_mps is not None:
        motion["closing_mps"] = round(float(event.closing_mps), 2)
    return motion


def _unit_score(value: float | int | None) -> float:
    return round(float(np.clip(0.0 if value is None else float(value), 0.0, 1.0)), 3)


def _risk_metric(event: RiskEvent) -> dict[str, Any]:
    # A passing (off-corridor) object's TTC is not a collision course, so its
    # TTC-score bar is withheld too — matching the suppressed collision TTC —
    # rather than showing a scary 85% next to a SAFE verdict.
    ttc_score_surfaced = (
        0.0
        if event.corridor_score < _ETA_SURFACE_MIN_CROSSING
        else ttc_score(event.collision_ttc_sec)
    )
    return {
        "risk_score": score_event(event),
        "factors": {
            "ttc_score": _unit_score(ttc_score_surfaced),
            "proximity_score": _unit_score(event.proximity_score),
            "approach_score": _unit_score(event.approach_score),
            "corridor_score": _unit_score(event.corridor_score),
            "brake_score": _unit_score(event.brake_score),
        },
    }


def _lane_obj_metric(event: RiskEvent) -> dict[str, Any]:
    return {
        "lane": event.lane,
        "lane_position": _display_lane_position(event.lane_position),
        "corridor_score": _unit_score(event.corridor_score),
    }


def _confidence_metric(event: RiskEvent) -> dict[str, float]:
    expansion_component = _component(event, "expansion")
    return {
        "risk_confidence": _unit_score(event.risk_confidence),
        "detection_confidence": _unit_score(event.detection_confidence),
        "lane_confidence": _unit_score(event.lane_confidence),
        "depth_confidence": _unit_score(event.depth_confidence),
        "flow_confidence": _unit_score(event.flow_confidence),
        "expansion_confidence": _unit_score(getattr(expansion_component, "confidence", 0.0)),
    }


def _traffic_light_metric(state: tuple[str, float]) -> dict[str, Any]:
    name, confidence = state
    return {"state": name, "confidence": round(float(confidence), 3)}


def _normalized_bbox(
    event: RiskEvent, frame_width: int, frame_height: int
) -> list[float] | None:
    """Return the bbox as ``[x1, y1, x2, y2]`` normalized to ``[0, 1]``.

    Coordinates are in the processed (resized) frame space; normalizing keeps
    them resolution-independent so the frontend can scale them onto the
    displayed video regardless of its render size.
    """

    if event.bbox is None or frame_width <= 0 or frame_height <= 0:
        return None
    x1, y1, x2, y2 = event.bbox
    return [
        round(float(x1) / frame_width, 4),
        round(float(y1) / frame_height, 4),
        round(float(x2) / frame_width, 4),
        round(float(y2) / frame_height, 4),
    ]


def _object_label(object_type: str | None) -> str | None:
    if not object_type:
        return None
    return object_type.replace("_", " ").title()


def _object_metric(
    event: RiskEvent, frame_width: int = 0, frame_height: int = 0
) -> dict[str, Any]:
    return {
        "display_id": event.display_id,
        "object_type": event.object_type,
        "raw_state": event.raw_state,
        "risk": _risk_metric(event),
        "eta": _eta_metric(event),
        "motion": _motion_metric(event),
        "lane": _lane_obj_metric(event),
        "confidence": _confidence_metric(event),
        "object_id": event.object_id,
        "bbox": _normalized_bbox(event, frame_width, frame_height),
    }


def _lane_metric(
    lane: LaneFrame | None, frame_width: int, frame_height: int
) -> dict[str, Any]:
    """Serialize the ego-lane corridor as a normalized polygon.

    Reuses the same ``_lane_corridor`` geometry the rendered saved-event
    overlays use, so the live canvas overlay matches the baked-in imagery.
    """

    if frame_width <= 0 or frame_height <= 0:
        return {"detected": False, "confidence": 0.0, "corridor": []}
    safe_lane = lane if isinstance(lane, LaneFrame) else None
    corridor = _lane_corridor(frame_width, frame_height, safe_lane)
    detected = bool(safe_lane is not None and safe_lane.detected)
    confidence = float(np.clip(safe_lane.confidence if safe_lane is not None else 0.25, 0.0, 1.0))
    norm = [
        [round(float(x) / frame_width, 4), round(float(y) / frame_height, 4)]
        for x, y in corridor
    ]
    return {
        "detected": detected,
        "confidence": round(confidence, 3),
        "corridor": norm,
    }


def _frame_row(
    *,
    event: RiskEvent,
    all_events: list[RiskEvent],
    raw_primary_score: float,
    traffic_light_state: tuple[str, float] = ("none", 0.0),
    lane: LaneFrame | None = None,
    frame_width: int = 0,
    frame_height: int = 0,
) -> dict[str, Any]:
    """Build the v6 client-facing row shared by timeline frames and events.

    ``primary.raw_primary_score`` is computed from the **raw** primary event (before
    hysteresis stabilization) so it matches the corresponding entry in
    ``objects[]`` (``objects[i].risk.risk_score`` where ``object_id`` equals
    ``primary.object_id``). The frame band is exposed as ``stabilized_state``.
    """

    return {
        "frame_index": event.frame_index,
        "timestamp_sec": float(event.timestamp_sec),
        "stabilized_state": event.raw_state,
        "primary": {
            "object_id": event.object_id,
            "raw_primary_score": raw_primary_score,
            "lane": event.lane,
        },
        "traffic_light": _traffic_light_metric(traffic_light_state),
        "lane_geometry": _lane_metric(lane, frame_width, frame_height),
        "objects": [
            _object_metric(item, frame_width, frame_height)
            for item in all_events
            if item.object_id is not None
        ],
    }


def _event_payload_base(
    *,
    event: RiskEvent,
    all_events: list[RiskEvent],
    raw_primary_score: float,
    traffic_light_state: tuple[str, float] = ("none", 0.0),
    lane: LaneFrame | None = None,
    frame_width: int = 0,
    frame_height: int = 0,
) -> dict[str, Any]:
    """Build the metadata-only event payload. RGB views are attached later.

    Carries the v6 client-facing row (see ``_frame_row``) plus snake_case
    diagnostics used only for saved-event dedup/ranking and event identity
    (``risk_score``, ``frame_index``, ``timestamp_sec``); those internal keys
    are stripped by ``_serialize_event`` before the payload reaches the
    frontend.
    """

    payload = _frame_row(
        event=event,
        all_events=all_events,
        raw_primary_score=raw_primary_score,
        traffic_light_state=traffic_light_state,
        lane=lane,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    payload.update(
        {
            "frame_index": event.frame_index,
            "timestamp_sec": event.timestamp_sec,
            "raw_primary_score": raw_primary_score,
            "risk_score": score_event(event),
        }
    )
    return payload


@dataclass
class _DeferredRender:
    """Inputs needed to materialize the RGB views for a saved event."""

    frame_bgr: np.ndarray
    primary_event: RiskEvent
    all_events: list[RiskEvent]
    lane: Any
    traffic_light_state: str = "none"


def _attach_render(payload: dict[str, Any], render: "_DeferredRender") -> dict[str, Any]:
    """Materialize RGB views and attach them to the payload in place."""

    annotated = annotate_frame(
        render.frame_bgr,
        render.primary_event,
        render.all_events,
        lane=render.lane,
        traffic_light_state=render.traffic_light_state,
    )
    payload["original_rgb"] = _to_rgb(render.frame_bgr)
    payload["overlay_rgb"] = _to_rgb(annotated)
    return payload


def _encode_preview_jpeg(annotated_bgr: np.ndarray, quality: int = 70) -> str | None:
    """Encode an annotated BGR frame as a base64 JPEG data URI for live preview."""

    try:
        ok, buffer = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


def _detect_lanes(frame_bgr: np.ndarray, lanenet: UFLDv2ONNX) -> RoadROI:
    """Run UFLDv2 once on a frame and convert its output to a RoadROI."""

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    lanes = lanenet.predict(rgb)
    return estimate_road_roi_from_lanes(
        lanes, width=frame_bgr.shape[1], height=frame_bgr.shape[0]
    )


def _endpoint_drift_px(a: RoadROI, b: RoadROI) -> float:
    """Max absolute x-distance between matching lane endpoints, in pixels.

    Returns 0.0 when either side lacks valid lines. Used to detect when a
    fresh UFLDv2 measurement disagrees so strongly with the Kalman prior
    that the filter has locked onto stale geometry and should be reset.
    """

    if not (a.detected and b.detected):
        return 0.0
    if a.left_line is None or a.right_line is None:
        return 0.0
    if b.left_line is None or b.right_line is None:
        return 0.0
    diffs = (
        abs(a.left_line[0] - b.left_line[0]),
        abs(a.left_line[2] - b.left_line[2]),
        abs(a.right_line[0] - b.right_line[0]),
        abs(a.right_line[2] - b.right_line[2]),
    )
    return float(max(diffs))


def _smooth_lane_confidence(prev: float | None, raw: float) -> float:
    """Asymmetric EMA of the lane-geometry confidence scalar.

    The Kalman filter smooths lane *geometry* but not the scalar confidence,
    which otherwise swings frame-to-frame and makes corridor/lane-trust gating
    jittery. Rises faster than it falls so a brief detection dip doesn't sharply
    drop lane trust. ``prev is None`` (first frame) passes ``raw`` through.
    """

    raw = float(np.clip(raw, 0.0, 1.0))
    if prev is None:
        return raw
    alpha = 0.5 if raw >= prev else 0.3
    return float(alpha * raw + (1.0 - alpha) * prev)


@dataclass
class FrameAnalysis:
    """Per-frame risk output plus the inputs needed to render or preview it.

    ``primary_event`` is the hysteresis-stabilized primary; ``raw_primary_score``
    is the raw pre-stabilization primary risk score, so it matches the matching
    entry in ``all_events``/``objects[]``. ``frame_bgr`` is the resized frame the
    pipeline actually ran on (not the caller's raw input).
    """

    primary_event: RiskEvent
    all_events: list[RiskEvent]
    raw_primary_score: float
    frame_bgr: np.ndarray
    lane: Any
    traffic_light_state: tuple[str, float] = ("none", 0.0)


class SpatialFrameAnalyzer:
    """Stateful per-frame driver for the spatial-risk pipeline.

    Holds every piece of cross-frame state — tracker, lane Kalman, hysteresis
    stabilizer, the three per-track smoothers, and the cached lane/flow/depth —
    so callers can feed frames from any source (a video file or a live camera)
    through identical logic.

    ``analyze_spatial_video`` is now a thin orchestrator on top of this class;
    the per-frame behaviour is byte-for-byte identical to the previous inline
    loop. Per-video diagnostics (``performance_stats``, ``depth_refresh``,
    ``performance_sample_logs``) accumulate on the instance and are read by the
    orchestrator after the loop.
    """

    def __init__(
        self,
        *,
        resize_max_side: int,
        depth_every: int = 10,
        adaptive_depth: bool = True,
        detect_every: int = 3,
        lane_every: int = 3,
        flow_every: int = 1,
        sensitivity: "str | RiskSensitivity" = "balanced",
        lane_reset_after_misses: int = 10,
        lane_drift_reset_px_ratio: float = 0.12,
        fps: float = 0.0,
    ) -> None:
        _ensure_required_models()

        self.resize_max_side = resize_max_side
        self.depth_every = depth_every
        self.adaptive_depth = adaptive_depth
        self.detect_every = detect_every
        self.lane_every = lane_every
        self.flow_every = flow_every
        # Score-band edges → SAFE/CAUTION/DANGER. Resolved once so a bad preset
        # name degrades to "balanced" rather than every frame.
        self.sensitivity = resolve_sensitivity(sensitivity)
        self.lane_reset_after_misses = lane_reset_after_misses
        self.lane_drift_reset_px_ratio = lane_drift_reset_px_ratio
        # Only used for the first frame's flow dt fallback (no previous
        # timestamp yet). Live sources can leave this 0.0 to get the 1/30s
        # default.
        self.fps = fps

        # Cross-frame perception state.
        self.previous_frame = None
        self.last_depth: DepthResult | None = None
        self.last_flow = None
        self.cached_road_roi: RoadROI | None = None
        self.lane_miss_streak = 0
        # EMA of lane-geometry confidence. The Kalman filter smooths the lane
        # geometry but not the scalar confidence, which otherwise swings
        # frame-to-frame and makes corridor/lane-trust gating jittery.
        self.lane_confidence_ema: float | None = None
        self.previous_timestamp_sec: float | None = None
        self.last_motion_depth_frame: int | None = None
        # Advisory traffic-light colour + confidence, refreshed on detection
        # frames and coasted on skipped frames.
        self.last_traffic_light_state: tuple[str, float] = ("none", 0.0)

        # Whether the previous frame had a near, in-corridor threat actively
        # tracked. When False the detector runs an extra lower-center pass to
        # recover a close lead vehicle the full-frame pass may have dropped
        # (Layer 1 near-band gating). True in the healthy case → no extra cost.
        self._near_threat_tracked = False
        # Track ids that should coast longer than the default window when a
        # detection is missed (last primary + previous-frame CAUTION/DANGER ids)
        # so a genuine threat does not vanish from the active set during a brief
        # YOLO dropout (Layer 2).
        self._hot_track_ids: set[int] = set()
        # Last near in-corridor CAUTION/DANGER primary, used to hold the banner
        # across a short dropout instead of falling to a distant low-risk object
        # (Layer 3 safety net). None when no recent near threat.
        self._last_strong_primary: dict[str, Any] | None = None

        # Per-track history / hysteresis.
        self.expansion_smoother = ExpansionSmoother()
        self.depth_smoother = DepthDeltaSmoother()
        self.ttc_imminence_smoother = TtcImminenceSmoother()
        self.confidence_smoother = ConfidenceSmoother()
        self.lane_kalman = LaneKalman()
        self.stabilizer = StateStabilizer(upgrade_frames=3, downgrade_frames=7)
        self.tracker = IoUTracker(iou_threshold=0.25)

        # Models (cached singletons).
        self.lanenet = get_lanenet_model()
        self.detector = get_detector()

        # Diagnostics accumulated across frames.
        self.processed_frames = 0
        self.performance_stats = _empty_performance_stats()
        self.performance_sample_logs: list[str] = []
        self.depth_refresh = {
            "runs": 0,
            "skips": 0,
            "initial_runs": 0,
            "periodic_runs": 0,
            "motion_triggered_runs": 0,
            "cooldown_frames": max(3, depth_every // 2),
        }

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
    ) -> FrameAnalysis:
        """Run the full perception + risk pipeline on a single frame."""

        t0 = time.perf_counter()

        frame = preprocess_frame(frame_bgr, max_side=self.resize_max_side)
        t_preprocess = time.perf_counter()

        # Lane detection runs on scheduled frames only; the most recent
        # committed UFLDv2 detection is cached and reused on in-between frames.
        # Kalman smooths endpoint jitter and coasts through brief misses.
        # The synthetic default ROI is never fed to the Kalman — it would
        # initialize the filter on fake geometry and lock the corridor for
        # the rest of the video.
        fi = frame_index
        did_lane = fi % max(self.lane_every, 1) == 0
        frame_h, frame_w = frame.gray.shape
        if did_lane:
            new_roi = _detect_lanes(frame.bgr, self.lanenet)
            if new_roi.detected:
                smoothed = apply_lane_kalman(new_roi, self.lane_kalman)
                if _endpoint_drift_px(new_roi, smoothed) > self.lane_drift_reset_px_ratio * frame_w:
                    self.lane_kalman.reset()
                    smoothed = apply_lane_kalman(new_roi, self.lane_kalman)
                self.cached_road_roi = smoothed
                self.lane_miss_streak = 0
            else:
                self.lane_miss_streak += 1
                if self.lane_miss_streak >= self.lane_reset_after_misses:
                    self.cached_road_roi = None
                    self.lane_kalman.reset()
                    self.lane_miss_streak = 0

        if self.cached_road_roi is not None:
            if did_lane:
                road_roi = self.cached_road_roi
            else:
                road_roi = apply_lane_kalman(self.cached_road_roi, self.lane_kalman, predict_only=True)
        else:
            road_roi = default_road_roi(frame.bgr.shape)

        lane = build_lane_frame(road_roi, width=frame_w, height=frame_h)
        # Temporal smoothing of the lane confidence scalar (rise faster than it
        # falls) so brief detection dips don't sharply drop lane trust.
        self.lane_confidence_ema = _smooth_lane_confidence(
            self.lane_confidence_ema, float(lane.confidence)
        )
        lane = replace(lane, confidence=self.lane_confidence_ema)
        t_lane = time.perf_counter()

        if self.previous_timestamp_sec is None:
            flow_dt_sec = 1.0 / self.fps if self.fps > 0.0 else 1.0 / 30.0
        else:
            flow_dt_sec = max(1.0 / 120.0, timestamp_sec - self.previous_timestamp_sec)

        # Flow every N frames: reuse previous result on skipped frames.
        did_flow = self.last_flow is None or fi % max(self.flow_every, 1) == 0
        if did_flow:
            flow = compute_velocity(self.previous_frame, frame)
            self.last_flow = flow
        else:
            flow = self.last_flow

        self.previous_frame = frame
        self.previous_timestamp_sec = timestamp_sec
        t_flow = time.perf_counter()

        # 1. Fast motion check: is the scene busy enough to re-run depth?
        quick_risk = compute_quick_risk(flow)
        is_periodic = fi % max(self.depth_every, 1) == 0
        is_initial_depth = self.last_depth is None
        motion_cooldown_ready = (
            self.last_motion_depth_frame is None
            or fi - self.last_motion_depth_frame >= self.depth_refresh["cooldown_frames"]
        )
        is_high_risk = self.adaptive_depth and quick_risk > 0.15
        is_motion_triggered_depth = (
            (not is_initial_depth)
            and (not is_periodic)
            and is_high_risk
            and motion_cooldown_ready
        )
        needs_depth = is_initial_depth or is_periodic or is_motion_triggered_depth
        depth_is_fresh = bool(needs_depth)
        if needs_depth:
            self.depth_refresh["runs"] += 1
            if is_initial_depth:
                self.depth_refresh["initial_runs"] += 1
            elif is_periodic:
                self.depth_refresh["periodic_runs"] += 1
            elif is_motion_triggered_depth:
                self.depth_refresh["motion_triggered_runs"] += 1
                self.last_motion_depth_frame = fi
            self.last_depth = estimate_frame_depth(frame)
        else:
            self.depth_refresh["skips"] += 1
        t_depth = time.perf_counter()

        # 2. Detect objects every N frames; between detections the tracker
        # propagates existing tracks so IDs and depth-Kalman history stay continuous.
        should_detect = fi % max(self.detect_every, 1) == 0
        if should_detect:
            # Pass the latest near_map so the corridor filter can admit
            # physically-close side-lane vehicles as cut-in candidates.
            # ``last_depth`` may be None on the very first frame before the
            # initial depth pass completes; the filter falls back to its
            # strict gate in that case.
            depth_near_map = self.last_depth.near_map if self.last_depth is not None else None
            # Run the lower-center recovery pass only when a near in-corridor
            # threat is NOT currently tracked — i.e. exactly when a close lead
            # vehicle may have dropped out — so we don't pay for a second YOLO
            # pass while the lead car is healthily tracked.
            raw_detections = self.detector.detect(
                frame.bgr, near_band=not self._near_threat_tracked
            )
            # Traffic lights are advisory-only: split them out so they never
            # enter the collision tracker, then classify the nearest one.
            lights = [d for d in raw_detections if d.class_name == "traffic_light"]
            collision_detections = [d for d in raw_detections if d.class_name != "traffic_light"]
            self.last_traffic_light_state = frame_light_state(frame.bgr, lights)
            detections = filter_relevant_detections(
                collision_detections,
                lane,
                near_map=depth_near_map,
            )
            active_tracks = self.tracker.update(
                detections,
                frame_index=fi,
                timestamp_sec=timestamp_sec,
                frame_shape=frame.bgr.shape,
                frame_bgr=frame.bgr,
                hot_ids=self._hot_track_ids,
            )
        else:
            active_tracks = self.tracker.propagate(hot_ids=self._hot_track_ids)
        t_yolo = time.perf_counter()

        _record_stage(self.performance_stats, "preprocess", t_preprocess - t0)
        _record_stage(self.performance_stats, "lane", t_lane - t_preprocess, active=did_lane)
        _record_stage(self.performance_stats, "flow", t_flow - t_lane, active=did_flow)
        _record_stage(self.performance_stats, "depth", t_depth - t_flow, active=needs_depth)
        _record_stage(self.performance_stats, "yolo", t_yolo - t_depth, active=should_detect)

        if self.processed_frames % 30 == 0:
            log_line = (
                f"[FRAME {fi:4d}] "
                f"preprocess={1000*(t_preprocess-t0):.0f}ms  "
                f"lane={'skip' if not did_lane else f'{1000*(t_lane-t_preprocess):.0f}ms':>7}  "
                f"flow={'skip' if not did_flow else f'{1000*(t_flow-t_lane):.0f}ms':>7}  "
                f"depth={'skip' if not needs_depth else f'{1000*(t_depth-t_flow):.0f}ms':>7}  "
                f"yolo={'skip' if not should_detect else f'{1000*(t_yolo-t_depth):.0f}ms':>7}"
            )
            self.performance_sample_logs.append(log_line)

        # 3. Per-object collision TTC + risk evaluation.
        primary_event, all_events = build_object_events(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            tracks=active_tracks,
            fields=SpatialFields(
                depth=self.last_depth,
                flow=flow,
                lane=lane,
                flow_dt_sec=flow_dt_sec,
                depth_is_fresh=depth_is_fresh,
                bgr=frame.bgr,
            ),
            expansion_smoother=self.expansion_smoother,
            depth_smoother=self.depth_smoother,
            ttc_imminence_smoother=self.ttc_imminence_smoother,
            confidence_smoother=self.confidence_smoother,
            sensitivity=self.sensitivity,
        )

        # 4. Smooth State Transitions (Hysteresis)
        # Note: physical ETA is preserved through SAFE stabilization so the
        # timeline can stay continuous while the scene is calm.
        # The primary's RAW risk score is captured before stabilization
        # mutates the event, so it stays consistent with the entry in
        # ``all_events`` (i.e. the ``objects[]`` row whose ``object_id`` equals
        # ``primary.object_id``). The frame-level ``stabilized_state`` carries
        # the hysteresis-smoothed band independently.
        # Frame banner state. The stabilizer is fed a single "banner intent"
        # derived from the RAW per-object states; the shown primary event/score
        # always stay on the real current object, never a frozen or dead-track
        # value. ``raw_banner_event`` is the frame's most severe raw object.
        raw_banner_event = max(
            all_events,
            key=lambda e: (
                {"SAFE": 0, "CAUTION": 1, "DANGER": 2}.get(e.raw_state, 0),
                score_event(e),
            ),
        )

        # A threat worth keeping the banner elevated: a near in-corridor
        # CAUTION/DANGER object that is not already receding (a car that has
        # passed or is pulling away is no longer a forward collision threat).
        frame_shape = frame.bgr.shape[:2]
        active_threat = any(
            e.raw_state in ("CAUTION", "DANGER")
            and _is_near_in_corridor(e, frame_shape)
            and not _is_receding(e)
            for e in all_events
        )
        # Dropout hold (Layer 3): an approaching near threat was the primary
        # within the last ~1s but is momentarily absent from the active set.
        # Hold only the BANNER BAND at CAUTION — never the primary pointer — so
        # a brief miss cannot flash SAFE while the panel keeps showing the real
        # current object.
        hold = (
            self._last_strong_primary is not None
            and 0.0
            < timestamp_sec - self._last_strong_primary["timestamp"]
            <= _STRONG_PRIMARY_HOLD_SEC
        )

        if active_threat:
            raw_stabilizer_event = raw_banner_event
            fast_clear = False
        elif hold:
            raw_stabilizer_event = replace(
                raw_banner_event, raw_state="CAUTION", collision_ttc_sec=None
            )
            fast_clear = False
        else:
            # Scene cleared: nothing is approaching. Feed SAFE monotonically so a
            # raw state oscillating at a band edge cannot keep resetting the
            # downgrade counter, and let an elevated banner fall faster than the
            # anti-flicker gap so a passed threat's DANGER does not linger.
            raw_stabilizer_event = replace(
                raw_banner_event, raw_state="SAFE", collision_ttc_sec=None
            )
            fast_clear = True

        raw_primary_score = score_event(primary_event)
        stabilized_state = stabilized_event_state(
            self.stabilizer, raw_stabilizer_event, fast_clear=fast_clear
        )
        primary_event = replace(primary_event, raw_state=stabilized_state)

        # Refresh cross-frame perception state for the next frame: which close
        # threat is tracked (Layer 1 gating), which ids should coast longer
        # (Layer 2), and the last approaching strong primary to hold on a
        # dropout (Layer 3). Driven by the RAW per-object events.
        self._update_threat_memory(all_events, timestamp_sec, frame_shape)

        self.processed_frames += 1

        return FrameAnalysis(
            primary_event=primary_event,
            all_events=all_events,
            raw_primary_score=raw_primary_score,
            frame_bgr=frame.bgr,
            lane=lane,
            traffic_light_state=self.last_traffic_light_state,
        )

    def _update_threat_memory(
        self,
        all_events: list[RiskEvent],
        timestamp_sec: float,
        frame_shape: tuple[int, int] | None = None,
    ) -> None:
        """Refresh near-threat / hot-id / strong-primary state for next frame."""

        # Layer 1 gating: is a close in-corridor object actively in the set? When
        # not (only far traffic remains), the next detection frame runs the
        # near-band recovery pass.
        self._near_threat_tracked = any(
            _is_near_in_corridor(e, frame_shape) for e in all_events
        )

        # Layer 2: coast the active threat ids longer. Any CAUTION/DANGER object
        # plus the remembered strong primary (so it keeps coasting through a
        # brief miss before the hold window lapses).
        hot: set[int] = {
            e.object_id
            for e in all_events
            if e.raw_state in ("CAUTION", "DANGER") and e.object_id is not None
        }
        strong = self._last_strong_primary
        if (
            strong is not None
            and strong.get("object_id") is not None
            and (timestamp_sec - strong["timestamp"]) <= _STRONG_PRIMARY_HOLD_SEC
        ):
            hot.add(strong["object_id"])
        self._hot_track_ids = hot

        # Layer 3: remember the strongest near in-corridor CAUTION/DANGER object
        # that is actually APPROACHING, as the threat to hold across a brief
        # dropout. A receding / passed object is never stored (so the banner is
        # not held on a car that has already pulled away).
        strong_now = [
            e
            for e in all_events
            if e.raw_state in ("CAUTION", "DANGER")
            and _is_near_in_corridor(e, frame_shape)
            and _is_approaching(e)
        ]
        if strong_now:
            best = max(strong_now, key=score_event)
            self._last_strong_primary = {
                "timestamp": timestamp_sec,
                "object_id": best.object_id,
                "distance_m": best.distance_m,
            }
            return

        # No approaching threat this frame. Drop the memory when it has aged out
        # of the hold window, OR when its track is still visible but has gone
        # non-threatening (receding / no longer approaching) — so a threat that
        # dissipated in view does not keep the banner elevated.
        mem = self._last_strong_primary
        if mem is None:
            return
        tid = mem.get("object_id")
        seen_now = any(e.object_id == tid for e in all_events)
        still_strong = any(
            e.object_id == tid
            and e.raw_state in ("CAUTION", "DANGER")
            and _is_approaching(e)
            for e in all_events
        )
        aged = (timestamp_sec - mem["timestamp"]) > _STRONG_PRIMARY_HOLD_SEC
        if aged or (seen_now and not still_strong):
            self._last_strong_primary = None


def analyze_spatial_video(
    video_path: str | Path,
    *,
    max_processed_frames: int,
    max_saved_events: int,
    resize_max_side: int,
    depth_every: int = 10,
    adaptive_depth: bool = True,
    detect_every: int = 3,
    lane_every: int = 3,
    flow_every: int = 1,
    sensitivity: "str | RiskSensitivity" = "balanced",
    lane_reset_after_misses: int = 4,
    lane_drift_reset_px_ratio: float = 0.12,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_every: int = 6,
) -> dict[str, Any]:
    """Run lane-relative spatial risk analysis and return the UI-compatible result shape."""

    _ensure_required_models()

    loader = VideoLoader(
        video_path,
        max_frames=None,
        start_sec=start_sec,
        end_sec=end_sec,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    # Clamp max_processed_frames to the video's actual frame count so short
    # videos are always analyzed in full regardless of the caller's default.
    if loader.frame_count > 0:
        window_end_frame = loader.end_frame if loader.end_frame is not None else loader.frame_count - 1
        available_frames = max(1, window_end_frame - loader.start_frame + 1)
        max_processed_frames = min(max_processed_frames, available_frames)
    loader.max_frames = max_processed_frames

    analyzer = SpatialFrameAnalyzer(
        resize_max_side=resize_max_side,
        depth_every=depth_every,
        adaptive_depth=adaptive_depth,
        detect_every=detect_every,
        lane_every=lane_every,
        flow_every=flow_every,
        sensitivity=sensitivity,
        lane_reset_after_misses=lane_reset_after_misses,
        lane_drift_reset_px_ratio=lane_drift_reset_px_ratio,
        fps=loader.fps,
    )

    saved_events: list[dict[str, Any]] = []
    pending_renders: dict[int, _DeferredRender] = {}
    frames: list[dict[str, Any]] = []
    preview_rows_buffer: list[dict[str, Any]] = []
    frame_width = 0
    frame_height = 0
    processing_start = time.perf_counter()

    for video_frame in loader.frames():
        analysis = analyzer.process_frame(
            video_frame.bgr,
            video_frame.frame_index,
            video_frame.timestamp_sec,
        )
        primary_event = analysis.primary_event
        all_events = analysis.all_events
        raw_primary_score = analysis.raw_primary_score
        lane = analysis.lane
        frame_height, frame_width = analysis.frame_bgr.shape[:2]

        # Build the metadata-only payload first; heavy RGB views are deferred
        # so we can skip them entirely on frames that won't be saved or
        # previewed.
        event_payload = _event_payload_base(
            event=primary_event,
            all_events=all_events,
            raw_primary_score=raw_primary_score,
            traffic_light_state=analysis.traffic_light_state,
            lane=lane,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        deferred = _DeferredRender(
            frame_bgr=analysis.frame_bgr,
            primary_event=primary_event,
            all_events=all_events,
            lane=lane,
            traffic_light_state=analysis.traffic_light_state[0],
        )

        # Deduplicate events within 1.0 second window
        new_score = score_event_payload(event_payload)
        replaced = False
        for i, saved in enumerate(saved_events):
            if abs(saved["timestamp_sec"] - primary_event.timestamp_sec) <= 1.0:
                if new_score > score_event_payload(saved):
                    saved_events[i] = event_payload
                    pending_renders[id(event_payload)] = deferred
                replaced = True
                break

        if not replaced:
            saved_events.append(event_payload)
            pending_renders[id(event_payload)] = deferred

        # Trim to top-N. Drop pending-render entries that get cut so we
        # don't render images we'll throw away.
        saved_events = sorted(saved_events, key=lambda item: score_event_payload(item), reverse=True)[:max_saved_events]
        live_ids = {id(item) for item in saved_events}
        pending_renders = {pid: ref for pid, ref in pending_renders.items() if pid in live_ids}

        frame_row = _frame_row(
            event=primary_event,
            all_events=all_events,
            raw_primary_score=raw_primary_score,
            traffic_light_state=analysis.traffic_light_state,
            lane=lane,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        frames.append(frame_row)
        if progress_callback is not None:
            preview_rows_buffer.append(frame_row)
        # ``analyzer.process_frame`` already advanced the processed-frame
        # counter, so read it back here for progress reporting.
        processed_frames = analyzer.processed_frames

        is_first_frame = (processed_frames == 1)
        if progress_callback is not None and (is_first_frame or processed_frames % max(1, int(progress_every)) == 0):
            annotated = annotate_frame(
                analysis.frame_bgr,
                primary_event,
                all_events,
                lane=lane,
                traffic_light_state=analysis.traffic_light_state[0],
            )
            preview_uri = _encode_preview_jpeg(annotated)
            progress_pct = min(100.0, round((processed_frames / max(1, max_processed_frames)) * 100.0, 1))
            try:
                progress_callback(
                    {
                        "type": "preview",
                        "progress": progress_pct,
                        "frameImage": preview_uri,
                        "frame": frame_row,
                        "frames": list(preview_rows_buffer),
                    }
                )
            except Exception:
                pass
            preview_rows_buffer.clear()

    if progress_callback is not None and preview_rows_buffer and frames:
        final_row = frames[-1]
        try:
            progress_callback(
                {
                    "type": "preview",
                    "progress": 100.0,
                    "frameImage": None,
                    "frame": final_row,
                    "frames": list(preview_rows_buffer),
                }
            )
        except Exception:
            pass
        preview_rows_buffer.clear()

    # Materialize the heavy RGB views only for events that survived the
    # top-N filter. Frames that never made the cut paid no rendering cost.
    for payload in saved_events:
        deferred = pending_renders.pop(id(payload), None)
        if deferred is not None:
            _attach_render(payload, deferred)
    pending_renders.clear()

    performance_summary = _build_performance_summary(
        analyzer.performance_stats,
        processed_frames=analyzer.processed_frames,
        elapsed_sec=time.perf_counter() - processing_start,
        lane_every=lane_every,
        flow_every=flow_every,
        depth_every=depth_every,
        adaptive_depth=adaptive_depth,
        detect_every=detect_every,
        depth_refresh=analyzer.depth_refresh,
    )
    performance_logs = _format_performance_summary(performance_summary)
    if analyzer.performance_sample_logs:
        performance_logs.extend(["", "SAMPLES", *analyzer.performance_sample_logs])

    peak_event = saved_events[0] if saved_events else None
    return {
        "fps": loader.fps,
        "frame_count": loader.frame_count,
        "processed_frames": analyzer.processed_frames,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "frames": frames,
        "events": saved_events,
        "peak_event": peak_event,
        # Echo the active score-band edges so the UI's Score chart can draw the
        # CAUTION/DANGER threshold lines that this analysis actually used.
        "sensitivity": {
            "caution_band": analyzer.sensitivity.caution_band,
            "danger_band": analyzer.sensitivity.danger_band,
        },
        "performance_summary": performance_summary,
        "performance_logs": performance_logs,
    }


def score_event_payload(payload: dict[str, Any]) -> float:
    score = payload.get("risk_score")
    if score is None:
        score = payload.get("raw_primary_score")
    return round(float(score or 0.0), 3)
