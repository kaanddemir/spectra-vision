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
    ConfidenceSmoother,
    DepthDeltaSmoother,
    ExpansionSmoother,
    RiskEvent,
    SpatialFields,
    StateStabilizer,
    TtcImminenceSmoother,
    build_object_events,
    compute_quick_risk,
    score_raw,
    stabilized_event_state,
)
from ..vision.depth import DepthResult, estimate_frame_depth
from ..vision.detection import get_detector, is_yolo_available
from ..vision.lanenet import UFLDv2ONNX, get_lanenet_model, is_lanenet_available
from ..vision.models import get_depth_model, is_depth_available
from ..vision.motion import compute_velocity
from ..vision.preprocessing import preprocess_frame
from ..vision.traffic_light import frame_light_state
from ..vision.road import (
    LaneKalman,
    RoadROI,
    apply_lane_kalman,
    build_lane_frame,
    default_road_roi,
    estimate_road_roi_from_lanes,
    filter_relevant_detections,
)
from .tracking import IoUTracker
from .overlay import annotate_frame



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
    ) -> None:
        self.source = str(source)
        self.max_frames = max_frames
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        self.start_frame = 0
        if self.start_sec > 0 and self.fps > 0:
            self.start_frame = int(self.start_sec * self.fps)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

    def frames(self) -> Iterator[VideoFrame]:
        frame_index = self.start_frame
        try:
            while self.max_frames is None or (frame_index - self.start_frame) < self.max_frames:
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


def _risk_score(event: RiskEvent) -> float:
    state_floor = {
        "SAFE": 0.06,
        "CAUTION": 0.42,
        "DANGER": 0.68,
    }.get(event.state, 0.0)
    signal = min(1.0, (0.52 * event.near_score) + (0.48 * event.closing_speed))
    # ETA pressure only contributes when the stabilized state is non-SAFE; a
    # calm scene with a far physical ETA should not inflate the risk score.
    if event.state == "SAFE" or event.ttc_sec is None:
        ttc_pressure = 0.0
    else:
        ttc_pressure = max(0.0, 3.0 - float(event.ttc_sec)) / 3.0
    return round(float(max(state_floor, (0.48 * signal) + (0.52 * ttc_pressure))), 3)


def _display_lane_position(lane_position: float) -> float:
    return round(float(np.clip(lane_position, -1.5, 1.5)), 3)


def _depth_component(event: RiskEvent) -> Any | None:
    for component in event.ttc_components:
        if component.name == "depth":
            return component
    return None


def _collision_eta_metric(event: RiskEvent) -> dict[str, Any]:
    depth_component = _depth_component(event)
    distance_m = event.distance_m
    closing_mps = event.closing_mps
    status = "estimating"
    display = "Estimating"
    sec: float | None = None

    if distance_m is None or closing_mps is None:
        status = "estimating"
        display = "Estimating"
    elif float(closing_mps) <= _MIN_CLOSING_FOR_DISPLAY_MPS:
        status = "not_closing"
        display = "No closing"
    else:
        raw_eta = float(distance_m) / max(float(closing_mps), 1e-6)
        confidence = float(getattr(depth_component, "confidence", 0.0) or 0.0)
        if raw_eta > _ETA_HORIZON_SEC:
            status = "beyond_horizon"
            display = f">{_ETA_HORIZON_SEC:.0f}s"
        elif confidence < _ETA_LOW_CONFIDENCE:
            status = "low_confidence"
            display = "Low confidence"
        else:
            status = "closing"
            sec = event.ttc_sec if event.ttc_sec is not None else raw_eta
            display = f"{float(sec):.1f}s"

    eta: dict[str, Any] = {
        "status": status,
        "display": display,
        "source": "depth_kalman",
    }
    if sec is not None:
        eta["sec"] = round(float(sec), 2)
    return eta


def _kinematics_metric(event: RiskEvent) -> dict[str, Any]:
    kinematics: dict[str, Any] = {}
    if event.distance_m is not None:
        kinematics["distanceM"] = round(float(event.distance_m), 2)
    if event.closing_mps is not None:
        kinematics["closingMps"] = round(float(event.closing_mps), 2)
    return kinematics


def _unit_score(value: float | int | None) -> float:
    return round(float(np.clip(0.0 if value is None else float(value), 0.0, 1.0)), 3)


def _risk_factors_metric(event: RiskEvent) -> dict[str, float]:
    return {
        "proximity": _unit_score(event.near_score),
        "approach": _unit_score(event.closing_speed),
        "crossing": _unit_score(event.crossing_risk),
        "brake": _unit_score(event.brake_score),
    }


def _confidence_metric(event: RiskEvent) -> dict[str, float]:
    return {
        "detection": _unit_score(event.detection_confidence),
        "tracking": _unit_score(event.tracking_confidence),
        "depth": _unit_score(event.depth_confidence),
    }


def _evidence_metric(event: RiskEvent, lane_position: float, confidence: dict[str, float]) -> dict[str, Any]:
    depth: dict[str, Any] = {
        "source": "depth_kalman",
        "status": "tracked" if event.distance_m is not None else "estimating",
        "confidence": confidence["depth"],
    }
    if event.distance_m is not None:
        depth["distanceM"] = round(float(event.distance_m), 2)
    if event.closing_mps is not None:
        depth["closingMps"] = round(float(event.closing_mps), 2)

    return {
        "detector": {
            "source": "yolo",
            "class": event.object_type,
            "confidence": confidence["detection"],
        },
        "depth": depth,
        "flow": {
            "expansionScore": round(float(np.clip(event.expansion_rate, 0.0, 1.0)), 3),
            "radialScore": round(float(np.clip(event.velocity_magnitude, 0.0, 1.0)), 3),
        },
        "lane": {
            "bucket": event.lane,
            "position": lane_position,
            "crossingRisk": round(float(event.crossing_risk), 3),
        },
    }


def _object_metric(event: RiskEvent) -> dict[str, Any]:
    lane_position = _display_lane_position(event.lane_position)
    confidence = _confidence_metric(event)
    return {
        "objectId": event.object_id,
        "displayId": event.display_id,
        "objectType": event.object_type,
        "rawRiskState": event.state,
        "riskScore": _risk_score(event),
        "lane": event.lane,
        "lanePosition": lane_position,
        "overallConfidence": _unit_score(event.confidence),
        "confidence": confidence,
        "collisionEta": _collision_eta_metric(event),
        "kinematics": _kinematics_metric(event),
        "riskFactors": _risk_factors_metric(event),
        "evidence": _evidence_metric(event, lane_position, confidence),
    }


def _event_payload_base(
    *,
    event: RiskEvent,
    all_events: list[RiskEvent],
    primary_risk_score: float,
) -> dict[str, Any]:
    """Build the metadata-only event payload. RGB views are attached later.

    Client-facing fields use schema names (``stabilized_risk_state``,
    ``primary_*``). The primary event's raw scoring inputs are kept on the
    payload as internal scratch fields because ``score_event_payload`` ranks
    saved events by them; ``_serialize_event`` drops these before the JSON
    leaves the server.

    ``primary_risk_score`` is computed from the **raw** primary event (before
    hysteresis stabilization) so it stays consistent with the matching entry
    in ``objects[]`` — i.e. ``payload.primary_risk_score`` always equals
    ``payload.objects[primary_object_id].riskScore``. The state band is
    decoupled in ``stabilized_risk_state``.
    """

    return {
        "frame_index": event.frame_index,
        "timestamp_sec": event.timestamp_sec,
        "stabilized_risk_state": event.state,
        "primary_object_id": event.object_id,
        "primary_risk_score": primary_risk_score,
        "primary_lane": event.lane,
        "risk_score": _risk_score(event),
        "collision_eta_sec": event.ttc_sec,
        "proximity_score": event.near_score,
        "approach_score": event.closing_speed,
        "objects": [_object_metric(item) for item in all_events if item.object_id is not None],
    }


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

    ``primary_event`` is the hysteresis-stabilized primary; ``primary_risk_score``
    is computed from the *raw* primary (before stabilization) so it matches the
    matching entry in ``all_events``/``objects[]``. ``frame_bgr`` is the resized
    frame the pipeline actually ran on (not the caller's raw input).
    """

    primary_event: RiskEvent
    all_events: list[RiskEvent]
    primary_risk_score: float
    frame_bgr: np.ndarray
    lane: Any
    traffic_light_state: str = "none"


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
        lane_reset_after_misses: int = 6,
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
        # Advisory traffic-light colour, refreshed on detection frames and
        # coasted on skipped frames.
        self.last_traffic_light_state = "none"

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
        quick_risk = compute_quick_risk(flow, frame.gray.shape[1], frame.gray.shape[0])
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
            raw_detections = self.detector.detect(frame.bgr)
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
            )
        else:
            active_tracks = self.tracker.propagate()
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

        # 3. Per-object collision ETA + risk evaluation.
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
        )

        # 4. Smooth State Transitions (Hysteresis)
        # Note: physical ETA is preserved through SAFE stabilization so the
        # timeline can stay continuous while the scene is calm.
        # The primary's RAW risk score is captured before stabilization
        # mutates the event, so it stays consistent with the entry in
        # ``all_events`` (and therefore with ``objects[primaryObjectId]``).
        # ``stabilized_risk_state`` carries the hysteresis-smoothed state
        # band independently.
        primary_risk_score = _risk_score(primary_event)
        # Stabilize on the frame's WORST raw-state object, not just the selected
        # primary. After the eligibility fix this is normally the primary, but
        # feeding the worst keeps the banner elevated when primary selection
        # flips between objects or a threat briefly drops out of the active set
        # for a frame — both of which otherwise let the stabilizer downgrade
        # prematurely while a real danger is still present.
        stabilizer_event = max(
            all_events,
            key=lambda e: (
                {"SAFE": 0, "CAUTION": 1, "DANGER": 2}.get(e.state, 0),
                _risk_score(e),
            ),
        )
        stabilized_state = stabilized_event_state(self.stabilizer, stabilizer_event)
        primary_event = replace(primary_event, state=stabilized_state)

        self.processed_frames += 1

        return FrameAnalysis(
            primary_event=primary_event,
            all_events=all_events,
            primary_risk_score=primary_risk_score,
            frame_bgr=frame.bgr,
            lane=lane,
            traffic_light_state=self.last_traffic_light_state,
        )


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
    lane_reset_after_misses: int = 4,
    lane_drift_reset_px_ratio: float = 0.12,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_every: int = 6,
) -> dict[str, Any]:
    """Run lane-relative spatial risk analysis and return the UI-compatible result shape."""

    _ensure_required_models()

    loader = VideoLoader(video_path, max_frames=None, start_sec=start_sec, end_sec=end_sec)
    # Clamp max_processed_frames to the video's actual frame count so short
    # videos are always analyzed in full regardless of the caller's default.
    if loader.frame_count > 0:
        max_processed_frames = min(max_processed_frames, loader.frame_count)
    loader.max_frames = max_processed_frames

    analyzer = SpatialFrameAnalyzer(
        resize_max_side=resize_max_side,
        depth_every=depth_every,
        adaptive_depth=adaptive_depth,
        detect_every=detect_every,
        lane_every=lane_every,
        flow_every=flow_every,
        lane_reset_after_misses=lane_reset_after_misses,
        lane_drift_reset_px_ratio=lane_drift_reset_px_ratio,
        fps=loader.fps,
    )

    saved_events: list[dict[str, Any]] = []
    pending_renders: dict[int, _DeferredRender] = {}
    frames: list[dict[str, Any]] = []
    preview_rows_buffer: list[dict[str, Any]] = []
    processing_start = time.perf_counter()

    for video_frame in loader.frames():
        analysis = analyzer.process_frame(
            video_frame.bgr,
            video_frame.frame_index,
            video_frame.timestamp_sec,
        )
        primary_event = analysis.primary_event
        all_events = analysis.all_events
        primary_risk_score = analysis.primary_risk_score
        lane = analysis.lane

        # Build the metadata-only payload first; heavy RGB views are deferred
        # so we can skip them entirely on frames that won't be saved or
        # previewed.
        event_payload = _event_payload_base(
            event=primary_event,
            all_events=all_events,
            primary_risk_score=primary_risk_score,
        )
        deferred = _DeferredRender(
            frame_bgr=analysis.frame_bgr,
            primary_event=primary_event,
            all_events=all_events,
            lane=lane,
            traffic_light_state=analysis.traffic_light_state,
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

        object_metrics = [_object_metric(ev) for ev in all_events if ev.object_id is not None]
        frame_row = {
            "frameIndex": primary_event.frame_index,
            "timestampSec": float(primary_event.timestamp_sec),
            "stabilizedRiskState": primary_event.state,
            "primaryObjectId": primary_event.object_id,
            "primaryRiskScore": primary_risk_score,
            "primaryLane": primary_event.lane,
            "trafficLight": analysis.traffic_light_state,
            "objects": object_metrics,
        }
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
                traffic_light_state=analysis.traffic_light_state,
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
        "frames": frames,
        "events": saved_events,
        "peak_event": peak_event,
        "performance_summary": performance_summary,
        "performance_logs": performance_logs,
    }


def score_event_payload(payload: dict[str, Any]) -> float:
    return score_raw(
        state=str(payload.get("stabilized_risk_state") or "").upper(),
        ttc_sec=payload.get("collision_eta_sec"),
        near_score=float(payload.get("proximity_score") or 0.0),
        closing_speed=float(payload.get("approach_score") or 0.0),
    )
