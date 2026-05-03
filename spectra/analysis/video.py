"""FastAPI-facing spatial-awareness risk analysis adapter."""

from __future__ import annotations

import base64
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np

from .risk import (
    DepthDeltaSmoother,
    ExpansionSmoother,
    RiskEvent,
    SpatialFields,
    StateStabilizer,
    build_object_events,
    compute_quick_risk,
    score_raw,
    stabilized_event_state,
)
from ..vision.depth import DepthResult, estimate_frame_depth
from ..vision.detection import get_detector
from ..vision.models import is_depth_available, is_flow_available
from ..vision.motion import compute_velocity, flow_to_rgb
from ..vision.preprocessing import preprocess_frame
from ..vision.road import (
    VanishingPointSmoother,
    RoadROI,
    build_lane_frame,
    compute_vanishing_point,
    default_road_roi,
    estimate_road_roi,
)
from .tracking import IoUTracker
from .overlay import annotate_frame


BAND_BY_STATE = {
    "SAFE": "low",
    "CAUTION": "medium",
    "DANGER": "critical",
}


@dataclass(frozen=True)
class VideoFrame:
    frame_index: int
    timestamp_sec: float
    bgr: np.ndarray


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
    if not is_depth_available():
        raise RuntimeError("Depth Anything ONNX model missing at models/depth_anything_v2_small.onnx")
    # Optical flow is now classical (DIS) and always available — no model file.


def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _depth_rgb(depth: DepthResult) -> np.ndarray:
    colorized = cv2.applyColorMap(depth.depth_map, cv2.COLORMAP_INFERNO)
    return _to_rgb(colorized)


def _road_tracking_rgb(frame_bgr: np.ndarray, road_roi: RoadROI | None = None) -> np.ndarray:
    """Generate an ADAS-style road tracking overlay from the active ROI."""

    road_roi = road_roi or default_road_roi(frame_bgr.shape)
    output = frame_bgr.copy()
    h, w = output.shape[:2]
    overlay = output.copy()

    pts = road_roi.polygon.astype(np.int32)
    cv2.fillPoly(overlay, [pts], (180, 140, 40))
    cv2.addWeighted(overlay, 0.25, output, 0.75, 0, output)
    cv2.polylines(output, [pts], True, (180, 180, 120), 1, cv2.LINE_AA)

    for line in (road_roi.left_line, road_roi.right_line):
        if line is None:
            continue
        x1, y1, x2, y2 = line
        cv2.line(output, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
    
    path_pts = []
    left_line = road_roi.left_line
    right_line = road_roi.right_line
    for i in range(10):
        t = i / 9.0
        curr_y = int(h * (0.60 + t * 0.39))
        if left_line is not None and right_line is not None:
            lx = _line_x_at_y(left_line, curr_y)
            rx = _line_x_at_y(right_line, curr_y)
            curr_x = int((lx + rx) / 2)
        else:
            curr_x = int(w * 0.5)
        path_pts.append([curr_x, curr_y])
    
    path_pts = np.array(path_pts, np.int32)
    cv2.polylines(output, [path_pts], False, (0, 255, 0), 2, cv2.LINE_AA)

    for i in range(1, 4):
        dist_y = int(h * (0.60 + i * 0.10))
        if left_line is not None and right_line is not None:
            x_start = _line_x_at_y(left_line, dist_y)
            x_end = _line_x_at_y(right_line, dist_y)
        else:
            width_at_y = int(w * (0.16 + i * 0.20))
            x_start = int(w * 0.5 - width_at_y / 2)
            x_end = int(w * 0.5 + width_at_y / 2)
        cv2.line(output, (x_start, dist_y), (x_end, dist_y), (200, 200, 200), 1, cv2.LINE_AA)

    return _to_rgb(output)


def _line_x_at_y(line: tuple[int, int, int, int], y: int) -> int:
    x1, y1, x2, y2 = line
    if y2 == y1:
        return int((x1 + x2) / 2)
    t = (y - y1) / float(y2 - y1)
    return int(round(x1 + ((x2 - x1) * t)))


def _risk_score(event: RiskEvent) -> float:
    state_floor = {
        "SAFE": 0.06,
        "CAUTION": 0.42,
        "DANGER": 0.68,
    }.get(event.state, 0.0)
    signal = min(1.0, (0.52 * event.near_score) + (0.48 * event.closing_speed))
    # TTC pressure only contributes when the stabilized state is non-SAFE; a
    # calm scene with a far TTC reading should not inflate the risk score.
    if event.state == "SAFE" or event.ttc_sec is None:
        ttc_pressure = 0.0
    else:
        ttc_pressure = max(0.0, 3.0 - float(event.ttc_sec)) / 3.0
    return round(float(max(state_floor, (0.48 * signal) + (0.52 * ttc_pressure))), 3)


def _lane_metric(event: RiskEvent) -> dict[str, Any]:
    return {
        "lane": event.lane,
        "score": _risk_score(event),
        "meanDepth": event.near_score,
        "motionEnergy": event.velocity_magnitude,
        "expansionEnergy": event.closing_speed,
        "structureSignal": event.confidence,
        "nearRatio": event.near_score,
        "ttcSec": event.ttc_sec,
        "estimated_ttc_sec": event.ttc_sec,
        "directionHint": event.direction,
        "objectType": event.object_type,
        "object_type": event.object_type,
        "objectId": event.object_id,
        "object_id": event.object_id,
        "riskState": event.state,
        "risk_state": event.state,
        "near_score": event.near_score,
        "closing_speed": event.closing_speed,
        "confidence_pct": round(event.confidence * 100.0, 1),
        "bbox": list(event.bbox) if event.bbox is not None else None,
        "expansionRate": event.expansion_rate,
        "crossingRisk": event.crossing_risk,
        "lateralVelocityNorm": event.lateral_velocity_norm,
        "lanePosition": event.lane_position,
        "ttcComponents": [
            {
                "name": component.name,
                "value": component.value,
                "confidence": round(component.confidence, 3),
            }
            for component in event.ttc_components
        ],
    }


def _object_metric(event: RiskEvent) -> dict[str, Any]:
    return {
        "objectId": event.object_id,
        "object_id": event.object_id,
        "objectType": event.object_type,
        "object_type": event.object_type,
        "riskState": event.state,
        "risk_state": event.state,
        "riskScore": _risk_score(event),
        "risk_score": _risk_score(event),
        "lane": event.lane,
        "ttcSec": event.ttc_sec,
        "estimated_ttc_sec": event.ttc_sec,
        "nearScore": event.near_score,
        "near_score": event.near_score,
        "closingSpeed": event.closing_speed,
        "closing_speed": event.closing_speed,
        "velocityMagnitude": event.velocity_magnitude,
        "velocity_magnitude": event.velocity_magnitude,
        "confidencePct": round(event.confidence * 100.0, 1),
        "confidence_pct": round(event.confidence * 100.0, 1),
        "bbox": list(event.bbox) if event.bbox is not None else None,
        "expansionRate": event.expansion_rate,
        "expansion_rate": event.expansion_rate,
        "crossingRisk": event.crossing_risk,
        "crossing_risk": event.crossing_risk,
        "lateralVelocityNorm": event.lateral_velocity_norm,
        "lateral_velocity_norm": event.lateral_velocity_norm,
        "lanePosition": event.lane_position,
        "lane_position": event.lane_position,
        "ttcComponents": [
            {
                "name": component.name,
                "value": component.value,
                "confidence": round(component.confidence, 3),
            }
            for component in event.ttc_components
        ],
    }


def _event_payload_base(
    *,
    event: RiskEvent,
    all_events: list[RiskEvent],
) -> dict[str, Any]:
    """Build the metadata-only event payload. RGB views are attached later."""

    band = BAND_BY_STATE.get(event.state, "low")
    confidence_pct = round(event.confidence * 100.0, 1)
    uncertainty_pct = round((1.0 - event.confidence) * 100.0, 1)
    summary = f"{event.state} in the {event.lane} lane"
    if event.ttc_sec is not None:
        summary += f", TTC {event.ttc_sec:.2f}s"
    summary += f", direction {event.direction}"

    return {
        "frame_index": event.frame_index,
        "timestamp_sec": event.timestamp_sec,
        "risk_score": _risk_score(event),
        "risk_band": band,
        "risk_state": event.state,
        "primary_lane": event.lane,
        "estimated_ttc_sec": event.ttc_sec,
        "confidence_pct": confidence_pct,
        "uncertainty_pct": uncertainty_pct,
        "heuristic_summary": summary,
        "reasons": [event.reason, f"{event.object_type} in {event.lane} lane", f"motion direction: {event.direction}"],
        "lane_metrics": [_lane_metric(item) for item in all_events],
        "objects": [_object_metric(item) for item in all_events if item.object_id is not None],
        "object_type": event.object_type,
        "approach": "approaching" if event.state in {"CAUTION", "DANGER"} else "stable",
        "lane": event.lane,
        "bbox": list(event.bbox) if event.bbox is not None else None,
        "object_id": event.object_id,
        "near_score": event.near_score,
        "closing_speed": event.closing_speed,
        "velocity_magnitude": event.velocity_magnitude,
        "expansion_rate": event.expansion_rate,
        "crossing_risk": event.crossing_risk,
        "lane_position": event.lane_position,
        "ttc_components": [
            {
                "name": component.name,
                "value": component.value,
                "confidence": round(component.confidence, 3),
            }
            for component in event.ttc_components
        ],
        "payload": {
            "risk_state": event.state,
            "object_type": event.object_type,
            "lane": event.lane,
            "direction": event.direction,
            "ttc_sec": event.ttc_sec,
            "near_score": event.near_score,
            "closing_speed": event.closing_speed,
            "confidence_pct": confidence_pct,
            "object_id": event.object_id,
            "lane_position": event.lane_position,
            "ttc_components": [
                {
                    "name": component.name,
                    "value": component.value,
                    "confidence": round(component.confidence, 3),
                }
                for component in event.ttc_components
            ],
        },
    }


@dataclass
class _DeferredRender:
    """Inputs needed to materialize the heavy RGB views for a saved event."""

    frame_bgr: np.ndarray
    primary_event: RiskEvent
    all_events: list[RiskEvent]
    lane: Any
    flow: np.ndarray
    depth: DepthResult | None
    road_roi: RoadROI | None


def _attach_render(payload: dict[str, Any], render: "_DeferredRender") -> dict[str, Any]:
    """Materialize the heavy RGB views and attach them to the payload in place."""

    annotated = annotate_frame(
        render.frame_bgr, render.primary_event, render.all_events, lane=render.lane
    )
    payload["original_rgb"] = _to_rgb(render.frame_bgr)
    payload["depth_rgb"] = _depth_rgb(render.depth) if render.depth is not None else _to_rgb(render.frame_bgr)
    payload["road_rgb"] = _road_tracking_rgb(render.frame_bgr, render.road_roi)
    payload["motion_rgb"] = flow_to_rgb(render.flow)
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


def analyze_spatial_video(
    video_path: str | Path,
    *,
    max_processed_frames: int,
    max_saved_events: int,
    resize_max_side: int,
    depth_every: int = 10,
    enable_road_roi: bool = False,
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

    previous_frame = None
    last_depth: DepthResult | None = None
    saved_events: list[dict[str, Any]] = []
    pending_renders: dict[int, _DeferredRender] = {}
    timeline_rows: list[dict[str, Any]] = []
    preview_rows_buffer: list[dict[str, Any]] = []
    expansion_smoother = ExpansionSmoother()
    depth_smoother = DepthDeltaSmoother()
    vp_smoother = VanishingPointSmoother()
    stabilizer = StateStabilizer(upgrade_frames=3, downgrade_frames=7)
    detector = get_detector()
    tracker = IoUTracker(iou_threshold=0.25, max_misses=5)
    processed_frames = 0
    previous_timestamp_sec: float | None = None

    for video_frame in loader.frames():
        frame = preprocess_frame(video_frame.bgr, max_side=resize_max_side)
        road_roi = estimate_road_roi(frame.bgr) if enable_road_roi else default_road_roi(frame.bgr.shape)
        frame_h, frame_w = frame.gray.shape
        raw_vp = compute_vanishing_point(road_roi, frame_w, frame_h)
        smoothed_vp = vp_smoother.update(raw_vp, road_roi.confidence)
        lane = build_lane_frame(
            road_roi,
            width=frame_w,
            height=frame_h,
            smoothed_vp=smoothed_vp,
        )

        if previous_timestamp_sec is None:
            flow_dt_sec = 1.0 / loader.fps if loader.fps > 0.0 else 1.0 / 30.0
        else:
            flow_dt_sec = max(1.0 / 120.0, video_frame.timestamp_sec - previous_timestamp_sec)
        flow = compute_velocity(previous_frame, frame)
        previous_frame = frame
        previous_timestamp_sec = video_frame.timestamp_sec

        # 1. Fast motion check: is the scene busy enough to re-run depth?
        quick_risk = compute_quick_risk(flow, frame.gray.shape[1], frame.gray.shape[0])
        is_high_risk = quick_risk > 0.15
        is_periodic = video_frame.frame_index % max(depth_every, 1) == 0
        needs_depth = last_depth is None or is_periodic or is_high_risk
        depth_is_fresh = bool(needs_depth)
        if needs_depth:
            last_depth = estimate_frame_depth(frame)

        # 2. Detect objects on the original (preprocessed) BGR frame and track
        # them across frames. The tracker links bboxes by IoU so per-object
        # scale expansion (TTC source) survives detection jitter.
        detections = detector.detect(frame.bgr)
        active_tracks = tracker.update(
            detections,
            frame_index=video_frame.frame_index,
            timestamp_sec=video_frame.timestamp_sec,
        )

        # 3. Per-object risk via scale-expansion TTC + lateral crossing.
        primary_event, all_events = build_object_events(
            frame_index=video_frame.frame_index,
            timestamp_sec=video_frame.timestamp_sec,
            tracks=active_tracks,
            fields=SpatialFields(
                depth=last_depth,
                flow=flow,
                lane=lane,
                flow_dt_sec=flow_dt_sec,
                depth_is_fresh=depth_is_fresh,
            ),
            expansion_smoother=expansion_smoother,
            depth_smoother=depth_smoother,
        )

        # 4. Smooth State Transitions (Hysteresis)
        # Note: TTC is preserved through SAFE stabilization so the timeline
        # chart stays continuous and the UI can still show a TTC reading
        # while the scene is calm.
        stabilized_state = stabilized_event_state(stabilizer, primary_event)
        primary_event = replace(primary_event, state=stabilized_state)

        # Build the metadata-only payload first; heavy RGB views are deferred
        # so we can skip them entirely on frames that won't be saved or
        # previewed.
        event_payload = _event_payload_base(event=primary_event, all_events=all_events)
        deferred = _DeferredRender(
            frame_bgr=frame.bgr,
            primary_event=primary_event,
            all_events=all_events,
            lane=lane,
            flow=flow.flow,
            depth=last_depth,
            road_roi=road_roi,
        )

        # Deduplicate events within 1.0 second window
        new_score = score_event_payload(event_payload)
        replaced = False
        kept = False
        for i, saved in enumerate(saved_events):
            if abs(saved["timestamp_sec"] - primary_event.timestamp_sec) <= 1.0:
                if new_score > score_event_payload(saved):
                    saved_events[i] = event_payload
                    pending_renders[id(event_payload)] = deferred
                    kept = True
                replaced = True
                break

        if not replaced:
            saved_events.append(event_payload)
            pending_renders[id(event_payload)] = deferred
            kept = True

        # Trim to top-N. Drop pending-render entries that get cut so we
        # don't render images we'll throw away.
        saved_events = sorted(saved_events, key=lambda item: score_event_payload(item), reverse=True)[:max_saved_events]
        live_ids = {id(item) for item in saved_events}
        pending_renders = {pid: ref for pid, ref in pending_renders.items() if pid in live_ids}

        # Aggregate lane-bucket scores by taking the max per object bucket.
        # Multiple objects can sit in the same lane, and only the worst one
        # should drive that lane's alert.
        object_metrics = [_object_metric(ev) for ev in all_events if ev.object_id is not None]
        lane_scores: dict[str, float] = {}
        for ev in all_events:
            z_key = str(ev.lane).lower().split()[0] if ev.lane else "center"
            score = _risk_score(ev)
            lane_scores[z_key] = max(lane_scores.get(z_key, 0.0), score)

        timeline_row = {
            "frameIndex": primary_event.frame_index,
            "timeSec": round(primary_event.timestamp_sec, 2),
            "riskState": primary_event.state,
            "riskBand": BAND_BY_STATE.get(primary_event.state, "low"),
            "lane": primary_event.lane,
            "direction": primary_event.direction,
            "ttcSec": primary_event.ttc_sec,
            "nearScore": primary_event.near_score,
            "closingSpeed": primary_event.closing_speed,
            "laneScores": lane_scores,
            "objects": object_metrics,
            "detections": object_metrics,
        }
        timeline_rows.append(timeline_row)
        if progress_callback is not None:
            preview_rows_buffer.append(timeline_row)
        processed_frames += 1

        is_first_frame = (processed_frames == 1)
        if progress_callback is not None and (is_first_frame or processed_frames % max(1, int(progress_every)) == 0):
            # Annotate only here — non-preview frames skip this entirely.
            annotated = annotate_frame(frame.bgr, primary_event, all_events, lane=lane)
            preview_uri = _encode_preview_jpeg(annotated)
            progress_pct = min(100.0, round((processed_frames / max(1, max_processed_frames)) * 100.0, 1))
            lane_metrics_payload = [_lane_metric(ev) for ev in all_events if ev.object_id is not None]
            object_metrics_payload = [_object_metric(ev) for ev in all_events if ev.object_id is not None]
            timeline_rows_payload = list(preview_rows_buffer)
            try:
                progress_callback(
                    {
                        "type": "preview",
                        "frameIndex": int(primary_event.frame_index),
                        "timestampSec": float(primary_event.timestamp_sec),
                        "progress": progress_pct,
                        "riskState": primary_event.state,
                        "ttcSec": None if primary_event.ttc_sec is None else float(primary_event.ttc_sec),
                        "lane": primary_event.lane,
                        "objectType": primary_event.object_type,
                        "nearScore": float(primary_event.near_score),
                        "closingSpeed": float(primary_event.closing_speed),
                        "crossingRisk": float(primary_event.crossing_risk),
                        "confidencePct": float(primary_event.confidence * 100.0),
                        "frame": preview_uri,
                        "laneMetrics": lane_metrics_payload,
                        "objects": object_metrics_payload,
                        "detections": object_metrics_payload,
                        "timelineRow": timeline_row,
                        "timelineRows": timeline_rows_payload,
                    }
                )
            except Exception:
                pass
            preview_rows_buffer.clear()

    if progress_callback is not None and preview_rows_buffer and timeline_rows:
        final_row = timeline_rows[-1]
        try:
            progress_callback(
                {
                    "type": "preview",
                    "frameIndex": int(final_row["frameIndex"]),
                    "timestampSec": float(final_row["timeSec"]),
                    "progress": 100.0,
                    "riskState": final_row["riskState"],
                    "ttcSec": None if final_row["ttcSec"] is None else float(final_row["ttcSec"]),
                    "lane": final_row["lane"],
                    "nearScore": float(final_row["nearScore"]),
                    "closingSpeed": float(final_row["closingSpeed"]),
                    "timelineRow": final_row,
                    "timelineRows": list(preview_rows_buffer),
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

    peak_event = saved_events[0] if saved_events else None
    return {
        "media_type": "video",
        "pipeline": "spatial_awareness",
        "fps": loader.fps,
        "frame_count": loader.frame_count,
        "processed_frames": processed_frames,
        "sampled_frames": processed_frames,
        "timeline_rows": timeline_rows,
        "events": saved_events,
        "peak_event": peak_event,
        "summary": None if peak_event is None else peak_event["heuristic_summary"],
        "analysis_mode": "lane_relative",
    }


def score_event_payload(payload: dict[str, Any]) -> float:
    return score_raw(
        state=str(payload.get("risk_state") or "").upper(),
        ttc_sec=payload.get("estimated_ttc_sec"),
        near_score=float(payload.get("near_score") or 0.0),
        closing_speed=float(payload.get("closing_speed") or 0.0),
    )
