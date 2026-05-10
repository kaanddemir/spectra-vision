"""FastAPI-facing spatial-awareness risk analysis adapter."""

from __future__ import annotations

import base64
import time
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
from ..vision.detection import get_detector, is_yolo_available
from ..vision.lanenet import UFLDv2ONNX, get_lanenet_model, is_lanenet_available
from ..vision.models import get_depth_model, is_depth_available
from ..vision.motion import compute_velocity
from ..vision.preprocessing import preprocess_frame
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
            "Depth Anything ONNX model missing at models/depth_anything_v2_small.onnx"
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


def _object_metric(event: RiskEvent) -> dict[str, Any]:
    return {
        "objectId": event.object_id,
        "objectType": event.object_type,
        "rawRiskState": event.state,
        "riskScore": _risk_score(event),
        "lane": event.lane,
        "ttcSec": event.ttc_sec,
        "nearScore": event.near_score,
        "closingSpeed": event.closing_speed,
        "crossingRisk": event.crossing_risk,
        "lanePosition": event.lane_position,
        "confidence": round(event.confidence, 4),
    }


def _event_payload_base(
    *,
    event: RiskEvent,
    all_events: list[RiskEvent],
    primary_risk_score: float,
) -> dict[str, Any]:
    """Build the metadata-only event payload. RGB views are attached later.

    Client-facing fields use the v2 schema names (``stabilized_risk_state``,
    ``primary_*``). The primary event's raw metrics (``ttc_sec``,
    ``near_score``, ``closing_speed``) are kept on the payload too because
    ``score_event_payload`` ranks saved events by them; ``_serialize_event``
    drops these internal scratch fields before the JSON leaves the server.

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
        "ttc_sec": event.ttc_sec,
        "near_score": event.near_score,
        "closing_speed": event.closing_speed,
        "objects": [_object_metric(item) for item in all_events if item.object_id is not None],
    }


@dataclass
class _DeferredRender:
    """Inputs needed to materialize the RGB views for a saved event."""

    frame_bgr: np.ndarray
    primary_event: RiskEvent
    all_events: list[RiskEvent]
    lane: Any
    road_roi: RoadROI | None


def _attach_render(payload: dict[str, Any], render: "_DeferredRender") -> dict[str, Any]:
    """Materialize RGB views and attach them to the payload in place."""

    annotated = annotate_frame(
        render.frame_bgr, render.primary_event, render.all_events, lane=render.lane
    )
    payload["original_rgb"] = _to_rgb(render.frame_bgr)
    payload["road_rgb"] = _road_tracking_rgb(render.frame_bgr, render.road_roi)
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


def analyze_spatial_video(
    video_path: str | Path,
    *,
    max_processed_frames: int,
    max_saved_events: int,
    resize_max_side: int,
    depth_every: int = 10,
    detect_every: int = 3,
    lane_every: int = 5,
    flow_every: int = 1,
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
    last_flow = None
    cached_road_roi: RoadROI | None = None
    saved_events: list[dict[str, Any]] = []
    pending_renders: dict[int, _DeferredRender] = {}
    frames: list[dict[str, Any]] = []
    preview_rows_buffer: list[dict[str, Any]] = []
    performance_logs: list[str] = []
    expansion_smoother = ExpansionSmoother()
    depth_smoother = DepthDeltaSmoother()
    lane_kalman = LaneKalman()
    lanenet = get_lanenet_model()
    stabilizer = StateStabilizer(upgrade_frames=3, downgrade_frames=7)
    detector = get_detector()
    tracker = IoUTracker(iou_threshold=0.25, max_misses=5)
    processed_frames = 0
    previous_timestamp_sec: float | None = None

    for video_frame in loader.frames():
        t0 = time.perf_counter()

        frame = preprocess_frame(video_frame.bgr, max_side=resize_max_side)
        t_preprocess = time.perf_counter()

        # Lane detection runs on scheduled frames only; the most recent
        # committed UFLDv2 detection is cached and reused on in-between frames.
        # Kalman smooths endpoint jitter and coasts through brief misses.
        fi = video_frame.frame_index
        if fi % max(lane_every, 1) == 0:
            new_roi = _detect_lanes(frame.bgr, lanenet)
            if new_roi.detected:
                cached_road_roi = new_roi
        road_roi = cached_road_roi or default_road_roi(frame.bgr.shape)
        road_roi = apply_lane_kalman(road_roi, lane_kalman)

        frame_h, frame_w = frame.gray.shape
        lane = build_lane_frame(road_roi, width=frame_w, height=frame_h)
        t_lane = time.perf_counter()

        if previous_timestamp_sec is None:
            flow_dt_sec = 1.0 / loader.fps if loader.fps > 0.0 else 1.0 / 30.0
        else:
            flow_dt_sec = max(1.0 / 120.0, video_frame.timestamp_sec - previous_timestamp_sec)

        # Flow every N frames: reuse previous result on skipped frames.
        if last_flow is None or fi % max(flow_every, 1) == 0:
            flow = compute_velocity(previous_frame, frame)
            last_flow = flow
        else:
            flow = last_flow

        previous_frame = frame
        previous_timestamp_sec = video_frame.timestamp_sec
        t_flow = time.perf_counter()

        # 1. Fast motion check: is the scene busy enough to re-run depth?
        quick_risk = compute_quick_risk(flow, frame.gray.shape[1], frame.gray.shape[0])
        is_high_risk = quick_risk > 0.15
        is_periodic = fi % max(depth_every, 1) == 0
        needs_depth = last_depth is None or is_periodic or is_high_risk
        depth_is_fresh = bool(needs_depth)
        if needs_depth:
            last_depth = estimate_frame_depth(frame)
        t_depth = time.perf_counter()

        # 2. Detect objects every N frames; between detections the tracker
        # propagates existing tracks so IDs and TTC history stay continuous.
        should_detect = (fi % max(detect_every, 1) == 0)
        if should_detect:
            detections = filter_relevant_detections(detector.detect(frame.bgr), lane)
            active_tracks = tracker.update(
                detections,
                frame_index=fi,
                timestamp_sec=video_frame.timestamp_sec,
            )
        else:
            active_tracks = tracker.propagate()
        t_yolo = time.perf_counter()

        if processed_frames % 30 == 0:
            log_line = (
                f"[FRAME {fi:4d}] "
                f"preprocess={1000*(t_preprocess-t0):.0f}ms  "
                f"lane={'skip' if fi % max(lane_every,1) != 0 else f'{1000*(t_lane-t_preprocess):.0f}ms':>7}  "
                f"flow={'skip' if fi % max(flow_every,1) != 0 else f'{1000*(t_flow-t_lane):.0f}ms':>7}  "
                f"depth={'skip' if not needs_depth else f'{1000*(t_depth-t_flow):.0f}ms':>7}  "
                f"yolo={'skip' if not should_detect else f'{1000*(t_yolo-t_depth):.0f}ms':>7}"
            )
            performance_logs.append(log_line)

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
        # The primary's RAW risk score is captured before stabilization
        # mutates the event, so it stays consistent with the entry in
        # ``all_events`` (and therefore with ``objects[primaryObjectId]``).
        # ``stabilized_risk_state`` carries the hysteresis-smoothed state
        # band independently.
        primary_risk_score = _risk_score(primary_event)
        stabilized_state = stabilized_event_state(stabilizer, primary_event)
        primary_event = replace(primary_event, state=stabilized_state)

        # Build the metadata-only payload first; heavy RGB views are deferred
        # so we can skip them entirely on frames that won't be saved or
        # previewed.
        event_payload = _event_payload_base(
            event=primary_event,
            all_events=all_events,
            primary_risk_score=primary_risk_score,
        )
        deferred = _DeferredRender(
            frame_bgr=frame.bgr,
            primary_event=primary_event,
            all_events=all_events,
            lane=lane,
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

        object_metrics = [_object_metric(ev) for ev in all_events if ev.object_id is not None]
        frame_row = {
            "frameIndex": primary_event.frame_index,
            "timestampSec": float(primary_event.timestamp_sec),
            "stabilizedRiskState": primary_event.state,
            "primaryObjectId": primary_event.object_id,
            "primaryRiskScore": primary_risk_score,
            "primaryLane": primary_event.lane,
            "objects": object_metrics,
        }
        frames.append(frame_row)
        if progress_callback is not None:
            preview_rows_buffer.append(frame_row)
        processed_frames += 1

        is_first_frame = (processed_frames == 1)
        if progress_callback is not None and (is_first_frame or processed_frames % max(1, int(progress_every)) == 0):
            annotated = annotate_frame(frame.bgr, primary_event, all_events, lane=lane)
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

    peak_event = saved_events[0] if saved_events else None
    return {
        "fps": loader.fps,
        "frame_count": loader.frame_count,
        "processed_frames": processed_frames,
        "frames": frames,
        "events": saved_events,
        "peak_event": peak_event,
        "performance_logs": performance_logs,
    }


def score_event_payload(payload: dict[str, Any]) -> float:
    return score_raw(
        state=str(payload.get("stabilized_risk_state") or "").upper(),
        ttc_sec=payload.get("ttc_sec"),
        near_score=float(payload.get("near_score") or 0.0),
        closing_speed=float(payload.get("closing_speed") or 0.0),
    )
