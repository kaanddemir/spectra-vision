"""FastAPI-facing zone-based risk analysis adapter."""

from __future__ import annotations

import base64
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .fusion import compute_quick_risk, fuse_frame_risk
from ..vision.depth_estimator import DepthResult, estimate_frame_depth
from ..vision.optical_flow import compute_velocity, flow_to_rgb
from ..vision.preprocess import preprocess_frame
from ..vision.road_roi import RoadROI, estimate_road_roi, fallback_road_roi
from .annotator import annotate_frame
from .risk_calculator import (
    MetricEmaSmoother,
    RiskEvent,
    StateStabilizer,
    score_raw,
    select_primary_event,
    stabilized_event_state,
)
from .video_loader import VideoLoader


BAND_BY_STATE = {
    "SAFE": "low",
    "CAUTION": "medium",
    "DANGER": "critical",
}


def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _depth_rgb(depth: DepthResult) -> np.ndarray:
    colorized = cv2.applyColorMap(depth.depth_map, cv2.COLORMAP_INFERNO)
    return _to_rgb(colorized)


def _region_overlay_rgb(frame_bgr: np.ndarray, events: list[RiskEvent]) -> np.ndarray:
    output = frame_bgr.copy()
    height, width = output.shape[:2]
    
    # Draw subtle top guides for zones
    w25 = int(width * 0.25)
    w75 = int(width * 0.75)
    
    # Just draw subtle markers at the top
    for x in (w25, w75):
        cv2.line(output, (x, 0), (x, 12), (70, 80, 95), 1)

    labels = [("Left", w25 // 2), ("Same", width // 2), ("Right", (width + w75) // 2)]
    for label, x in labels:
        cv2.putText(output, label, (x - 20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 210), 1, cv2.LINE_AA)

    return _to_rgb(output)


def _road_tracking_rgb(frame_bgr: np.ndarray, road_roi: RoadROI | None = None) -> np.ndarray:
    """Generate an ADAS-style road tracking overlay from the active ROI."""

    road_roi = road_roi or fallback_road_roi(frame_bgr.shape)
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
    ttc_pressure = 0.0 if event.ttc_sec is None else max(0.0, 3.0 - float(event.ttc_sec)) / 3.0
    return round(float(max(state_floor, (0.48 * signal) + (0.52 * ttc_pressure))), 3)


def _zone_metric(event: RiskEvent) -> dict[str, Any]:
    return {
        "zone": event.zone,
        "score": _risk_score(event),
        "meanDepth": event.near_score,
        "motionEnergy": event.velocity_magnitude,
        "expansionEnergy": event.closing_speed,
        "structureSignal": event.confidence,
        "nearRatio": event.near_score,
        "ttcSec": event.ttc_sec,
        "directionHint": event.direction,
        "objectType": event.object_type,
    }


def _event_payload(
    *,
    event: RiskEvent,
    all_events: list[RiskEvent],
    frame_bgr: np.ndarray,
    annotated_bgr: np.ndarray,
    region_overlay_rgb: np.ndarray,
    depth_rgb: np.ndarray,
    motion_rgb: np.ndarray,
    road_roi: RoadROI | None = None,
) -> dict[str, Any]:
    band = BAND_BY_STATE.get(event.state, "low")
    confidence_pct = round(event.confidence * 100.0, 1)
    uncertainty_pct = round((1.0 - event.confidence) * 100.0, 1)
    summary = f"{event.state} in the {event.zone} zone"
    if event.ttc_sec is not None:
        summary += f", TTC {event.ttc_sec:.2f}s"
    summary += f", direction {event.direction}"


    return {
        "frame_index": event.frame_index,
        "timestamp_sec": event.timestamp_sec,
        "risk_score": _risk_score(event),
        "risk_band": band,
        "risk_state": event.state,
        "primary_zone": event.zone,
        "estimated_ttc_sec": event.ttc_sec,
        "confidence_pct": confidence_pct,
        "uncertainty_pct": uncertainty_pct,
        "heuristic_summary": summary,
        "reasons": [event.reason, f"{event.object_type} in {event.zone}", f"motion direction: {event.direction}"],
        "zone_metrics": [_zone_metric(item) for item in all_events],
        "object_type": event.object_type,
        "approach": "approaching" if event.state in {"CAUTION", "DANGER"} else "stable",
        "lane": event.zone,
        "bbox": list(event.bbox) if event.bbox is not None else None,
        "near_score": event.near_score,
        "closing_speed": event.closing_speed,
        "velocity_magnitude": event.velocity_magnitude,
        "payload": {
            "risk_state": event.state,
            "object_type": event.object_type,
            "zone": event.zone,
            "direction": event.direction,
            "ttc_sec": event.ttc_sec,
            "near_score": event.near_score,
            "closing_speed": event.closing_speed,
            "confidence_pct": confidence_pct,
        },
        "original_rgb": _to_rgb(frame_bgr),
        "depth_rgb": depth_rgb,
        "segmentation_rgb": region_overlay_rgb,
        "road_rgb": _road_tracking_rgb(frame_bgr, road_roi),
        "motion_rgb": motion_rgb,
        "overlay_rgb": _to_rgb(annotated_bgr),
    }


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


def analyze_zone_video(
    video_path: str | Path,
    *,
    max_processed_frames: int,
    max_saved_events: int,
    resize_max_side: int,
    depth_every: int = 10,
    enable_road_roi: bool = False,
    use_depth_model: bool = True,
    use_flow_model: bool = True,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_every: int = 6,
) -> dict[str, Any]:
    """Run zone-based risk analysis and return the UI-compatible result shape."""

    loader = VideoLoader(video_path, max_frames=None, start_sec=start_sec, end_sec=end_sec)
    # Clamp max_processed_frames to the video's actual frame count so short
    # videos are always analyzed in full regardless of the caller's default.
    if loader.frame_count > 0:
        max_processed_frames = min(max_processed_frames, loader.frame_count)
    loader.max_frames = max_processed_frames

    previous_frame = None
    last_depth: DepthResult | None = None
    saved_events: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    preview_rows_buffer: list[dict[str, Any]] = []
    metric_smoother = MetricEmaSmoother()
    stabilizer = StateStabilizer(upgrade_frames=3, downgrade_frames=5)
    processed_frames = 0

    for video_frame in loader.frames():
        frame = preprocess_frame(video_frame.bgr, max_side=resize_max_side)
        road_roi = estimate_road_roi(frame.bgr) if enable_road_roi else fallback_road_roi(frame.bgr.shape)
        flow = compute_velocity(previous_frame, frame, use_flow_model=use_flow_model)
        previous_frame = frame

        # 1. Fast Analysis: Estimate quick risk from motion (optical flow)
        quick_risk = compute_quick_risk(flow, frame.gray.shape[1], frame.gray.shape[0])

        # 2. Decisions:
        # - High risk: if motion-based risk is above threshold
        # - Periodic: recompute depth every N frames even if low risk
        is_high_risk = quick_risk > 0.15
        is_periodic = video_frame.frame_index % max(depth_every, 1) == 0

        # DL model is expensive — only run on schedule, not on every high-risk frame.
        # Classical cues are cheap enough to re-run on high-risk frames.
        needs_depth = last_depth is None or is_periodic or (not use_depth_model and is_high_risk)
        if needs_depth:
            last_depth = estimate_frame_depth(frame, use_depth_model=use_depth_model)

        # 3. Final Fusion: Combine flow and current (or reused) depth
        _, raw_events = fuse_frame_risk(
            frame_index=video_frame.frame_index,
            timestamp_sec=video_frame.timestamp_sec,
            depth=last_depth,
            flow=flow,
            road_roi=road_roi if enable_road_roi else None,
        )
        all_events = metric_smoother.smooth_events(raw_events)
        primary_event = select_primary_event(all_events)

        # 4. Smooth State Transitions (Hysteresis)
        stabilized_state = stabilized_event_state(stabilizer, primary_event)
        primary_event = replace(
            primary_event,
            state=stabilized_state,
            ttc_sec=None if stabilized_state == "SAFE" else primary_event.ttc_sec,
        )

        annotated = annotate_frame(frame.bgr, primary_event, all_events)
        event_payload = _event_payload(
            event=primary_event,
            all_events=all_events,
            frame_bgr=frame.bgr,
            annotated_bgr=annotated,
            region_overlay_rgb=_region_overlay_rgb(frame.bgr, all_events),
            depth_rgb=_depth_rgb(last_depth),
            motion_rgb=flow_to_rgb(flow.flow),
            road_roi=road_roi,
        )
        # Deduplicate events within 1.0 second window
        new_score = score_event_payload(event_payload)
        replaced = False
        for i, saved in enumerate(saved_events):
            if abs(saved["timestamp_sec"] - primary_event.timestamp_sec) <= 1.0:
                if new_score > score_event_payload(saved):
                    saved_events[i] = event_payload
                replaced = True
                break

        if not replaced:
            saved_events.append(event_payload)

        saved_events = sorted(saved_events, key=lambda item: score_event_payload(item), reverse=True)[:max_saved_events]

        # Map all zone scores for this frame to allow dynamic UI updates during playback
        zone_scores = {}
        for ev in all_events:
            z_key = str(ev.zone).lower().split()[0] # "left", "center", "right"
            zone_scores[z_key] = _risk_score(ev)

        timeline_row = {
            "frameIndex": primary_event.frame_index,
            "timeSec": round(primary_event.timestamp_sec, 2),
            "riskState": primary_event.state,
            "riskBand": BAND_BY_STATE.get(primary_event.state, "low"),
            "zone": primary_event.zone,
            "direction": primary_event.direction,
            "ttcSec": primary_event.ttc_sec,
            "nearScore": primary_event.near_score,
            "closingSpeed": primary_event.closing_speed,
            "zoneScores": zone_scores,
        }
        timeline_rows.append(timeline_row)
        if progress_callback is not None:
            preview_rows_buffer.append(timeline_row)
        processed_frames += 1

        is_first_frame = (processed_frames == 1)
        if progress_callback is not None and (is_first_frame or processed_frames % max(1, int(progress_every)) == 0):
            preview_uri = _encode_preview_jpeg(annotated)
            progress_pct = min(100.0, round((processed_frames / max(1, max_processed_frames)) * 100.0, 1))
            zone_metrics_payload = [
                {
                    "zone": ev.zone,
                    "score": _risk_score(ev),
                    "estimated_ttc_sec": None if ev.ttc_sec is None else float(ev.ttc_sec),
                    "near_score": float(ev.near_score),
                    "closing_speed": float(ev.closing_speed),
                }
                for ev in all_events
            ]
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
                        "zone": primary_event.zone,
                        "nearScore": float(primary_event.near_score),
                        "closingSpeed": float(primary_event.closing_speed),
                        "frame": preview_uri,
                        "zoneMetrics": zone_metrics_payload,
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
                    "zone": final_row["zone"],
                    "nearScore": float(final_row["nearScore"]),
                    "closingSpeed": float(final_row["closingSpeed"]),
                    "timelineRow": final_row,
                    "timelineRows": list(preview_rows_buffer),
                }
            )
        except Exception:
            pass
        preview_rows_buffer.clear()

    peak_event = saved_events[0] if saved_events else None
    return {
        "media_type": "video",
        "pipeline": "zone_risk",
        "fps": loader.fps,
        "frame_count": loader.frame_count,
        "processed_frames": processed_frames,
        "sampled_frames": processed_frames,
        "timeline_rows": timeline_rows,
        "events": saved_events,
        "peak_event": peak_event,
        "summary": None if peak_event is None else peak_event["heuristic_summary"],
        "analysis_mode": "zone",
    }


def score_event_payload(payload: dict[str, Any]) -> float:
    return score_raw(
        state=str(payload.get("risk_state") or "").upper(),
        ttc_sec=payload.get("estimated_ttc_sec"),
        near_score=float(payload.get("near_score") or 0.0),
        closing_speed=float(payload.get("closing_speed") or 0.0),
    )
