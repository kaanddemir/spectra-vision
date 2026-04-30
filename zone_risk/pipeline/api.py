"""FastAPI-facing zone-based risk analysis adapter."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .fusion import compute_quick_risk, fuse_frame_risk
from ..vision.depth_estimator import DepthResult, estimate_frame_depth
from ..vision.optical_flow import compute_velocity, flow_to_rgb
from ..vision.preprocess import preprocess_frame
from .annotator import annotate_frame
from .risk_calculator import RiskEvent
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


def _road_tracking_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """Generate a professional ADAS-style road tracking overlay."""
    output = frame_bgr.copy()
    h, w = output.shape[:2]
    overlay = output.copy()

    # 1. Draw Road ROI (Trapezoid)
    # Define a perspective-based road region
    pts = np.array([
        [int(w * 0.42), int(h * 0.60)], # Top-left
        [int(w * 0.58), int(h * 0.60)], # Top-right
        [int(w * 0.95), h],            # Bottom-right
        [int(w * 0.05), h]             # Bottom-left
    ], np.int32)
    
    cv2.fillPoly(overlay, [pts], (180, 140, 40)) # Soft blue-ish gold or amber? Let's use a tech-blue (180, 100, 40 is BGR)
    cv2.addWeighted(overlay, 0.25, output, 0.75, 0, output)

    # 2. Draw Lane Lines
    # Left lane
    cv2.line(output, (int(w * 0.42), int(h * 0.60)), (int(w * 0.10), h), (255, 255, 255), 2, cv2.LINE_AA)
    # Right lane
    cv2.line(output, (int(w * 0.58), int(h * 0.60)), (int(w * 0.90), h), (255, 255, 255), 2, cv2.LINE_AA)
    
    # 3. Predicted Path (Center Curve)
    path_pts = []
    for i in range(10):
        t = i / 9.0
        curr_y = int(h * (0.60 + t * 0.40))
        # Add a slight curve for aesthetics
        curve = np.sin(t * 2) * 15
        curr_x = int(w * 0.5 + curve)
        path_pts.append([curr_x, curr_y])
    
    path_pts = np.array(path_pts, np.int32)
    cv2.polylines(output, [path_pts], False, (0, 255, 0), 2, cv2.LINE_AA) # Green path

    # 4. Horizontal markers
    for i in range(1, 4):
        dist_y = int(h * (0.60 + i * 0.10))
        width_at_y = int(w * (0.16 + i * 0.20))
        x_start = int(w * 0.5 - width_at_y / 2)
        x_end = int(w * 0.5 + width_at_y / 2)
        cv2.line(output, (x_start, dist_y), (x_end, dist_y), (200, 200, 200), 1, cv2.LINE_AA)

    return _to_rgb(output)


def _hazard_score(event: RiskEvent) -> float:
    base = {
        "SAFE": 0.18,
        "CAUTION": 0.56,
        "DANGER": 0.88,
    }.get(event.state, 0.0)
    signal = min(1.0, (0.52 * event.near_score) + (0.48 * event.closing_speed))
    return round(float(max(base, signal)), 3)


def _zone_metric(event: RiskEvent) -> dict[str, Any]:
    return {
        "zone": event.zone,
        "score": _hazard_score(event),
        "mean_depth": event.near_score,
        "motion_energy": event.velocity_magnitude,
        "expansion_energy": event.closing_speed,
        "structure_signal": event.confidence,
        "near_ratio": event.near_score,
        "estimated_ttc_sec": event.ttc_sec,
        "direction_hint": event.direction,
        "object_type": event.object_type,
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
        "hazard_score": _hazard_score(event),
        "hazard_band": band,
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
        "road_rgb": _road_tracking_rgb(frame_bgr),
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
    start_sec: float = 0.0,
    end_sec: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_every: int = 6,
) -> dict[str, Any]:
    """Run zone-based risk analysis and return the UI-compatible result shape."""

    loader = VideoLoader(video_path, max_frames=max_processed_frames, start_sec=start_sec, end_sec=end_sec)

    previous_gray: np.ndarray | None = None
    last_depth: DepthResult | None = None
    saved_events: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    from .risk_calculator import StateStabilizer
    stabilizer = StateStabilizer(upgrade_frames=3, downgrade_frames=5)
    processed_frames = 0

    _STATE_RANK = {"SAFE": 0, "CAUTION": 1, "DANGER": 2}
    _RANK_STATE = {v: k for k, v in _STATE_RANK.items()}
    window_min_ttc: float | None = None
    window_worst_rank: int = 0
    window_worst_zone: str | None = None

    for video_frame in loader.frames():
        frame = preprocess_frame(video_frame.bgr, max_side=resize_max_side)
        flow = compute_velocity(previous_gray, frame.gray)
        previous_gray = frame.gray

        # 1. Fast Analysis: Estimate quick risk from motion (optical flow)
        quick_risk = compute_quick_risk(flow, frame.gray.shape[1], frame.gray.shape[0])

        # 2. Decisions:
        # - High risk: if motion-based risk is above threshold
        # - Periodic: recompute depth every N frames even if low risk
        is_high_risk = quick_risk > 0.15
        is_periodic = video_frame.frame_index % max(depth_every, 1) == 0

        if last_depth is None or is_high_risk or is_periodic:
            last_depth = estimate_frame_depth(frame)

        # 3. Final Fusion: Combine flow and current (or reused) depth
        primary_event, all_events = fuse_frame_risk(
            frame_index=video_frame.frame_index,
            timestamp_sec=video_frame.timestamp_sec,
            depth=last_depth,
            flow=flow,
        )

        # 4. Smooth State Transitions (Hysteresis)
        stabilized_state = stabilizer.process(primary_event.state)
        # Update primary_event with stabilized state
        from dataclasses import replace
        primary_event = replace(primary_event, state=stabilized_state)

        annotated = annotate_frame(frame.bgr, primary_event, all_events)
        event_payload = _event_payload(
            event=primary_event,
            all_events=all_events,
            frame_bgr=frame.bgr,
            annotated_bgr=annotated,
            region_overlay_rgb=_region_overlay_rgb(frame.bgr, all_events),
            depth_rgb=_depth_rgb(last_depth),
            motion_rgb=flow_to_rgb(flow.flow),
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
            zone_scores[z_key] = _hazard_score(ev)

        timeline_rows.append(
            {
                "Frame": primary_event.frame_index,
                "Time (s)": round(primary_event.timestamp_sec, 2),
                "State": primary_event.state,
                "Band": BAND_BY_STATE.get(primary_event.state, "low"),
                "Zone": primary_event.zone,
                "Direction": primary_event.direction,
                "TTC (s)": primary_event.ttc_sec,
                "Near": primary_event.near_score,
                "Closing": primary_event.closing_speed,
                "ZoneScores": zone_scores,
            }
        )
        processed_frames += 1

        # Track worst-case TTC and state across the interval since the last preview push
        if primary_event.ttc_sec is not None:
            if window_min_ttc is None or primary_event.ttc_sec < window_min_ttc:
                window_min_ttc = float(primary_event.ttc_sec)
        cur_rank = _STATE_RANK.get(primary_event.state, 0)
        if cur_rank >= window_worst_rank:
            window_worst_rank = cur_rank
            window_worst_zone = primary_event.zone

        is_first_frame = (processed_frames == 1)
        if progress_callback is not None and (is_first_frame or processed_frames % max(1, int(progress_every)) == 0):
            preview_uri = _encode_preview_jpeg(annotated)
            progress_pct = min(100.0, round((processed_frames / max(1, max_processed_frames)) * 100.0, 1))
            zone_metrics_payload = [
                {
                    "zone": ev.zone,
                    "score": _hazard_score(ev),
                    "estimated_ttc_sec": None if ev.ttc_sec is None else float(ev.ttc_sec),
                    "near_score": float(ev.near_score),
                    "closing_speed": float(ev.closing_speed),
                }
                for ev in all_events
            ]
            worst_state = _RANK_STATE.get(window_worst_rank, primary_event.state)
            worst_zone = window_worst_zone or primary_event.zone
            window_ttc = window_min_ttc
            timeline_row_payload = {
                "Time (s)": round(float(primary_event.timestamp_sec), 2),
                "State": worst_state,
                "TTC (s)": None if window_ttc is None else float(window_ttc),
                "Zone": worst_zone,
            }
            try:
                progress_callback(
                    {
                        "type": "preview",
                        "frameIndex": int(primary_event.frame_index),
                        "timestampSec": float(primary_event.timestamp_sec),
                        "progress": progress_pct,
                        "riskState": worst_state,
                        "ttcSec": None if window_ttc is None else float(window_ttc),
                        "zone": worst_zone,
                        "frame": preview_uri,
                        "zoneMetrics": zone_metrics_payload,
                        "timelineRow": timeline_row_payload,
                    }
                )
            except Exception:
                pass
            # Reset interval aggregators for the next push window
            window_min_ttc = None
            window_worst_rank = 0
            window_worst_zone = None

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
    state = str(payload.get("risk_state") or "").upper()
    state_weight = {"SAFE": 0.0, "CAUTION": 1.0, "DANGER": 2.0}.get(state, 0.0)
    ttc = payload.get("estimated_ttc_sec")
    ttc_weight = 0.0 if ttc is None else max(0.0, 3.0 - float(ttc)) / 3.0
    near = float(payload.get("near_score") or 0.0)
    closing = float(payload.get("closing_speed") or 0.0)
    return state_weight + ttc_weight + near + closing
