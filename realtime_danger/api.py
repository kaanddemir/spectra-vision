"""FastAPI-facing realtime danger analysis adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


from .annotator import annotate_frame
from .depth_estimator import DepthResult, estimate_frame_depth
from .fusion import fuse_frame_risk
from .optical_flow import compute_velocity, flow_to_rgb
from .preprocess import preprocess_frame
from .risk_calculator import RiskEvent, score_event
from .vehicle_detector import Detection, VehicleDetector
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


def _region_overlay_rgb(frame_bgr: np.ndarray, detections: list[Detection], events: list[RiskEvent]) -> np.ndarray:
    output = frame_bgr.copy()
    height, width = output.shape[:2]
    if detections:
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (96, 165, 250), 2)
            cv2.putText(
                output,
                detection.label,
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (230, 236, 245),
                1,
                cv2.LINE_AA,
            )
    else:
        for x in (width // 3, 2 * width // 3):
            cv2.line(output, (x, 0), (x, height), (96, 165, 250), 2)
        labels = [("left", width // 6), ("center", width // 2), ("right", 5 * width // 6)]
        for label, x in labels:
            cv2.putText(output, label, (x - 28, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 236, 245), 1)

    for event in events:
        if event.bbox is None:
            continue
        x1, y1, x2, y2 = event.bbox
        color = (40, 50, 255) if event.state == "DANGER" else (0, 180, 255) if event.state == "CAUTION" else (80, 210, 120)
        cv2.rectangle(output, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), color, 1)
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
        "motion_rgb": motion_rgb,
        "overlay_rgb": _to_rgb(annotated_bgr),
    }


def analyze_realtime_video(
    video_path: str | Path,
    *,
    max_processed_frames: int,
    max_saved_events: int,
    resize_max_side: int,
    depth_every: int = 3,
    detect_every: int = 3,
    yolo_model: str = "yolov8s.pt",
    enable_yolo: bool = True,
) -> dict[str, Any]:
    """Run realtime danger analysis and return the UI-compatible result shape."""

    loader = VideoLoader(video_path, max_frames=max_processed_frames)
    detector = VehicleDetector(model_name=yolo_model, enabled=enable_yolo)

    previous_gray: np.ndarray | None = None
    last_depth: DepthResult | None = None
    last_detections: list[Detection] = []
    saved_events: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    processed_frames = 0

    try:
        for video_frame in loader.frames():
            frame = preprocess_frame(video_frame.bgr, max_side=resize_max_side)
            flow = compute_velocity(previous_gray, frame.gray)
            previous_gray = frame.gray

            if last_depth is None or video_frame.frame_index % max(depth_every, 1) == 0:
                last_depth = estimate_frame_depth(frame)

            if video_frame.frame_index % max(detect_every, 1) == 0:
                last_detections = detector.detect(frame.bgr)

            primary_event, all_events = fuse_frame_risk(
                frame_index=video_frame.frame_index,
                timestamp_sec=video_frame.timestamp_sec,
                depth=last_depth,
                flow=flow,
                detections=last_detections,
            )


            annotated = annotate_frame(frame.bgr, primary_event, last_detections)
            event_payload = _event_payload(
                event=primary_event,
                all_events=all_events,
                frame_bgr=frame.bgr,
                annotated_bgr=annotated,
                region_overlay_rgb=_region_overlay_rgb(frame.bgr, last_detections, all_events),
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
                }
            )
            processed_frames += 1

    finally:
        pass

    peak_event = saved_events[0] if saved_events else None
    return {
        "media_type": "video",
        "pipeline": "realtime_danger",
        "fps": loader.fps,
        "frame_count": loader.frame_count,
        "processed_frames": processed_frames,
        "sampled_frames": processed_frames,
        "timeline_rows": timeline_rows,
        "events": saved_events,
        "peak_event": peak_event,
        "summary": None if peak_event is None else peak_event["heuristic_summary"],
        "yolo_enabled": detector.enabled,
    }


def score_event_payload(payload: dict[str, Any]) -> float:
    state = str(payload.get("risk_state") or "").upper()
    state_weight = {"SAFE": 0.0, "CAUTION": 1.0, "DANGER": 2.0}.get(state, 0.0)
    ttc = payload.get("estimated_ttc_sec")
    ttc_weight = 0.0 if ttc is None else max(0.0, 3.0 - float(ttc)) / 3.0
    near = float(payload.get("near_score") or 0.0)
    closing = float(payload.get("closing_speed") or 0.0)
    return state_weight + ttc_weight + near + closing
