"""Draw per-object risk overlays on video frames."""

from __future__ import annotations

import cv2
import numpy as np

from .risk import RiskEvent
from ..vision.road import LaneFrame, lane_edges_at_y


COLORS = {
    "SAFE": (80, 210, 120),
    "CAUTION": (0, 180, 255),
    "DANGER": (40, 50, 255),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _readable_lane(lane: str | None) -> str:
    if not lane:
        return ""
    value = str(lane).lower().strip()
    if "left" in value:
        return "Left"
    if "right" in value:
        return "Right"
    if "center" in value:
        return "Same Lane"
    return value.title()


def _draw_metric_bar(
    img: np.ndarray,
    x: int,
    y: int,
    width: int,
    label: str,
    pct: int,
    color: tuple[int, int, int],
) -> None:
    pct = max(0, min(100, int(pct)))
    cv2.putText(img, label, (x, y), _FONT, 0.34, (170, 180, 195), 1, cv2.LINE_AA)
    pct_text = f"{pct}%"
    (pw, _), _ = cv2.getTextSize(pct_text, _FONT, 0.34, 1)
    cv2.putText(img, pct_text, (x + width - pw, y), _FONT, 0.34, (220, 225, 235), 1, cv2.LINE_AA)
    track_y = y + 4
    cv2.rectangle(img, (x, track_y), (x + width, track_y + 3), (38, 46, 60), thickness=-1)
    fill_w = int(round(width * pct / 100.0))
    if fill_w > 0:
        cv2.rectangle(img, (x, track_y), (x + fill_w, track_y + 3), color, thickness=-1)


def _draw_bbox(output: np.ndarray, event: RiskEvent) -> None:
    if event.bbox is None:
        return
    x1, y1, x2, y2 = event.bbox
    color = COLORS.get(event.state, (160, 170, 190))
    thickness = 2 if event.state == "DANGER" else 1

    cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    label_parts = []
    if event.object_id is not None:
        label_parts.append(f"#{event.object_id}")
    label_parts.append(event.object_type.upper())
    if event.ttc_sec is not None:
        label_parts.append(f"{event.ttc_sec:.1f}s")
    label = " ".join(label_parts)

    (tw, th), baseline = cv2.getTextSize(label, _FONT, 0.42, 1)
    pad = 4
    label_y1 = max(0, y1 - th - 2 * pad)
    label_y2 = label_y1 + th + 2 * pad
    label_x2 = min(output.shape[1], x1 + tw + 2 * pad)
    cv2.rectangle(output, (x1, label_y1), (label_x2, label_y2), color, thickness=-1)
    cv2.putText(output, label, (x1 + pad, label_y2 - pad - 1), _FONT, 0.42, (15, 18, 24), 1, cv2.LINE_AA)


def _default_corridor(width: int, height: int) -> np.ndarray:
    return np.array(
        [
            (int(width * 0.30), height),
            (int(width * 0.46), int(height * 0.50)),
            (int(width * 0.54), int(height * 0.50)),
            (int(width * 0.70), height),
        ],
        np.int32,
    )


def _lane_corridor(width: int, height: int, lane: LaneFrame | None) -> np.ndarray:
    if lane is None or not lane.detected:
        return _default_corridor(width, height)

    y_bottom = height - 1
    y_top = int(height * 0.58)
    left_bottom, right_bottom = lane_edges_at_y(lane, y_bottom)
    left_top, right_top = lane_edges_at_y(lane, y_top)
    return np.array(
        [
            (int(round(left_bottom)), y_bottom),
            (int(round(left_top)), y_top),
            (int(round(right_top)), y_top),
            (int(round(right_bottom)), y_bottom),
        ],
        np.int32,
    )


def _draw_dashed_line(
    output: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    *,
    thickness: int,
    dash_px: int = 14,
    gap_px: int = 9,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    length = float(np.hypot(x2 - x1, y2 - y1))
    if length <= 1.0:
        return
    step = dash_px + gap_px
    for start in np.arange(0.0, length, step):
        end = min(start + dash_px, length)
        t0 = start / length
        t1 = end / length
        a = (int(round(x1 + (x2 - x1) * t0)), int(round(y1 + (y2 - y1) * t0)))
        b = (int(round(x1 + (x2 - x1) * t1)), int(round(y1 + (y2 - y1) * t1)))
        cv2.line(output, a, b, color, thickness, cv2.LINE_AA)


def _draw_lane_overlay(
    output: np.ndarray,
    lane: LaneFrame | None,
    state_color: tuple[int, int, int],
) -> None:
    height, width = output.shape[:2]
    corridor = _lane_corridor(width, height, lane)
    confidence = float(np.clip(lane.confidence if lane is not None else 0.25, 0.0, 1.0))
    detected = bool(lane is not None and lane.detected and confidence >= 0.35)

    # Slightly separate non-ego-lane space without making the video feel like
    # a debug mask. The lane itself stays readable but unobtrusive.
    outside_mask = np.ones((height, width), dtype=np.uint8)
    cv2.fillPoly(outside_mask, [corridor], 0)
    darkened = output.copy()
    darkened[outside_mask.astype(bool)] = (darkened[outside_mask.astype(bool)] * 0.94).astype(np.uint8)
    cv2.addWeighted(darkened, 0.52, output, 0.48, 0, output)

    if detected:
        fill_color = (
            int(0.55 * state_color[0] + 0.45 * 120),
            int(0.55 * state_color[1] + 0.45 * 190),
            int(0.55 * state_color[2] + 0.45 * 185),
        )
    else:
        fill_color = (90, 112, 112)

    left_bottom, left_top, right_top, right_bottom = [tuple(map(int, p)) for p in corridor]
    inner_bottom_ratio = 0.18
    inner_top_ratio = 0.30
    for idx, t0 in enumerate(np.linspace(0.0, 0.82, 5)):
        t1 = min(1.0, t0 + 0.24)
        outer_lb = (
            int(round(left_bottom[0] + (left_top[0] - left_bottom[0]) * t0)),
            int(round(left_bottom[1] + (left_top[1] - left_bottom[1]) * t0)),
        )
        outer_rb = (
            int(round(right_bottom[0] + (right_top[0] - right_bottom[0]) * t0)),
            int(round(right_bottom[1] + (right_top[1] - right_bottom[1]) * t0)),
        )
        outer_lt = (
            int(round(left_bottom[0] + (left_top[0] - left_bottom[0]) * t1)),
            int(round(left_bottom[1] + (left_top[1] - left_bottom[1]) * t1)),
        )
        outer_rt = (
            int(round(right_bottom[0] + (right_top[0] - right_bottom[0]) * t1)),
            int(round(right_bottom[1] + (right_top[1] - right_bottom[1]) * t1)),
        )
        inset0 = inner_bottom_ratio + ((inner_top_ratio - inner_bottom_ratio) * t0)
        inset1 = inner_bottom_ratio + ((inner_top_ratio - inner_bottom_ratio) * t1)
        lb = (
            int(round(outer_lb[0] + (outer_rb[0] - outer_lb[0]) * inset0)),
            outer_lb[1],
        )
        rb = (
            int(round(outer_rb[0] + (outer_lb[0] - outer_rb[0]) * inset0)),
            outer_rb[1],
        )
        lt = (
            int(round(outer_lt[0] + (outer_rt[0] - outer_lt[0]) * inset1)),
            outer_lt[1],
        )
        rt = (
            int(round(outer_rt[0] + (outer_lt[0] - outer_rt[0]) * inset1)),
            outer_rt[1],
        )
        band = np.array([lb, lt, rt, rb], dtype=np.int32)
        band_overlay = output.copy()
        cv2.fillPoly(band_overlay, [band], fill_color)
        base_alpha = 0.24 if detected else 0.11
        alpha = max(0.035, base_alpha * (1.0 - (idx * 0.17)) * (0.62 + (0.38 * confidence)))
        cv2.addWeighted(band_overlay, alpha, output, 1.0 - alpha, 0, output)

    edge_color = (214, 228, 222) if detected else (115, 130, 128)
    edge_shadow = (42, 64, 62)
    edge_thickness = 1 if confidence < 0.80 else 2
    if confidence >= 0.55:
        cv2.line(output, left_bottom, left_top, edge_shadow, edge_thickness + 1, cv2.LINE_AA)
        cv2.line(output, right_top, right_bottom, edge_shadow, edge_thickness + 1, cv2.LINE_AA)
        cv2.line(output, left_bottom, left_top, edge_color, edge_thickness, cv2.LINE_AA)
        cv2.line(output, right_top, right_bottom, edge_color, edge_thickness, cv2.LINE_AA)
    else:
        _draw_dashed_line(output, left_bottom, left_top, edge_shadow, thickness=edge_thickness + 1)
        _draw_dashed_line(output, right_top, right_bottom, edge_shadow, thickness=edge_thickness + 1)
        _draw_dashed_line(output, left_bottom, left_top, edge_color, thickness=edge_thickness)
        _draw_dashed_line(output, right_top, right_bottom, edge_color, thickness=edge_thickness)


def annotate_frame(
    frame_bgr: np.ndarray,
    primary_event: RiskEvent,
    object_events: list[RiskEvent],
    lane: LaneFrame | None = None,
) -> np.ndarray:
    output = frame_bgr.copy()
    height, width = output.shape[:2]
    color = COLORS.get(primary_event.state, (220, 220, 220))

    # 1. Forward collision corridor. This is presentation only; risk uses the
    # same lane geometry upstream and adjusts trust via lane confidence.
    _draw_lane_overlay(output, lane, color)

    # 2. Per-object bboxes — paint SAFE first, then CAUTION/DANGER on top so
    # the worst object stays visually dominant when bboxes overlap.
    state_order = {"SAFE": 0, "CAUTION": 1, "DANGER": 2}
    for event in sorted(object_events, key=lambda e: state_order.get(e.state, 0)):
        _draw_bbox(output, event)

    ttc_str = "--" if primary_event.ttc_sec is None else f"{primary_event.ttc_sec:.1f}s"
    proximity_pct = int(round((primary_event.near_score or 0) * 100))
    approach_pct = int(round(min(1.0, max(0.0, primary_event.closing_speed or 0.0)) * 100))
    crossing_pct = int(round((primary_event.crossing_risk or 0) * 100))
    confidence_pct = int(round((primary_event.confidence or 0) * 100))
    lane_lbl = _readable_lane(primary_event.lane)
    obj_lbl = (primary_event.object_type or "scene").upper()

    # 3. Telemetry HUD card. Only visible in CAUTION/DANGER states to keep 
    # the SAFE view clean, matching the event timeline logic.
    if primary_event.state in {"CAUTION", "DANGER"}:
        card_x, card_y = 12, 10
        card_w = min(220, max(160, width - 24))
        pad = 10
        inner_x = card_x + pad
        inner_w = card_w - 2 * pad

        # background
        card_y2 = card_y + 64
        panel = output.copy()
        cv2.rectangle(panel, (card_x, card_y), (card_x + card_w, card_y2), (8, 12, 20), thickness=-1)
        cv2.addWeighted(panel, 0.78, output, 0.22, 0, output)
        cv2.rectangle(output, (card_x, card_y), (card_x + card_w, card_y2), (40, 50, 65), 1, cv2.LINE_AA)

        # row 1 — state pill + TTC
        pill_h = 18
        state_text = primary_event.state
        (stw, sth), _ = cv2.getTextSize(state_text, _FONT, 0.45, 1)
        pill_w = stw + 14
        pill_y = card_y + 10
        cv2.rectangle(output, (inner_x, pill_y), (inner_x + pill_w, pill_y + pill_h), color, thickness=-1)
        cv2.putText(
            output,
            state_text,
            (inner_x + 7, pill_y + pill_h - 5),
            _FONT,
            0.45,
            (15, 18, 24),
            1,
            cv2.LINE_AA,
        )
        ttc_label = f"TTC {ttc_str}"
        cv2.putText(
            output,
            ttc_label,
            (inner_x + pill_w + 12, pill_y + pill_h - 4),
            _FONT,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        # row 2 — ID | object | lane
        sub_parts = []
        if primary_event.object_id is not None:
            sub_parts.append(f"#{primary_event.object_id}")
        sub_parts.append(obj_lbl)
        if lane_lbl and primary_event.bbox is not None:
            sub_parts.append(lane_lbl)
        sub_text = " | ".join(sub_parts)
        cv2.putText(
            output,
            sub_text,
            (inner_x, card_y + 50),
            _FONT,
            0.4,
            (200, 210, 225),
            1,
            cv2.LINE_AA,
        )

    return output
