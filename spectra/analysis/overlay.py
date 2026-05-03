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
_SEP_COLOR = (55, 68, 90)


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


def _hud_row(img, segments, colors, font_scale, thickness, y_text, x0=24, sep=True, sep_thickness=2):
    gap = 20
    (_, th), baseline = cv2.getTextSize("Ag", _FONT, font_scale, thickness)
    sep_y1 = y_text - th - 3
    sep_y2 = y_text + baseline + 3
    x = x0
    for i, (text, col) in enumerate(zip(segments, colors)):
        if i > 0 and sep:
            sep_x = x - gap // 2
            cv2.line(img, (sep_x, sep_y1), (sep_x, sep_y2), _SEP_COLOR, sep_thickness, cv2.LINE_AA)
        cv2.putText(img, text, (x, y_text), _FONT, font_scale, col, thickness, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(text, _FONT, font_scale, thickness)
        x += tw + gap


def _draw_bbox(output: np.ndarray, event: RiskEvent) -> None:
    if event.bbox is None:
        return
    x1, y1, x2, y2 = event.bbox
    color = COLORS.get(event.state, (160, 170, 190))
    thickness = 2 if event.state == "DANGER" else 1

    cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    label_parts = [event.object_type.upper()]
    if event.ttc_sec is not None:
        label_parts.append(f"{event.ttc_sec:.1f}s")
    if event.object_id is not None:
        label_parts.append(f"#{event.object_id}")
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


def _component_summary(event: RiskEvent) -> str:
    if not event.ttc_components:
        return "TTCx --"
    parts = []
    for component in event.ttc_components:
        label = component.name[:1].upper()
        value = "--" if component.value is None else f"{component.value:.1f}"
        parts.append(f"{label}:{value}")
    return "TTCx " + " ".join(parts)


def annotate_frame(
    frame_bgr: np.ndarray,
    primary_event: RiskEvent,
    object_events: list[RiskEvent],
    lane: LaneFrame | None = None,
) -> np.ndarray:
    output = frame_bgr.copy()
    height, width = output.shape[:2]
    color = COLORS.get(primary_event.state, (220, 220, 220))

    # 1. Forward collision corridor. Use detected lane geometry when present;
    # otherwise keep the historical fixed perspective corridor.
    cone_pts = _lane_corridor(width, height, lane)
    overlay = output.copy()
    cv2.fillPoly(overlay, [cone_pts], color)
    cv2.addWeighted(overlay, 0.10, output, 0.90, 0, output)
    cv2.polylines(output, [cone_pts], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)

    # 2. Per-object bboxes — paint SAFE first, then CAUTION/DANGER on top so
    # the worst object stays visually dominant when bboxes overlap.
    state_order = {"SAFE": 0, "CAUTION": 1, "DANGER": 2}
    for event in sorted(object_events, key=lambda e: state_order.get(e.state, 0)):
        _draw_bbox(output, event)

    ttc_str = "--" if primary_event.ttc_sec is None else f"{primary_event.ttc_sec:.1f}s"
    near_pct = int(round((primary_event.near_score or 0) * 100))
    expansion_pct = int(round(min(1.0, max(0.0, primary_event.expansion_rate)) * 100))
    crossing_pct = int(round((primary_event.crossing_risk or 0) * 100))
    lane_lbl = _readable_lane(primary_event.lane)
    obj_lbl = (primary_event.object_type or "scene").upper()

    header_segs = [primary_event.state, f"TTC {ttc_str}", obj_lbl]
    header_cols = [color, color, color]
    if lane_lbl and primary_event.bbox is not None:
        header_segs.append(lane_lbl)
        header_cols.append(color)

    detail_segs = [
        f"Expansion {expansion_pct}%",
        f"Crossing {crossing_pct}%",
        f"Nearness {near_pct}%",
        _component_summary(primary_event),
    ]
    detail_cols = [(190, 205, 220)] * len(detail_segs)

    box_y1, box_y2 = 10, 54
    max_w = 0
    for row_segs, scale in [(header_segs, 0.52), (detail_segs, 0.38)]:
        tw_total = 0
        for s in row_segs:
            (tw, _), _ = cv2.getTextSize(s, _FONT, scale, 1)
            tw_total += tw + 20
        max_w = max(max_w, tw_total)

    box_w = max_w + 12
    cv2.rectangle(output, (12, box_y1), (12 + box_w, box_y2), (8, 12, 20), thickness=-1)
    cv2.rectangle(output, (12, box_y1), (12 + box_w, box_y2), (40, 50, 65), thickness=1)

    _hud_row(output, header_segs, header_cols, 0.52, 1, 32, x0=12 + 10, sep=False)
    _hud_row(output, detail_segs, detail_cols, 0.38, 1, 47, x0=12 + 10, sep=False)

    return output
