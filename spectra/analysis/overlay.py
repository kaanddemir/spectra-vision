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
    proximity_pct = int(round((primary_event.near_score or 0) * 100))
    approach_pct = int(round(min(1.0, max(0.0, primary_event.closing_speed or 0.0)) * 100))
    crossing_pct = int(round((primary_event.crossing_risk or 0) * 100))
    confidence_pct = int(round((primary_event.confidence or 0) * 100))
    lane_lbl = _readable_lane(primary_event.lane)
    obj_lbl = (primary_event.object_type or "scene").upper()

    # Compact card pinned to top-left with a fixed footprint so it never
    # overflows narrow previews. Layout:
    #   row 1 — STATE pill + TTC
    #   row 2 — object · lane
    #   rows 3-6 — Prox / Appr / Cross / Conf bars
    card_x, card_y = 12, 10
    card_w = min(220, max(160, width - 24))
    pad = 10
    inner_x = card_x + pad
    inner_w = card_w - 2 * pad

    # background
    card_y2 = card_y + 116
    panel = output.copy()
    cv2.rectangle(panel, (card_x, card_y), (card_x + card_w, card_y2), (8, 12, 20), thickness=-1)
    cv2.addWeighted(panel, 0.78, output, 0.22, 0, output)
    cv2.rectangle(output, (card_x, card_y), (card_x + card_w, card_y2), (40, 50, 65), 1, cv2.LINE_AA)

    # row 1 — state pill + TTC
    pill_h = 18
    state_text = primary_event.state
    (stw, sth), _ = cv2.getTextSize(state_text, _FONT, 0.45, 1)
    pill_w = stw + 14
    pill_y = card_y + 8
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
        (inner_x + pill_w + 10, pill_y + pill_h - 4),
        _FONT,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    # row 2 — object · lane
    sub_parts = [obj_lbl]
    if lane_lbl and primary_event.bbox is not None:
        sub_parts.append(lane_lbl)
    sub_text = "  ·  ".join(sub_parts)
    cv2.putText(
        output,
        sub_text,
        (inner_x, card_y + 44),
        _FONT,
        0.4,
        (200, 210, 225),
        1,
        cv2.LINE_AA,
    )

    # rows 3-6 — metric bars
    metrics = [
        ("Prox", proximity_pct),
        ("Appr", approach_pct),
        ("Cross", crossing_pct),
        ("Conf", confidence_pct),
    ]
    bar_y = card_y + 60
    for label, pct in metrics:
        _draw_metric_bar(output, inner_x, bar_y, inner_w, label, pct, color)
        bar_y += 14

    return output
