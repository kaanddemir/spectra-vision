"""Draw zone-based risk information on video frames."""

from __future__ import annotations

import cv2
import numpy as np

from .risk_calculator import RiskEvent


COLORS = {
    "SAFE": (80, 210, 120),
    "CAUTION": (0, 180, 255),
    "DANGER": (40, 50, 255),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_SEP_COLOR = (55, 68, 90)


def _readable_zone(zone: str | None) -> str:
    if not zone:
        return ""
    z = str(zone).lower().replace("_", " ").strip()
    if "left" in z:
        return "Left Lane"
    if "right" in z:
        return "Right Lane"
    if "center" in z:
        return "Same Lane"
    return z.title()


def _hud_row(img, segments, colors, font_scale, thickness, y_text, x0=24, sep=True, sep_thickness=2):
    """Draw text segments, optionally separated by vertical lines."""
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


def annotate_frame(
    frame_bgr: np.ndarray,
    primary_event: RiskEvent,
    zone_events: list[RiskEvent],
) -> np.ndarray:
    output = frame_bgr.copy()
    color = COLORS.get(primary_event.state, (220, 220, 220))

    for event in zone_events:
        if event.bbox is None:
            continue
        x1, y1, x2, y2 = event.bbox
        zone_color = COLORS.get(event.state, (100, 120, 140))
        cv2.rectangle(output, (x1, y1), (x2, y2), zone_color, 1)
        cv2.putText(output, _readable_zone(event.zone), (x1 + 8, 24),
                    _FONT, 0.42, (200, 210, 225), 1, cv2.LINE_AA)

    if primary_event.bbox is not None:
        x1, y1, x2, y2 = primary_event.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

    ttc_str = "--" if primary_event.ttc_sec is None else f"{primary_event.ttc_sec:.1f}s"
    near_pct = int(round((primary_event.near_score or 0) * 100))
    vel_pct  = int(round((primary_event.velocity_magnitude or 0) * 100))
    zone_lbl = _readable_zone(primary_event.zone)

    header_segs  = [primary_event.state, f"TTC {ttc_str}"]
    header_cols  = [color, color]
    if zone_lbl:
        header_segs.append(zone_lbl)
        header_cols.append(color)

    detail_segs  = [f"Nearness {near_pct}%", f"Closing Speed {vel_pct}%"]
    detail_cols  = [(190, 205, 220), (190, 205, 220)]

    box_y1, box_y2 = 12, 82
    box_w = min(output.shape[1] - 24, 620)
    cv2.rectangle(output, (12, box_y1), (12 + box_w, box_y2), (8, 12, 20), thickness=-1)

    _hud_row(output, header_segs, header_cols, 0.68, 2, 42, sep=False)
    _hud_row(output, detail_segs, detail_cols, 0.48, 1, 68, sep=False)

    return output
