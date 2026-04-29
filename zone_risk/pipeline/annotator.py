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
    height, width = output.shape[:2]
    color = COLORS.get(primary_event.state, (220, 220, 220))

    # 1. Draw Collision Cone (Trapezoid) with dynamic risk color
    # Match the logic in fusion.py: Bottom 50%, Top 10%
    pt_bl = (int(width * 0.25), height)
    pt_br = (int(width * 0.75), height)
    pt_tl = (int(width * 0.45), 0)
    pt_tr = (int(width * 0.55), 0)
    
    # Use risk-based color for the cone
    cone_color = color # Use the already determined primary state color
    
    # Draw subtle cone lines and fill
    cone_pts = np.array([pt_bl, pt_tl, pt_tr, pt_br], np.int32)
    overlay = output.copy()
    cv2.fillPoly(overlay, [cone_pts], cone_color)
    cv2.addWeighted(overlay, 0.12, output, 0.88, 0, output)
    cv2.polylines(output, [cone_pts], isClosed=False, color=cone_color, thickness=1, lineType=cv2.LINE_AA)

    # 2. Draw Zones
    for event in zone_events:
        if event.bbox is None:
            continue
        x1, y1, x2, y2 = event.bbox
        zone_color = COLORS.get(event.state, (100, 120, 140))
        
        # Subtle zone vertical dividers
        cv2.line(output, (x1, 0), (x1, height), (60, 70, 85), 1, cv2.LINE_AA)
        
        # Draw small zone label
        cv2.putText(output, _readable_zone(event.zone), (x1 + 8, height - 12),
                    _FONT, 0.38, (200, 210, 225), 1, cv2.LINE_AA)

    # (Primary event rectangle removed to localize risk visualization)

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

    # 3. Draw HUD (Top-Left)
    box_y1, box_y2 = 10, 54
    # Calculate box width based on text
    max_w = 0
    for row_segs, scale in [(header_segs, 0.52), (detail_segs, 0.38)]:
        tw_total = 0
        for s in row_segs:
            (tw, _), _ = cv2.getTextSize(s, _FONT, scale, 1)
            tw_total += tw + 20
        max_w = max(max_w, tw_total)
    
    box_w = max_w + 12
    cv2.rectangle(output, (12, box_y1), (12 + box_w, box_y2), (8, 12, 20), thickness=-1)
    cv2.rectangle(output, (12, box_y1), (12 + box_w, box_y2), (40, 50, 65), thickness=1) # Subtle border

    # Draw rows centered in the box
    _hud_row(output, header_segs, header_cols, 0.52, 1, 32, x0=12+10, sep=False)
    _hud_row(output, detail_segs, detail_cols, 0.38, 1, 47, x0=12+10, sep=False)

    return output
