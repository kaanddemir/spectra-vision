"""Draw risk information on video frames."""

from __future__ import annotations

import cv2
import numpy as np

from .risk_calculator import RiskEvent
from .vehicle_detector import Detection


COLORS = {
    "SAFE": (80, 210, 120),
    "CAUTION": (0, 180, 255),
    "DANGER": (40, 50, 255),
}


def annotate_frame(
    frame_bgr: np.ndarray,
    primary_event: RiskEvent,
    detections: list[Detection],
) -> np.ndarray:
    output = frame_bgr.copy()
    color = COLORS.get(primary_event.state, (220, 220, 220))

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (100, 120, 140), 1)
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(output, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 226, 235), 1)

    if primary_event.bbox is not None:
        x1, y1, x2, y2 = primary_event.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

    ttc = "--" if primary_event.ttc_sec is None else f"{primary_event.ttc_sec:.2f}s"
    header = f"{primary_event.state} | TTC {ttc} | {primary_event.zone}/{primary_event.direction}"
    cv2.rectangle(output, (12, 12), (min(output.shape[1] - 12, 610), 82), (8, 12, 20), thickness=-1)
    cv2.putText(output, header, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    detail = (
        f"{primary_event.object_type} | near {primary_event.near_score:.2f} | "
        f"velocity {primary_event.velocity_magnitude:.2f}"
    )
    cv2.putText(output, detail, (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (232, 236, 241), 1, cv2.LINE_AA)

    return output

