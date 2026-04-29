"""Optional YOLO vehicle detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "motorcycle", "bicycle"}


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    label: str
    confidence: float
    id: int | None = None


class VehicleDetector:
    """YOLOv8 detector wrapper.

    If ultralytics is unavailable, detection is disabled and the rest of the
    pipeline still runs with zone-level risk.
    """

    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        enabled: bool = True,
        confidence_threshold: float = 0.35,
    ) -> None:
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names: dict[int, str] = {}

        if not enabled:
            return

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception:
            self.enabled = False
            return

        self.model = YOLO(model_name)
        names = getattr(self.model, "names", {})
        self.class_names = {int(key): str(value) for key, value in dict(names).items()}

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if not self.enabled or self.model is None:
            return []

        results = self.model.track(frame_bgr, persist=True, verbose=False, conf=self.confidence_threshold)
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                label = self.class_names.get(cls_id, str(cls_id))
                if label not in VEHICLE_CLASS_NAMES:
                    continue
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue
                obj_id = int(box.id[0]) if box.id is not None else None
                detections.append(Detection(bbox=(x1, y1, x2, y2), label=label, confidence=confidence, id=obj_id))

        return detections


def zone_detections(width: int, height: int) -> Iterable[Detection]:
    """Fallback regions when no object detector is active."""

    third = width // 3
    yield Detection((0, 0, third, height), "left zone", 0.0, id=101)
    yield Detection((third, 0, 2 * third, height), "center zone", 0.0, id=102)
    yield Detection((2 * third, 0, width, height), "right zone", 0.0, id=103)

