"""YOLOv8 object detection wrapper with M1 MPS / CPU device selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# COCO classes that are road-relevant for collision risk. Other detections
# are filtered out so the pipeline only reasons about traffic participants.
RELEVANT_CLASSES: dict[str, str] = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "train": "train",
    "truck": "truck",
    "traffic light": "traffic light",
    "stop sign": "stop sign",
}


# Per-class collision priors. Larger objects get higher trust on their
# expansion signal because they cover more pixels and tracking is stabler.
CLASS_RISK_WEIGHT: dict[str, float] = {
    "person": 1.10,
    "bicycle": 1.05,
    "motorcycle": 1.05,
    "car": 1.00,
    "bus": 0.95,
    "truck": 0.95,
    "train": 0.90,
    "traffic light": 0.40,
    "stop sign": 0.40,
}


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    class_name: str
    confidence: float


class ObjectDetector:
    """Lazy-loaded YOLOv8 detector. Returns [] if Ultralytics is unavailable."""

    def __init__(
        self,
        *,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.30,
        iou: float = 0.45,
        image_size: int = 640,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = int(image_size)
        self._device = device
        self._model = None
        self._load_attempted = False
        self._available = False

    def _resolve_device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _try_load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from ultralytics import YOLO

            model_path = self.model_name
            local_models = Path(__file__).resolve().parents[2] / "models" / self.model_name
            if local_models.exists():
                model_path = str(local_models)

            self._model = YOLO(model_path)
            self._device = self._resolve_device()
            try:
                self._model.to(self._device)
            except Exception:
                self._device = "cpu"
                self._model.to("cpu")
            self._available = True
        except Exception as exc:
            self._model = None
            self._available = False
            print(
                "[ObjectDetector] YOLO unavailable — every frame will report SAFE. "
                f"Install with `pip install -r requirements.txt`. ({exc})"
            )

    @property
    def available(self) -> bool:
        if not self._load_attempted:
            self._try_load()
        return self._available

    @property
    def device(self) -> str:
        return self._device or "cpu"

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if not self.available or self._model is None:
            return []

        try:
            results = self._model.predict(
                source=frame_bgr,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.image_size,
                device=self._device,
                verbose=False,
            )
        except Exception as exc:
            print(f"[ObjectDetector] inference failed: {exc}")
            return []

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return []

        names = result.names if hasattr(result, "names") else self._model.names
        xyxy = boxes.xyxy.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        conf = boxes.conf.detach().cpu().numpy()

        detections: list[Detection] = []
        height, width = frame_bgr.shape[:2]
        for box, c, p in zip(xyxy, cls, conf):
            class_name = names.get(int(c), str(c)) if isinstance(names, dict) else names[int(c)]
            if class_name not in RELEVANT_CLASSES:
                continue
            x1 = max(0, int(round(float(box[0]))))
            y1 = max(0, int(round(float(box[1]))))
            x2 = min(width, int(round(float(box[2]))))
            y2 = min(height, int(round(float(box[3]))))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    class_name=RELEVANT_CLASSES[class_name],
                    confidence=float(p),
                )
            )

        return detections


_GLOBAL_DETECTOR: ObjectDetector | None = None


def get_detector() -> ObjectDetector:
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None:
        _GLOBAL_DETECTOR = ObjectDetector()
    return _GLOBAL_DETECTOR
