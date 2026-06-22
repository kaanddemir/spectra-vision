"""YOLOv8 object detection wrapper with M1 MPS / CPU device selection.

Loading mirrors the lane (``lanenet.py``) and depth (``models.py``) pattern:

- ``is_yolo_available()`` — fast file + import check, no model load.
- ``get_detector()`` — returns the cached singleton, lazy-loads on first
  call, raises ``RuntimeError`` if the load itself fails. The caller
  (``_ensure_required_models()`` in ``analysis/video.py``) drives the
  load eagerly at startup so model issues surface there rather than
  mid-video.

Device fallback (MPS → CUDA → CPU) and the ``model.to(device)`` retry are
internal recoveries; they are not load failures.
"""

from __future__ import annotations

import threading
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
    # Traffic lights are detected for the advisory colour-state cue. They are
    # NOT collision participants — callers split them out before tracking
    # (see analysis/video.py).
    "traffic light": "traffic_light",
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
    "traffic_light": 0.0,  # advisory only, never a collision participant
}

CLASS_MIN_CONFIDENCE: dict[str, float] = {
    "person": 0.45,
    "bicycle": 0.45,
    "motorcycle": 0.45,
    "car": 0.50,
    "bus": 0.50,
    "train": 0.55,
    "truck": 0.50,
    "traffic_light": 0.40,
}


_DEFAULT_YOLO_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "yolov8n.pt"


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    class_name: str
    confidence: float


def is_yolo_available() -> bool:
    """Whether Ultralytics is importable and the YOLO weights are present."""

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return _DEFAULT_YOLO_MODEL_PATH.is_file()


class ObjectDetector:
    """YOLOv8 detector. Construction is cheap; load happens explicitly."""

    def __init__(
        self,
        *,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.45,
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

    def _load(self) -> None:
        """Load weights and place them on the best available device.

        Lets ``ImportError`` and any YOLO file errors propagate so
        ``get_detector()`` can convert them into a single ``RuntimeError``.
        Device-level fallback (MPS/CUDA → CPU) is handled internally
        because that is recovery, not failure.
        """

        from ultralytics import YOLO

        local_models = Path(__file__).resolve().parents[2] / "models" / self.model_name
        model_path = str(local_models) if local_models.exists() else self.model_name
        self._model = YOLO(model_path)
        self._device = self._resolve_device()
        try:
            self._model.to(self._device)
        except Exception:
            self._device = "cpu"
            self._model.to("cpu")

    @property
    def device(self) -> str:
        return self._device or "cpu"

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if self._model is None:
            raise RuntimeError("YOLOv8 detector is not loaded.")

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
            raise RuntimeError(f"YOLOv8 inference failed: {exc}") from exc

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
            normalized_class = RELEVANT_CLASSES[class_name]
            if float(p) < CLASS_MIN_CONFIDENCE.get(normalized_class, self.confidence):
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
                    class_name=normalized_class,
                    confidence=float(p),
                )
            )

        return detections


_GLOBAL_DETECTOR: ObjectDetector | None = None
_DETECTOR_LOCK = threading.Lock()


def get_detector() -> ObjectDetector:
    """Return the cached detector, loading it on first call.

    Raises ``RuntimeError`` if Ultralytics fails to import or if model
    weights cannot be loaded — matches the lane/depth ONNX patterns so
    backend issues surface uniformly through ``_ensure_required_models``.
    """

    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is not None:
        return _GLOBAL_DETECTOR

    with _DETECTOR_LOCK:
        if _GLOBAL_DETECTOR is not None:
            return _GLOBAL_DETECTOR
        instance = ObjectDetector()
        try:
            instance._load()
        except Exception as exc:
            raise RuntimeError(
                f"YOLOv8 detector failed to load: {exc}. "
                "Install Ultralytics (`pip install -r requirements.txt`) "
                "and ensure models/yolov8n.pt exists."
            ) from exc
        _GLOBAL_DETECTOR = instance
        return _GLOBAL_DETECTOR
