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

# Per-class acceptance floors applied AFTER YOLO's own ``conf`` pre-filter.
# Vehicle floors (car/truck/bus) are deliberately low: a very close lead vehicle
# whose bbox fills the bottom of the frame is a hard case for yolov8n and its
# confidence sags, yet it is the single most safety-relevant object. The
# downstream ego-corridor filter (``detection_corridor_score``) drops far/side
# junk, so a low YOLO floor here mainly recovers in-path boxes rather than
# flooding the tracker. The detector's predict-time ``conf`` must stay <= the
# smallest floor below, or these classes are filtered before we ever see them.
# NOTE: the low vehicle floor applies only to NEAR/LARGE boxes (the close lead
# vehicle it is meant to recover). Far, small vehicle boxes keep the stricter
# ``_VEHICLE_FAR_MIN_CONFIDENCE`` so low-confidence distant traffic does not
# flood the tracker with jittery boxes that produce non-physical depth velocities.
CLASS_MIN_CONFIDENCE: dict[str, float] = {
    "person": 0.45,
    "bicycle": 0.45,
    "motorcycle": 0.45,
    "car": 0.35,
    "bus": 0.35,
    "train": 0.55,
    "truck": 0.35,
    "traffic_light": 0.40,
}

# Vehicle classes whose low near-floor is gated by box size/position.
_VEHICLE_CLASSES = frozenset({"car", "bus", "truck"})
# Stricter floor for far/small vehicle boxes (restores the pre-recovery 0.50).
_VEHICLE_FAR_MIN_CONFIDENCE = 0.50


def _vehicle_box_is_near_or_large(
    bbox: tuple[int, int, int, int],
    *,
    frame_h: int,
) -> bool:
    """Whether a vehicle box is close enough to deserve the low confidence floor.

    Mirrors the near/large test in ``road.detection_corridor_score`` so the
    detector and corridor filter agree on what "close lead vehicle" means.
    """

    if frame_h <= 0:
        return False
    _, y1, _, y2 = bbox
    bottom_frac = float(y2) / float(frame_h)
    height_frac = float(y2 - y1) / float(frame_h)
    return bottom_frac >= 0.76 or height_frac >= 0.20


def _class_min_confidence(
    normalized_class: str,
    bbox: tuple[int, int, int, int],
    *,
    frame_h: int,
    default: float,
) -> float:
    """Per-class acceptance floor, raised for far/small vehicle boxes."""

    floor = CLASS_MIN_CONFIDENCE.get(normalized_class, default)
    if normalized_class in _VEHICLE_CLASSES and not _vehicle_box_is_near_or_large(
        bbox, frame_h=frame_h
    ):
        floor = max(floor, _VEHICLE_FAR_MIN_CONFIDENCE)
    return floor


_DEFAULT_YOLO_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "yolov8n.pt"


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    class_name: str
    confidence: float


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _merge_detections(
    primary: list[Detection],
    extra: list[Detection],
    *,
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Merge a second-pass detection list into ``primary`` via greedy NMS.

    An ``extra`` detection is appended only when it does not overlap an existing
    same-class detection above ``iou_threshold``; when it does overlap, the
    higher-confidence box replaces the lower one. This keeps the near-band pass
    from double-counting a vehicle already found in the full-frame pass.
    """

    merged = list(primary)
    for cand in extra:
        best_idx = -1
        best_iou = iou_threshold
        for idx, existing in enumerate(merged):
            if existing.class_name != cand.class_name:
                continue
            iou = _bbox_iou(existing.bbox, cand.bbox)
            if iou >= best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx < 0:
            merged.append(cand)
        elif cand.confidence > merged[best_idx].confidence:
            merged[best_idx] = cand
    return merged


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
        # Predict-time floor. Kept at/below the smallest vehicle entry in
        # ``CLASS_MIN_CONFIDENCE`` (0.35) so low-confidence close vehicles reach
        # the per-class gate instead of being dropped inside YOLO. Per-class
        # floors then restore the effective threshold for every other class.
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

    def detect(self, frame_bgr: np.ndarray, *, near_band: bool = False) -> list[Detection]:
        """Detect road participants in ``frame_bgr``.

        With ``near_band=True`` a second inference pass runs on the lower-center
        crop (where a very close lead vehicle lives) and is merged into the
        full-frame result. The crop is fed to YOLO at the same ``imgsz``, so the
        close vehicle occupies far more of the network input and is recovered
        even when the full-frame pass drops it. Callers gate this on the threat
        actually being absent to avoid paying for a second pass every frame.
        """

        if self._model is None:
            raise RuntimeError("YOLOv8 detector is not loaded.")

        height, width = frame_bgr.shape[:2]
        detections = self._detect_region(frame_bgr, x_offset=0, y_offset=0, full_w=width, full_h=height)

        if near_band:
            # Lower-center band: bottom 45% of rows, central 70% of columns.
            y0 = int(height * 0.55)
            x0 = int(width * 0.15)
            x1 = int(width * 0.85)
            crop = frame_bgr[y0:height, x0:x1]
            if crop.shape[0] >= 16 and crop.shape[1] >= 16:
                band = self._detect_region(
                    crop, x_offset=x0, y_offset=y0, full_w=width, full_h=height
                )
                detections = _merge_detections(detections, band)

        return detections

    def _detect_region(
        self,
        image_bgr: np.ndarray,
        *,
        x_offset: int,
        y_offset: int,
        full_w: int,
        full_h: int,
    ) -> list[Detection]:
        """Run YOLO on ``image_bgr`` and map boxes back into full-frame coords.

        ``image_bgr`` may be the whole frame (offsets 0) or a crop; detections
        are offset by ``(x_offset, y_offset)`` and clamped to the full frame.
        """

        try:
            results = self._model.predict(
                source=image_bgr,
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
        for box, c, p in zip(xyxy, cls, conf):
            class_name = names.get(int(c), str(c)) if isinstance(names, dict) else names[int(c)]
            if class_name not in RELEVANT_CLASSES:
                continue
            normalized_class = RELEVANT_CLASSES[class_name]
            x1 = max(0, int(round(float(box[0]))) + x_offset)
            y1 = max(0, int(round(float(box[1]))) + y_offset)
            x2 = min(full_w, int(round(float(box[2]))) + x_offset)
            y2 = min(full_h, int(round(float(box[3]))) + y_offset)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            # Size-gated floor: far/small vehicle boxes keep the stricter floor;
            # near/large ones get the low recovery floor. Evaluated on the
            # full-frame box so the crop pass uses the same geometry.
            min_conf = _class_min_confidence(
                normalized_class, (x1, y1, x2, y2), frame_h=full_h, default=self.confidence
            )
            if float(p) < min_conf:
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
