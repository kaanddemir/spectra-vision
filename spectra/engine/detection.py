"""D-FINE object detector run through ONNX Runtime, plus class-agnostic duplicate suppression.

The ONNX graph is exported once from the official D-FINE checkpoint (Apache-2.0 code, COCO-trained
weights) and contains the model and its post-processor: inputs ``images`` (1, 3, S, S) RGB in [0, 1]
and ``orig_target_sizes`` (1, 2) as (width, height); outputs ``labels``, ``boxes`` (xyxy in the given
size) and ``scores`` for a fixed number of queries. Nothing is downloaded: the file path and its
sha256 are explicit and checked on load.

DETR-style detectors skip non-maximum suppression and sometimes report one vehicle twice under two
classes (car and truck), which fragments tracks. ``suppress_duplicates`` keeps the highest-scoring
box among vehicle boxes overlapping above an IoU threshold, whatever their class.

The camera vehicle's own hood is sometimes detected as a car. It is deliberately not removed here: a
geometric rule (large box touching the bottom edge) would also remove a real vehicle at the moment of
a collision, which is the footage this product exists for. Hood suppression belongs to the
segmentation model's ego-vehicle mask.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from spectra.engine.adapters import ComponentFailure
from spectra.engine.contracts import Box, ComponentInfo, Detection, ObjectCategory, Role
from spectra.engine.video import Frame, file_sha256

COCO_TO_CATEGORY = {0: ObjectCategory.PERSON, 1: ObjectCategory.BICYCLE, 2: ObjectCategory.CAR,
                    3: ObjectCategory.MOTORCYCLE, 5: ObjectCategory.BUS, 7: ObjectCategory.TRUCK}
VEHICLES = frozenset({ObjectCategory.CAR, ObjectCategory.MOTORCYCLE, ObjectCategory.BUS, ObjectCategory.TRUCK})



def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against many."""
    ix = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter)


def suppress_duplicates(detections: Sequence[Detection], iou_threshold: float) -> list[Detection]:
    """Greedy class-agnostic NMS over vehicle detections; other categories pass through unchanged."""
    vehicles = sorted((d for d in detections if d.category in VEHICLES), key=lambda d: -d.score)
    others = [d for d in detections if d.category not in VEHICLES]
    if not vehicles:
        return list(detections)
    boxes = np.array([[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in vehicles])
    keep = np.ones(len(vehicles), bool)
    for i in range(len(vehicles)):
        if keep[i] and i + 1 < len(vehicles):
            rest = np.arange(i + 1, len(vehicles))
            keep[rest[_iou(boxes[i], boxes[rest]) > iou_threshold]] = False
    return [v for v, k in zip(vehicles, keep) if k] + others


class DFineOnnxDetector:
    """``Detector`` adapter. Returns road users above ``score_threshold`` in frame-image pixels."""

    def __init__(self, model_path: Path | str, sha256: str, *, input_size: int = 640, score_threshold: float = 0.1,
                 duplicate_iou: Optional[float] = 0.7, threads: int = 4, name: str = "dfine-n-coco",
                 interval_s: Optional[float] = None) -> None:
        self.model_path, self.expected_sha256 = Path(model_path), sha256
        self.input_size, self.score_threshold, self.duplicate_iou = input_size, score_threshold, duplicate_iou
        self.threads = threads
        self.info = ComponentInfo(
            role=Role.DETECTOR, name=name, version="onnx-1", checkpoint_sha256=sha256, interval_s=interval_s,
            params={"input_size": input_size, "score_threshold": score_threshold, "duplicate_iou": duplicate_iou,
                    "categories": sorted(c.value for c in COCO_TO_CATEGORY.values()), "runtime": "onnxruntime-cpu"})
        self._session = None

    def load(self) -> None:
        import onnxruntime as ort
        if not self.model_path.is_file():
            raise FileNotFoundError(f"detector model not found: {self.model_path}")
        actual = file_sha256(self.model_path)
        if actual != self.expected_sha256:
            raise ValueError(f"detector model {self.model_path.name} has sha256 {actual}, expected "
                             f"{self.expected_sha256}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(self.model_path), options, providers=["CPUExecutionProvider"])
        names = {i.name for i in self._session.get_inputs()}
        if names != {"images", "orig_target_sizes"}:
            raise ValueError(f"unexpected detector inputs {sorted(names)}")

    def close(self) -> None:
        self._session = None

    def detect(self, frame: Frame) -> list[Detection]:
        import cv2
        if self._session is None:
            raise RuntimeError("detector used before load()")
        h, w = frame.image.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(frame.image, (self.input_size, self.input_size),
                                      interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        labels, boxes, scores = self._session.run(
            None, {"images": x, "orig_target_sizes": np.array([[w, h]], dtype=np.int64)})
        return self._parse(labels[0], boxes[0], scores[0], w, h)

    def _parse(self, labels: np.ndarray, boxes: np.ndarray, scores: np.ndarray, w: int, h: int) -> list[Detection]:
        if not (len(labels) == len(boxes) == len(scores)) or boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ComponentFailure(f"invalid output: shapes {labels.shape}, {boxes.shape}, {scores.shape}")
        if not (np.all(np.isfinite(boxes)) and np.all(np.isfinite(scores))):
            raise ComponentFailure("invalid output: non-finite boxes or scores")
        out = []
        for label, box, score in zip(labels.astype(int), boxes.astype(np.float64), scores.astype(np.float64)):
            category = COCO_TO_CATEGORY.get(int(label))
            if category is None or score < self.score_threshold:
                continue
            x1, y1, x2, y2 = np.clip(box[0], 0, w), np.clip(box[1], 0, h), np.clip(box[2], 0, w), np.clip(box[3], 0, h)
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(Detection(box=Box(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                                 score=float(min(score, 1.0)), category=category))
        if self.duplicate_iou is not None:
            out = suppress_duplicates(out, self.duplicate_iou)
        return sorted(out, key=lambda d: (-d.score, d.box.x1, d.box.y1))
