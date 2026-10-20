"""RF-DETR through ONNX Runtime behind the engine's ``Detector`` protocol, for comparisons only.

The graph is RF-DETR's own ONNX export (``RFDETR.export(format="onnx")``): input ``input`` (1, 3, S, S)
ImageNet-normalised RGB; outputs ``dets`` (1, Q, 4) normalised cxcywh and ``labels`` (1, Q, 91) class logits
over sparse COCO ids. Preprocessing and decoding follow ``rfdetr.export._runtime`` (bilinear resize without
antialiasing, independent sigmoids, top-Q query/class pairs before thresholding), reimplemented here so the
runtime needs neither torch nor ``rfdetr``.

Everything after decoding (road-user classes, score floor, clipping, class-agnostic duplicate suppression)
is the engine's D-FINE adapter, so both detectors reach the tracker through identical post-processing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.detectors.rfdetr import COCO_CATEGORY_IDS
from spectra.engine.detection import DFineOnnxDetector
from spectra.engine.video import Frame

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
_CONTIGUOUS = np.full(91, -1)
_CONTIGUOUS[list(COCO_CATEGORY_IDS)] = np.arange(80)


def decode(boxes_cwh: np.ndarray, logits: np.ndarray, width: int, height: int, threshold: float):
    """(xyxy pixels, scores, zero-based contiguous COCO labels; -1 for unused ids), best first."""
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -88, 88)))
    flat = scores.ravel()
    q = logits.shape[0]
    top = np.argpartition(-flat, q - 1)[:q] if flat.size > q else np.arange(flat.size)
    top = top[np.argsort(-flat[top], kind="stable")]
    top = top[flat[top] > threshold]
    query, cls = np.divmod(top, logits.shape[1])
    cx, cy, w, h = boxes_cwh[query].T
    scale = np.array([width, height, width, height], np.float32)
    xyxy = np.clip(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1) * scale, 0, scale)
    return xyxy, flat[top], _CONTIGUOUS[cls]


class RfDetrOnnxDetector(DFineOnnxDetector):
    def __init__(self, model_path: Path | str, sha256: str, *, input_size: int = 512, name: str = "rfdetr-s-coco",
                 **kw) -> None:
        super().__init__(model_path, sha256, input_size=input_size, name=name, **kw)
        self.info = self.info.model_copy(update={"version": "rfdetr-onnx-1"})

    def load(self) -> None:
        import onnxruntime as ort
        from spectra.engine.video import file_sha256
        if not self.model_path.is_file():
            raise FileNotFoundError(f"detector model not found: {self.model_path}")
        actual = file_sha256(self.model_path)
        if actual != self.expected_sha256:
            raise ValueError(f"detector model {self.model_path.name} has sha256 {actual}, expected {self.expected_sha256}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(self.model_path), options, providers=["CPUExecutionProvider"])
        if [i.name for i in self._session.get_inputs()] != ["input"]:
            raise ValueError("unexpected RF-DETR inputs")

    def detect(self, frame: Frame) -> list:
        import cv2
        if self._session is None:
            raise RuntimeError("detector used before load()")
        h, w = frame.image.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(frame.image, (self.input_size, self.input_size),
                                      interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        x = ((rgb.transpose(2, 0, 1).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)[None]
        dets, logits = self._session.run(None, {"input": np.ascontiguousarray(x)})
        xyxy, scores, labels = decode(dets[0], logits[0], w, h, self.score_threshold)
        return self._parse(labels, xyxy, scores, w, h)
