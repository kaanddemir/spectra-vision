"""Detector adapters with a common interface, for benchmarking only.

Each adapter returns detections as an (N, 6) float array of
``x1, y1, x2, y2, score, coco_class_id`` in source-frame pixels.
Adapters import their heavy dependencies lazily, inside ``load``.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # zero-based COCO ids


class Detector(Protocol):
    name: str

    def load(self) -> None: ...

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray: ...
