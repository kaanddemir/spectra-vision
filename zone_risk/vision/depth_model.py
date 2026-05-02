"""ONNX-based monocular depth estimation (Depth Anything V2 Small).

Falls back gracefully if onnxruntime is not installed or the model file
is missing — callers should call `is_available()` first or rely on
`get_model()` returning None.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "depth_anything_v2_small.onnx"
_MODEL_PATH = Path(os.environ.get("SPECTRA_DEPTH_MODEL", str(_DEFAULT_MODEL_PATH)))

_INPUT_SIZE = 518
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_model_lock = threading.Lock()
_model_singleton: Optional["DepthAnythingONNX"] = None
_load_failed = False


class DepthAnythingONNX:
    """Thin wrapper around an ONNX Runtime depth-estimation session."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = os.cpu_count() or 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(model_path), sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Return a near-map (float32, [0, 1], larger = closer)."""

        height, width = rgb.shape[:2]
        resized = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - _MEAN) / _STD
        input_tensor = np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        depth = outputs[0]
        if depth.ndim == 4:
            depth = depth[0, 0]
        elif depth.ndim == 3:
            depth = depth[0]

        depth = cv2.resize(depth.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        d_min = float(depth.min())
        d_max = float(depth.max())
        if d_max - d_min < 1e-6:
            return np.zeros_like(depth, dtype=np.float32)
        return (depth - d_min) / (d_max - d_min)


def is_available() -> bool:
    """Whether the ONNX runtime and model file are usable."""

    if _load_failed:
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return _MODEL_PATH.is_file()


def get_model() -> Optional[DepthAnythingONNX]:
    """Return a cached depth model instance, or None if unavailable."""

    global _model_singleton, _load_failed
    if _load_failed:
        return None
    if _model_singleton is not None:
        return _model_singleton

    with _model_lock:
        if _model_singleton is not None:
            return _model_singleton
        if not is_available():
            _load_failed = True
            return None
        try:
            _model_singleton = DepthAnythingONNX(_MODEL_PATH)
        except Exception:
            _load_failed = True
            return None
        return _model_singleton
