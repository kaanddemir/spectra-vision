"""ONNX-based optical flow estimation (NeuFlow v2).

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


_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "neuflow_v2.onnx"
_MODEL_PATH = Path(os.environ.get("SPECTRA_FLOW_MODEL", str(_DEFAULT_MODEL_PATH)))
_DEFAULT_INPUT_HW = (432, 768)

_model_lock = threading.Lock()
_model_singleton: Optional["NeuFlowONNX"] = None
_load_failed = False


class NeuFlowONNX:
    """Thin wrapper around an ONNX Runtime optical-flow session."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = os.cpu_count() or 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(model_path), sess_options, providers=providers)
        inputs = self.session.get_inputs()
        if len(inputs) < 2:
            raise RuntimeError(f"NeuFlow ONNX expects two image inputs; got {len(inputs)}.")
        self.input_names = [inp.name for inp in inputs[:2]]

        first_shape = inputs[0].shape
        if len(first_shape) >= 4 and isinstance(first_shape[2], int) and isinstance(first_shape[3], int):
            self.input_h = int(first_shape[2])
            self.input_w = int(first_shape[3])
        else:
            self.input_h, self.input_w = _DEFAULT_INPUT_HW

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        return np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)

    def predict(self, prev_rgb: np.ndarray, curr_rgb: np.ndarray) -> np.ndarray:
        """Return dense optical flow at the source resolution as float32 (H, W, 2)."""

        height, width = curr_rgb.shape[:2]
        prev_in = self._preprocess(prev_rgb)
        curr_in = self._preprocess(curr_rgb)

        outputs = self.session.run(None, {self.input_names[0]: prev_in, self.input_names[1]: curr_in})
        flow = outputs[0]
        if flow.ndim == 4:
            flow = flow[0]
        flow = np.transpose(flow, (1, 2, 0)).astype(np.float32)

        scale_x = width / float(self.input_w)
        scale_y = height / float(self.input_h)
        flow_resized = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
        flow_resized[..., 0] *= scale_x
        flow_resized[..., 1] *= scale_y
        return flow_resized.astype(np.float32)


def is_available() -> bool:
    """Whether the ONNX runtime and model file are usable."""

    if _load_failed:
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return _MODEL_PATH.is_file()


def get_model() -> Optional[NeuFlowONNX]:
    """Return a cached flow model instance, or None if unavailable."""

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
            _model_singleton = NeuFlowONNX(_MODEL_PATH)
        except Exception:
            _load_failed = True
            return None
        return _model_singleton
