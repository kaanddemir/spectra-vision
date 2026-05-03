"""ONNX model wrappers for depth and optical flow.

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


_DEFAULT_DEPTH_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "depth_anything_v2_small.onnx"
_DEPTH_MODEL_PATH = Path(os.environ.get("SPECTRA_DEPTH_MODEL", str(_DEFAULT_DEPTH_MODEL_PATH)))

_INPUT_SIZE = 518
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_depth_model_lock = threading.Lock()
_depth_model_singleton: Optional["DepthAnythingONNX"] = None
_depth_load_failed = False


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


def is_depth_available() -> bool:
    """Whether the ONNX runtime and model file are usable."""

    if _depth_load_failed:
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return _DEPTH_MODEL_PATH.is_file()


def get_depth_model() -> Optional[DepthAnythingONNX]:
    """Return a cached depth model instance, or None if unavailable."""

    global _depth_model_singleton, _depth_load_failed
    if _depth_load_failed:
        return None
    if _depth_model_singleton is not None:
        return _depth_model_singleton

    with _depth_model_lock:
        if _depth_model_singleton is not None:
            return _depth_model_singleton
        if not is_depth_available():
            _depth_load_failed = True
            return None
        try:
            _depth_model_singleton = DepthAnythingONNX(_DEPTH_MODEL_PATH)
        except Exception:
            _depth_load_failed = True
            return None
        return _depth_model_singleton


_DEFAULT_FLOW_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "neuflow_v2.onnx"
_FLOW_MODEL_PATH = Path(os.environ.get("SPECTRA_FLOW_MODEL", str(_DEFAULT_FLOW_MODEL_PATH)))
_DEFAULT_FLOW_INPUT_HW = (432, 768)

_flow_model_lock = threading.Lock()
_flow_model_singleton: Optional["NeuFlowONNX"] = None
_flow_load_failed = False


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
            self.input_h, self.input_w = _DEFAULT_FLOW_INPUT_HW

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


def is_flow_available() -> bool:
    """Whether the ONNX runtime and flow model file are usable."""

    if _flow_load_failed:
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return _FLOW_MODEL_PATH.is_file()


def get_flow_model() -> Optional[NeuFlowONNX]:
    """Return a cached flow model instance, or None if unavailable."""

    global _flow_model_singleton, _flow_load_failed
    if _flow_load_failed:
        return None
    if _flow_model_singleton is not None:
        return _flow_model_singleton

    with _flow_model_lock:
        if _flow_model_singleton is not None:
            return _flow_model_singleton
        if not is_flow_available():
            _flow_load_failed = True
            return None
        try:
            _flow_model_singleton = NeuFlowONNX(_FLOW_MODEL_PATH)
        except Exception:
            _flow_load_failed = True
            return None
        return _flow_model_singleton
