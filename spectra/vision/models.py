"""ONNX model wrappers for depth estimation, plus the shared CoreML
provider builder used by every Spectra ONNX model (depth and UFLDv2).

Optical flow is computed classically (OpenCV DIS) in ``motion.py``. YOLO
runs through Ultralytics + PyTorch in ``detection.py``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np


_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_DEFAULT_DEPTH_MODEL_PATH = _MODELS_DIR / "depth_anything_v2_small.onnx"
_DEPTH_MODEL_PATH = Path(os.environ.get("SPECTRA_DEPTH_MODEL", str(_DEFAULT_DEPTH_MODEL_PATH)))

_INPUT_SIZE = int(os.environ.get("SPECTRA_DEPTH_INPUT", "256"))
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_COREML_CACHE_DIR = Path(
    os.environ.get("SPECTRA_COREML_CACHE_DIR", str(_MODELS_DIR / ".coreml_cache"))
)

_depth_model_lock = threading.Lock()
_depth_model_singleton: "DepthAnythingONNX" | None = None
_depth_load_failed = False
_depth_load_error: str | None = None

_DEPTH_UNAVAILABLE_MESSAGE = (
    "Depth Anything ONNX model unavailable. Install onnxruntime and ensure "
    "models/depth_anything_v2_small.onnx exists."
)


def build_coreml_providers() -> list:
    """Shared ONNX provider config for every Spectra ONNX model.

    Prefers CoreML on Apple Silicon (routes the model to ANE/GPU) and falls
    back to CPU when CoreML is unavailable. Uses ``MLProgram`` with a
    persistent ``ModelCacheDirectory`` so the CoreML graph compile cost
    (~2-5s for Depth Anything, similar for UFLDv2) is paid once and reused
    across restarts. Different models share the cache dir — CoreML keys
    cached graphs by hash so there is no collision.
    """

    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]

    providers: list = []
    if "CoreMLExecutionProvider" in available:
        _COREML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        providers.append(
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "RequireStaticInputShapes": "1",
                    "EnableOnSubgraphs": "0",
                    "ModelCacheDirectory": str(_COREML_CACHE_DIR),
                },
            )
        )
    providers.append("CPUExecutionProvider")
    return providers


class DepthAnythingONNX:
    """Thin wrapper around an ONNX Runtime depth-estimation session."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = os.cpu_count() or 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = build_coreml_providers()
        try:
            self.session = ort.InferenceSession(str(model_path), sess_options, providers=providers)
        except Exception:
            # CoreML init can throw for graphs that don't satisfy
            # RequireStaticInputShapes; retry on CPU only.
            if not any(
                isinstance(p, tuple) and p[0] == "CoreMLExecutionProvider" for p in providers
            ):
                raise
            self.session = ort.InferenceSession(
                str(model_path), sess_options, providers=["CPUExecutionProvider"]
            )
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

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return _DEPTH_MODEL_PATH.is_file()


def get_depth_model() -> DepthAnythingONNX:
    """Return a cached depth model instance, or raise if unavailable."""

    global _depth_model_singleton, _depth_load_failed, _depth_load_error
    if _depth_load_failed:
        raise RuntimeError(_depth_load_error or _DEPTH_UNAVAILABLE_MESSAGE)
    if _depth_model_singleton is not None:
        return _depth_model_singleton

    with _depth_model_lock:
        if _depth_load_failed:
            raise RuntimeError(_depth_load_error or _DEPTH_UNAVAILABLE_MESSAGE)
        if _depth_model_singleton is not None:
            return _depth_model_singleton
        if not is_depth_available():
            _depth_load_failed = True
            _depth_load_error = _DEPTH_UNAVAILABLE_MESSAGE
            raise RuntimeError(_depth_load_error)
        try:
            _depth_model_singleton = DepthAnythingONNX(_DEPTH_MODEL_PATH)
        except Exception as exc:
            _depth_load_failed = True
            _depth_load_error = f"Depth Anything ONNX model failed to load: {exc}"
            raise RuntimeError(_depth_load_error) from exc
        return _depth_model_singleton

