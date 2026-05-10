"""ONNX lane detection wrapper (UFLDv2, CULane ResNet-18 export).

Expected model file: ``models/ufld_v2_culane_r18.onnx``

How to obtain it:
    The upstream repository (cfzd/Ultra-Fast-Lane-Detection-v2) ships with a
    PyTorch ``.pth`` checkpoint and an export script. Run the repo's
    ``deploy/export_onnx.py`` against ``culane_res18.pth`` and copy the
    resulting ``.onnx`` file into this project's ``models/`` directory under
    the name above. Override the path with ``SPECTRA_LANENET_MODEL`` if your
    layout differs.

    UFLDv2 is a hard requirement. When the model file is absent,
    ``_ensure_required_models()`` in ``analysis/video.py`` raises
    ``RuntimeError`` at startup. There is no classical (Hough) fallback —
    benchmarking on real dashcam footage showed Hough traces the road's
    outer edges instead of the ego corridor, which is more dangerous than
    a clear failure.

Expected I/O contract (from upstream CULane R18 config):

    Input  : (1, 3, 320, 1600) float32, ImageNet-normalized RGB
    Outputs: 4 tensors keyed roughly as
        loc_row   : (1, num_grid_row,   num_cls_row, num_lane_row)
        loc_col   : (1, num_grid_col,   num_cls_col, num_lane_col)
        exist_row : (1, 2,              num_cls_row, num_lane_row)
        exist_col : (1, 2,              num_cls_col, num_lane_col)

    With CULane defaults: num_grid_row=200, num_cls_row=72, num_lane_row=4,
    num_grid_col=100, num_cls_col=81, num_lane_col=4. Lane indices map to
    [left-left, left, right, right-right].

``_decode_lanes`` resolves outputs by shape rather than by name, so it tolerates
exports that rename tensors. If your export uses a different anchor count or
omits the existence head, the decoder degrades gracefully (anchors are
resampled linearly; existence is gated on softmax confidence).

The session prefers ``CoreMLExecutionProvider`` on Apple Silicon — the same
provider routes Depth Anything V2 to the ANE/GPU. CPU is a transparent
fallback when CoreML is unavailable.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np

from .models import build_coreml_providers


_DEFAULT_LANENET_PATH = Path(__file__).resolve().parents[2] / "models" / "ufld_v2_culane_r18.onnx"
_LANENET_PATH = Path(os.environ.get("SPECTRA_LANENET_MODEL", str(_DEFAULT_LANENET_PATH)))

_INPUT_H = int(os.environ.get("SPECTRA_LANENET_INPUT_H", "320"))
_INPUT_W = int(os.environ.get("SPECTRA_LANENET_INPUT_W", "1600"))
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# CULane preset: the network only attends to the lower portion of the frame
# (sky/horizon are cropped before resize to match training distribution).
_CROP_RATIO = float(os.environ.get("SPECTRA_LANENET_CROP_RATIO", "0.6"))

# Row anchor sample positions in the cropped image (normalized).
# Defaults follow the upstream CULane config — 72 anchors spanning the
# bottom 60% of the cropped region.
_ROW_ANCHOR_RANGE = (0.42, 1.0)
_NUM_ROW_ANCHORS = 72
_NUM_LANES = 4

_lanenet_lock = threading.Lock()
_lanenet_singleton: "UFLDv2ONNX" | None = None
_lanenet_load_failed = False
_lanenet_load_error: str | None = None

_LANENET_UNAVAILABLE_MESSAGE = (
    "UFLDv2 ONNX model unavailable. Install onnxruntime and ensure "
    "models/ufld_v2_culane_r18.onnx exists."
)


class UFLDv2ONNX:
    """Thin wrapper around an ONNX Runtime UFLDv2 session.

    Produces a list of up to 4 lane curves, each as an ``(N, 2)`` array of
    ``(x, y)`` image coordinates in the original frame's pixel space.
    """

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = os.cpu_count() or 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = build_coreml_providers()
        try:
            self.session = ort.InferenceSession(str(model_path), sess_options, providers=providers)
        except Exception:
            if not any(isinstance(provider, tuple) and provider[0] == "CoreMLExecutionProvider" for provider in providers):
                raise
            self.session = ort.InferenceSession(str(model_path), sess_options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Pre-compute row anchor y-positions in the cropped resized image.
        self._row_anchors_norm = np.linspace(
            _ROW_ANCHOR_RANGE[0], _ROW_ANCHOR_RANGE[1], _NUM_ROW_ANCHORS, dtype=np.float32
        )

    def predict(self, rgb: np.ndarray) -> list[np.ndarray]:
        """Return up to 4 lane polylines in original-frame coordinates.

        Lane index convention (CULane): 0=left-left, 1=left (ego-left),
        2=right (ego-right), 3=right-right. Missing lanes are returned as
        empty arrays so callers can rely on a stable list length.
        """

        original_h, original_w = rgb.shape[:2]

        # Top-crop to match training distribution (sky removed).
        crop_top = int(original_h * (1.0 - _CROP_RATIO))
        cropped = rgb[crop_top:, :, :]
        cropped_h = cropped.shape[0]

        resized = cv2.resize(cropped, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - _MEAN) / _STD
        input_tensor = np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        return self._decode_lanes(
            outputs,
            original_w=original_w,
            original_h=original_h,
            crop_top=crop_top,
            cropped_h=cropped_h,
        )

    def _decode_lanes(
        self,
        outputs: list[np.ndarray],
        *,
        original_w: int,
        original_h: int,
        crop_top: int,
        cropped_h: int,
    ) -> list[np.ndarray]:
        """Convert the row-anchor logits into per-lane polylines.

        Resolves outputs by shape rather than name so this works across
        different export naming conventions. The row-anchor head is the one
        whose last dim equals ``_NUM_LANES`` and whose third-to-last dim is
        large (the grid). Existence head shares lane/anchor dims but has
        size 2 on its second axis.
        """

        # First pass: pick loc_row as the largest grid (shape[1]) head.
        # Both row and col location heads exist; the row head has the
        # larger grid (CULane: 200 row vs 100 col). Existence heads share
        # the lane axis (shape[-1]==4) but have shape[1]==2.
        loc_row = None
        for tensor in outputs:
            if tensor.ndim != 4 or tensor.shape[-1] != _NUM_LANES:
                continue
            if tensor.shape[1] == 2:
                continue
            if loc_row is None or tensor.shape[1] > loc_row.shape[1]:
                loc_row = tensor

        if loc_row is None:
            return [np.empty((0, 2), dtype=np.float32) for _ in range(_NUM_LANES)]

        num_grid = loc_row.shape[1]
        num_cls = loc_row.shape[2]

        # Second pass: existence head has the same num_cls as loc_row.
        # Without this constraint the col-existence head (which has a
        # different num_cls) silently shadowed the row head, producing a
        # mask whose length did not match the grid index array.
        exist_row = None
        for tensor in outputs:
            if tensor.ndim != 4 or tensor.shape[-1] != _NUM_LANES:
                continue
            if tensor.shape[1] == 2 and tensor.shape[2] == num_cls:
                exist_row = tensor
                break

        # Soft-argmax over the grid dimension yields a continuous column
        # index per (anchor, lane) — much smoother than hard argmax.
        logits = loc_row[0]  # (num_grid, num_cls, num_lane)
        # Numerical-stability softmax over grid axis.
        logits = logits - logits.max(axis=0, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.clip(probs.sum(axis=0, keepdims=True), 1e-6, None)
        grid_idx = np.arange(num_grid, dtype=np.float32)[:, None, None]
        col_idx = (probs * grid_idx).sum(axis=0)  # (num_cls, num_lane)

        if exist_row is not None:
            exist_logits = exist_row[0]  # (2, num_cls, num_lane)
            exist_score = np.exp(exist_logits[1]) / (
                np.exp(exist_logits[0]) + np.exp(exist_logits[1]) + 1e-6
            )
            valid = exist_score > 0.5
        else:
            # No explicit existence head — gate on softmax confidence instead.
            valid = probs.max(axis=0) > 0.30

        # Map (num_cls anchors) to actual y-positions in the cropped image,
        # then offset by crop_top to land in original-frame coordinates.
        # If the model emits a different anchor count we resample linearly.
        if num_cls == _NUM_ROW_ANCHORS:
            anchors_norm = self._row_anchors_norm
        else:
            anchors_norm = np.linspace(
                _ROW_ANCHOR_RANGE[0], _ROW_ANCHOR_RANGE[1], num_cls, dtype=np.float32
            )
        ys_in_cropped = anchors_norm * cropped_h
        ys_in_frame = ys_in_cropped + float(crop_top)

        # Grid index → x in original frame width.
        # The grid spans the full input width (in cropped, pre-resize space).
        x_scale = float(original_w) / float(num_grid)

        lanes: list[np.ndarray] = []
        for lane_i in range(_NUM_LANES):
            mask = valid[:, lane_i]
            if int(mask.sum()) < 2:
                lanes.append(np.empty((0, 2), dtype=np.float32))
                continue
            xs = col_idx[mask, lane_i] * x_scale
            ys = ys_in_frame[mask]
            lanes.append(np.stack([xs, ys], axis=1).astype(np.float32))

        return lanes


def is_lanenet_available() -> bool:
    """Whether the ONNX runtime and UFLDv2 model file are usable."""

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return _LANENET_PATH.is_file()


def get_lanenet_model() -> UFLDv2ONNX:
    """Return a cached UFLDv2 instance, or raise if unavailable."""

    global _lanenet_singleton, _lanenet_load_failed, _lanenet_load_error
    if _lanenet_load_failed:
        raise RuntimeError(_lanenet_load_error or _LANENET_UNAVAILABLE_MESSAGE)
    if _lanenet_singleton is not None:
        return _lanenet_singleton

    with _lanenet_lock:
        if _lanenet_load_failed:
            raise RuntimeError(_lanenet_load_error or _LANENET_UNAVAILABLE_MESSAGE)
        if _lanenet_singleton is not None:
            return _lanenet_singleton
        if not is_lanenet_available():
            _lanenet_load_failed = True
            _lanenet_load_error = _LANENET_UNAVAILABLE_MESSAGE
            raise RuntimeError(_lanenet_load_error)
        try:
            _lanenet_singleton = UFLDv2ONNX(_LANENET_PATH)
        except Exception as exc:
            _lanenet_load_failed = True
            _lanenet_load_error = f"UFLDv2 ONNX model failed to load: {exc}"
            raise RuntimeError(_lanenet_load_error) from exc
        return _lanenet_singleton
