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

# Anchor sample positions, normalized to the ORIGINAL image (upstream CULane
# config). Row anchors are y-fractions where the row head samples a lane's x;
# col anchors are x-fractions where the col head samples a lane's y. Both span
# the bottom portion of the frame the network was trained/cropped on.
_ROW_ANCHOR_RANGE = (0.42, 1.0)
_NUM_ROW_ANCHORS = 72
_COL_ANCHOR_RANGE = (0.0, 1.0)
_NUM_COL_ANCHORS = 81
_NUM_LANES = 4

# Upstream UFLDv2 ``pred2coords`` decodes the two ego-lane boundaries (indices
# 1, 2) from the ROW head and the two outer lanes (indices 0, 3) from the COL
# head — each lane from one head, not a fusion. The ego corridor that
# downstream risk uses is therefore driven entirely by the row head.
_ROW_LANE_INDICES = (1, 2)
_COL_LANE_INDICES = (0, 3)

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

        # Pre-compute anchor fractions (relative to the ORIGINAL image).
        self._row_anchors_norm = np.linspace(
            _ROW_ANCHOR_RANGE[0], _ROW_ANCHOR_RANGE[1], _NUM_ROW_ANCHORS, dtype=np.float32
        )
        self._col_anchors_norm = np.linspace(
            _COL_ANCHOR_RANGE[0], _COL_ANCHOR_RANGE[1], _NUM_COL_ANCHORS, dtype=np.float32
        )

    def predict(self, rgb: np.ndarray) -> list[np.ndarray]:
        """Return up to 4 lane polylines in original-frame coordinates.

        Lane index convention (CULane): 0=left-left, 1=left (ego-left),
        2=right (ego-right), 3=right-right. Missing lanes are returned as
        empty arrays so callers can rely on a stable list length.
        """

        original_h, original_w = rgb.shape[:2]

        # Top-crop to match training distribution (sky removed). The crop only
        # affects what the network sees; lane coordinates are decoded back into
        # the full original frame because the anchors are full-image fractions.
        crop_top = int(original_h * (1.0 - _CROP_RATIO))
        cropped = rgb[crop_top:, :, :]

        resized = cv2.resize(cropped, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - _MEAN) / _STD
        input_tensor = np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        return self._decode_lanes(
            outputs,
            original_w=original_w,
            original_h=original_h,
        )

    def _decode_lanes(
        self,
        outputs: list[np.ndarray],
        *,
        original_w: int,
        original_h: int,
    ) -> list[np.ndarray]:
        """Decode the row/col logits into per-lane polylines (upstream pred2coords).

        Outputs are resolved by shape, not name, so renamed exports still work.
        Per upstream UFLDv2: the two ego lanes (indices 1, 2) come from the ROW
        head, the two outer lanes (0, 3) from the COL head. For each kept anchor
        the grid location is the argmax refined by a local softmax window (not a
        global expectation, which biases toward the image center). Anchors map
        to full-image fractions, so no crop offset is re-applied.
        """

        locs = [
            t for t in outputs
            if t.ndim == 4 and t.shape[-1] == _NUM_LANES and t.shape[1] != 2
        ]
        exists = [
            t for t in outputs
            if t.ndim == 4 and t.shape[-1] == _NUM_LANES and t.shape[1] == 2
        ]
        if not locs:
            return [np.empty((0, 2), dtype=np.float32) for _ in range(_NUM_LANES)]

        # Row head has the larger grid (CULane: 200 row vs 100 col).
        loc_row = max(locs, key=lambda t: t.shape[1])
        loc_col = min(locs, key=lambda t: t.shape[1]) if len(locs) > 1 else None
        if loc_col is loc_row:
            loc_col = None
        exist_row = self._match_exist(exists, loc_row.shape[2])
        exist_col = self._match_exist(exists, loc_col.shape[2]) if loc_col is not None else None

        lanes: list[np.ndarray] = [np.empty((0, 2), dtype=np.float32) for _ in range(_NUM_LANES)]

        row_anchors = self._anchors_for(loc_row.shape[2], self._row_anchors_norm, _ROW_ANCHOR_RANGE)
        for lane_i in _ROW_LANE_INDICES:
            lanes[lane_i] = self._decode_lane(
                loc_row[0], exist_row, lane_i,
                anchors_norm=row_anchors, axis="row",
                original_w=original_w, original_h=original_h,
                valid_divisor=2,
            )

        if loc_col is not None:
            col_anchors = self._anchors_for(loc_col.shape[2], self._col_anchors_norm, _COL_ANCHOR_RANGE)
            for lane_i in _COL_LANE_INDICES:
                lanes[lane_i] = self._decode_lane(
                    loc_col[0], exist_col, lane_i,
                    anchors_norm=col_anchors, axis="col",
                    original_w=original_w, original_h=original_h,
                    valid_divisor=4,
                )

        return lanes

    @staticmethod
    def _match_exist(exists: list[np.ndarray], num_cls: int) -> np.ndarray | None:
        for tensor in exists:
            if tensor.shape[2] == num_cls:
                return tensor
        return None

    @staticmethod
    def _anchors_for(num_cls: int, precomputed: np.ndarray, anchor_range: tuple[float, float]) -> np.ndarray:
        if num_cls == precomputed.shape[0]:
            return precomputed
        return np.linspace(anchor_range[0], anchor_range[1], num_cls, dtype=np.float32)

    @staticmethod
    def _decode_lane(
        loc: np.ndarray,
        exist: np.ndarray | None,
        lane_i: int,
        *,
        anchors_norm: np.ndarray,
        axis: str,
        original_w: int,
        original_h: int,
        valid_divisor: int,
    ) -> np.ndarray:
        """Decode one lane from a single head (row or col).

        ``loc`` is ``(num_grid, num_cls, num_lane)``. For a row head the grid
        spans x (anchor gives y); for a col head the grid spans y (anchor gives
        x). Returns an ``(N, 2)`` ``(x, y)`` polyline, empty if the lane fails
        the upstream majority existence gate.
        """

        num_grid, num_cls, _ = loc.shape
        logits = loc[:, :, lane_i]  # (num_grid, num_cls)

        if exist is not None:
            exist_lane = exist[0, :, :, lane_i]  # (2, num_cls)
            valid = exist_lane[1] > exist_lane[0]
        else:
            # No existence head — gate on per-anchor peak confidence instead.
            shifted = logits - logits.max(axis=0, keepdims=True)
            probs = np.exp(shifted)
            probs /= np.clip(probs.sum(axis=0, keepdims=True), 1e-6, None)
            valid = probs.max(axis=0) > 0.30

        # Upstream lane-level gate: keep the lane only if a majority (row) /
        # quarter (col) of its anchors exist, which rejects noisy partials.
        if int(valid.sum()) <= num_cls // valid_divisor:
            return np.empty((0, 2), dtype=np.float32)

        peak = np.argmax(logits, axis=0)  # (num_cls,)
        points: list[tuple[float, float]] = []
        for k in range(num_cls):
            if not valid[k]:
                continue
            lo = max(0, int(peak[k]) - 1)
            hi = min(num_grid - 1, int(peak[k]) + 1)
            idx = np.arange(lo, hi + 1)
            window = logits[idx, k]
            window = window - window.max()
            w = np.exp(window)
            w /= np.clip(w.sum(), 1e-6, None)
            grid_pos = float((w * idx).sum()) + 0.5
            loc_frac = grid_pos / float(num_grid - 1)
            if axis == "row":
                x = loc_frac * original_w
                y = float(anchors_norm[k]) * original_h
            else:  # col head: grid is y, anchor is x
                x = float(anchors_norm[k]) * original_w
                y = loc_frac * original_h
            points.append((x, y))

        if len(points) < 2:
            return np.empty((0, 2), dtype=np.float32)
        pts = np.array(points, dtype=np.float32)
        # Sort by y so the polyline is monotonic top→bottom for the line fit.
        return pts[np.argsort(pts[:, 1])]


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
