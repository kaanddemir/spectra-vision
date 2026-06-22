"""Depth Anything V2 ONNX estimation for the lane-relative risk pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import models
from .preprocessing import PreprocessedFrame


@dataclass(frozen=True)
class DepthResult:
    depth_m: np.ndarray
    depth_map: np.ndarray
    near_map: np.ndarray


_MODEL_ABSOLUTE_NEAR_WEIGHT = 0.34
_MODEL_ROW_EXCESS_NEAR_WEIGHT = 1.0 - _MODEL_ABSOLUTE_NEAR_WEIGHT
_MODEL_ROW_EXCESS_FLOOR = 0.08
_MODEL_ROW_EXCESS_SCALE_FLOOR = 0.22


def _calibrate_model_near_map(near_map: np.ndarray) -> np.ndarray:
    """Convert relative AI depth into a conservative obstacle-nearness signal."""

    near = np.nan_to_num(near_map.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    near = np.clip(near, 0.0, 1.0)

    row_baseline = np.median(near, axis=1, keepdims=True).astype(np.float32)
    row_excess = np.clip(near - row_baseline, 0.0, 1.0)
    high_excess = float(np.percentile(row_excess, 98))
    excess_scale = max(high_excess, _MODEL_ROW_EXCESS_SCALE_FLOOR)
    excess_denominator = max(excess_scale - _MODEL_ROW_EXCESS_FLOOR, 1e-6)
    obstacle_excess = np.clip(
        (row_excess - _MODEL_ROW_EXCESS_FLOOR) / excess_denominator,
        0.0,
        1.0,
    )

    calibrated = (
        _MODEL_ABSOLUTE_NEAR_WEIGHT * near
        + _MODEL_ROW_EXCESS_NEAR_WEIGHT * obstacle_excess
    )
    return np.clip(calibrated, 0.0, 1.0).astype(np.float32)


def _depth_from_model(rgb: np.ndarray) -> DepthResult:
    model = models.get_depth_model()
    depth_m = np.clip(model.predict(rgb).astype(np.float32), 0.0, models._METRIC_MAX_DEPTH_M)
    near_map = np.clip(1.0 - (depth_m / models._METRIC_MAX_DEPTH_M), 0.0, 1.0).astype(np.float32)
    depth_map = np.clip(near_map * 255.0, 0.0, 255.0).astype(np.uint8)
    return DepthResult(
        depth_m=depth_m.astype(np.float32),
        depth_map=depth_map,
        near_map=near_map,
    )


def estimate_frame_depth(frame: PreprocessedFrame) -> DepthResult:
    """Estimate metric depth plus a normalized nearness compatibility map."""

    return _depth_from_model(frame.denoised_rgb)
