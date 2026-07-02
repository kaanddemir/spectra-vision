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

    return _depth_from_model(frame.rgb)
