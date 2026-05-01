"""Monocular depth estimation for the zone-based risk pipeline.

Prefers the Depth Anything V2 ONNX model when available, otherwise falls
back to classical monocular cues (texture + edges + position + atmosphere).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import depth_model
from .depth_cues import estimate_depth
from .edge_detector import detect_edges
from .preprocess import PreprocessedFrame
from .texture_analyzer import analyze_texture


@dataclass(frozen=True)
class DepthResult:
    depth_map: np.ndarray
    near_map: np.ndarray


_MODEL_ABSOLUTE_NEAR_WEIGHT = 0.34
_MODEL_ROW_EXCESS_NEAR_WEIGHT = 1.0 - _MODEL_ABSOLUTE_NEAR_WEIGHT
_MODEL_ROW_EXCESS_FLOOR = 0.08
_MODEL_ROW_EXCESS_SCALE_FLOOR = 0.22


def _calibrate_model_near_map(near_map: np.ndarray) -> np.ndarray:
    """Convert relative AI depth into a conservative obstacle-nearness signal.

    Depth Anything returns relative depth. On dashcam video, the road plane can
    dominate that signal: bottom/side road pixels are genuinely near, but they
    are not obstacles. Keep a small amount of absolute nearness, then emphasize
    only pixels that are nearer than the same-row baseline.
    """

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


def _depth_from_model(rgb: np.ndarray) -> DepthResult | None:
    model = depth_model.get_model()
    if model is None:
        return None
    try:
        near_map = _calibrate_model_near_map(model.predict(rgb))
    except Exception:
        return None
    depth_map = np.clip(near_map * 255.0, 0.0, 255.0).astype(np.uint8)
    return DepthResult(depth_map=depth_map, near_map=near_map.astype(np.float32))


def _depth_from_classical(
    frame: PreprocessedFrame,
    *,
    texture_weight: float,
    position_weight: float,
    edge_weight: float,
    atmosphere_weight: float,
    bilateral_sigma: int,
) -> DepthResult:
    edge_map = detect_edges(frame.enhanced_gray)
    texture_map = analyze_texture(frame.enhanced_gray)
    depth_map = estimate_depth(
        frame.denoised_rgb,
        frame.enhanced_gray,
        texture_map,
        edge_map,
        texture_weight=texture_weight,
        position_weight=position_weight,
        edge_weight=edge_weight,
        atmosphere_weight=atmosphere_weight,
        bilateral_sigma=bilateral_sigma,
    )
    near_map = depth_map.astype(np.float32) / 255.0
    return DepthResult(depth_map=depth_map, near_map=near_map)


def estimate_frame_depth(
    frame: PreprocessedFrame,
    texture_weight: float = 0.35,
    position_weight: float = 0.30,
    edge_weight: float = 0.20,
    atmosphere_weight: float = 0.15,
    bilateral_sigma: int = 75,
    use_depth_model: bool = True,
) -> DepthResult:
    """Estimate a normalized near-depth map.

    Larger values mean closer regions.
    """

    if use_depth_model:
        model_result = _depth_from_model(frame.denoised_rgb)
        if model_result is not None:
            return model_result

    return _depth_from_classical(
        frame,
        texture_weight=texture_weight,
        position_weight=position_weight,
        edge_weight=edge_weight,
        atmosphere_weight=atmosphere_weight,
        bilateral_sigma=bilateral_sigma,
    )
