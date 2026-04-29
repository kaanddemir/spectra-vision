"""Monocular depth estimation for the zone-based risk pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .depth_cues import estimate_depth
from .edge_detector import detect_edges
from .texture_analyzer import analyze_texture

from .preprocess import PreprocessedFrame


@dataclass(frozen=True)
class DepthResult:
    raw_depth: np.ndarray
    depth_map: np.ndarray
    near_map: np.ndarray


def estimate_frame_depth(
    frame: PreprocessedFrame,
    texture_weight: float = 0.35,
    position_weight: float = 0.30,
    edge_weight: float = 0.20,
    atmosphere_weight: float = 0.15,
    bilateral_sigma: int = 75,
) -> DepthResult:
    """Estimate a normalized near-depth map.

    Current zone-risk semantics treat larger values as nearer regions.
    """

    edge_map = detect_edges(frame.enhanced_gray)
    texture_map = analyze_texture(frame.enhanced_gray)
    raw_depth, depth_map = estimate_depth(
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
    return DepthResult(raw_depth=raw_depth, depth_map=depth_map, near_map=near_map)
