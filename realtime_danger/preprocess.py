"""Frame preprocessing for real-time danger detection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from depth_project.preprocess import preprocess_image


@dataclass(frozen=True)
class PreprocessedFrame:
    bgr: np.ndarray
    rgb: np.ndarray
    gray: np.ndarray
    enhanced_gray: np.ndarray
    denoised_rgb: np.ndarray
    scale: float


def resize_preserving_aspect(frame_bgr: np.ndarray, max_side: int) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    longest_edge = max(height, width)
    if longest_edge <= max_side:
        return frame_bgr

    scale = max_side / float(longest_edge)
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)


def preprocess_frame(frame_bgr: np.ndarray, max_side: int = 720) -> PreprocessedFrame:
    """Resize and enhance a BGR frame."""

    original_height, original_width = frame_bgr.shape[:2]
    resized_bgr = resize_preserving_aspect(frame_bgr, max_side=max_side)
    height, width = resized_bgr.shape[:2]
    scale = width / float(original_width) if original_width else 1.0
    if original_height:
        scale = min(scale, height / float(original_height))

    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    enhanced_gray, _, denoised_rgb = preprocess_image(rgb, gray)

    return PreprocessedFrame(
        bgr=resized_bgr,
        rgb=rgb,
        gray=gray,
        enhanced_gray=enhanced_gray,
        denoised_rgb=denoised_rgb,
        scale=scale,
    )
