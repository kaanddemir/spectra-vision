"""Frame preprocessing for zone-based risk detection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .image_preprocess import preprocess_image


@dataclass(frozen=True)
class PreprocessedFrame:
    bgr: np.ndarray
    gray: np.ndarray
    enhanced_gray: np.ndarray
    denoised_rgb: np.ndarray


def resize_preserving_aspect(frame_bgr: np.ndarray, max_side: int) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    longest_edge = max(height, width)
    if longest_edge <= max_side:
        return frame_bgr

    scale = max_side / float(longest_edge)
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)


def preprocess_frame(frame_bgr: np.ndarray, max_side: int = 640) -> PreprocessedFrame:
    """Resize and enhance a BGR frame."""

    resized_bgr = resize_preserving_aspect(frame_bgr, max_side=max_side)
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    enhanced_gray, denoised_rgb = preprocess_image(rgb, gray)

    return PreprocessedFrame(
        bgr=resized_bgr,
        gray=gray,
        enhanced_gray=enhanced_gray,
        denoised_rgb=denoised_rgb,
    )
