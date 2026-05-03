"""Frame preprocessing for lane-relative risk detection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


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
    """Resize a BGR frame and produce the views the pipeline needs.

    Heavy denoising / CLAHE has been removed: Depth Anything V2 and the optical
    flow stage are robust to raw frames, and `fastNlMeansDenoising` was the
    single biggest CPU sink on M1.
    """

    resized_bgr = resize_preserving_aspect(frame_bgr, max_side=max_side)
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

    return PreprocessedFrame(
        bgr=resized_bgr,
        gray=gray,
        enhanced_gray=gray,
        denoised_rgb=rgb,
    )
