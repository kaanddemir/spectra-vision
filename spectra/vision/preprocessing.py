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


def preprocess_image(image_rgb: np.ndarray, image_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enhance luminance, denoise, and convert to LAB space."""

    denoised_gray = cv2.fastNlMeansDenoising(image_gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(denoised_gray)

    lab_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    _, a_channel, b_channel = cv2.split(lab_image)
    enhanced_lab = cv2.merge((enhanced_gray, a_channel, b_channel))
    denoised_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    return enhanced_gray, denoised_rgb


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
