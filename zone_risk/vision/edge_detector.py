"""Edge detection primitives used for depth cue estimation."""

from __future__ import annotations

import cv2
import numpy as np


def _normalize_to_unit(image: np.ndarray) -> np.ndarray:
    """Normalize an image to the range [0, 1].

    Args:
        image: Input array.

    Returns:
        Float32 normalized array.
    """

    image = image.astype(np.float32)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value - min_value < 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)


def detect_edges(image_gray: np.ndarray) -> np.ndarray:
    """Compute a fused edge magnitude map using Sobel, Canny, and LoG.

    Args:
        image_gray: Input grayscale image.

    Returns:
        Float32 edge magnitude map normalized to [0, 1].
    """

    gray = image_gray.astype(np.uint8)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel_norm = _normalize_to_unit(sobel_magnitude)

    median_intensity = float(np.median(gray))
    lower = int(max(0, (1.0 - 0.33) * median_intensity))
    upper = int(min(255, (1.0 + 0.33) * median_intensity))
    canny_edges = cv2.Canny(gray, lower, upper)
    canny_norm = canny_edges.astype(np.float32) / 255.0

    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.2)
    log_response = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
    log_norm = _normalize_to_unit(np.abs(log_response))

    fused_edges = 0.5 * sobel_norm + 0.3 * canny_norm + 0.2 * log_norm
    return _normalize_to_unit(fused_edges)
