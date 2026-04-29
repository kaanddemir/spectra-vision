"""Texture feature extraction using Gabor responses and local energy."""

from __future__ import annotations

from functools import lru_cache
from typing import List

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


@lru_cache(maxsize=1)
def build_gabor_filter_bank() -> List[np.ndarray]:
    """Create a compact Gabor filter bank with 4 orientations and 2 frequencies.

    Returns:
        A list of Gabor kernels.
    """

    kernels: List[np.ndarray] = []
    orientations = np.linspace(0.0, np.pi, 4, endpoint=False)
    frequencies = [0.1, 0.2]

    for theta in orientations:
        for frequency in frequencies:
            wavelength = max(3.0, 1.0 / frequency)
            sigma = 0.56 * wavelength
            kernel_size = int(max(15, round(wavelength * 4)))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getGaborKernel(
                (kernel_size, kernel_size),
                sigma=sigma,
                theta=float(theta),
                lambd=float(wavelength),
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F,
            )
            kernels.append(kernel)

    return kernels


def analyze_texture(image_gray: np.ndarray, window_size: int = 15) -> np.ndarray:
    """Compute a texture density map from Gabor responses and local variance.

    Args:
        image_gray: Input grayscale image.
        window_size: Sliding window size for local variance.

    Returns:
        Float32 texture density map normalized to [0, 1].
    """

    gray_float = image_gray.astype(np.float32) / 255.0
    gabor_responses = []

    for kernel in build_gabor_filter_bank():
        response = cv2.filter2D(gray_float, cv2.CV_32F, kernel)
        gabor_responses.append(np.abs(response))

    max_response = np.max(np.stack(gabor_responses, axis=0), axis=0)
    max_response_norm = _normalize_to_unit(max_response)

    mean = cv2.boxFilter(max_response_norm, ddepth=cv2.CV_32F, ksize=(window_size, window_size))
    mean_sq = cv2.boxFilter(max_response_norm ** 2, ddepth=cv2.CV_32F, ksize=(window_size, window_size))
    local_variance = np.maximum(mean_sq - (mean ** 2), 0.0)
    texture_energy = _normalize_to_unit(local_variance)

    texture_density = 0.6 * max_response_norm + 0.4 * texture_energy
    return _normalize_to_unit(texture_density)
