"""Image preprocessing routines."""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_image(image_rgb: np.ndarray, image_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enhance luminance, denoise, and convert to LAB space.

    Args:
        image_rgb: Input RGB image.
        image_gray: Input grayscale image.

    Returns:
        A tuple of:
            - CLAHE-enhanced grayscale image
            - Denoised RGB image reconstructed from the enhanced LAB image
    """

    denoised_gray = cv2.fastNlMeansDenoising(image_gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(denoised_gray)

    lab_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    _, a_channel, b_channel = cv2.split(lab_image)
    enhanced_lab = cv2.merge((enhanced_gray, a_channel, b_channel))
    denoised_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    return enhanced_gray, denoised_rgb
