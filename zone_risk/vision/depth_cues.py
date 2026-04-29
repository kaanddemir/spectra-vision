"""Classical monocular depth cue fusion."""

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


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """Apply a grayscale guided filter.

    Args:
        guide: Guidance image normalized to [0, 1].
        src: Source image normalized to [0, 1].
        radius: Window radius.
        eps: Regularization term.

    Returns:
        Filtered float32 image in [0, 1].
    """

    kernel = (radius, radius)
    mean_i = cv2.boxFilter(guide, cv2.CV_32F, kernel)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, kernel)
    corr_i = cv2.boxFilter(guide * guide, cv2.CV_32F, kernel)
    corr_ip = cv2.boxFilter(guide * src, cv2.CV_32F, kernel)

    var_i = corr_i - (mean_i * mean_i)
    cov_ip = corr_ip - (mean_i * mean_p)

    a = cov_ip / (var_i + eps)
    b = mean_p - (a * mean_i)

    mean_a = cv2.boxFilter(a, cv2.CV_32F, kernel)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, kernel)
    return (mean_a * guide) + mean_b


def _compute_position_cue(shape: tuple[int, int]) -> np.ndarray:
    """Compute a perspective-inspired vertical position cue.

    Args:
        shape: Image shape as (height, width).

    Returns:
        Float32 cue map in [0, 1], where larger values are nearer.
    """

    height, width = shape
    vertical = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(height, 1)
    return np.repeat(vertical, width, axis=1)


def _compute_atmospheric_near_cue(image_gray: np.ndarray) -> np.ndarray:
    """Estimate a near-depth cue from atmospheric perspective.

    Args:
        image_gray: Grayscale image.

    Returns:
        Float32 near cue in [0, 1].
    """

    gray = image_gray.astype(np.float32) / 255.0
    local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=5)
    local_sq_mean = cv2.GaussianBlur(gray * gray, (0, 0), sigmaX=5)
    local_variance = np.maximum(local_sq_mean - (local_mean * local_mean), 0.0)
    local_contrast = np.sqrt(local_variance)
    local_contrast = _normalize_to_unit(local_contrast)

    brightness = _normalize_to_unit(gray)
    inverse_contrast = 1.0 - local_contrast
    far_score = _normalize_to_unit(0.6 * inverse_contrast + 0.4 * brightness)
    return 1.0 - far_score


def estimate_depth(
    image_rgb: np.ndarray,
    image_gray: np.ndarray,
    texture_map: np.ndarray,
    edge_map: np.ndarray,
    texture_weight: float = 0.35,
    position_weight: float = 0.30,
    edge_weight: float = 0.20,
    atmosphere_weight: float = 0.15,
    bilateral_sigma: int = 75,
) -> np.ndarray:
    """Fuse classical monocular cues into a depth estimate.

    Args:
        image_rgb: Original RGB image.
        image_gray: Grayscale image.
        texture_map: Texture density map normalized to [0, 1].
        edge_map: Edge density map normalized to [0, 1].
        texture_weight: Weight for the texture cue.
        position_weight: Weight for the vertical position cue.
        edge_weight: Weight for the edge cue.
        atmosphere_weight: Weight for the atmospheric perspective cue.
        bilateral_sigma: Bilateral filter strength for post-processing.

    Returns:
        Final uint8 depth map in [0, 255].
    """

    weights = np.array(
        [texture_weight, position_weight, edge_weight, atmosphere_weight],
        dtype=np.float32,
    )
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("Depth cue weights must sum to a positive value.")
    weights /= weight_sum

    texture_cue = _normalize_to_unit(texture_map)
    position_cue = _compute_position_cue(image_gray.shape)
    edge_cue = _normalize_to_unit(edge_map)
    atmospheric_cue = _compute_atmospheric_near_cue(image_gray)

    fused_depth = (
        weights[0] * texture_cue
        + weights[1] * position_cue
        + weights[2] * edge_cue
        + weights[3] * atmospheric_cue
    ).astype(np.float32)
    fused_depth = _normalize_to_unit(fused_depth)

    bilateral_filtered = cv2.bilateralFilter(
        fused_depth,
        d=9,
        sigmaColor=float(bilateral_sigma),
        sigmaSpace=float(bilateral_sigma),
    )

    guide = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    guided = guided_filter(guide, bilateral_filtered.astype(np.float32), radius=8, eps=1e-3)
    guided = _normalize_to_unit(guided)
    final_depth = np.clip(guided * 255.0, 0, 255).astype(np.uint8)

    return final_depth
