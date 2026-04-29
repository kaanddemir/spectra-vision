"""Optical-flow velocity estimation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FlowResult:
    flow: np.ndarray
    magnitude_norm: np.ndarray
    divergence_norm: np.ndarray


def _normalize_to_unit(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def empty_flow(shape: tuple[int, int]) -> FlowResult:
    height, width = shape
    flow = np.zeros((height, width, 2), dtype=np.float32)
    zeros = np.zeros((height, width), dtype=np.float32)
    return FlowResult(
        flow=flow,
        magnitude_norm=zeros,
        divergence_norm=zeros,
    )


def compute_velocity(previous_gray: np.ndarray | None, current_gray: np.ndarray) -> FlowResult:
    """Compute dense optical flow between two grayscale frames."""

    if previous_gray is None:
        return empty_flow(current_gray.shape)
    if previous_gray.shape != current_gray.shape:
        raise ValueError("Optical flow requires grayscale frames with matching shapes.")

    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    magnitude, _ = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=False)
    grad_x = cv2.Sobel(flow_x, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(flow_y, cv2.CV_32F, 0, 1, ksize=3)
    divergence_positive = np.clip(grad_x + grad_y, 0.0, None)

    return FlowResult(
        flow=flow.astype(np.float32),
        magnitude_norm=_normalize_to_unit(magnitude),
        divergence_norm=_normalize_to_unit(divergence_positive),
    )


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """Convert raw flow vectors into an HSV-based RGB visualization.
    Hue represents direction, Saturation is max, Value is magnitude.
    """
    height, width = flow.shape[:2]
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hue 0-180 in opencv
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
