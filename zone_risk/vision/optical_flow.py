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


_MAX_VELOCITY_PX = 25.0
_MAX_DIVERGENCE = 1.5


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

    height, width = current_gray.shape
    winsize = max(9, min(31, width // 35))

    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=winsize,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]

    # Subtract dominant background flow (camera/ego motion) so magnitude
    # reflects relative object motion rather than camera motion.
    ego_x = float(np.median(flow_x[::16, ::16]))
    ego_y = float(np.median(flow_y[::16, ::16]))
    flow_x = flow_x - ego_x
    flow_y = flow_y - ego_y
    flow = np.stack((flow_x, flow_y), axis=-1).astype(np.float32)

    magnitude, _ = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=False)
    grad_x = cv2.Sobel(flow_x, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(flow_y, cv2.CV_32F, 0, 1, ksize=3)
    divergence_positive = np.clip(grad_x + grad_y, 0.0, None)

    # Absolute normalization: values stay comparable across frames so a quiet
    # scene does not get min-max stretched into spurious "fast motion".
    magnitude_norm = np.clip(magnitude / _MAX_VELOCITY_PX, 0.0, 1.0).astype(np.float32)
    divergence_norm = np.clip(divergence_positive / _MAX_DIVERGENCE, 0.0, 1.0).astype(np.float32)

    return FlowResult(
        flow=flow,
        magnitude_norm=magnitude_norm,
        divergence_norm=divergence_norm,
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
