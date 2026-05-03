"""Optical-flow velocity estimation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import models
from .preprocessing import PreprocessedFrame


@dataclass(frozen=True)
class FlowResult:
    flow: np.ndarray
    magnitude_norm: np.ndarray
    divergence_norm: np.ndarray


_MAX_VELOCITY_PX = 25.0
_MAX_DIVERGENCE = 1.5

_EGO_MAX_CORNERS = 400
_EGO_QUALITY_LEVEL = 0.01
_EGO_MIN_DISTANCE = 8
_EGO_MIN_INLIERS = 18
_EGO_RANSAC_THRESHOLD_PX = 3.0


def empty_flow(shape: tuple[int, int]) -> FlowResult:
    height, width = shape
    flow = np.zeros((height, width, 2), dtype=np.float32)
    zeros = np.zeros((height, width), dtype=np.float32)
    return FlowResult(
        flow=flow,
        magnitude_norm=zeros,
        divergence_norm=zeros,
    )


def _estimate_ego_homography(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
) -> np.ndarray | None:
    """Fit a perspective ego-motion model from sparse tracks with RANSAC.

    Returns the 3×3 homography that maps the previous frame onto the current
    frame, or ``None`` if the fit is unreliable. RANSAC rejects foreground
    object motion so a moving vehicle does not corrupt the ego model.
    """

    corners = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=_EGO_MAX_CORNERS,
        qualityLevel=_EGO_QUALITY_LEVEL,
        minDistance=_EGO_MIN_DISTANCE,
        blockSize=7,
    )
    if corners is None or len(corners) < _EGO_MIN_INLIERS:
        return None

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        corners,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_pts is None or status is None:
        return None

    good = status.flatten().astype(bool)
    if int(good.sum()) < _EGO_MIN_INLIERS:
        return None

    src = corners[good].reshape(-1, 2)
    dst = next_pts[good].reshape(-1, 2)

    homography, inlier_mask = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=_EGO_RANSAC_THRESHOLD_PX,
        maxIters=2000,
        confidence=0.99,
    )
    if homography is None or inlier_mask is None:
        return None
    if int(inlier_mask.sum()) < _EGO_MIN_INLIERS:
        return None

    return homography.astype(np.float32)


def _ego_flow_field(homography: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    points = np.stack((xs.ravel(), ys.ravel()), axis=1).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(points, homography).reshape(height, width, 2)
    return (warped[..., 0] - xs).astype(np.float32), (warped[..., 1] - ys).astype(np.float32)


def _subtract_ego_motion(
    flow_x: np.ndarray,
    flow_y: np.ndarray,
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    homography = _estimate_ego_homography(previous_gray, current_gray)
    if homography is not None:
        ego_x_field, ego_y_field = _ego_flow_field(homography, current_gray.shape)
        return flow_x - ego_x_field, flow_y - ego_y_field

    # Translation-only compensation when the homography fit is unreliable
    # (low texture, abrupt scene change, or too few tracked features).
    ego_x = float(np.median(flow_x[::16, ::16]))
    ego_y = float(np.median(flow_y[::16, ::16]))
    return flow_x - ego_x, flow_y - ego_y


def _flow_from_neural_model(
    previous_frame: PreprocessedFrame,
    current_frame: PreprocessedFrame,
) -> np.ndarray:
    model = models.get_flow_model()
    if model is None:
        raise RuntimeError("NeuFlow ONNX model missing at models/neuflow_v2.onnx")
    return model.predict(previous_frame.denoised_rgb, current_frame.denoised_rgb)


def compute_velocity(
    previous_frame: PreprocessedFrame | None,
    current_frame: PreprocessedFrame,
) -> FlowResult:
    """Compute dense optical flow between two preprocessed frames.

    Uses NeuFlow ONNX as the required dense-flow source, then applies
    RANSAC ego-motion subtraction and exposes a stable FlowResult shape for
    TTC and motion overlays.
    """

    if previous_frame is None:
        return empty_flow(current_frame.gray.shape)
    if previous_frame.gray.shape != current_frame.gray.shape:
        raise ValueError("Optical flow requires frames with matching shapes.")

    flow = _flow_from_neural_model(previous_frame, current_frame)

    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    flow_x, flow_y = _subtract_ego_motion(
        flow_x, flow_y, previous_frame.gray, current_frame.gray
    )
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
