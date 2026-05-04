"""Optical-flow velocity estimation (classical DIS, M1-friendly)."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

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
_EGO_FIELD_DOWNSAMPLE = 8

_FLOW_MAX_SIDE = 320

_dis_lock = threading.Lock()
_dis_singleton: cv2.DISOpticalFlow | None = None


def _get_dis() -> cv2.DISOpticalFlow:
    """Return a cached DIS optical-flow instance.

    DIS is a classical, edge-aware dense optical flow algorithm that runs
    well on Apple Silicon CPU. The MEDIUM preset is a good speed/quality
    trade-off for driving footage.
    """

    global _dis_singleton
    if _dis_singleton is not None:
        return _dis_singleton
    with _dis_lock:
        if _dis_singleton is None:
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            dis.setUseSpatialPropagation(True)
            _dis_singleton = dis
        return _dis_singleton


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
    """Fit a perspective ego-motion model from sparse tracks with RANSAC."""

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
    """Build the ego-motion flow field on a coarse grid, then upscale.

    `perspectiveTransform` over every pixel is wasteful — the ego flow is
    smooth, so we sample it on a 1/N grid and bilinear-resize back. This is
    visually identical to the dense version but ~50× cheaper.
    """

    height, width = shape
    step = _EGO_FIELD_DOWNSAMPLE
    small_h = max(2, (height + step - 1) // step)
    small_w = max(2, (width + step - 1) // step)

    ys = np.linspace(0.0, height - 1.0, small_h, dtype=np.float32)
    xs = np.linspace(0.0, width - 1.0, small_w, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(points, homography).reshape(small_h, small_w, 2)

    ego_x_small = (warped[..., 0] - grid_x).astype(np.float32)
    ego_y_small = (warped[..., 1] - grid_y).astype(np.float32)

    ego_x = cv2.resize(ego_x_small, (width, height), interpolation=cv2.INTER_LINEAR)
    ego_y = cv2.resize(ego_y_small, (width, height), interpolation=cv2.INTER_LINEAR)
    return ego_x, ego_y


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

    ego_x = float(np.median(flow_x[::16, ::16]))
    ego_y = float(np.median(flow_y[::16, ::16]))
    return flow_x - ego_x, flow_y - ego_y


def _flow_from_dis(previous_gray: np.ndarray, current_gray: np.ndarray) -> np.ndarray:
    """Compute DIS flow on a downscaled pair, then upscale.

    The TTC pipeline only needs bbox-level radial percentile and a coarse
    divergence map, so a 320-px-long-side computation is plenty.
    """

    height, width = current_gray.shape
    longest = max(height, width)
    if longest > _FLOW_MAX_SIDE:
        scale = _FLOW_MAX_SIDE / float(longest)
        small_w = max(2, int(round(width * scale)))
        small_h = max(2, int(round(height * scale)))
        prev_small = cv2.resize(previous_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(current_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        flow_small = _get_dis().calc(prev_small, curr_small, None)
        flow = cv2.resize(flow_small, (width, height), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= width / float(small_w)
        flow[..., 1] *= height / float(small_h)
        return flow.astype(np.float32)

    return _get_dis().calc(previous_gray, current_gray, None).astype(np.float32)


def compute_velocity(
    previous_frame: PreprocessedFrame | None,
    current_frame: PreprocessedFrame,
) -> FlowResult:
    """Compute dense optical flow between two preprocessed frames.

    Uses DIS (classical, OpenCV) on grayscale, with RANSAC ego-motion
    subtraction layered on top. The output shape matches the previous
    NeuFlow-based implementation so the rest of the pipeline is unchanged.
    """

    if previous_frame is None:
        return empty_flow(current_frame.gray.shape)
    if previous_frame.gray.shape != current_frame.gray.shape:
        raise ValueError("Optical flow requires frames with matching shapes.")

    flow = _flow_from_dis(previous_frame.gray, current_frame.gray)

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

    magnitude_norm = np.clip(magnitude / _MAX_VELOCITY_PX, 0.0, 1.0).astype(np.float32)
    divergence_norm = np.clip(divergence_positive / _MAX_DIVERGENCE, 0.0, 1.0).astype(np.float32)

    return FlowResult(
        flow=flow,
        magnitude_norm=magnitude_norm,
        divergence_norm=divergence_norm,
    )


