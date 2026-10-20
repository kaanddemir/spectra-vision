"""BoT-SORT multi-object tracking through Roboflow ``trackers`` (Apache-2.0).

Behaviour of the library this adapter relies on (checked in ``trackers`` 2.6.1):

* With a ``timestamp`` on every update, lost tracks expire after ``lost_track_buffer / 30`` seconds of
  source time, not after a frame count, and the Kalman step is scaled by the real elapsed time, so
  variable frame rates and per-component sampling intervals are handled. ``frame_rate`` only sets the
  Kalman time unit.
* Camera motion compensation estimates an affine transform from the frame image passed in and masks
  out the current detections, so image and boxes must share coordinates. Detections arrive in source
  display pixels; they are scaled into the frame image's pixels for the update and the output refers
  back to the original detections, so the stored boxes are always the detector's.
* ``sparseOptFlow`` (the default) fits ``cv2.estimateAffinePartial2D`` (a similarity: scale, rotation,
  translation) from the previous frame image to the current one and returns an exact identity both on
  the first frame and when it fails, so an exact identity cannot be told apart from "no estimate". The
  adapter reports the transform through ``camera_motion()``, wrapping the library's ``estimate`` call
  without altering what the tracker receives, and reports ``None`` for an identity or a counted failure.
* The library discards RANSAC's inlier mask, so the fit's quality is lost. ``QualityCMC`` replaces the
  tracker's estimator with a copy of its sparse-optical-flow method (``trackers`` 2.6.1,
  ``trackers/utils/cmc.py``, Apache-2.0) that returns the same matrix and also keeps the number of
  correspondences, the inliers and their median residual. The copy is only valid for that version, so
  loading refuses any other.
* The tracker skips motion compensation when it has neither tracks nor detections, and the estimator
  then compares the next frame with the last one it saw, so ``since_t`` is the time of the last frame the
  estimator ran on, not of the previous update. Empty scenes therefore have no motion estimate.

Only confirmed tracks are reported; detections the tracker has not (yet) given an identity are left out
of the track list but stay in the frame's detections.
"""
from __future__ import annotations

import copy
from importlib import metadata
from typing import Optional, Sequence

import numpy as np

from spectra.engine.contracts import CameraMotion, ComponentInfo, Detection, ObjectCategory, Role, TrackedObject
from spectra.engine.video import Frame

_CATEGORIES = tuple(ObjectCategory)
CMC_VERSION = "2.6.1"  # the trackers release QualityCMC copies its estimator from


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _quality_cmc_class():
    """``QualityCMC`` is built on first use, so importing this module does not import ``trackers``."""
    import cv2
    from trackers.utils.cmc import CMC, logger

    class QualityCMC(CMC):
        """``trackers.utils.cmc.CMC`` whose sparse optical flow also records the fit quality in ``quality``:
        ``(matches, inliers, median inlier residual in frame-image pixels)``, or None when there is no fit.

        ``_estimate_sparse_optflow`` is the library's method (2.6.1) unchanged except for the lines marked
        ``quality``; the returned matrix and ``frames_failed`` are the library's.
        """

        quality: Optional[tuple[int, int, float]] = None

        def estimate(self, frame_bgr, dets_xyxy=None):
            self.quality = None
            return super().estimate(frame_bgr, dets_xyxy)

        def _estimate_sparse_optflow(self, frame_bgr: np.ndarray) -> np.ndarray:
            img_h, img_w = frame_bgr.shape[:2]
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            affine_mtx = np.eye(2, 3, dtype=np.float32)
            if self.downscale > 1:
                new_w = max(1, img_w // self.downscale)
                new_h = max(1, img_h // self.downscale)
                frame = cv2.resize(frame, (new_w, new_h))
            keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)
            if not self._initialized:
                self._prev_frame_gray = frame.copy()
                self._prev_points = copy.copy(keypoints)
                self._initialized = True
                return affine_mtx
            if self._prev_frame_gray is None or self._prev_points is None or keypoints is None:
                self._prev_frame_gray = frame.copy()
                self._prev_points = None if keypoints is None else keypoints.copy()
                return affine_mtx
            if self._prev_frame_gray.shape != frame.shape:
                logger.debug("CMC: frame size changed (%s -> %s); re-syncing sparse optical flow",
                             self._prev_frame_gray.shape, frame.shape)
                self._prev_frame_gray = frame.copy()
                self._prev_points = keypoints.copy()
                return affine_mtx
            matched, status, _err = cv2.calcOpticalFlowPyrLK(self._prev_frame_gray, frame, self._prev_points, None)
            if status is None or matched is None:
                self._prev_frame_gray = frame.copy()
                self._prev_points = copy.copy(keypoints)
                return affine_mtx
            good = status.reshape(-1) != 0
            prev_pts_np = self._prev_points[good]
            curr_pts_np = matched[good]
            if np.size(prev_pts_np, 0) > 4:
                affine_est, inlier_mask = cv2.estimateAffinePartial2D(prev_pts_np, curr_pts_np, method=cv2.RANSAC)
                if affine_est is not None:
                    affine_mtx = affine_est.astype(np.float32)
                    if self.downscale > 1:
                        affine_mtx[0, 2] *= self.downscale
                        affine_mtx[1, 2] *= self.downscale
                    self.quality = _fit_quality(affine_est, prev_pts_np, curr_pts_np, inlier_mask,  # quality
                                                self.downscale)
            else:
                logger.warning("CMC: not enough matching points for motion estimation")
                self.frames_failed += 1
            self._prev_frame_gray = frame.copy()
            self._prev_points = None if keypoints is None else keypoints.copy()
            return affine_mtx

    return QualityCMC


def _fit_quality(affine: np.ndarray, prev: np.ndarray, curr: np.ndarray, inlier_mask: Optional[np.ndarray],
                 downscale: int) -> tuple[int, int, float]:
    """(correspondences, RANSAC inliers, median inlier distance from the fit in frame-image pixels)."""
    p, q = prev.reshape(-1, 2).astype(float), curr.reshape(-1, 2).astype(float)
    keep = (np.ones(len(p), bool) if inlier_mask is None else inlier_mask.reshape(-1) != 0)
    if not keep.any():
        return len(p), 0, 0.0
    err = np.hypot(*(p[keep] @ affine[:, :2].T + affine[:, 2] - q[keep]).T)
    return len(p), int(keep.sum()), float(np.median(err)) * downscale


class BoTSortTracker:
    """``Tracker`` adapter around ``trackers.BoTSORTTracker``."""

    def __init__(self, *, lost_seconds: float = 2.0, reference_rate: float = 10.0, camera_motion: bool = True,
                 cmc_method: str = "sparseOptFlow", track_activation_threshold: float = 0.7,
                 high_confidence_threshold: float = 0.6, minimum_consecutive_frames: int = 2) -> None:
        self.params = {
            "lost_seconds": lost_seconds, "reference_rate": reference_rate, "camera_motion": camera_motion,
            "cmc_method": cmc_method, "track_activation_threshold": track_activation_threshold,
            "high_confidence_threshold": high_confidence_threshold,
            "minimum_consecutive_frames": minimum_consecutive_frames,
        }
        self.info = ComponentInfo(role=Role.TRACKER, name="botsort",
                                  version=f"trackers-{_version('trackers')}+supervision-{_version('supervision')}"
                                          "+motion-2", params=self.params)
        self._tracker = None
        self._motion: Optional[CameraMotion] = None
        self._estimate: Optional[np.ndarray] = None  # transform from the latest update, image pixels
        self._quality: Optional[tuple[int, int, float]] = None  # its fit quality, image pixels
        self._ran = False  # whether the estimator ran during the latest update
        self._last_t: Optional[float] = None  # time of the last frame the estimator ran on

    def load(self) -> None:
        import trackers
        p = self.params
        if p["camera_motion"] and (p["cmc_method"] != "sparseOptFlow" or _version("trackers") != CMC_VERSION):
            raise RuntimeError(f"camera-motion quality is implemented for trackers {CMC_VERSION} with "
                               f"sparseOptFlow; found trackers {_version('trackers')} with {p['cmc_method']}")
        self._tracker = trackers.BoTSORTTracker(
            lost_track_buffer=int(round(p["lost_seconds"] * 30)), frame_rate=p["reference_rate"],
            track_activation_threshold=p["track_activation_threshold"],
            high_conf_det_threshold=p["high_confidence_threshold"],
            minimum_consecutive_frames=p["minimum_consecutive_frames"],
            enable_cmc=p["camera_motion"], cmc_method=p["cmc_method"])
        if self._tracker.cmc is not None:
            cmc = self._tracker.cmc = _quality_cmc_class()(self._tracker.cmc.cfg)
            estimate = cmc.estimate

            def recorded(frame_bgr, dets_xyxy=None):
                failures = cmc.frames_failed
                matrix = estimate(frame_bgr, dets_xyxy)
                self._ran = True
                identity = np.array_equal(matrix, np.eye(2, 3, dtype=matrix.dtype))
                failed = identity or cmc.frames_failed != failures or cmc.quality is None
                self._estimate = None if failed else np.array(matrix, float)
                self._quality = None if failed else cmc.quality
                return matrix

            cmc.estimate = recorded
        self._motion = self._estimate = self._last_t = None

    def reset(self) -> None:
        if self._tracker is not None:
            self._tracker.reset()
        self._motion = self._estimate = self._last_t = None

    def close(self) -> None:
        self._tracker = None
        self._motion = self._estimate = self._last_t = None

    def camera_motion(self) -> Optional[CameraMotion]:
        """Image motion estimated during the latest ``update``, in source display pixels; None if none."""
        return self._motion

    def update(self, frame: Frame, detections: Sequence[Detection]) -> list[TrackedObject]:
        import supervision as sv
        if self._tracker is None:
            raise RuntimeError("tracker used before load()")
        if detections:
            xyxy = np.array([[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in detections]) * frame.scale
            sd = sv.Detections(xyxy=xyxy.astype(np.float32),
                               confidence=np.array([d.score for d in detections], np.float32),
                               class_id=np.array([_CATEGORIES.index(d.category) for d in detections]),
                               data={"source_index": np.arange(len(detections))})
        else:
            sd = sv.Detections.empty()
        image = frame.image if self.params["camera_motion"] else None
        self._estimate, self._ran = None, False
        out = self._tracker.update(sd, frame=image, timestamp=frame.t)
        self._motion = None
        if self._estimate is not None and self._last_t is not None and self._last_t < frame.t:
            m = self._estimate
            # scale and rotation are resolution independent; translation goes from image to source pixels
            matches, inliers, residual = self._quality
            self._motion = CameraMotion(since_t=self._last_t, scale=float(np.hypot(m[0, 0], m[1, 0])),
                                        rotation=float(np.arctan2(m[1, 0], m[0, 0])),
                                        tx=float(m[0, 2] / frame.scale), ty=float(m[1, 2] / frame.scale),
                                        matches=matches, inliers=inliers, residual_px=residual / frame.scale)
        if self._ran:
            self._last_t = frame.t
        if not len(out) or out.tracker_id is None:
            return []
        tracked = []
        for i, tid in zip(out.data["source_index"], out.tracker_id):
            if tid >= 0:
                d = detections[int(i)]
                tracked.append(TrackedObject(track_id=int(tid), box=d.box, score=d.score, category=d.category))
        return sorted(tracked, key=lambda o: o.track_id)
