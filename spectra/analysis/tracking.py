"""Lightweight multi-signal object tracker.

Maintains short per-track history (bbox + timestamp) so downstream code can
compute scale expansion (TTC source) and lateral velocity (path crossing).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import cv2
import numpy as np

from ..vision.detection import Detection


@dataclass
class TrackSample:
    frame_index: int
    timestamp_sec: float
    bbox: tuple[int, int, int, int]


@dataclass
class Track:
    track_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    frame_index: int
    timestamp_sec: float
    age: int = 0
    hits: int = 1
    confirmed: bool = False
    display_id: int | None = None
    misses: int = 0
    # Normalized HS colour histogram of the last seen crop, EMA-smoothed. Used
    # as an appearance gate during re-identification so a longer lost-track
    # window does not relabel a different object onto an existing ID.
    appearance: np.ndarray | None = None
    history: Deque[TrackSample] = field(default_factory=lambda: deque(maxlen=12))

    def previous_sample(self, min_dt: float = 0.05) -> TrackSample | None:
        """Return the most recent history sample at least ``min_dt`` ago."""

        if not self.history:
            return None
        for sample in reversed(self.history):
            if (self.timestamp_sec - sample.timestamp_sec) >= min_dt:
                return sample
        return self.history[0]


_APPEARANCE_EMA_ALPHA = 0.3
_REASSOC_APPEARANCE_MIN = 0.30


def _appearance_descriptor(
    frame_bgr: np.ndarray | None,
    bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Normalized 8x8 Hue-Saturation histogram of the bbox crop.

    A cheap, illumination-tolerant colour signature. Returns ``None`` when no
    frame is available or the crop is degenerate, in which case appearance is
    simply not used (the geometric matcher still runs).
    """

    if frame_bgr is None:
        return None
    h_full, w_full = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w_full - 1, int(x1)))
    x2 = max(0, min(w_full, int(x2)))
    y1 = max(0, min(h_full - 1, int(y1)))
    y2 = max(0, min(h_full, int(y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.astype(np.float32)


def _appearance_similarity(
    a: np.ndarray | None,
    b: np.ndarray | None,
) -> float | None:
    """Colour similarity in ``[0, 1]`` (1 == identical), or ``None`` if unknown."""

    if a is None or b is None:
        return None
    dist = float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))
    return max(0.0, 1.0 - dist)


def _blend_appearance(track: Track, new_desc: np.ndarray | None) -> None:
    """EMA-update a track's appearance signature with a fresh observation."""

    if new_desc is None:
        return
    if track.appearance is None:
        track.appearance = new_desc
        return
    blended = (
        (1.0 - _APPEARANCE_EMA_ALPHA) * track.appearance
        + _APPEARANCE_EMA_ALPHA * new_desc
    )
    cv2.normalize(blended, blended, alpha=1.0, norm_type=cv2.NORM_L1)
    track.appearance = blended.astype(np.float32)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _bbox_size(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float(x1 + x2) * 0.5, float(y1 + y2) * 0.5)


def _bbox_area(bbox: tuple[int, int, int, int]) -> float:
    width, height = _bbox_size(bbox)
    return width * height


def _center_distance_ratio(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    aw, ah = _bbox_size(a)
    bw, bh = _bbox_size(b)
    scale = max(1.0, ((aw * aw + ah * ah) ** 0.5 + (bw * bw + bh * bh) ** 0.5) * 0.5)
    return (((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5) / scale


def _scale_compatible(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    min_area_ratio: float = 0.45,
    max_area_ratio: float = 2.60,
    max_aspect_ratio_delta: float = 2.20,
) -> bool:
    aw, ah = _bbox_size(a)
    bw, bh = _bbox_size(b)
    area_ratio = _bbox_area(b) / max(_bbox_area(a), 1.0)
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return False
    aspect_a = aw / ah
    aspect_b = bw / bh
    aspect_delta = max(aspect_a, aspect_b) / max(min(aspect_a, aspect_b), 1e-6)
    return aspect_delta <= max_aspect_ratio_delta


def _predict_bbox(track: Track, timestamp_sec: float) -> tuple[int, int, int, int]:
    previous = track.previous_sample(min_dt=0.001)
    if previous is None:
        return track.bbox

    dt = track.timestamp_sec - previous.timestamp_sec
    future_dt = timestamp_sec - track.timestamp_sec
    if dt <= 0.0 or future_dt <= 0.0:
        return track.bbox

    # Use a conservative constant-velocity bbox prediction. Cap the horizon so
    # one stale motion sample cannot pull a track across the whole frame.
    ratio = min(3.0, future_dt / dt)
    predicted = []
    for current, prior in zip(track.bbox, previous.bbox):
        predicted.append(int(round(float(current) + (float(current) - float(prior)) * ratio)))
    x1, y1, x2, y2 = predicted
    if x2 <= x1 or y2 <= y1:
        return track.bbox
    return (x1, y1, x2, y2)


def _predict_bbox_full(track: Track, timestamp_sec: float) -> tuple[int, int, int, int]:
    """Constant-velocity bbox prediction over the FULL elapsed gap.

    Unlike ``_predict_bbox`` (capped at ~3 frames for live frame-to-frame
    matching), this extrapolates the last-known velocity across the whole
    occlusion gap so a lost track can be re-associated with a detection that
    has moved a long way in the image. Used only for re-identification, where
    scale compatibility and a gap-scaled center threshold guard against bad
    matches.
    """

    previous = track.previous_sample(min_dt=0.001)
    if previous is None:
        return track.bbox

    dt = track.timestamp_sec - previous.timestamp_sec
    future_dt = timestamp_sec - track.timestamp_sec
    if dt <= 0.0 or future_dt <= 0.0:
        return track.bbox

    ratio = future_dt / dt
    predicted = []
    for current, prior in zip(track.bbox, previous.bbox):
        predicted.append(int(round(float(current) + (float(current) - float(prior)) * ratio)))
    x1, y1, x2, y2 = predicted
    if x2 <= x1 or y2 <= y1:
        return track.bbox
    return (x1, y1, x2, y2)


class IoUTracker:
    """Greedy multi-signal matcher with per-track history for TTC."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.25,
        predicted_iou_threshold: float = 0.12,
        center_distance_threshold: float = 0.70,
        max_misses: int = 8,
        confirm_hits: int = 3,
        fast_confirm_confidence: float = 0.70,
        fast_confirm_height_ratio: float = 0.18,
        coast_limit: int = 2,
        hot_coast_limit: int = 6,
        max_lost_sec: float = 2.5,
        reassoc_gap_gain: float = 0.8,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.predicted_iou_threshold = float(predicted_iou_threshold)
        self.center_distance_threshold = float(center_distance_threshold)
        self.max_misses = int(max_misses)
        self.confirm_hits = int(confirm_hits)
        self.fast_confirm_confidence = float(fast_confirm_confidence)
        self.fast_confirm_height_ratio = float(fast_confirm_height_ratio)
        # A confirmed track that misses a detection still emits risk for up to
        # ``coast_limit`` detection frames (using its last bbox) so a one-off
        # YOLO miss does not make a live threat vanish from the active set.
        self.coast_limit = int(coast_limit)
        # "Hot" tracks (the active threat: last primary + recent CAUTION/DANGER
        # ids, supplied by the caller) coast for a longer window so a near lead
        # vehicle that the detector briefly drops stays in the active set —
        # keeping the banner elevated — instead of vanishing after 2 frames and
        # letting primary selection fall to a distant low-risk object. Capped
        # below ``max_misses`` so demotion/re-id timing is unchanged.
        self.hot_coast_limit = int(hot_coast_limit)
        # Confirmed tracks that exhaust ``max_misses`` are demoted to a lost
        # pool instead of deleted, so a re-appearing detection within
        # ``max_lost_sec`` is re-associated to the original ID rather than
        # minting a new one (prevents #1 -> #7 relabeling across short gaps).
        # The window is generous (2.5s) because re-id is appearance-gated: a
        # colour-mismatched detection cannot hijack a stale ID even if its
        # predicted position happens to line up.
        self.max_lost_sec = float(max_lost_sec)
        self.reassoc_gap_gain = float(reassoc_gap_gain)
        self._next_id = 1
        self._next_display_id_by_class: dict[str, int] = {}
        self._tracks: dict[int, Track] = {}
        self._lost_tracks: dict[int, Track] = {}

    def _is_fast_confirm(self, det: Detection, frame_height: int | None) -> bool:
        if frame_height is None or frame_height <= 0:
            return False
        _, y1, _, y2 = det.bbox
        height_ratio = max(0.0, float(y2 - y1)) / float(frame_height)
        return det.confidence >= self.fast_confirm_confidence and height_ratio >= self.fast_confirm_height_ratio

    def _emittable(self, hot_ids: set[int] | None = None) -> list[Track]:
        """Live tracks that should produce risk this frame.

        A confirmed track emits while actively detected (misses == 0) and for a
        short coast window after a miss (misses <= coast_limit), using its last
        bbox. Tracks deeper in a miss streak stay live for re-matching but go
        silent. ``hot_ids`` (the active threat) get the longer
        ``hot_coast_limit`` window so a near lead vehicle the detector briefly
        drops stays visible instead of vanishing from the active set. Applied
        identically on detection and propagate frames so a threat never flickers
        between the two.
        """

        hot = hot_ids or set()
        emittable: list[Track] = []
        for track in self._tracks.values():
            if not track.confirmed:
                continue
            limit = self.hot_coast_limit if track.track_id in hot else self.coast_limit
            if track.misses <= limit:
                emittable.append(track)
        return emittable

    def _ensure_display_id(self, track: Track) -> None:
        if track.display_id is not None:
            return
        next_id = self._next_display_id_by_class.get(track.class_name, 1)
        track.display_id = next_id
        self._next_display_id_by_class[track.class_name] = next_id + 1

    def _confirm_if_ready(
        self,
        track: Track,
        det: Detection,
        frame_height: int | None,
    ) -> None:
        if track.hits >= self.confirm_hits or self._is_fast_confirm(det, frame_height):
            track.confirmed = True
            self._ensure_display_id(track)

    def _match_score(
        self,
        track: Track,
        det: Detection,
        *,
        timestamp_sec: float,
    ) -> float | None:
        direct_iou = _iou(track.bbox, det.bbox)
        predicted_bbox = _predict_bbox(track, timestamp_sec)
        predicted_iou = _iou(predicted_bbox, det.bbox)
        center_ratio = _center_distance_ratio(predicted_bbox, det.bbox)
        scale_ok = _scale_compatible(predicted_bbox, det.bbox)

        accepted = (
            direct_iou >= self.iou_threshold
            or predicted_iou >= self.predicted_iou_threshold
            or (scale_ok and center_ratio <= self.center_distance_threshold)
        )
        if not accepted:
            return None

        center_score = max(0.0, 1.0 - (center_ratio / max(self.center_distance_threshold, 1e-6)))
        match_iou = max(direct_iou, predicted_iou)
        area_ratio = _bbox_area(det.bbox) / max(_bbox_area(predicted_bbox), 1.0)
        scale_score = max(0.0, 1.0 - min(abs(area_ratio - 1.0), 1.0))
        return (2.0 * match_iou) + center_score + (0.25 * scale_score) - (0.03 * track.misses)

    def update(
        self,
        detections: list[Detection],
        *,
        frame_index: int,
        timestamp_sec: float,
        frame_shape: tuple[int, int] | tuple[int, int, int] | None = None,
        frame_bgr: np.ndarray | None = None,
        hot_ids: set[int] | None = None,
    ) -> list[Track]:
        if frame_bgr is not None and frame_shape is None:
            frame_shape = frame_bgr.shape
        frame_height = int(frame_shape[0]) if frame_shape is not None and len(frame_shape) >= 1 else None
        # Appearance signatures for every detection (None when no frame given),
        # reused across the match, re-id and new-track passes below.
        det_descriptors: list[np.ndarray | None] = [
            _appearance_descriptor(frame_bgr, det.bbox) for det in detections
        ]
        active_tracks = list(self._tracks.values())
        unmatched_dets = list(range(len(detections)))
        matched_track_ids: set[int] = set()

        # Build all candidate (track, det) pairs and pick greedily by the
        # strongest multi-signal score. IoU remains the primary signal, while
        # predicted IoU and center/scale compatibility keep IDs stable across
        # sparse YOLO frames and box jitter.
        candidates: list[tuple[float, int, int]] = []
        for det_idx, det in enumerate(detections):
            for track in active_tracks:
                if track.class_name != det.class_name:
                    continue
                score = self._match_score(
                    track,
                    det,
                    timestamp_sec=timestamp_sec,
                )
                if score is not None:
                    candidates.append((score, det_idx, track.track_id))
        candidates.sort(reverse=True)

        used_dets: set[int] = set()
        for _, det_idx, track_id in candidates:
            if det_idx in used_dets or track_id in matched_track_ids:
                continue
            track = self._tracks[track_id]
            det = detections[det_idx]
            track.history.append(
                TrackSample(
                    frame_index=track.frame_index,
                    timestamp_sec=track.timestamp_sec,
                    bbox=track.bbox,
                )
            )
            track.bbox = det.bbox
            track.confidence = det.confidence
            track.frame_index = frame_index
            track.timestamp_sec = timestamp_sec
            track.age += 1
            track.hits += 1
            _blend_appearance(track, det_descriptors[det_idx])
            self._confirm_if_ready(track, det, frame_height)
            track.misses = 0
            matched_track_ids.add(track_id)
            used_dets.add(det_idx)

        unmatched_dets = [i for i in unmatched_dets if i not in used_dets]

        # Re-associate surviving unmatched detections with recently-lost tracks
        # before minting new IDs, so a short occlusion/detection gap does not
        # relabel the same physical object (e.g. #1 -> #7).
        self._expire_lost_tracks(timestamp_sec)
        revived_ids: set[int] = set()
        reassoc_updated_ids: set[int] = set()
        if unmatched_dets:
            # Loose second pass over leftover detections, before minting new IDs.
            # The pool is confirmed tracks that did NOT match this frame —
            # coasting live tracks (a YOLO miss or a corridor-filter drop froze
            # their bbox) plus the lost pool. When such a track reappears at a
            # jumped position the tight live matcher misses it, so without this
            # it would spawn a fresh ID (e.g. truck #2 -> #4). The full-gap
            # scorer reconnects the detection to the original ID instead.
            pool: list[tuple[int, Track, bool]] = []  # (id, track, is_lost)
            for tid, track in self._tracks.items():
                if tid in matched_track_ids or not track.confirmed:
                    continue
                pool.append((tid, track, False))
            for lid, lost in self._lost_tracks.items():
                pool.append((lid, lost, True))

            if pool:
                is_lost_by_id = {tid: is_lost for tid, _t, is_lost in pool}
                reassoc_candidates: list[tuple[float, int, int]] = []
                for det_idx in unmatched_dets:
                    det = detections[det_idx]
                    for tid, track, _is_lost in pool:
                        if track.class_name != det.class_name:
                            continue
                        score = self._reassoc_score(
                            track,
                            det,
                            timestamp_sec=timestamp_sec,
                            det_appearance=det_descriptors[det_idx],
                        )
                        if score is not None:
                            reassoc_candidates.append((score, det_idx, tid))
                reassoc_candidates.sort(reverse=True)
                used_reassoc_dets: set[int] = set()
                assigned_ids: set[int] = set()
                for _, det_idx, tid in reassoc_candidates:
                    if det_idx in used_reassoc_dets or tid in assigned_ids:
                        continue
                    if is_lost_by_id[tid]:
                        track = self._lost_tracks.pop(tid, None)
                        if track is None:
                            continue
                        self._tracks[tid] = track
                        revived_ids.add(tid)
                    else:
                        track = self._tracks.get(tid)
                        if track is None:
                            continue
                        reassoc_updated_ids.add(tid)
                    det = detections[det_idx]
                    track.history.append(
                        TrackSample(
                            frame_index=track.frame_index,
                            timestamp_sec=track.timestamp_sec,
                            bbox=track.bbox,
                        )
                    )
                    track.bbox = det.bbox
                    track.confidence = det.confidence
                    track.frame_index = frame_index
                    track.timestamp_sec = timestamp_sec
                    track.age += 1
                    track.hits += 1
                    track.misses = 0
                    track.confirmed = True  # only confirmed tracks enter the pool
                    _blend_appearance(track, det_descriptors[det_idx])
                    assigned_ids.add(tid)
                    used_reassoc_dets.add(det_idx)
                unmatched_dets = [i for i in unmatched_dets if i not in used_reassoc_dets]

        new_track_ids: set[int] = set()
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track_id = self._next_id
            self._next_id += 1
            track = Track(
                track_id=track_id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                age=1,
                hits=1,
                confirmed=False,
                misses=0,
                appearance=det_descriptors[det_idx],
            )
            self._confirm_if_ready(track, det, frame_height)
            self._tracks[track_id] = track
            new_track_ids.add(track_id)

        # Age out tracks that were not updated this frame. Confirmed tracks that
        # exhaust ``max_misses`` are demoted to the lost pool (for re-id) rather
        # than deleted; unconfirmed tracks are dropped outright.
        updated_ids = matched_track_ids | new_track_ids | revived_ids | reassoc_updated_ids
        for track_id, track in list(self._tracks.items()):
            if track_id in updated_ids:
                continue
            track.misses += 1
            if track.misses > self.max_misses:
                del self._tracks[track_id]
                if track.confirmed:
                    self._lost_tracks[track_id] = track

        # Emit freshly-updated tracks plus confirmed coasting tracks (missed
        # this frame but recently seen) so a one-frame detection miss does not
        # make a live threat vanish from the active set on a detection frame.
        return self._emittable(hot_ids)

    def _reassoc_score(
        self,
        lost: Track,
        det: Detection,
        *,
        timestamp_sec: float,
        det_appearance: np.ndarray | None = None,
    ) -> float | None:
        """Score a lost track against a fresh detection for re-identification.

        IoU is ~0 across a multi-frame gap, so the signal is constant-velocity
        predicted-center proximity plus scale compatibility, gated by colour
        appearance. Returns ``None`` when the pair is incompatible; otherwise a
        higher score is a better match (negated center distance plus an
        appearance bonus).
        """

        predicted = _predict_bbox_full(lost, timestamp_sec)
        gap = max(0.0, timestamp_sec - lost.timestamp_sec)
        # A vehicle that the detector lost while it was approaching keeps growing
        # in the image, so by the time it is re-detected its bbox can be much
        # larger than the last-seen box. Widen the allowed area growth with the
        # gap length (and floor the shrink side) so re-id can reconnect a fast
        # close-range cut-in to its original ID instead of minting a new one.
        # This is the re-id path only; the tight live matcher is unchanged.
        grow_slack = min(gap, self.max_lost_sec) * 1.6
        if not _scale_compatible(
            predicted,
            det.bbox,
            min_area_ratio=0.30,
            max_area_ratio=2.60 + grow_slack,
        ):
            return None
        center_ratio = _center_distance_ratio(predicted, det.bbox)
        threshold = self.center_distance_threshold + (
            min(gap, self.max_lost_sec) * self.reassoc_gap_gain
        )
        if center_ratio > threshold:
            return None
        # Appearance gate: when both signatures exist, a clearly different
        # colour profile rejects the match outright (protects the long re-id
        # window), and a closer match earns a score bonus. Across a SHORT gap the
        # same object's colour can shift (a close car fills the frame, lighting
        # changes), so the min similarity is relaxed for brief gaps where the
        # geometric prediction is still trustworthy.
        appearance_min = _REASSOC_APPEARANCE_MIN if gap > 0.7 else 0.18
        similarity = _appearance_similarity(lost.appearance, det_appearance)
        if similarity is not None:
            if similarity < appearance_min:
                return None
            return -center_ratio + (0.5 * similarity)
        return -center_ratio

    def _expire_lost_tracks(self, timestamp_sec: float) -> None:
        for lost_id in list(self._lost_tracks.keys()):
            if (timestamp_sec - self._lost_tracks[lost_id].timestamp_sec) > self.max_lost_sec:
                del self._lost_tracks[lost_id]

    def propagate(self, hot_ids: set[int] | None = None) -> list[Track]:
        """Return emittable tracks without consuming a detection frame.

        Called on frames where YOLO is intentionally skipped. Miss counters are
        not incremented (we chose not to look), and emission uses the same coast
        gate as ``update`` so a track does not flicker between detection and
        propagate frames.
        """
        return self._emittable(hot_ids)

    def reset(self) -> None:
        self._tracks.clear()
        self._lost_tracks.clear()
        self._next_id = 1
        self._next_display_id_by_class.clear()
