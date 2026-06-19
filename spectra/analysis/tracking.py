"""Lightweight multi-signal object tracker.

Maintains short per-track history (bbox + timestamp) so downstream code can
compute scale expansion (TTC source) and lateral velocity (path crossing).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

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
    history: Deque[TrackSample] = field(default_factory=lambda: deque(maxlen=12))

    def previous_sample(self, min_dt: float = 0.05) -> TrackSample | None:
        """Return the most recent history sample at least ``min_dt`` ago."""

        if not self.history:
            return None
        for sample in reversed(self.history):
            if (self.timestamp_sec - sample.timestamp_sec) >= min_dt:
                return sample
        return self.history[0]


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


class IoUTracker:
    """Greedy multi-signal matcher with per-track history for TTC."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.25,
        predicted_iou_threshold: float = 0.12,
        center_distance_threshold: float = 0.70,
        max_misses: int = 5,
        confirm_hits: int = 2,
        fast_confirm_confidence: float = 0.70,
        fast_confirm_height_ratio: float = 0.18,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.predicted_iou_threshold = float(predicted_iou_threshold)
        self.center_distance_threshold = float(center_distance_threshold)
        self.max_misses = int(max_misses)
        self.confirm_hits = int(confirm_hits)
        self.fast_confirm_confidence = float(fast_confirm_confidence)
        self.fast_confirm_height_ratio = float(fast_confirm_height_ratio)
        self._next_id = 1
        self._next_display_id = 1
        self._tracks: dict[int, Track] = {}

    def _is_fast_confirm(self, det: Detection, frame_height: int | None) -> bool:
        if frame_height is None or frame_height <= 0:
            return False
        _, y1, _, y2 = det.bbox
        height_ratio = max(0.0, float(y2 - y1)) / float(frame_height)
        return det.confidence >= self.fast_confirm_confidence and height_ratio >= self.fast_confirm_height_ratio

    @staticmethod
    def _visible(tracks: list[Track]) -> list[Track]:
        return [track for track in tracks if track.confirmed]

    def _ensure_display_id(self, track: Track) -> None:
        if track.display_id is not None:
            return
        track.display_id = self._next_display_id
        self._next_display_id += 1

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
    ) -> list[Track]:
        frame_height = int(frame_shape[0]) if frame_shape is not None and len(frame_shape) >= 1 else None
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
            self._confirm_if_ready(track, det, frame_height)
            track.misses = 0
            matched_track_ids.add(track_id)
            used_dets.add(det_idx)

        unmatched_dets = [i for i in unmatched_dets if i not in used_dets]
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
            )
            self._confirm_if_ready(track, det, frame_height)
            self._tracks[track_id] = track
            new_track_ids.add(track_id)

        # Age out tracks that were not updated this frame. Risk is only
        # emitted for tracks that received a detection in this frame.
        updated_ids = matched_track_ids | new_track_ids
        for track_id, track in list(self._tracks.items()):
            if track_id in updated_ids:
                continue
            track.misses += 1
            if track.misses > self.max_misses:
                del self._tracks[track_id]

        return self._visible([self._tracks[tid] for tid in updated_ids])

    def propagate(self) -> list[Track]:
        """Return all surviving tracks without consuming a detection frame.

        Called on frames where YOLO is intentionally skipped. Tracks are
        returned as-is so risk estimation continues; miss counters are not
        incremented because we chose not to look, not because objects vanished.
        """
        return self._visible(list(self._tracks.values()))

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._next_display_id = 1
