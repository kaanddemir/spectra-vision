"""Lightweight IoU-based multi-object tracker.

Maintains short per-track history (bbox + timestamp) so downstream code can
compute scale expansion (TTC source) and lateral velocity (path crossing).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from ..vision.object_detector import Detection


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


class IoUTracker:
    """Greedy IoU matcher with per-track history for expansion-rate TTC."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.25,
        max_misses: int = 5,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_misses = int(max_misses)
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

    def update(
        self,
        detections: list[Detection],
        *,
        frame_index: int,
        timestamp_sec: float,
    ) -> list[Track]:
        active_tracks = list(self._tracks.values())
        unmatched_dets = list(range(len(detections)))
        matched_track_ids: set[int] = set()

        # Build all candidate (track, det) pairs above the IoU threshold and
        # pick them greedily by descending IoU, requiring class agreement.
        candidates: list[tuple[float, int, int]] = []
        for det_idx, det in enumerate(detections):
            for track in active_tracks:
                if track.class_name != det.class_name:
                    continue
                iou = _iou(track.bbox, det.bbox)
                if iou >= self.iou_threshold:
                    candidates.append((iou, det_idx, track.track_id))
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
            track.misses = 0
            matched_track_ids.add(track_id)
            used_dets.add(det_idx)

        unmatched_dets = [i for i in unmatched_dets if i not in used_dets]
        new_track_ids: set[int] = set()
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = Track(
                track_id=track_id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                age=1,
                misses=0,
            )
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

        return [self._tracks[tid] for tid in updated_ids]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
