"""Video frame loading helpers for the zone-based risk pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoFrame:
    frame_index: int
    timestamp_sec: float
    bgr: np.ndarray


class VideoLoader:
    """Small wrapper around OpenCV video capture."""

    def __init__(self, source: str | Path, max_frames: int | None = None) -> None:
        self.source = str(source)
        self.max_frames = max_frames
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def frames(self) -> Iterator[VideoFrame]:
        frame_index = 0
        try:
            while self.max_frames is None or frame_index < self.max_frames:
                ok, frame = self.capture.read()
                if not ok:
                    break

                timestamp_sec = frame_index / self.fps if self.fps > 0.0 else float(frame_index)
                yield VideoFrame(frame_index=frame_index, timestamp_sec=timestamp_sec, bgr=frame)
                frame_index += 1
        finally:
            self.close()

    def close(self) -> None:
        self.capture.release()

