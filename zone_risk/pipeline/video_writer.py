"""Annotated video and event-log writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class VideoWriter:
    def __init__(self, output_path: str | Path, fps: float, frame_size: tuple[int, int]) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, fps if fps > 0 else 25.0, frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {self.output_path}")

    def write(self, frame_bgr: np.ndarray) -> None:
        self.writer.write(frame_bgr)

    def close(self) -> None:
        self.writer.release()


class JsonlEventWriter:
    def __init__(self, output_path: str | Path | None) -> None:
        self.handle = None
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open("w", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        if self.handle is None:
            return
        self.handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

