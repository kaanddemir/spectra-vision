"""Deterministic stand-in components that follow the adapter interfaces without any model."""
from __future__ import annotations

from typing import Optional

import numpy as np

from spectra.engine.adapters import ComponentFailure, Segmentation
from spectra.engine.adapters import AnalyzerOutput
from spectra.engine.contracts import (Basis, Box, ComponentInfo, Detection, EgoLaneObservation, EventDraft, EventType,
                                      EvidenceItem, EvidenceState, LaneBoundary, ObjectCategory, ReasonCode, Role,
                                      TrackedObject)
from spectra.engine.contracts import VideoInfo, TimestampSource
from spectra.engine.resources import ResourceGuard
from tests.engine.media import read_index


class _Fake:
    role: Role

    def __init__(self, name: Optional[str] = None, interval_s: Optional[float] = None, fail_on=(), **params):
        self.info = ComponentInfo(role=self.role, name=name or f"fake-{self.role.value}", version="1",
                                  params=params, interval_s=interval_s)
        self.fail_on = set(fail_on)
        self.loaded = self.closed = 0
        self.calls: list[float] = []

    def load(self) -> None:
        self.loaded += 1

    def close(self) -> None:
        self.closed += 1

    def _call(self, frame) -> int:
        self.calls.append(frame.t)
        i = read_index(frame.image) if frame.scale == 1.0 else len(self.calls) - 1
        if i in self.fail_on:
            raise ComponentFailure(f"fake failure on frame {i}")
        return i


class FakeDetector(_Fake):
    """One car whose box position encodes the frame number, in image pixels."""
    role = Role.DETECTOR

    def detect(self, frame):
        i = self._call(frame)
        h, w = frame.image.shape[:2]
        x = (i % 10) * w / 20
        return [Detection(box=Box(x1=x, y1=h * 0.25, x2=x + w / 4, y2=h * 0.75), score=0.9,
                          category=ObjectCategory.CAR)]


class FakeTracker(_Fake):
    role = Role.TRACKER

    def update(self, frame, detections):
        self.calls.append(frame.t)
        return [TrackedObject(track_id=k, box=d.box, score=d.score, category=d.category)
                for k, d in enumerate(detections)]


class FakeSegmenter(_Fake):
    role = Role.SEGMENTER
    CLASSES = ("road", "lane_marking", "ego_vehicle")

    def segment(self, frame):
        i = self._call(frame)
        mask = np.zeros((24, 32), np.uint8)
        mask[12:, :] = 0
        mask[12:20, 8 + i % 4] = 1
        mask[12:20, 24 - i % 4] = 1
        mask[20:, :] = 2
        return Segmentation(mask=mask, classes=self.CLASSES)


class FakeEgoLane(_Fake):
    role = Role.EGO_LANE

    def estimate(self, frame, segmentation):
        self.calls.append(frame.t)
        cols = np.flatnonzero((segmentation.mask == 1).any(0))
        w = segmentation.mask.shape[1]
        left, right = cols[0] / w, cols[-1] / w
        offset = (0.5 - (left + right) / 2) / (right - left)
        return EgoLaneObservation(left=LaneBoundary(near=(left, 0.8), far=(left, 0.5)),
                                  right=LaneBoundary(near=(right, 0.8), far=(right, 0.5)),
                                  offset=offset, confidence=0.5)


class FakeAnalyzer(_Fake):
    """Proposes one approach spanning the frames where track 0 exists, and rejects one pattern."""
    role = Role.EVENT_ANALYZER

    def analyze(self, frames, width, height):
        self.calls.append(len(frames))
        times = [f.t for f in frames if f.tracks]
        out = AnalyzerOutput()
        out.reject(ReasonCode.OTHER_LANE)
        if times:
            out.drafts.append(EventDraft(type=EventType.RAPID_APPROACH, state=EvidenceState.SUPPORTED,
                                         t_start=times[0], t_peak=times[len(times) // 2], t_end=times[-1], rank=0.5,
                                         track_ids=(0,), analyzer=self.info.name,
                                         evidence=(EvidenceItem(basis=Basis.OBSERVED, name="track", t=times[0],
                                                                track_id=0),)))
        return out


SHA = "a" * 64

def video(**kw):
    base = dict(file_name="clip.mp4", size_bytes=10, sha256=SHA, container="mp4", codec="h264", width=1280,
                height=720, coded_width=1280, coded_height=720, rotation=0, time_base="1/1000",
                timestamp_source=TimestampSource.CONTAINER)
    return VideoInfo(**{**base, **kw})


def calm(**kw):
    return ResourceGuard(pressure=lambda: 1, rss=lambda: 100.0, rss_budget_mb=1000, **kw)
