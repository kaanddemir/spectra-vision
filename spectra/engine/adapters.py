"""Interfaces that perception and event components implement.

Adapters translate library-specific outputs into the contracts in ``contracts``; nothing downstream sees
model class ids, tensors or tracker internals. Every component:

* exposes ``info`` (a ``ComponentInfo``) naming the model, version, checkpoint hash, parameters and
  sampling interval, so a cache made with one configuration is never reused for another;
* acquires heavy resources in ``load`` and releases them in ``close``. The runner calls ``load`` once
  per run, after validating the video, and always calls ``close``. Loading must not download anything;
* raises ``ComponentFailure`` for a recoverable problem with one frame (the frame is recorded as failed
  and the run continues). Any other exception fails the run. An event analyser raises
  ``AnalyzerUnavailable`` when its inputs are not in the records and ``ComponentFailure`` for any other
  recoverable problem; either is recorded in its report and the run completes.

Coordinates: a detector returns boxes in the pixels of the ``Frame.image`` it was given and the runner
rescales them to source display pixels (``Frame.scale``). A tracker receives and returns source display
pixels. An ego-lane estimator returns boundaries normalised by the segmentation mask size.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from spectra.engine.contracts import (CameraMotion, ComponentInfo, Detection, EgoLaneObservation, EventDraft,
                                      FrameRecord, ReasonCode, TrackedObject, UnresolvedPattern)
from spectra.engine.video import Frame


class ComponentFailure(Exception):
    """A recoverable failure on one frame."""


class AnalyzerUnavailable(ComponentFailure):
    """An event analyser's inputs are not in the records (e.g. no lane relations were computed)."""


@dataclass
class AnalyzerOutput:
    """What an event analyser returns: its proposals, and how many patterns it saw but did not propose,
    by reason. Every pattern the analyser considered is one or the other. Patterns rejected because a
    signal was missing are also listed with their time (``unresolved``)."""
    drafts: list[EventDraft] = field(default_factory=list)
    rejected: Counter = field(default_factory=Counter)  # ReasonCode -> count
    unresolved: list[UnresolvedPattern] = field(default_factory=list)

    def reject(self, reason: ReasonCode, n: int = 1) -> None:
        self.rejected[reason] += n

    def unresolve(self, pattern: UnresolvedPattern) -> None:
        self.unresolved.append(pattern)
        self.reject(pattern.reason)

    def extend(self, other: "AnalyzerOutput") -> None:
        self.drafts.extend(other.drafts)
        self.rejected.update(other.rejected)
        self.unresolved.extend(other.unresolved)


@dataclass(frozen=True)
class Segmentation:
    mask: np.ndarray  # (H, W) uint8 class indexes
    classes: tuple[str, ...]  # canonical name of each index, e.g. ("road", "lane_marking", ...)


@runtime_checkable
class Component(Protocol):
    info: ComponentInfo

    def load(self) -> None: ...

    def close(self) -> None: ...


class Detector(Component, Protocol):
    def detect(self, frame: Frame) -> Sequence[Detection]: ...


class Tracker(Component, Protocol):
    def update(self, frame: Frame, detections: Sequence[Detection]) -> Sequence[TrackedObject]:
        """Called on every frame the detector succeeded on, in time order, with detections in source display
        pixels. ``frame`` is available for camera motion compensation (its image is ``frame.scale`` times
        source size); ``frame.t`` gives the true time step on variable-rate sources."""
        ...


@runtime_checkable
class ReportsCameraMotion(Protocol):
    def camera_motion(self) -> Optional[CameraMotion]:
        """Image motion estimated during the latest tracker update, or None when there is no estimate."""
        ...


class Segmenter(Component, Protocol):
    def segment(self, frame: Frame) -> Segmentation: ...


class EgoLaneEstimator(Component, Protocol):
    def estimate(self, frame: Frame, segmentation: Segmentation) -> EgoLaneObservation:
        """Boundaries in coordinates normalised by the segmentation mask size (resolution independent)."""
        ...


class EventAnalyzer(Component, Protocol):
    def analyze(self, frames: Sequence[FrameRecord], width: int, height: int) -> AnalyzerOutput:
        """Runs once after all frames, on perception records only; never on pixels or models."""
        ...
