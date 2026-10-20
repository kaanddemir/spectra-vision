"""Versioned ground-truth annotation contract.

Missing knowledge is expressed as ``None``; validators never invent values.

Every event has an origin. It is the event's own ``origin`` if set, otherwise it follows from the clip's
``provenance.method``: ``manual`` is human-confirmed, ``synthetic`` is exact by construction, ``imported:*``
and ``converted:*`` are derived from a dataset's own labels, and anything else is a draft. Only
human-confirmed and synthetic events are ground truth by default (see ``metrics.evaluate``).

A dataset's original labels are kept verbatim in ``source_labels``; events derived from them point back
through ``source_refs``. Schema 1.0 files load unchanged and are read as 1.1.
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

ANNOTATION_SCHEMA_VERSION = "1.1"
_READABLE_VERSIONS = ("1.0", ANNOTATION_SCHEMA_VERSION)


class EventType(str, Enum):
    COLLISION = "collision"
    APPROACH = "approach"
    CUT_IN = "cut_in"


class Light(str, Enum):
    DAY = "day"
    DUSK = "dusk"
    NIGHT = "night"


class Weather(str, Enum):
    CLEAR = "clear"
    RAIN = "rain"
    SNOW = "snow"
    FOG = "fog"


class RoadShape(str, Enum):
    STRAIGHT = "straight"
    CURVE = "curve"
    INTERSECTION = "intersection"


class Split(str, Enum):
    DEV = "dev"
    VALIDATION = "validation"
    TEST = "test"


class Origin(str, Enum):
    HUMAN_CONFIRMED = "human_confirmed"
    SYNTHETIC = "synthetic"
    SOURCE_DERIVED = "source_derived"
    DRAFT = "draft"


GROUND_TRUTH_ORIGINS = frozenset({Origin.HUMAN_CONFIRMED, Origin.SYNTHETIC})


def origin_of_method(method: str) -> Origin:
    if method == "manual":
        return Origin.HUMAN_CONFIRMED
    if method == "synthetic":
        return Origin.SYNTHETIC
    if method.startswith(("imported:", "converted:")):
        return Origin.SOURCE_DERIVED
    return Origin.DRAFT


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Conditions(_Strict):
    light: Optional[Light] = None
    weather: Optional[Weather] = None
    road: Optional[RoadShape] = None
    occlusion: Optional[bool] = None


class Provenance(_Strict):
    annotator: str = Field(min_length=1)
    method: str = Field(min_length=1, description="e.g. 'manual', 'converted:ccd'")
    created: str = Field(min_length=1, description="ISO-8601 date")


class BoxAt(_Strict):
    t: float = Field(ge=0)
    xyxy: tuple[float, float, float, float]
    label: Optional[str] = None
    participant_id: Optional[str] = Field(default=None, description="identity in the labelling source")

    @model_validator(mode="after")
    def _ordered(self) -> "BoxAt":
        x1, y1, x2, y2 = self.xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("box must satisfy x2 > x1 and y2 > y1")
        return self


class Event(_Strict):
    event_id: str = Field(min_length=1)
    type: EventType
    t_start: Optional[float] = Field(default=None, ge=0)
    t_peak: Optional[float] = Field(default=None, ge=0)
    t_end: Optional[float] = Field(default=None, ge=0)
    ego_involved: Optional[bool] = None
    ambiguous: bool = False
    involved: tuple[BoxAt, ...] = ()
    participant_ids: tuple[str, ...] = Field(default=(), description="involved road users, source identities")
    origin: Optional[Origin] = Field(default=None, description="overrides the origin implied by the clip")
    reviewed_by: Optional[str] = None
    source_refs: tuple[str, ...] = Field(default=(), description="label_id values in the clip's source_labels")
    conditions: Conditions = Conditions()
    note: Optional[str] = None

    @model_validator(mode="after")
    def _timing(self) -> "Event":
        if self.origin is Origin.HUMAN_CONFIRMED and not self.reviewed_by:
            raise ValueError("an event confirmed on its own needs reviewed_by")
        if self.t_start is None and self.t_peak is None and self.t_end is None:
            raise ValueError("an event needs at least one of t_start, t_peak, t_end")
        known = [t for t in (self.t_start, self.t_peak, self.t_end) if t is not None]
        if known != sorted(known):
            raise ValueError("timestamps must satisfy t_start <= t_peak <= t_end")
        return self

    def anchor(self) -> float:
        """Best available single time for matching: peak, else midpoint, else the one known bound."""
        if self.t_peak is not None:
            return self.t_peak
        if self.t_start is not None and self.t_end is not None:
            return (self.t_start + self.t_end) / 2
        return self.t_start if self.t_start is not None else self.t_end  # type: ignore[return-value]


Scalar = Union[str, int, float, bool, None]


class SourceLabel(_Strict):
    """One label exactly as the dataset states it. Times are converted; raw values stay in ``fields``."""
    label_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    record: str = Field(min_length=1, description="file and key or line the label came from")
    label: str = Field(min_length=1, description="the dataset's own class name or code")
    fields: dict[str, Scalar] = Field(default_factory=dict)
    t_start: Optional[float] = Field(default=None, ge=0)
    t_end: Optional[float] = Field(default=None, ge=0)
    participant_ids: tuple[str, ...] = ()
    boxes: tuple[BoxAt, ...] = ()
    review_for: tuple["EventType", ...] = Field(
        default=(), description="event types a reviewer should look for; not a label of that type")
    note: Optional[str] = None

    @model_validator(mode="after")
    def _interval(self) -> "SourceLabel":
        if self.t_start is not None and self.t_end is not None and self.t_end < self.t_start:
            raise ValueError(f"source label {self.label_id}: t_end before t_start")
        return self


class Unmeasurable(_Strict):
    type: EventType
    reason: str = Field(min_length=1)


class Licence(_Strict):
    dataset_id: str = Field(min_length=1)
    terms: str = Field(min_length=1, description="identifier or short statement of the governing terms")
    evaluation_allowed: Optional[bool] = None
    training_allowed: Optional[bool] = None


class ClipAnnotation(_Strict):
    schema_version: str = ANNOTATION_SCHEMA_VERSION
    clip_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    licence: Licence
    split: Split
    duration_s: float = Field(gt=0)
    fps_nominal: Optional[float] = Field(default=None, gt=0)
    variable_frame_rate: Optional[bool] = None
    frame_timestamps_s: Optional[tuple[float, ...]] = None
    conditions: Conditions = Conditions()
    events: tuple[Event, ...] = ()
    negative_for: tuple[EventType, ...] = Field(
        default=(), description="event types verified absent from the whole clip"
    )
    unmeasurable: tuple[Unmeasurable, ...] = Field(
        default=(), description="event types that cannot be judged on this clip, with the reason"
    )
    source_labels: tuple[SourceLabel, ...] = ()
    provenance: Provenance

    @model_validator(mode="before")
    @classmethod
    def _upgrade(cls, data):
        if isinstance(data, dict) and data.get("schema_version") in _READABLE_VERSIONS:
            data = {**data, "schema_version": ANNOTATION_SCHEMA_VERSION}
        return data

    @property
    def origin(self) -> Origin:
        return origin_of_method(self.provenance.method)

    def origin_of(self, event: Event) -> Origin:
        return event.origin or self.origin

    @model_validator(mode="after")
    def _consistency(self) -> "ClipAnnotation":
        if self.schema_version != ANNOTATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version!r}")
        if self.negative_for and self.origin not in GROUND_TRUTH_ORIGINS:
            raise ValueError("negative_for needs a manual or synthetic clip; absence cannot be imported or drafted")
        unmeasurable = {u.type for u in self.unmeasurable}
        if unmeasurable & set(self.negative_for):
            raise ValueError("an event type cannot be both negative and unmeasurable")
        label_ids = [s.label_id for s in self.source_labels]
        if len(label_ids) != len(set(label_ids)):
            raise ValueError("duplicate label_id in source_labels")
        for s in self.source_labels:
            for t in (s.t_start, s.t_end):
                if t is not None and t > self.duration_s + 1e-6:
                    raise ValueError(f"source label {s.label_id} exceeds clip duration")
        ids = [e.event_id for e in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate event_id in clip")
        for e in self.events:
            for t in (e.t_start, e.t_peak, e.t_end):
                if t is not None and t > self.duration_s + 1e-6:
                    raise ValueError(f"event {e.event_id} exceeds clip duration")
            if e.type in self.negative_for:
                raise ValueError(f"clip is marked negative for {e.type.value} but has such an event")
            missing = set(e.source_refs) - set(label_ids)
            if missing:
                raise ValueError(f"event {e.event_id} refers to unknown source labels {sorted(missing)}")
        ts = self.frame_timestamps_s
        if ts is not None and any(b < a for a, b in zip(ts, ts[1:])):
            raise ValueError("frame_timestamps_s must be non-decreasing")
        return self

    def ground_truth(self, event_type: EventType, origins=GROUND_TRUTH_ORIGINS) -> list[Event]:
        return [e for e in self.events if e.type == event_type and self.origin_of(e) in origins]

    def is_evaluated_for(self, event_type: EventType, origins=GROUND_TRUTH_ORIGINS) -> bool:
        """A clip counts for a type if it is verified negative or has events of that type from an accepted
        origin, and the type is not marked unmeasurable."""
        if any(u.type == event_type for u in self.unmeasurable):
            return False
        return event_type in self.negative_for or bool(self.ground_truth(event_type, origins))


def load_annotations(path: Path) -> list[ClipAnnotation]:
    """Load a JSON Lines file, one clip per line."""
    clips = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip():
            try:
                clips.append(ClipAnnotation.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{n}: {exc}") from exc
    ids = [c.clip_id for c in clips]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate clip_id")
    return clips


def dump_annotations(clips: list[ClipAnnotation], path: Path) -> None:
    Path(path).write_text("".join(c.model_dump_json(exclude_defaults=False) + "\n" for c in clips))


def json_schema() -> str:
    return json.dumps(ClipAnnotation.model_json_schema(), indent=2)
