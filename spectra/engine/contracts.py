"""Typed, versioned data contracts for analysis results.

Conventions shared by every model here:

* Time ``t`` is seconds of source presentation time since the video stream's start time, taken from
  decoded frame timestamps. It is never ``frame_index / fps``. When a stream carries no timestamps the
  video's ``timestamp_source`` says so and times derive from an explicitly supplied rate.
* Pixel coordinates are in the display-oriented source frame (after applying container rotation, before
  any analysis downscaling): origin at the top-left corner, x to the right, y down, continuous edges, so
  a full-frame box is ``(0, 0, width, height)``. Normalised coordinates divide by display width and
  height and may fall outside ``[0, 1]`` when a quantity is extrapolated beyond the frame.
* ``None`` means unknown. A component that ran and found nothing reports an empty tuple with status
  ``ok``; one that did not run reports a different status and no payload.
* Event proposals are reviewable moments, never findings: nothing here states that a collision, near
  miss or cut-in happened. Human decisions live in ``ReviewDecision`` records, which are never part of an
  analysis result or the perception cache.
* Machine-relevant reasons are enumerated codes (``LaneReason``, ``ReasonCode``, ``WarningCode``); prose
  is for people only.
"""
from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from typing import Any, Iterable, Literal, Optional, Sequence

from pydantic import (BaseModel, ConfigDict, Field, PlainValidator, TypeAdapter, field_serializer, field_validator,
                      model_validator)
from typing_extensions import Annotated

SCHEMA_VERSION = "1.0"

_SHA256 = r"^[0-9a-f]{64}$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def canonical_json(value: BaseModel | dict | list) -> str:
    """Deterministic JSON: sorted keys, no whitespace, no NaN or infinity."""
    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: BaseModel | dict | list) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


# --- Source video ---------------------------------------------------------------------------------


class TimestampSource(str, Enum):
    CONTAINER = "container_pts"  # per-frame presentation timestamps from the container
    ASSUMED_RATE = "assumed_rate"  # stream has no timestamps; t = decode index / assumed_fps


class VideoInfo(_Strict):
    file_name: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=_SHA256)
    container: str
    codec: str
    width: int = Field(gt=0, description="display width, after rotation")
    height: int = Field(gt=0, description="display height, after rotation")
    coded_width: int = Field(gt=0)
    coded_height: int = Field(gt=0)
    rotation: Literal[0, 90, 180, 270] = Field(description="counter-clockwise degrees applied for display")
    time_base: Optional[str] = Field(default=None, description="stream time base as 'num/den'")
    start_s: float = Field(default=0.0, description="stream start time subtracted from every timestamp")
    duration_s: Optional[float] = Field(default=None, ge=0, description="container-declared, may be absent")
    nominal_fps: Optional[float] = Field(default=None, gt=0, description="container-declared; never used as time")
    timestamp_source: TimestampSource
    assumed_fps: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _rate(self) -> "VideoInfo":
        if (self.timestamp_source is TimestampSource.ASSUMED_RATE) != (self.assumed_fps is not None):
            raise ValueError("assumed_fps is required with, and only with, assumed-rate timestamps")
        swapped = self.rotation in (90, 270)
        if (self.width, self.height) != ((self.coded_height, self.coded_width) if swapped
                                         else (self.coded_width, self.coded_height)):
            raise ValueError("display size must equal coded size after rotation")
        return self


class TimingReport(_Strict):
    """What the decoder actually delivered over the analysed span."""
    frames_decoded: int = Field(ge=0)
    frames_analysed: int = Field(ge=0)
    first_t: Optional[float] = None
    last_t: Optional[float] = None
    median_interval_s: Optional[float] = Field(default=None, ge=0)
    max_interval_s: Optional[float] = Field(default=None, ge=0)
    max_interval_start: Optional[float] = Field(default=None, ge=0,
                                                description="time of the frame before the longest gap")
    variable_frame_rate: Optional[bool] = None
    missing_timestamps: int = Field(default=0, ge=0, description="frames dropped for lacking a timestamp")
    non_monotonic: int = Field(default=0, ge=0, description="frames dropped for a repeated or earlier timestamp")
    decode_errors: int = Field(default=0, ge=0)
    ended_early: Optional[str] = Field(default=None, description="why decoding stopped before the stream end")


# --- Perception -----------------------------------------------------------------------------------


class ObjectCategory(str, Enum):
    PERSON = "person"
    BICYCLE = "bicycle"
    MOTORCYCLE = "motorcycle"
    CAR = "car"
    BUS = "bus"
    TRUCK = "truck"


class Box(_Strict):
    """Axis-aligned box in display-oriented source pixels (see module docstring)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _ordered(self) -> "Box":
        if not (self.x2 > self.x1 and self.y2 > self.y1):
            raise ValueError("box must satisfy x2 > x1 and y2 > y1")
        return self

    def normalized(self, width: int, height: int) -> tuple[float, float, float, float]:
        return self.x1 / width, self.y1 / height, self.x2 / width, self.y2 / height


class Detection(_Strict):
    box: Box
    score: float = Field(ge=0, le=1)
    category: ObjectCategory
    ego_overlap: Optional[float] = Field(default=None, ge=0, le=1,
                                         description="fraction of the box the segmenter labels as the camera's "
                                                     "own vehicle; None when no ego-vehicle mask was available")
    ego_vehicle: bool = Field(default=False, description="judged to be the camera's own vehicle (hood) and "
                                                         "withheld from tracking; kept here, never deleted")

    @model_validator(mode="after")
    def _ego_needs_overlap(self) -> "Detection":
        if self.ego_vehicle and self.ego_overlap is None:
            raise ValueError("ego_vehicle requires a measured ego_overlap")
        return self


class TrackedObject(_Strict):
    track_id: int = Field(ge=0, description="unique within one analysis run only")
    box: Box
    score: Optional[float] = Field(default=None, ge=0, le=1)
    category: ObjectCategory
    predicted: bool = Field(default=False, description="box extrapolated by the tracker, not observed in this frame")


class SegmentationRef(_Strict):
    """A class-index mask stored as a lossless image next to the result, verified by hash."""
    path: str = Field(min_length=1, description="relative to the cache entry directory")
    sha256: str = Field(pattern=_SHA256)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    classes: tuple[str, ...] = Field(min_length=1, description="canonical class name for each mask value")

    @field_validator("path")
    @classmethod
    def _relative(cls, v: str) -> str:
        if v.startswith(("/", "\\")) or ".." in v.split("/"):
            raise ValueError("segmentation path must be relative and inside the cache entry")
        return v


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


class LaneBoundary(_Strict):
    """One lane boundary in normalised display coordinates, near end first.

    ``near`` and ``far`` are the ends of ``points`` (optional), which sample the boundary so curves are
    represented. Beyond ``evidence_y`` (normalised row of the furthest marking evidence) the points are a
    bounded extrapolation of the fitted road model, not observations. ``lateral`` is the boundary's ground position relative to the
    camera in camera heights (+ right), from the flat-road model; ``None`` when not estimated.
    """
    near: tuple[float, float]
    far: tuple[float, float]
    points: tuple[tuple[float, float], ...] = ()
    lateral: Optional[float] = None
    track_id: Optional[int] = Field(default=None, ge=0, description="boundary identity across frames")
    measured: bool = Field(default=True, description="False: carried over from earlier frames, not seen now")
    age_s: float = Field(default=0.0, ge=0, description="time since the boundary was last measured")
    evidence_y: Optional[float] = Field(default=None, description="normalised row of the furthest evidence")

    @model_validator(mode="after")
    def _finite(self) -> "LaneBoundary":
        if not _finite(*self.near, *self.far, *(v for p in self.points for v in p)) or (
                self.lateral is not None and not math.isfinite(self.lateral)):
            raise ValueError("boundary coordinates must be finite")
        return self


class EgoLaneState(str, Enum):
    ESTIMATED = "estimated"  # both ego boundaries supported; offset given
    AMBIGUOUS = "ambiguous"  # lane structure seen but which lane the camera occupies is not decidable
    UNAVAILABLE = "unavailable"  # not enough lane evidence


class LaneChange(_Strict):
    """The camera crossing a tracked lane boundary. An observation for later analysis, not an event.

    ``crossing`` while the camera is in the boundary's band, ``completed`` once it stayed on the other side,
    ``cancelled`` on the one frame where a crossing ended without a lane change (the camera returned, or the
    tracked marking changed). Absence of a later phase never means completion."""
    direction: Literal["left", "right", "unknown"]
    phase: Literal["crossing", "completed", "cancelled"]
    t_start: float = Field(ge=0, description="when the lateral movement towards the crossed boundary began")
    boundary_id: Optional[int] = Field(default=None, ge=0)


class LaneReason(str, Enum):
    """Why an ego-lane observation is ambiguous, unavailable or limited."""
    SHARP_BEND = "sharp_bend"  # bend too sharp for lane-change decisions
    SHALLOW_ROAD = "shallow_road"  # too little visible road between horizon and bonnet
    BOUNDARY_MISSING = "boundary_missing"  # one ego boundary is not tracked
    IMPLAUSIBLE_WIDTH = "implausible_width"  # lane width outside the plausible range
    WIDTH_CHANGED = "width_changed"  # lane width differs from recent frames
    ON_BOUNDARY = "on_boundary"  # the camera is within a boundary's band
    NO_HORIZON = "no_horizon"  # no horizon estimate yet
    NO_EVIDENCE = "no_evidence"  # no lane evidence at all


class EgoLaneObservation(_Strict):
    """The lane the camera occupies, derived from segmentation plus temporal tracking.

    Positions refer to the camera, not the vehicle centre: the camera's mounting offset within the vehicle
    is not observable from the video. ``offset`` is in lane widths (+ right of the lane centre) and is
    measured at the camera's own ground position, so it does not depend on camera yaw or the image centre.
    """
    state: EgoLaneState = EgoLaneState.ESTIMATED
    left: Optional[LaneBoundary] = None
    right: Optional[LaneBoundary] = None
    offset: Optional[float] = Field(default=None, description="camera position in lane widths, + right of centre")
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    width: Optional[float] = Field(default=None, gt=0, description="lane width in camera heights")
    horizon_y: Optional[float] = Field(default=None, description="normalised row of the road horizon used")
    curvature: Optional[float] = Field(default=None, description="road bend term of the lane model, + bends "
                                                                  "right; relative units, not 1/m")
    lateral_rate: Optional[float] = Field(default=None, description="d(offset)/dt in lane widths per second")
    far_y: Optional[float] = Field(default=None, description="normalised row of the furthest lane evidence")
    lane_change: Optional[LaneChange] = None
    reasons: tuple[LaneReason, ...] = Field(default=(), description="the estimator's stated assumptions are in "
                                                                     "its ComponentInfo, not repeated here")

    @model_validator(mode="after")
    def _offset_needs_both(self) -> "EgoLaneObservation":
        if self.offset is not None and (self.left is None or self.right is None):
            raise ValueError("offset requires both boundaries")
        if self.state is EgoLaneState.UNAVAILABLE and (self.offset is not None or self.confidence):
            raise ValueError("an unavailable ego lane carries no offset or confidence")
        return self


class LaneRelation(str, Enum):
    EGO_LANE = "ego_lane"
    LEFT_ADJACENT = "left_adjacent"
    RIGHT_ADJACENT = "right_adjacent"
    BEYOND = "beyond_adjacent"  # on the road, further than one lane from the ego lane
    OFF_ROAD = "off_road"
    AMBIGUOUS = "ambiguous"  # straddling a boundary or beyond the lane evidence


class TrackLaneRelation(_Strict):
    """Where one tracked object's ground contact point lies relative to the ego lane."""
    track_id: int = Field(ge=0)
    relation: LaneRelation
    lateral: Optional[float] = Field(default=None, description="contact point relative to the ego-lane centre, "
                                                               "in lane widths at that row, + right")
    lane_t: float = Field(ge=0, description="time of the ego-lane observation used")
    beyond_evidence: bool = Field(default=False, description="contact point lies where the lane boundaries are "
                                                             "extrapolated from the road model, not observed")


class CameraMotion(_Strict):
    """Global image motion from the tracker's previous frame to this one, as estimated for its motion
    compensation: a RANSAC similarity fit to sparse optical flow over the whole image, moving objects
    included. It aligns images; it is not measured vehicle motion. Forward driving makes the near road
    expand, so ``scale`` mixes parallax with any zoom and is not a camera zoom measurement.

    A previous-frame point ``p`` (display-oriented source pixels) maps to
    ``scale * R(rotation) @ p + (tx, ty)``, rotation in radians, positive clockwise on screen (y down).

    ``matches``, ``inliers`` and ``residual_px`` describe the fit, so a degenerate estimate (blur, a
    featureless or mostly moving scene) can be told apart from real image motion. They describe agreement
    among the tracked points, not whether the motion is the camera's.
    """
    since_t: float = Field(ge=0, description="time of the previous frame the motion is measured from")
    scale: float = Field(gt=0)
    rotation: float
    tx: float
    ty: float
    matches: int = Field(ge=0, description="point correspondences the fit was given")
    inliers: int = Field(ge=0, description="correspondences RANSAC kept")
    residual_px: float = Field(ge=0, description="median distance of the inliers from the fitted transform, "
                                                 "source pixels")

    @model_validator(mode="after")
    def _counts(self) -> "CameraMotion":
        if self.inliers > self.matches:
            raise ValueError("inliers cannot exceed matches")
        return self


class Role(str, Enum):
    DETECTOR = "detector"
    TRACKER = "tracker"
    SEGMENTER = "segmenter"
    EGO_LANE = "ego_lane"
    EVENT_ANALYZER = "event_analyzer"


class OutputStatus(str, Enum):
    OK = "ok"  # ran; an empty payload is a negative at the component's operating point
    NOT_SCHEDULED = "not_scheduled"  # not due on this frame at its sampling interval
    INPUT_UNAVAILABLE = "input_unavailable"  # an upstream component did not produce output
    FAILED = "failed"  # ran and raised a recoverable error or returned invalid output


_PAYLOAD = {Role.DETECTOR: "detections", Role.TRACKER: "tracks", Role.SEGMENTER: "segmentation",
            Role.EGO_LANE: "ego_lane"}


class FrameRecord(_Strict):
    """Everything known about one analysed frame. Absent roles were not configured for the run."""
    index: int = Field(ge=0, description="position among analysed frames")
    decode_index: Optional[int] = Field(default=None, ge=0,
                                        description="position among decoded frames, unknown after a seek; not a time")
    t: float = Field(ge=0)
    pts: Optional[int] = None
    status: dict[Role, OutputStatus] = Field(default_factory=dict)
    errors: dict[Role, str] = Field(default_factory=dict)
    detections: Optional[tuple[Detection, ...]] = None
    tracks: Optional[tuple[TrackedObject, ...]] = None
    segmentation: Optional[SegmentationRef] = None
    ego_lane: Optional[EgoLaneObservation] = None
    lane_relations: Optional[tuple[TrackLaneRelation, ...]] = Field(
        default=None, description="derived from tracks and the latest ego lane; None when not computed")
    camera_motion: Optional[CameraMotion] = Field(
        default=None, description="from the tracker's motion compensation; None when it was not estimated or "
                                  "the estimate failed")

    @model_validator(mode="after")
    def _consistent(self) -> "FrameRecord":
        if Role.EVENT_ANALYZER in self.status:
            raise ValueError("event analysers do not report per frame")
        for role, field in _PAYLOAD.items():
            ok = self.status.get(role) is OutputStatus.OK
            if ok != (getattr(self, field) is not None):
                raise ValueError(f"{field} must be present exactly when {role.value} status is ok")
        for role in self.errors:
            if self.status.get(role) is not OutputStatus.FAILED:
                raise ValueError(f"error recorded for {role.value} without failed status")
        if self.tracks is not None and len({o.track_id for o in self.tracks}) != len(self.tracks):
            raise ValueError("duplicate track_id in frame")
        if self.camera_motion is not None:
            if self.status.get(Role.TRACKER) is not OutputStatus.OK:
                raise ValueError("camera motion is reported by a tracker that ran")
            if self.camera_motion.since_t >= self.t:
                raise ValueError("camera motion must be measured from an earlier frame")
        if self.lane_relations is not None:
            ids = {o.track_id for o in self.tracks or ()}
            if any(r.track_id not in ids for r in self.lane_relations):
                raise ValueError("lane relation for a track not in the frame")
        return self


# --- Configuration, provenance and results --------------------------------------------------------


class ComponentInfo(_Strict):
    """Identity of one component: everything that changes its output must be in here."""
    role: Role
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    checkpoint_sha256: Optional[str] = Field(default=None, pattern=_SHA256)
    params: dict[str, Any] = Field(default_factory=dict)
    interval_s: Optional[float] = Field(default=None, gt=0,
                                        description="minimum time between runs; None = every analysed frame")

    @field_validator("params")
    @classmethod
    def _json_params(cls, v: dict[str, Any]) -> dict[str, Any]:
        canonical_json(v)  # rejects values without a deterministic JSON form
        return v


class RunConfig(_Strict):
    interval_s: Optional[float] = Field(default=None, gt=0, description="analysed-frame spacing; None = every frame")
    start_s: float = Field(default=0.0, ge=0)
    end_s: Optional[float] = Field(default=None, gt=0)
    max_width: Optional[int] = Field(default=None, ge=16, description="downscale wider frames before analysis")
    assumed_fps: Optional[float] = Field(default=None, gt=0, description="only for streams without timestamps")

    @model_validator(mode="after")
    def _span(self) -> "RunConfig":
        if self.end_s is not None and self.end_s <= self.start_s:
            raise ValueError("end_s must be after start_s")
        return self


class Provenance(_Strict):
    """Everything that produced the perception records: engine, decoder and perception components."""
    engine_version: str
    decoder: dict[str, str] = Field(description="decoding library versions")
    components: tuple[ComponentInfo, ...] = ()

    @model_validator(mode="after")
    def _perception_only(self) -> "Provenance":
        roles = [c.role for c in self.components]
        if Role.EVENT_ANALYZER in roles or len(roles) != len(set(roles)):
            raise ValueError("provenance lists at most one component per perception role and no analysers")
        return self


class FrameSequence(Sequence[FrameRecord]):
    """Read-only frame records that are not all held in memory, e.g. streamed from a cache entry.

    Implementations must have run ``check_frames`` over their records for ``checked_roles``.
    """
    checked_roles: frozenset[Role] = frozenset()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Sequence) and len(self) == len(other) and all(a == b for a, b in zip(self, other))

    __hash__ = None  # type: ignore[assignment]


def check_frames(frames: Iterable[FrameRecord], configured: set[Role]) -> int:
    """Single pass over frame records: indexes 0..n-1, strictly increasing times, configured roles only."""
    n, last = 0, None
    for f in frames:
        if f.index != n:
            raise ValueError("frame indexes must be 0..n-1 in order")
        if last is not None and f.t <= last:
            raise ValueError("frame times must be strictly increasing")
        if set(f.status) != configured:
            raise ValueError(f"frame {f.index} status roles differ from the configured components")
        n, last = n + 1, f.t
    return n


_FRAME_TUPLE = TypeAdapter(tuple[FrameRecord, ...])


def _frames_in(v: Any) -> Sequence[FrameRecord]:
    return v if isinstance(v, FrameSequence) else _FRAME_TUPLE.validate_python(v)


class WarningCode(str, Enum):
    ENDED_EARLY = "ended_early"  # decoding stopped before the stream end
    DECODE_ERRORS = "decode_errors"  # packets that could not be decoded were skipped
    TIMESTAMPS_DROPPED = "timestamps_dropped"  # frames dropped for missing or out-of-order timestamps
    FRAME_GAP = "frame_gap"  # an unusually long stretch without frames
    ASSUMED_RATE = "assumed_rate"  # no container timestamps: times assume a frame rate


class RunWarning(_Strict):
    code: WarningCode
    message: str = Field(min_length=1, description="for people; the code is what software reads")
    t: Optional[float] = Field(default=None, ge=0, description="where it applies, when it applies at one time")


class PerceptionResult(_Strict):
    """Per-frame perception for one completed run; what the perception cache stores."""
    schema_version: str = SCHEMA_VERSION
    video: VideoInfo
    config: RunConfig
    provenance: Provenance
    timing: TimingReport
    frames: Annotated[Sequence[FrameRecord], PlainValidator(_frames_in)] = Field(
        description="a tuple, or a FrameSequence read lazily from a cache entry")
    warnings: tuple[RunWarning, ...] = ()

    @field_serializer("frames")
    def _frames_out(self, frames: Sequence[FrameRecord], info) -> list:
        return [f.model_dump(mode=info.mode) for f in frames]

    @model_validator(mode="after")
    def _consistent(self) -> "PerceptionResult":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version!r}")
        if self.timing.frames_analysed != len(self.frames):
            raise ValueError("timing.frames_analysed must equal the number of frame records")
        configured = {c.role for c in self.provenance.components}
        if not (isinstance(self.frames, FrameSequence) and self.frames.checked_roles == configured):
            check_frames(self.frames, configured)
        return self


# --- Events and evidence --------------------------------------------------------------------------


class EventType(str, Enum):
    RAPID_APPROACH = "rapid_approach"
    CUT_IN = "cut_in"
    IMPACT = "impact"


class EventSubtype(str, Enum):
    """Impact proposals only."""
    CAMERA_VEHICLE = "camera_vehicle"  # involving the camera's own vehicle
    ROAD_USERS = "road_users"  # between other road users


class EvidenceState(str, Enum):
    SUPPORTED = "supported"  # every condition the analyser checks held on observed data; not a verdict
    UNCERTAIN = "uncertain"  # the pattern was seen but a condition could not be checked; see reasons


class Basis(str, Enum):
    OBSERVED = "observed"  # read from the perception records, e.g. the track was in the camera's lane at t
    ESTIMATED = "estimated"  # computed under a stated model, e.g. apparent time-to-contact
    CONTEXT = "context"  # shown to reviewers, never decisive, e.g. box size


class ReasonCode(str, Enum):
    """Why a proposal is uncertain, or why a pattern an analyser saw was not proposed."""
    # uncertain proposals
    LANE_NOT_ESTABLISHED = "lane_not_established"  # no observed lane relation for most of the pattern
    BOUNDARIES_REIDENTIFIED = "boundaries_reidentified"  # the camera's lane boundaries changed identity
    ENTRY_NOT_OBSERVED = "entry_not_observed"  # the transition itself fell in an unobserved stretch
    IDENTITY_JUMP = "identity_jump"  # the box size stepped as if the track switched objects
    NOT_VALIDATED = "not_validated"  # the analyser is experimental and has not been validated
    # patterns not proposed
    OTHER_LANE = "other_lane"  # observed in another lane for most of the pattern
    PASSES_BESIDE = "passes_beside"  # its fitted path passes beside the camera
    EGO_LANE_CHANGE = "ego_lane_change"  # the camera completed a lane change: every relation shifted with it
    NOT_CLOSE = "not_close"  # the approach ended far from the camera
    NO_JOLT = "no_jolt"  # reliable camera-motion estimates show no jolt
    MOTION_ESTIMATE_MISSING = "motion_estimate_missing"  # camera motion missing or unreliable when it mattered
    MOTION_BASELINE_INSUFFICIENT = "motion_baseline_insufficient"  # too few earlier estimates to judge a jolt
    MERGED = "merged"  # overlapped another proposal of the same analyser and was merged into it


class EvidenceItem(_Strict):
    """One piece of evidence for a proposal: a pointer into the analysed data or a value derived from it,
    never a copy of pixels."""
    basis: Basis
    name: str = Field(min_length=1, description="what this is, e.g. 'apparent_tau_s', 'lane_relation'")
    t: float = Field(ge=0)
    t_end: Optional[float] = Field(default=None, ge=0, description="given for an interval")
    track_id: Optional[int] = Field(default=None, ge=0)
    value: Optional[float | str] = Field(default=None, description="a number, or an enum value such as "
                                                                    "'left_adjacent'")
    unit: Optional[str] = Field(default=None, description="e.g. 's', 'lane_widths', 'fraction_of_height', 'z'")
    note: str = Field(default="", description="assumption or method, one sentence")

    @model_validator(mode="after")
    def _shape(self) -> "EvidenceItem":
        if self.t_end is not None and self.t_end < self.t:
            raise ValueError("evidence t_end precedes t")
        if self.unit is not None and not isinstance(self.value, float):
            raise ValueError("a unit needs a numeric value")
        return self


class EventDraft(_Strict):
    """A proposal as an analyser returns it, before the run binds it to the analysis (``bind``)."""
    type: EventType
    subtype: Optional[EventSubtype] = None
    state: EvidenceState
    experimental: bool = Field(default=False, description="from an analyser that is off by default")
    t_start: float = Field(ge=0)
    t_peak: float = Field(ge=0)
    t_end: float = Field(ge=0)
    track_ids: tuple[int, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    reasons: tuple[ReasonCode, ...] = Field(default=(), description="why the proposal is uncertain")
    rank: Optional[float] = Field(default=None, ge=0, le=1, description="orders proposals; not a probability")
    analyzer: str = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> "EventDraft":
        if not self.t_start <= self.t_peak <= self.t_end:
            raise ValueError("a proposal must satisfy t_start <= t_peak <= t_end")
        if (self.type is EventType.IMPACT) != (self.subtype is not None):
            raise ValueError("a subtype is required for, and only for, impact proposals")
        if (self.state is EvidenceState.UNCERTAIN) != bool(self.reasons):
            raise ValueError("an uncertain proposal must give reasons, and only an uncertain one")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("duplicate reason")
        if list(self.track_ids) != sorted(set(self.track_ids)):
            raise ValueError("track_ids must be sorted and unique")
        return self

    def bind(self, analysis_key: str) -> "EventProposal":
        return EventProposal(event_id=event_id(analysis_key, self), **dict(self))


def event_id(analysis_key: str, draft: EventDraft) -> str:
    """Stable within one analysis: the same records and analysers give the same identifiers."""
    return digest([analysis_key, draft.analyzer, draft.type.value, draft.subtype and draft.subtype.value,
                   list(draft.track_ids), round(draft.t_start, 2)])[:16]


class EventProposal(EventDraft):
    """A machine-proposed moment awaiting human review. Not a finding: ``supported`` says only that the
    analyser's own conditions held, and only a reviewer confirms anything."""
    event_id: str = Field(pattern=r"^[0-9a-f]{16}$")


MISSING_SIGNAL = frozenset({ReasonCode.MOTION_ESTIMATE_MISSING, ReasonCode.MOTION_BASELINE_INSUFFICIENT})


class UnresolvedPattern(_Strict):
    """A pattern an analyser could not judge because a signal it needs was missing or unreliable. It is not
    proposed, because a missing signal is not evidence, and it is not evidence that nothing happened."""
    reason: ReasonCode
    t_start: float = Field(ge=0)
    t_end: float = Field(ge=0)
    track_ids: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _shape(self) -> "UnresolvedPattern":
        if self.reason not in MISSING_SIGNAL:
            raise ValueError(f"{self.reason.value} is not a missing-signal reason")
        if self.t_end < self.t_start:
            raise ValueError("interval must satisfy t_start <= t_end")
        return self


class AnalyzerStatus(str, Enum):
    OK = "ok"
    UNAVAILABLE = "unavailable"  # its inputs were not in the records, e.g. no lane relations
    FAILED = "failed"  # it raised a recoverable error


class AnalyzerReport(_Strict):
    """What one configured analyser did. ``considered`` patterns were either proposed or rejected for a
    reason (merged proposals count as rejected), so ``considered = proposals + sum(rejected)``. Run times
    are on the run outcome, so the same records and analysers always give the same result."""
    analyzer: ComponentInfo
    status: AnalyzerStatus
    reason: Optional[str] = Field(default=None, description="why it was unavailable or failed")
    considered: int = Field(default=0, ge=0)
    proposals: int = Field(default=0, ge=0)
    rejected: dict[ReasonCode, int] = Field(default_factory=dict)
    unresolved: tuple[UnresolvedPattern, ...] = Field(default=(), description="every pattern rejected for a "
                                                                              "missing signal, with its time")

    @model_validator(mode="after")
    def _consistent(self) -> "AnalyzerReport":
        if self.analyzer.role is not Role.EVENT_ANALYZER:
            raise ValueError("reports are for event analysers")
        if (self.status is AnalyzerStatus.OK) == (self.reason is not None):
            raise ValueError("a reason is required for, and only for, an analyser that did not complete")
        if self.status is not AnalyzerStatus.OK and (self.considered or self.proposals or self.rejected):
            raise ValueError("an analyser that did not complete reports no counts")
        if any(n <= 0 for n in self.rejected.values()):
            raise ValueError("rejected counts must be positive")
        if self.considered != self.proposals + sum(self.rejected.values()):
            raise ValueError("considered must equal proposals plus rejected")
        for reason in MISSING_SIGNAL:
            if sum(u.reason is reason for u in self.unresolved) != self.rejected.get(reason, 0):
                raise ValueError(f"every {reason.value} rejection must be listed as unresolved")
        return self


class Verdict(str, Enum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNSURE = "unsure"


class ReviewDecision(_Strict):
    """A person's decision about one proposal. Stored with project data, never in the cache. The proposal's
    type and interval are copied in, so a decision can be matched again after re-analysis changes ids."""
    schema_version: str = SCHEMA_VERSION
    analysis_key: str = Field(pattern=_SHA256)
    event_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    type: EventType
    subtype: Optional[EventSubtype] = None
    t_start: float = Field(ge=0, description="the proposal's interval")
    t_end: float = Field(ge=0)
    verdict: Verdict
    reviewer: str = Field(min_length=1)
    decided_at: str = Field(min_length=1, description="ISO-8601 timestamp")
    corrected_t_start: Optional[float] = Field(default=None, ge=0, description="reviewer-corrected interval")
    corrected_t_end: Optional[float] = Field(default=None, ge=0)
    note: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> "ReviewDecision":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version!r}")
        if self.t_end < self.t_start:
            raise ValueError("interval must satisfy t_start <= t_end")
        if (self.corrected_t_start is not None and self.corrected_t_end is not None
                and self.corrected_t_end < self.corrected_t_start):
            raise ValueError("corrected interval must satisfy t_start <= t_end")
        return self

    @classmethod
    def of(cls, proposal: EventProposal, analysis_key: str, **decision) -> "ReviewDecision":
        return cls(analysis_key=analysis_key, event_id=proposal.event_id, type=proposal.type,
                   subtype=proposal.subtype, t_start=proposal.t_start, t_end=proposal.t_end, **decision)


class AnalysisResult(_Strict):
    """Perception plus event proposals and one report per configured analyser, from one completed run.
    Cancelled, failed and resource-stopped runs produce none."""
    schema_version: str = SCHEMA_VERSION
    analysis_key: str = Field(pattern=_SHA256, description="analysis_key(perception_key, analysers); review "
                                                          "decisions refer to it")
    perception_key: str = Field(pattern=_SHA256)
    perception: PerceptionResult
    reports: tuple[AnalyzerReport, ...] = ()
    events: tuple[EventProposal, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> "AnalysisResult":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version!r}")
        names = [r.analyzer.name for r in self.reports]
        if len(names) != len(set(names)):
            raise ValueError("reports must name each analyser once")
        if self.analysis_key != analysis_key(self.perception_key, [r.analyzer for r in self.reports]):
            raise ValueError("analysis_key does not match the perception key and analysers")
        ids = [e.event_id for e in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate event_id")
        reports = {r.analyzer.name: r for r in self.reports}
        frames = self.perception.frames
        counts: dict[str, int] = {}
        for e in self.events:
            r = reports.get(e.analyzer)
            if r is None or r.status is not AnalyzerStatus.OK:
                raise ValueError(f"event {e.event_id} names an analyser without a completed report")
            if e.event_id != event_id(self.analysis_key, e):
                raise ValueError(f"event {e.event_id} is not bound to this analysis")
            if not frames or not (frames[0].t <= e.t_start and e.t_end <= frames[-1].t):
                raise ValueError(f"event {e.event_id} lies outside the analysed span")
            counts[e.analyzer] = counts.get(e.analyzer, 0) + 1
        for r in self.reports:
            if counts.get(r.analyzer.name, 0) != r.proposals:
                raise ValueError(f"report {r.analyzer.name} counts {r.proposals} proposals, the result has "
                                 f"{counts.get(r.analyzer.name, 0)}")
        return self


def analysis_key(perception_key: str, analyzers: Iterable[ComponentInfo]) -> str:
    """Identifies one set of proposals: the perception records plus every analyser's identity."""
    return digest({"perception": perception_key, "analyzers": [a.model_dump(mode="json") for a in analyzers]})
