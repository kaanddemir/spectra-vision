"""Typed contracts for review evidence built from an engine ``AnalysisResult``.

The engine decides what may have happened; this layer only arranges its results for review. Nothing here
re-judges a proposal: the ``EventProposal`` is embedded unchanged, its state and reasons are the engine's,
and ``supported`` still means only that the analyser's own conditions held. Human decisions are
``ReviewDecision`` records kept with project data, never in these artifacts.

Times are seconds of source presentation time, exactly as in the engine contracts (see
``spectra.engine.contracts``): never ``frame_index / fps``. A review interval is a span to show a
reviewer, with context around the proposal; it is not the time of any incident.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spectra.engine.contracts import (AnalyzerStatus, Box, ComponentInfo, EgoLaneState, EventProposal, EventSubtype,
                                      EventType, EvidenceState, LaneBoundary, LaneRelation, ObjectCategory,
                                      ReasonCode, RunWarning, UnresolvedPattern, VideoInfo)

EVIDENCE_VERSION = "1"

_SHA256 = r"^[0-9a-f]{64}$"
_EVENT_ID = r"^[0-9a-f]{16}$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ReviewConfig(_Strict):
    """How review intervals are built. Every field changes the output, so all of them are in the build key."""
    pre_s: float = Field(default=3.0, ge=0, description="context before the earliest evidence")
    post_s: float = Field(default=2.0, ge=0, description="context after the latest evidence")
    max_context_s: float = Field(default=20.0, ge=0, description="cap on pre plus post context; the proposal "
                                                                 "and its evidence are never cut")
    gap_s: float = Field(default=1.0, gt=0, description="a longer stretch without analysed frames is a gap")
    overlay_max_age_s: float = Field(default=0.25, ge=0, description="how long an analysed frame's records may be "
                                                                     "shown after its own time")


class Note(str, Enum):
    """Machine-readable limits of the material in a package, for the interface to explain. None of them
    changes the engine's proposal."""
    EXPERIMENTAL = "experimental"  # from an analyser that is off by default and not validated
    CLIPPED_AT_VIDEO_START = "clipped_at_video_start"  # less context before than configured: the video starts
    CLIPPED_AT_VIDEO_END = "clipped_at_video_end"  # less context after than configured: the video ends
    CONTEXT_CAPPED = "context_capped"  # context shortened to max_context_s
    OUTSIDE_ANALYSIS = "outside_analysis"  # part of the interval was not analysed: no records or overlays there
    FRAME_GAP = "frame_gap"  # the interval contains a stretch without analysed frames
    TRACK_ENDS_EARLY = "track_ends_early"  # a related track's last observation is before the interval end
    TRACK_PREDICTED = "track_predicted"  # some related boxes were extrapolated by the tracker, not observed
    LANE_UNOBSERVED = "lane_unobserved"  # no observed lane relation for a related track in the proposal's span
    EGO_LANE_UNAVAILABLE = "ego_lane_unavailable"  # the camera's lane was not estimated in the proposal's span
    MOTION_MISSING = "motion_missing"  # some tracker frames in the span have no camera-motion estimate
    RUN_WARNING = "run_warning"  # a decoding or timing warning applies to the interval
    UNRESOLVED_NEARBY = "unresolved_nearby"  # an analyser could not judge a pattern overlapping the interval


class ReviewInterval(_Strict):
    """The span to show a reviewer: the proposal and all its evidence, plus context, within the video."""
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    evidence_start_s: float = Field(ge=0, description="earliest of the proposal start and its evidence")
    evidence_end_s: float = Field(ge=0, description="latest of the proposal end and its evidence")

    @model_validator(mode="after")
    def _ordered(self) -> "ReviewInterval":
        if not self.start_s <= self.evidence_start_s <= self.evidence_end_s <= self.end_s:
            raise ValueError("a review interval must contain its evidence span")
        return self


class Gap(_Strict):
    """A stretch without analysed frames: the records say nothing about it."""
    start_s: float = Field(ge=0, description="last analysed frame before the gap")
    end_s: float = Field(ge=0, description="first analysed frame after it")


class TrackSummary(_Strict):
    """What the records hold for one related track inside the review interval; read, not judged."""
    track_id: int = Field(ge=0)
    category: Optional[ObjectCategory] = Field(default=None, description="most frequent; None when never seen")
    first_s: Optional[float] = Field(default=None, ge=0, description="first observed box in the interval")
    last_s: Optional[float] = Field(default=None, ge=0, description="last observed box in the interval")
    observed: int = Field(default=0, ge=0, description="frames with an observed box")
    predicted: int = Field(default=0, ge=0, description="frames with a box extrapolated by the tracker")
    ends_early: bool = Field(default=False, description="not observed after last_s although the tracker ran on "
                                                        "later frames in the interval (predicted boxes are not "
                                                        "observations)")


class RelationRun(_Strict):
    """Consecutive analysed frames with the same lane relation for one track. ``relation`` None: the track
    was in the frame but no relation was computed (no lane evidence), which is unknown, not 'elsewhere'."""
    track_id: int = Field(ge=0)
    relation: Optional[LaneRelation] = None
    beyond_evidence: bool = False
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    frames: int = Field(ge=1)


class LaneStateRun(_Strict):
    """Consecutive ego-lane observations in the same state."""
    state: EgoLaneState
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    frames: int = Field(ge=1)


class LaneSummary(_Strict):
    """Lane records inside the review interval. ``available`` False: no ego-lane estimator was configured."""
    available: bool
    ego_lane: tuple[LaneStateRun, ...] = ()
    relations: tuple[RelationRun, ...] = ()


class RelationKind(str, Enum):
    BASED_ON = "based_on"  # this impact proposal was built on the other's approach episode
    BASIS_OF = "basis_of"  # the other (an impact proposal) was built on this proposal's approach episode
    SHARES_TRACK = "shares_track"  # another proposal about a common track whose span overlaps this one's


class Relation(_Strict):
    """A known link between two proposals of the same analysis. A link is not a merged or confirmed incident:
    both proposals stay independent and each is reviewed on its own."""
    kind: RelationKind
    event_id: str = Field(pattern=_EVENT_ID)


class EvidencePackage(_Strict):
    """Everything a reviewer needs for one proposal, by reference to the source video and the records."""
    evidence_version: str = EVIDENCE_VERSION
    analysis_key: str = Field(pattern=_SHA256)
    perception_key: str = Field(pattern=_SHA256)
    build_key: str = Field(pattern=_SHA256, description="analysis, configuration and evidence version")
    proposal: EventProposal = Field(description="the engine's proposal, unchanged")
    analyzer: ComponentInfo = Field(description="identity of the analyser that proposed it")
    video: VideoInfo = Field(description="the analysed source video; media is resolved against its hash")
    analysed_start_s: float = Field(ge=0)
    analysed_end_s: float = Field(ge=0)
    review: ReviewInterval
    tracks: tuple[TrackSummary, ...] = ()
    lanes: LaneSummary
    gaps: tuple[Gap, ...] = ()
    warnings: tuple[RunWarning, ...] = Field(default=(), description="run warnings at a time in the interval, or "
                                                                     "for the whole run")
    unresolved: tuple[UnresolvedPattern, ...] = Field(default=(), description="patterns any analyser could not "
                                                                              "judge, overlapping the interval")
    related: tuple[Relation, ...] = ()
    notes: tuple[Note, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> "EvidencePackage":
        if self.evidence_version != EVIDENCE_VERSION:
            raise ValueError(f"unsupported evidence_version {self.evidence_version!r}")
        if self.analyzer.name != self.proposal.analyzer:
            raise ValueError("analyser identity does not match the proposal")
        if [t.track_id for t in self.tracks] != list(self.proposal.track_ids):
            raise ValueError("track summaries must cover the proposal's tracks, in order")
        if list(self.notes) != sorted(set(self.notes), key=list(Note).index):
            raise ValueError("notes must be unique and in declaration order")
        return self


class Coverage(str, Enum):
    """What one analyser's outcome means for the timeline. Missing data is never a negative finding."""
    PROPOSED = "proposed"  # it ran and proposed moments
    NO_PROPOSAL = "no_proposal"  # it ran over the analysed span and proposed nothing at its operating point
    UNAVAILABLE = "unavailable"  # its inputs were not in the records: it says nothing either way
    FAILED = "failed"  # it raised an error: it says nothing either way


class AnalyzerCoverage(_Strict):
    name: str
    version: str
    experimental: bool
    status: AnalyzerStatus
    coverage: Coverage
    reason: Optional[str] = Field(default=None, description="why it was unavailable or failed")
    proposals: int = Field(default=0, ge=0)
    considered: int = Field(default=0, ge=0)
    rejected: dict[ReasonCode, int] = Field(default_factory=dict)
    unresolved: tuple[UnresolvedPattern, ...] = Field(default=(), description="patterns it could not judge: "
                                                                              "insufficient evidence, not absence")


class AnalysisState(str, Enum):
    COMPLETE = "complete"  # the run finished and decoded its whole span
    INCOMPLETE = "incomplete"  # the run finished, but decoding stopped early: the tail was not analysed
    CANCELLED = "cancelled"  # no result: the run was cancelled
    RESOURCE_STOPPED = "resource_stopped"  # no result: the memory safeguards stopped it
    FAILED = "failed"  # no result: the video could not be analysed or an error occurred


class TimelineEntry(_Strict):
    event_id: str = Field(pattern=_EVENT_ID)
    type: EventType
    subtype: Optional[EventSubtype] = None
    title_key: str = Field(description="stable key for a translated title, e.g. 'event.impact.camera_vehicle'")
    title: str = Field(description="English fallback title")
    t_start: float = Field(ge=0)
    t_peak: float = Field(ge=0)
    t_end: float = Field(ge=0)
    review: ReviewInterval
    state: EvidenceState
    reasons: tuple[ReasonCode, ...] = ()
    experimental: bool
    analyzer: str
    track_ids: tuple[int, ...] = ()
    rank: Optional[float] = Field(default=None, ge=0, le=1, description="orders proposals; not a probability")
    package: str = Field(description="path of the evidence package, relative to the build directory")
    segment: int = Field(ge=0, description="index of the review segment holding the interval")
    related: tuple[Relation, ...] = ()
    notes: tuple[Note, ...] = ()


class ReviewSegment(_Strict):
    """Overlapping review intervals joined into one span to play; each proposal stays its own entry."""
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    event_ids: tuple[str, ...] = Field(min_length=1)


class Timeline(_Strict):
    """All proposals of one analysis in chronological order, with what each analyser covered. A timeline
    without entries is not a statement that nothing happened: see ``state`` and ``analyzers``."""
    evidence_version: str = EVIDENCE_VERSION
    state: AnalysisState
    message: Optional[str] = Field(default=None, description="for people: why the analysis is not complete")
    analysis_key: Optional[str] = Field(default=None, pattern=_SHA256, description="None without a result")
    perception_key: Optional[str] = Field(default=None, pattern=_SHA256)
    build_key: Optional[str] = Field(default=None, pattern=_SHA256)
    config: ReviewConfig = ReviewConfig()
    video: Optional[VideoInfo] = None
    analysed_start_s: Optional[float] = Field(default=None, ge=0)
    analysed_end_s: Optional[float] = Field(default=None, ge=0)
    analyzers: tuple[AnalyzerCoverage, ...] = ()
    entries: tuple[TimelineEntry, ...] = ()
    segments: tuple[ReviewSegment, ...] = ()
    gaps: tuple[Gap, ...] = Field(default=(), description="every stretch without analysed frames")
    warnings: tuple[RunWarning, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> "Timeline":
        if self.evidence_version != EVIDENCE_VERSION:
            raise ValueError(f"unsupported evidence_version {self.evidence_version!r}")
        has_result = self.state in (AnalysisState.COMPLETE, AnalysisState.INCOMPLETE)
        if has_result != (self.analysis_key is not None):
            raise ValueError("an analysis key is present exactly when the run produced a result")
        if not has_result and (self.entries or self.analyzers):
            raise ValueError("a run without a result has no entries or analyser coverage")
        keys = [entry_order(e) for e in self.entries]
        if keys != sorted(keys) or len({e.event_id for e in self.entries}) != len(self.entries):
            raise ValueError("entries must be unique and in chronological order")
        ids = {e.event_id for e in self.entries}
        if sorted(i for s in self.segments for i in s.event_ids) != sorted(ids):
            raise ValueError("every entry is in exactly one segment")
        if any(r.event_id not in ids for e in self.entries for r in e.related):
            raise ValueError("relations must point at entries of the same timeline")
        return self


def entry_order(e: TimelineEntry) -> tuple:
    """Chronological, then a fixed type order, then the identifier: the same entries always sort alike."""
    return (e.t_start, e.t_peak, list(EventType).index(e.type), e.event_id)


class TrackOverlay(_Strict):
    track_id: int = Field(ge=0)
    box: Box = Field(description="display-oriented source pixels")
    category: ObjectCategory
    score: Optional[float] = Field(default=None, ge=0, le=1)
    predicted: bool = Field(description="extrapolated by the tracker, not observed in this frame")
    related: bool = Field(description="one of the proposal's tracks")
    relation: Optional[LaneRelation] = Field(default=None, description="None: not computed for this frame")


class LaneOverlay(_Strict):
    state: EgoLaneState
    left: Optional[LaneBoundary] = Field(default=None, description="normalised; beyond evidence_y extrapolated")
    right: Optional[LaneBoundary] = None
    observed_t: float = Field(ge=0, description="time of the lane observation shown")


class OverlayFrame(_Strict):
    """Records of one analysed frame, to draw over the source video. Valid from ``t`` until ``valid_until``,
    never interpolated: between analysed frames objects move, and their drawn positions would be invented."""
    t: float = Field(ge=0)
    valid_until: float = Field(ge=0)
    tracks: Optional[tuple[TrackOverlay, ...]] = Field(default=None, description="None: tracker did not run here")
    lane: Optional[LaneOverlay] = Field(default=None, description="None: no ego-lane observation for this frame")


class Overlay(_Strict):
    """Optional drawing data for one review interval; overlays are off unless the interface turns them on."""
    event_id: str = Field(pattern=_EVENT_ID)
    width: int = Field(gt=0, description="display width the boxes refer to")
    height: int = Field(gt=0)
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    frames: tuple[OverlayFrame, ...] = ()
