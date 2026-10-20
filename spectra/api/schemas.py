"""HTTP response models owned by the API. Engine and evidence contracts are returned unchanged."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from spectra.evidence.contracts import Gap, Overlay
from spectra.evidence.media import ClipFile


class _Response(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelFile(_Response):
    role: Literal["detector", "segmenter"]
    file: str
    sha256: str


class ModelRegistryHealth(_Response):
    available: bool
    message: Optional[str] = None
    models: list[ModelFile] = []


class MemoryHealth(_Response):
    pressure_level: Optional[int]
    state: Literal["normal", "warning", "critical", "unavailable"]


class DatabaseHealth(_Response):
    available: bool
    schema_version: Optional[int] = None
    message: Optional[str] = None


class AnalysisReadiness(_Response):
    available: bool
    reasons: list[str]


class Health(_Response):
    status: Literal["ok", "degraded"]
    version: str
    database: DatabaseHealth
    models: ModelRegistryHealth
    memory: MemoryHealth
    analysis: AnalysisReadiness
    library_folders: int


class LibraryFolderInfo(_Response):
    id: str
    name: str
    path: str
    source: Literal["startup", "added"]
    available: bool
    registered_videos: int


class FolderSelection(_Response):
    """The outcome of the folder dialog: no folder when it was cancelled."""
    folder: Optional[LibraryFolderInfo]
    created: bool


class LibraryEntryInfo(_Response):
    name: str
    path: str = Field(description="relative to the library folder; use it to browse or register")
    kind: Literal["directory", "video"]
    size_bytes: Optional[int] = None
    modified_at: Optional[str] = None
    video_id: Optional[str] = Field(default=None, description="set when this file is already registered")


class LibraryListing(_Response):
    folder: LibraryFolderInfo
    path: str
    parent: Optional[str] = Field(description="path of the enclosing folder; null at the library root")
    entries: list[LibraryEntryInfo]
    truncated: bool


class VideoReference(BaseModel):
    """A file chosen inside a configured library folder."""
    model_config = ConfigDict(extra="forbid")
    folder_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    path: str = Field(min_length=1, max_length=4096)


class ProbeStatus(_Response):
    state: Literal["pending", "ready", "failed"]
    error: Optional[str] = None
    message: Optional[str] = None
    probed_at: Optional[str] = None


class MediaInfo(_Response):
    sha256: str
    container: str
    codec: str
    audio_codecs: list[str]
    width: int
    height: int
    rotation: int
    duration_s: Optional[float]
    nominal_fps: Optional[float] = Field(description="container-declared; times use presentation timestamps")
    start_s: float
    timestamp_source: str


class SourceState(_Response):
    status: Literal["available", "missing", "changed", "outside_library"]
    message: Optional[str] = None


class BrowserCheck(_Response):
    playable: bool
    checked_at: str
    user_agent: Optional[str] = None
    detail: Optional[str] = None


class PlaybackInfo(_Response):
    assessment: Literal["likely", "uncertain", "unsupported", "unknown"]
    reasons: list[str]
    mime_type: Optional[str] = None
    browser_check: Optional[BrowserCheck] = Field(
        default=None, description="latest in-browser load result for the current file contents")


class Eligibility(_Response):
    eligible: bool
    reasons: list[str]


class Video(_Response):
    id: str
    file_name: str
    folder_id: Optional[str] = None
    path: Optional[str] = Field(default=None, description="relative to its library folder")
    size_bytes: int
    modified_at: str
    registered_at: str
    probe: ProbeStatus
    media: Optional[MediaInfo] = None
    source: SourceState
    playback: PlaybackInfo
    analysis: Eligibility
    media_url: Optional[str] = Field(default=None, description="set when the file may be loaded for playback")


class PlaybackReport(BaseModel):
    """The application's in-browser load result for a registered video."""
    model_config = ConfigDict(extra="forbid")
    playable: bool
    detail: Optional[str] = Field(default=None, max_length=500)
    sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$",
                                  description="contents the browser loaded (media.sha256); refused when the "
                                              "file has changed since")


AnalysisStateName = Literal["queued", "running", "completed", "incomplete", "cancelled", "resource_stopped",
                            "failed"]


class AnalysisRequest(BaseModel):
    """Analyse a registered video with the application's fixed analysis profile."""
    model_config = ConfigDict(extra="forbid")
    video_id: str = Field(pattern=r"^[0-9a-f]{32}$")


class AnalysisProgress(_Response):
    stage: Optional[str] = None
    frames_analysed: int = 0
    media_time_s: Optional[float] = Field(default=None, description="source time of the latest analysed frame")
    fraction: Optional[float] = Field(default=None, description="null while the analysed span is unknown")


class AnalysisResultSummary(_Response):
    cache_hit: bool
    perception_key: Optional[str]
    analysis_key: Optional[str]
    proposals: int = Field(description="event proposals for human review, not confirmed incidents")
    notes: list[str]


class EvidenceStatus(_Response):
    state: Literal["not_applicable", "pending", "ready", "failed"] = Field(
        description="not_applicable without a result; failed evidence is rebuilt when the timeline is requested")
    build_key: Optional[str] = None
    message: Optional[str] = None


class Analysis(_Response):
    id: str
    video_id: str
    file_name: str
    source_sha256: str
    state: AnalysisStateName
    revision: int = Field(description="increases with every change; the SSE event id")
    queue_position: Optional[int] = Field(default=None, description="1 for the next queued analysis")
    cancel_requested: bool
    progress: AnalysisProgress
    result: Optional[AnalysisResultSummary] = Field(
        default=None, description="present for completed and incomplete analyses")
    evidence: EvidenceStatus
    stop_reason: Optional[str] = Field(
        default=None, description="cancelled, interrupted, shutdown, video, models, resources or error")
    message: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


ReviewDecisionName = Literal["not_reviewed", "accepted", "rejected"]


class ReviewDecision(_Response):
    """A person's decision on one event proposal. It is not the analyser's status, and accepting a proposal
    does not confirm that an incident occurred or determine fault."""
    event_id: str
    decision: ReviewDecisionName
    revision: int = Field(description="increases with every change; 0 when no decision was ever recorded")
    updated_at: Optional[str] = None


class ReviewDecisions(_Response):
    analysis_id: str
    decisions: list[ReviewDecision] = Field(
        description="proposals with a recorded decision; every other proposal is not reviewed")


class ReviewDecisionUpdate(BaseModel):
    """Set a proposal's decision. ``expected_revision`` is the revision the change is based on (0 when none
    was recorded); a change based on an older revision is refused with 409."""
    model_config = ConfigDict(extra="forbid")
    decision: ReviewDecisionName
    expected_revision: int = Field(ge=0)


DESCRIPTION_MAX = 2000


class EventEdit(_Response):
    """A person's corrections to one proposal. Times are analysis times in seconds; null times mean the proposed
    times apply. The proposal itself, its evidence and the analyser's status are unchanged."""
    event_id: str
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    description: Optional[str] = Field(default=None, description="plain text written by the reviewer")
    revision: int = Field(default=0, description="increases with every change; 0 when never edited")
    updated_at: Optional[str] = None


class EventEdits(_Response):
    analysis_id: str
    edits: list[EventEdit] = Field(description="proposals with a recorded edit; every other proposal is unedited")


class EventEditUpdate(BaseModel):
    """Replace a proposal's edit. Give both times or neither (keep the proposed times); an empty or null
    description means none. Refused with 409 when it changed since ``expected_revision``."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    t_start: Optional[float]
    t_end: Optional[float]
    description: Optional[str] = Field(max_length=DESCRIPTION_MAX, pattern=r"^[^\x00]*$")
    expected_revision: int = Field(ge=0)


class OverlayLayers(_Response):
    tracks: bool = Field(description="the tracker ran in this analysis; false means the layer does not exist")
    lane: bool = Field(description="ego-lane estimation ran in this analysis")


class OverlayTiming(_Response):
    """How to place overlay frames on the player's timeline."""
    stream_start_s: float = Field(description="container time of t = 0 (VideoInfo.start_s); add it to t for "
                                               "container-level positions")
    max_age_s: float = Field(description="longest time an analysed frame's records are shown")
    analysed_start_s: float
    analysed_end_s: float
    timestamp_source: str


class OverlayResponse(_Response):
    """Recorded observations for a span. Draw ``frames[k]`` while t is in [frames[k].t, frames[k].valid_until];
    at any other time show 'not analysed here'. Positions are never interpolated."""
    overlay: "Overlay"
    layers: OverlayLayers
    timing: OverlayTiming
    gaps: list["Gap"] = Field(description="stretches inside the span without analysed frames")


class ClipResponse(_Response):
    clip: "ClipFile"
    media_url: str
    mime_type: str
    browser_playback: Literal["unverified"] = Field(
        default="unverified", description="a valid stream copy may still not play in a browser")
