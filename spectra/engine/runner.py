"""Single-process analysis run: validate the video, decode sampled frames, run perception components on
their own schedules, cache the perception records, then run event analysers on them. Every configured analyser gets a report
in the result, including one that was unavailable or failed.

A run moves PENDING -> RUNNING -> COMPLETED | CANCELLED | FAILED exactly once. Only a completed run
carries a result. Cancellation is cooperative: it is checked between frames and between components. A
run cancelled or failed during perception leaves no cache entry; perception committed before a later
cancellation stays cached because it is complete and valid. Components are loaded one after another after
the video is validated, never on a cache hit, and are always closed.

Per frame the segmenter runs first (when due), then the ego-lane estimator, then the detector. With a
segmenter that labels the camera's own vehicle, each detection gets its overlap with the most recent such
mask and detections judged to be the hood are kept in the record but withheld from the tracker. With an
ego-lane estimator and a tracker, each frame's tracks are related to the most recent ego lane.

With a cache, frame records are streamed into the entry as they are produced and the result reads them
back lazily, so memory does not grow with the length of the video.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from spectra.engine import ENGINE_VERSION
from spectra.engine.adapters import (AnalyzerOutput, AnalyzerUnavailable, ComponentFailure, Detector, EgoLaneEstimator,
                                     EventAnalyzer, ReportsCameraMotion, Segmentation, Segmenter, Tracker)
from spectra.engine.cache import CacheWriter, PerceptionCache, perception_key
from spectra.engine.contracts import (AnalysisResult, AnalyzerReport, AnalyzerStatus, Box, ComponentInfo, Detection,
                                      EgoLaneObservation, EventDraft, EventProposal, FrameRecord, OutputStatus,
                                      PerceptionResult, Provenance, Role, RunConfig, RunWarning, TimestampSource,
                                      TrackedObject, WarningCode, analysis_key)
from spectra.engine.lanes import mark_ego_vehicle, relate_tracks
from spectra.engine.resources import ResourceExhausted, ResourceGuard
from spectra.engine.video import EPS, GAP_WARNING_S, Frame, VideoError, VideoSource, decoder_versions


class RunState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


TERMINAL = (RunState.COMPLETED, RunState.CANCELLED, RunState.FAILED)


class StopReason(str, Enum):
    """Why a run failed, for software; ``error`` says it for people."""
    VIDEO = "video"  # the video cannot be analysed
    RESOURCES = "resources"  # stopped by the memory safeguards
    ERROR = "error"  # an unexpected error, a worker crash or a timeout


@dataclass(frozen=True)
class Progress:
    state: RunState
    frames_analysed: int = 0
    t: Optional[float] = None  # media time of the latest analysed frame
    fraction: Optional[float] = None  # non-decreasing in [0, 1]; None while the span length is unknown
    stage: str = ""


@dataclass(frozen=True)
class RunOutcome:
    state: RunState
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
    cache_hit: bool = False
    notes: tuple[str, ...] = ()
    analyzer_seconds: tuple[tuple[str, float], ...] = ()  # wall time per analyser, kept out of the result
    stop: Optional[StopReason] = None  # set for, and only for, a failed run

    def __post_init__(self) -> None:
        if (self.state is RunState.COMPLETED) != (self.result is not None):
            raise ValueError("only a completed run has a result")
        if (self.state is RunState.FAILED) != (self.stop is not None):
            raise ValueError("a stop reason is required for, and only for, a failed run")
        if self.state not in TERMINAL:
            raise ValueError("an outcome needs a terminal state")


class CancelToken:
    """Cooperative cancel flag; pass a ``multiprocessing`` event to share it with a worker process."""

    def __init__(self, event=None) -> None:
        self._event = event if event is not None else threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


class Cancelled(Exception):
    pass


class _Schedule:
    """Grid schedule per component: due on the first analysed frame at or after each start + k * interval."""

    def __init__(self, start_s: float) -> None:
        self.start, self.next = start_s, {}

    def due(self, key: str, interval: Optional[float], t: float) -> bool:
        if interval is None:
            return True
        if t + EPS < self.next.get(key, -math.inf):
            return False
        self.next[key] = self.start + (math.floor((t - self.start) / interval + EPS) + 1) * interval
        return True


class AnalysisRun:
    def __init__(self, video: Path | str, config: RunConfig = RunConfig(), *, detector: Optional[Detector] = None,
                 tracker: Optional[Tracker] = None, segmenter: Optional[Segmenter] = None,
                 ego_lane: Optional[EgoLaneEstimator] = None, analyzers: Sequence[EventAnalyzer] = (),
                 cache: Optional[PerceptionCache] = None, on_progress: Optional[Callable[[Progress], None]] = None,
                 cancel: Optional[CancelToken] = None, guard: Optional[ResourceGuard] = None,
                 max_consecutive_failures: int = 10, ego_mask_max_age_s: float = 1.0,
                 ego_vehicle_threshold: float = 0.5, lane_max_age_s: float = 0.5,
                 cache_only: bool = False) -> None:
        if tracker is not None and detector is None:
            raise ValueError("a tracker needs a detector")
        if ego_lane is not None and segmenter is None:
            raise ValueError("an ego-lane estimator needs a segmenter")
        if segmenter is not None and cache is None:
            raise ValueError("segmentation masks are stored in the cache; pass a cache")
        expected = ((detector, Role.DETECTOR), (tracker, Role.TRACKER), (segmenter, Role.SEGMENTER),
                    (ego_lane, Role.EGO_LANE), *((a, Role.EVENT_ANALYZER) for a in analyzers))
        for component, role in expected:
            if component is not None and component.info.role is not role:
                raise ValueError(f"{component.info.name} declares role {component.info.role.value}, "
                                 f"expected {role.value}")
        self.video, self.config = Path(video), config
        self.detector, self.tracker, self.segmenter, self.ego_lane = detector, tracker, segmenter, ego_lane
        self.analyzers = tuple(analyzers)
        self.cache, self.on_progress = cache, on_progress
        self.cache_only = cache_only
        self.cancel = cancel or CancelToken()
        self.guard = guard or ResourceGuard()
        self.max_consecutive_failures = max_consecutive_failures
        self.ego_mask_max_age_s, self.ego_vehicle_threshold = ego_mask_max_age_s, ego_vehicle_threshold
        self.lane_max_age_s = lane_max_age_s
        self._mask: Optional[tuple[float, np.ndarray, tuple[str, ...]]] = None  # latest segmentation
        self._lane: Optional[tuple[float, EgoLaneObservation]] = None  # latest ego-lane observation
        self.state = RunState.PENDING
        self._fraction, self._frames, self._t = 0.0, 0, None
        self._loaded: list = []
        self._failures: dict[Role, int] = {}
        self._notes: list[str] = []
        self._seconds: list[tuple[str, float]] = []

    # --- lifecycle --------------------------------------------------------------------------------

    def run(self) -> RunOutcome:
        if self.state is not RunState.PENDING:
            raise RuntimeError("a run can only be started once")
        outcome = None
        try:
            self._set(RunState.RUNNING, stage="validating video")
            self.guard.check(force=True)
            source = VideoSource(self.video, assumed_fps=self.config.assumed_fps)
            if self.config.assumed_fps and source.info.timestamp_source is TimestampSource.CONTAINER:
                self._notes.append("assumed_fps ignored: the container has frame timestamps")
            perception_components = [c for c in (self.detector, self.tracker, self.segmenter, self.ego_lane) if c]
            provenance = Provenance(engine_version=ENGINE_VERSION, decoder=decoder_versions(),
                                    components=tuple(c.info for c in perception_components))
            key = perception_key(source.info, self.config, provenance)
            perception, hit = None, False
            if self.cache is not None:
                found = self.cache.lookup(key)
                if found.reason:
                    action = " (cache-only)" if self.cache_only else " and recomputed"
                    self._notes.append(f"cache entry rejected{action}: {found.reason}")
                perception, hit = found.result, found.result is not None
            if perception is None:
                if self.cache_only:
                    raise RuntimeError("cache-only run requires a valid perception cache hit")
                perception = self._perceive(source, provenance, key, perception_components)
            reports, drafts = self._analyze(perception)
            akey = analysis_key(key, [a.info for a in self.analyzers])
            events = sorted((d.bind(akey) for d in drafts), key=lambda e: (e.t_start, e.event_id))
            result = AnalysisResult(analysis_key=akey, perception_key=key, perception=perception, reports=reports,
                                    events=tuple(events))
            self._fraction = 1.0
            outcome = RunOutcome(RunState.COMPLETED, result, cache_hit=hit, notes=self._all_notes(),
                                 analyzer_seconds=tuple(self._seconds))
        except Cancelled:
            outcome = RunOutcome(RunState.CANCELLED, notes=self._all_notes())
        except VideoError as exc:
            outcome = RunOutcome(RunState.FAILED, error=str(exc), notes=self._all_notes(), stop=StopReason.VIDEO)
        except ResourceExhausted as exc:
            outcome = RunOutcome(RunState.FAILED, error=str(exc), notes=self._all_notes(), stop=StopReason.RESOURCES)
        except Exception as exc:  # unexpected: reported as a failed run, never as a result
            outcome = RunOutcome(RunState.FAILED, error=f"{type(exc).__name__}: {exc}", notes=self._all_notes(),
                                 stop=StopReason.ERROR)
        finally:
            close_errors = self._close_all()
        if close_errors:
            outcome = RunOutcome(outcome.state, outcome.result, outcome.error, outcome.cache_hit,
                                 outcome.notes + tuple(close_errors), outcome.analyzer_seconds, outcome.stop)
        try:
            self._set(outcome.state, stage="done")
        except Exception:  # the outcome is decided; an observer failing now cannot change it
            pass
        return outcome

    def _set(self, state: RunState, stage: str = "", frames: Optional[int] = None, t: Optional[float] = None) -> None:
        self.state = state
        if frames is not None:
            self._frames, self._t = frames, t if t is not None else self._t
        if self.on_progress is not None:
            self.on_progress(Progress(state, self._frames, self._t, self._fraction, stage))

    def _all_notes(self) -> tuple[str, ...]:
        return tuple(self._notes) + tuple(self.guard.warnings)

    def _check_cancel(self) -> None:
        if self.cancel.cancelled:
            raise Cancelled

    def _load(self, component) -> None:
        self._check_cancel()
        self._loaded.append(component)  # closed even if load fails half-way
        component.load()

    def _close_all(self) -> list[str]:
        errors = []
        while self._loaded:
            component = self._loaded.pop()
            try:
                component.close()
            except Exception as exc:
                errors.append(f"closing {component.info.name} failed: {type(exc).__name__}: {exc}")
        return errors

    # --- perception -------------------------------------------------------------------------------

    def _perceive(self, source: VideoSource, provenance: Provenance, key: str, components: list) -> PerceptionResult:
        for component in components:
            self._set(RunState.RUNNING, stage=f"loading {component.info.name}")
            self._load(component)
        cfg = self.config
        info = source.info
        end = cfg.end_s if cfg.end_s is not None else info.duration_s
        span = end - cfg.start_s if end is not None and end > cfg.start_s else None
        writer = self.cache.begin(key) if self.cache is not None else None
        schedule, records, count = _Schedule(cfg.start_s), [], 0
        frames = source.frames(cfg.interval_s, cfg.start_s, cfg.end_s, cfg.max_width)
        try:
            for frame in frames:
                self._check_cancel()
                self.guard.check()
                record = self._frame(count, frame, schedule, writer, info.width, info.height)
                if writer is not None:
                    writer.append_frame(record)
                else:
                    records.append(record)
                count += 1
                if span:
                    self._fraction = max(self._fraction, min(1.0, (frame.t - cfg.start_s) / span))
                self._set(RunState.RUNNING, "analysing frames", count, frame.t)
            frames.close()
            if not count:
                raise VideoError("no frames in the requested time span")
            warnings = []
            stats = source.stats
            if stats.ended_early:
                warnings.append(RunWarning(code=WarningCode.ENDED_EARLY,
                                           message=f"video ended early: {stats.ended_early}"))
            if stats.decode_errors:
                warnings.append(RunWarning(code=WarningCode.DECODE_ERRORS, message=f"{stats.decode_errors} packets "
                                           "could not be decoded and were skipped"))
            if stats.missing_timestamps or stats.non_monotonic:
                warnings.append(RunWarning(code=WarningCode.TIMESTAMPS_DROPPED,
                                           message=f"{stats.missing_timestamps + stats.non_monotonic} frames dropped "
                                                   "for missing or out-of-order timestamps"))
            timing = stats.report(count)
            gap = timing.max_interval_s
            if gap is not None and gap > max(GAP_WARNING_S, 10 * timing.median_interval_s):
                warnings.append(RunWarning(code=WarningCode.FRAME_GAP, t=timing.max_interval_start,
                                           message=f"no frames for {gap:.2f} s after {timing.max_interval_start:.3f} s "
                                                   "(variable frame rate or damaged data)"))
            if info.timestamp_source is TimestampSource.ASSUMED_RATE:
                warnings.append(RunWarning(code=WarningCode.ASSUMED_RATE, message="no container timestamps: times "
                                           f"assume {info.assumed_fps} frames per second"))
            result = PerceptionResult(video=info, config=cfg, provenance=provenance, timing=timing,
                                      frames=writer.staged_frames() if writer is not None else tuple(records),
                                      warnings=tuple(warnings))
            if writer is None:
                return result
            writer.commit(result)
            stored = self.cache.lookup(key)
            if stored.result is None:
                raise RuntimeError(f"cache entry unreadable straight after writing: {stored.reason}")
            return stored.result
        except BaseException:
            if writer is not None:
                writer.abort()
            raise
        finally:
            frames.close()

    def _frame(self, index: int, frame: Frame, schedule: _Schedule, writer: Optional[CacheWriter],
               width: int, height: int) -> FrameRecord:
        status: dict[Role, OutputStatus] = {}
        errors: dict[Role, str] = {}
        payload: dict = {}

        def attempt(role: Role, call):
            try:
                value = call()
            except ComponentFailure as exc:
                status[role], errors[role] = OutputStatus.FAILED, str(exc) or "failed"
                self._count_failure(role, errors[role])
                return None
            status[role] = OutputStatus.OK
            self._failures[role] = 0
            return value

        seg = None
        if self.segmenter is not None:
            if schedule.due("segmenter", self.segmenter.info.interval_s, frame.t):
                def segment():
                    out = self.segmenter.segment(frame)
                    if not isinstance(out, Segmentation):
                        raise ComponentFailure(f"invalid output: expected Segmentation, got {type(out).__name__}")
                    try:
                        return out, writer.write_mask(index, out.mask, out.classes)
                    except ValueError as exc:
                        raise ComponentFailure(f"invalid output: {exc}") from exc
                got = attempt(Role.SEGMENTER, segment)
                if got is not None:
                    seg, payload["segmentation"] = got
                    self._mask = (frame.t, seg.mask, seg.classes)
            else:
                status[Role.SEGMENTER] = OutputStatus.NOT_SCHEDULED
        if self.ego_lane is not None:
            upstream = status[Role.SEGMENTER]
            if upstream is OutputStatus.OK:
                def estimate():
                    out = self.ego_lane.estimate(frame, seg)
                    if not isinstance(out, EgoLaneObservation):
                        raise ComponentFailure(f"invalid output: expected EgoLaneObservation, got {type(out).__name__}")
                    return out
                lane = attempt(Role.EGO_LANE, estimate)
                if lane is not None:
                    payload["ego_lane"] = lane
                    self._lane = (frame.t, lane)
            else:
                status[Role.EGO_LANE] = (OutputStatus.NOT_SCHEDULED if upstream is OutputStatus.NOT_SCHEDULED
                                         else OutputStatus.INPUT_UNAVAILABLE)
        if self.detector is not None:
            if schedule.due("detector", self.detector.info.interval_s, frame.t):
                dets = attempt(Role.DETECTOR, lambda: _detections(self.detector.detect(frame), frame, width, height))
                if dets is not None:
                    payload["detections"] = self._mark_ego(dets, frame.t, width, height)
            else:
                status[Role.DETECTOR] = OutputStatus.NOT_SCHEDULED
        if self.tracker is not None:
            upstream = status[Role.DETECTOR]
            if upstream is OutputStatus.OK:
                visible = tuple(d for d in payload["detections"] if not d.ego_vehicle)
                tracks = attempt(Role.TRACKER, lambda: _tracks(self.tracker.update(frame, visible), width, height))
                if tracks is not None:
                    payload["tracks"] = tracks
                    if isinstance(self.tracker, ReportsCameraMotion):
                        payload["camera_motion"] = self.tracker.camera_motion()
                    relations = self._relate(tracks, frame.t, width, height)
                    if relations is not None:
                        payload["lane_relations"] = relations
            else:
                status[Role.TRACKER] = (OutputStatus.NOT_SCHEDULED if upstream is OutputStatus.NOT_SCHEDULED
                                        else OutputStatus.INPUT_UNAVAILABLE)
        return FrameRecord(index=index, decode_index=frame.decode_index, t=frame.t, pts=frame.pts, status=status,
                           errors=errors, **payload)

    def _recent_mask(self, t: float, max_age: float) -> Optional[tuple[np.ndarray, tuple[str, ...]]]:
        if self._mask is None or t - self._mask[0] > max_age:
            return None
        return self._mask[1], self._mask[2]

    def _mark_ego(self, dets: tuple[Detection, ...], t: float, width: int, height: int) -> tuple[Detection, ...]:
        """Attach ego-vehicle overlap from the latest mask; the camera's own bonnet does not move in the image."""
        recent = self._recent_mask(t, self.ego_mask_max_age_s)
        if recent is None or "ego_vehicle" not in recent[1]:
            return dets
        mask, classes = recent
        return mark_ego_vehicle(dets, mask == classes.index("ego_vehicle"), width, height,
                                self.ego_vehicle_threshold)

    def _relate(self, tracks, t: float, width: int, height: int):
        if self.ego_lane is None or self._lane is None or t - self._lane[0] > self.lane_max_age_s:
            return None
        recent = self._recent_mask(t, self.lane_max_age_s)
        off_road = None
        if recent is not None and "undrivable" in recent[1]:
            off_road = recent[0] == recent[1].index("undrivable")
        return relate_tracks(tracks, self._lane[1], self._lane[0], width, height, off_road=off_road)

    def _count_failure(self, role: Role, error: str) -> None:
        self._failures[role] = self._failures.get(role, 0) + 1
        if self._failures[role] > self.max_consecutive_failures:
            raise RuntimeError(f"{role.value} failed on {self._failures[role]} consecutive frames; last error: {error}")

    # --- events -----------------------------------------------------------------------------------

    def _analyze(self, perception: PerceptionResult) -> tuple[tuple[AnalyzerReport, ...], list[EventDraft]]:
        """One report per configured analyser. An unavailable or failed analyser reports why and proposes
        nothing; an invalid return value is a programming error and fails the run."""
        reports, drafts = [], []
        video = perception.video
        for analyzer in self.analyzers:
            info = analyzer.info
            self._set(RunState.RUNNING, stage=f"running {info.name}", frames=len(perception.frames))
            self.guard.check()
            self._load(analyzer)
            started = time.perf_counter()
            status, reason = AnalyzerStatus.OK, None
            try:
                out = analyzer.analyze(perception.frames, video.width, video.height)
            except AnalyzerUnavailable as exc:
                status, reason = AnalyzerStatus.UNAVAILABLE, str(exc) or "inputs unavailable"
            except ComponentFailure as exc:
                status, reason = AnalyzerStatus.FAILED, str(exc) or "failed"
            self._seconds.append((info.name, round(time.perf_counter() - started, 4)))
            if status is AnalyzerStatus.OK:
                if not isinstance(out, AnalyzerOutput) or any(
                        not isinstance(d, EventDraft) or isinstance(d, EventProposal) or d.analyzer != info.name
                        for d in out.drafts):
                    raise TypeError(f"analyser {info.name} returned an invalid output")
                rejected = {k: v for k, v in sorted(out.rejected.items()) if v}
                reports.append(AnalyzerReport(analyzer=info, status=status, proposals=len(out.drafts),
                                              considered=len(out.drafts) + sum(rejected.values()),
                                              rejected=rejected,
                                              unresolved=tuple(sorted(out.unresolved, key=lambda u: u.t_start))))
                drafts.extend(out.drafts)
            else:
                reports.append(AnalyzerReport(analyzer=info, status=status, reason=reason))
            self._check_cancel()
        return tuple(reports), drafts


def _clip_box(box: Box, scale: float, width: int, height: int) -> Optional[Box]:
    x1, x2 = (float(np.clip(v / scale, 0, width)) for v in (box.x1, box.x2))
    y1, y2 = (float(np.clip(v / scale, 0, height)) for v in (box.y1, box.y2))
    return Box(x1=x1, y1=y1, x2=x2, y2=y2) if x2 > x1 and y2 > y1 else None


def _detections(out, frame: Frame, width: int, height: int) -> tuple[Detection, ...]:
    """Validate detector output and convert boxes from image to source display pixels."""
    result = []
    for d in out:
        if not isinstance(d, Detection):
            raise ComponentFailure(f"invalid output: expected Detection, got {type(d).__name__}")
        box = _clip_box(d.box, frame.scale, width, height)
        if box is not None:  # boxes entirely outside the frame carry no information
            result.append(d.model_copy(update={"box": box}))
    return tuple(result)


def _tracks(out, width: int, height: int) -> tuple[TrackedObject, ...]:
    result = []
    for o in out:
        if not isinstance(o, TrackedObject):
            raise ComponentFailure(f"invalid output: expected TrackedObject, got {type(o).__name__}")
        box = _clip_box(o.box, 1.0, width, height)
        if box is not None:
            result.append(o.model_copy(update={"box": box}))
    if len({o.track_id for o in result}) != len(result):
        raise ComponentFailure("invalid output: duplicate track_id")
    return tuple(result)


__all__ = ["AnalysisRun", "CancelToken", "ComponentInfo", "Progress", "RunOutcome", "RunState"]
