"""Run one analysis in a dedicated child process.

Model runtimes keep memory after inference and can crash natively; a spawned process that exits after
each run returns all of it to the system and turns a crash into a failed run instead of a dead server.
Only one worker runs at a time per caller; the interface is a picklable job description in, progress
events and an outcome out, and a cancel flag shared with the child. The perception records are not sent
back: the child commits them to the cache and the parent reads that entry lazily by its key.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from spectra.engine.contracts import AnalysisResult, AnalyzerReport, EventProposal, RunConfig
from spectra.engine.runner import CancelToken, Progress, RunOutcome, RunState, StopReason


@dataclass(frozen=True)
class DetectorSpec:
    model_path: str
    sha256: str
    input_size: int = 640
    score_threshold: float = 0.1
    duplicate_iou: Optional[float] = 0.7
    threads: int = 4
    interval_s: Optional[float] = None


@dataclass(frozen=True)
class TrackerSpec:
    lost_seconds: float = 2.0
    reference_rate: float = 10.0
    camera_motion: bool = True


@dataclass(frozen=True)
class SegmenterSpec:
    model_path: str
    sha256: str
    threads: int = 4
    interval_s: Optional[float] = 0.2


@dataclass(frozen=True)
class AnalyzerSpec:
    """A built-in analyzer name and constructor parameters (no dynamic imports).

    Impact's optional ``approach`` parameter is an ApproachAnalyzer parameter dict.
    Set it explicitly to match a separately configured approach analyzer when needed.
    """
    name: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class WorkerJob:
    video: str
    cache_root: str
    config: dict = field(default_factory=dict)  # RunConfig fields
    detector: Optional[DetectorSpec] = None
    tracker: Optional[TrackerSpec] = None
    segmenter: Optional[SegmenterSpec] = None
    lanes: bool = False  # estimate the ego lane from the segmentation
    events: bool = False  # run the event analysers on the perception records
    policy: str = "standard"
    analyzers: Optional[tuple[AnalyzerSpec, ...]] = None
    cache_only: bool = False  # fail a cache miss before loading perception components


def build_analyzers(job: WorkerJob) -> list:
    """Build explicit specs, or reproduce the original events/lanes defaults exactly.

    None preserves the events switch; an explicit tuple overrides it, including an
    empty tuple to disable analyzers. Order is preserved because it is part of identity.
    """
    from spectra.engine.events import ApproachAnalyzer, CutInAnalyzer, ImpactAnalyzer

    table = {cls.name: cls for cls in (ApproachAnalyzer, CutInAnalyzer, ImpactAnalyzer)}
    specs = job.analyzers
    if specs is None:
        specs = () if not job.events else (
            (AnalyzerSpec(ApproachAnalyzer.name),)
            + ((AnalyzerSpec(CutInAnalyzer.name),) if job.lanes else ())
            + (AnalyzerSpec(ImpactAnalyzer.name),))
    analyzers = []
    for spec in specs:
        if spec.name not in table:
            raise ValueError(f"unknown analyzer: {spec.name}")
        params = dict(spec.params)
        if spec.name == ImpactAnalyzer.name and "approach" in params:
            params["approach"] = ApproachAnalyzer(**params["approach"])
        analyzers.append(table[spec.name](**params))
    return analyzers


def _child(job: WorkerJob, events, cancel_event) -> None:
    from spectra.engine.cache import PerceptionCache
    from spectra.engine.detection import DFineOnnxDetector
    from spectra.engine.lanes import LaneGeometryEstimator, SegnetOnnxSegmenter
    from spectra.engine.resources import ResourceGuard
    from spectra.engine.runner import AnalysisRun
    from spectra.engine.tracking import BoTSortTracker

    try:
        detector = tracker = None
        if job.detector is not None:
            d = asdict(job.detector)
            detector = DFineOnnxDetector(d.pop("model_path"), d.pop("sha256"), **d)
        if job.tracker is not None:
            tracker = BoTSortTracker(**asdict(job.tracker))
        segmenter = ego_lane = None
        if job.segmenter is not None:
            s = asdict(job.segmenter)
            segmenter = SegnetOnnxSegmenter(s.pop("model_path"), s.pop("sha256"), **s)
        if job.lanes:
            ego_lane = LaneGeometryEstimator()
        analyzers = build_analyzers(job)
        last = [0.0]

        def progress(p: Progress) -> None:
            now = time.monotonic()
            if p.state is RunState.RUNNING and p.stage == "analysing frames" and now - last[0] < 0.25:
                return
            last[0] = now
            events.put(("progress", (p.state.value, p.frames_analysed, p.t, p.fraction, p.stage)))

        outcome = AnalysisRun(job.video, RunConfig(**job.config), detector=detector, tracker=tracker,
                              segmenter=segmenter, ego_lane=ego_lane, analyzers=analyzers,
                              cache=PerceptionCache(job.cache_root), on_progress=progress,
                              cancel=CancelToken(cancel_event), guard=ResourceGuard(policy=job.policy),
                              cache_only=job.cache_only).run()
        summary = None
        if outcome.result is not None:
            r = outcome.result
            summary = (r.perception_key, r.analysis_key, [a.model_dump_json() for a in r.reports],
                       [e.model_dump_json() for e in r.events])
        events.put(("outcome", (outcome.state.value, outcome.error, outcome.cache_hit, outcome.notes, summary,
                                outcome.analyzer_seconds, outcome.stop and outcome.stop.value)))
    except BaseException as exc:  # building components failed; report rather than die silently
        events.put(("outcome", (RunState.FAILED.value, f"{type(exc).__name__}: {exc}", False, (), None, (),
                                StopReason.ERROR.value)))


def run_in_worker(job: WorkerJob, on_progress: Optional[Callable[[Progress], None]] = None,
                  cancel: Optional[CancelToken] = None, timeout_s: Optional[float] = None) -> RunOutcome:
    """Run ``job`` in a spawned process and wait for it. Never raises for analysis problems.

    Script callers must put calls behind ``if __name__ == "__main__":`` so that
    spawn can safely import the calling module. Invalid RunConfig is rejected in
    the parent, before allocating multiprocessing resources or loading models.
    """
    try:
        RunConfig(**job.config)
    except (TypeError, ValueError) as exc:
        return RunOutcome(RunState.FAILED, error=f"invalid run config: {exc}", stop=StopReason.ERROR)
    if not Path(job.video).is_file():
        return RunOutcome(RunState.FAILED, error=f"video file not found: {Path(job.video).name}",
                          stop=StopReason.VIDEO)
    ctx = mp.get_context("spawn")
    events, cancel_event = ctx.Queue(), ctx.Event()
    child = ctx.Process(target=_child, args=(job, events, cancel_event), daemon=True)
    deadline = time.monotonic() + timeout_s if timeout_s else None
    child.start()
    outcome = None
    try:
        while outcome is None:
            if cancel is not None and cancel.cancelled:
                cancel_event.set()
            try:
                kind, payload = events.get(timeout=0.2)
            except queue.Empty:
                if not child.is_alive():
                    try:  # the outcome may have arrived just before the child exited
                        kind, payload = events.get(timeout=0.5)
                    except queue.Empty:
                        outcome = RunOutcome(RunState.FAILED, error=f"analysis worker exited with code "
                                                                     f"{child.exitcode} without a result",
                                             stop=StopReason.ERROR)
                        break
                elif deadline is not None and time.monotonic() > deadline:
                    outcome = RunOutcome(RunState.FAILED, error=f"analysis worker timed out after {timeout_s} s",
                                         stop=StopReason.ERROR)
                    break
                else:
                    continue
            if kind == "progress" and on_progress is not None:
                state, frames, t, fraction, stage = payload
                on_progress(Progress(RunState(state), frames, t, fraction, stage))
            elif kind == "outcome":
                state, error, hit, notes, summary, seconds, stop = payload
                parsed = None
                if summary is not None:
                    parsed, problem = _open_result(job.cache_root, *summary)
                    if parsed is None:
                        state, error, stop = RunState.FAILED.value, problem, StopReason.ERROR.value
                outcome = RunOutcome(RunState(state), parsed, error, hit, tuple(notes), tuple(map(tuple, seconds)),
                                     stop and StopReason(stop))
    finally:
        child.join(10)
        if child.is_alive():
            child.terminate()
            child.join(5)
        events.close()
    return outcome


def _open_result(cache_root: str, key: str, akey: str, reports: list, events: list):
    """The completed run's result, with perception read lazily from the entry the child committed."""
    from spectra.engine.cache import PerceptionCache
    found = PerceptionCache(cache_root).lookup(key)
    if found.result is None:
        return None, f"the worker's cache entry could not be read: {found.reason or 'missing'}"
    return AnalysisResult(analysis_key=akey, perception_key=key, perception=found.result,
                          reports=tuple(AnalyzerReport.model_validate_json(r) for r in reports),
                          events=tuple(EventProposal.model_validate_json(e) for e in events)), None
