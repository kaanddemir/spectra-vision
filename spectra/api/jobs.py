"""Analysis jobs: a persisted FIFO queue executed one worker process at a time.

The database is the queue. One runner thread claims the oldest queued analysis, verifies the
source (full SHA-256 through ``check_source``), checks models and memory pressure, then calls
``run_in_worker``, which spawns a fresh process for the whole analysis. The API process never loads
a model. Progress is persisted, throttled, with a revision number that clients and the SSE stream
follow. Cancellation uses ``CancelToken``; shutdown cancels the running job and waits for its worker.
Nothing is retried automatically, and work interrupted by a restart is never resumed.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from spectra.api.evidence import result_record
from spectra.api.schemas import Analysis, AnalysisProgress, AnalysisResultSummary, EvidenceStatus
from spectra.api.store import Database
from spectra.api.system import ModelRegistryCheck, PressureReader
from spectra.api.videos import VideoNotFound, VideoRegistry
from spectra.engine.contracts import RunConfig, VideoInfo
from spectra.engine.models import default_specs
from spectra.engine.runner import CancelToken, Progress, RunOutcome, RunState, StopReason
from spectra.engine.worker import TrackerSpec, WorkerJob, run_in_worker
from spectra.evidence.media import SourceStatus, check_source

log = logging.getLogger(__name__)

JobFactory = Callable[[Path, VideoInfo], WorkerJob]
ResultHandler = Callable[[str, RunOutcome], None]
WorkerRunner = Callable[..., RunOutcome]
TERMINAL_STATES = ("completed", "incomplete", "cancelled", "resource_stopped", "failed")
CRITICAL_PRESSURE = 4
PROGRESS_INTERVAL_S = 0.5
ANALYSIS_INTERVAL_S = 0.1  # 10 Hz detection and tracking; segmentation runs at its registry default (5 Hz)


def standard_job(models_dir: Path, cache_dir: Path) -> JobFactory:
    """The application's fixed profile, matching the documented CLI operating point.

    Models are resolved and hash-verified for every job, so a replaced weight file is refused.
    """
    def make(path: Path, info: VideoInfo) -> WorkerJob:
        detector, segmenter = default_specs(models_dir)
        config = RunConfig(interval_s=ANALYSIS_INTERVAL_S)
        return WorkerJob(video=str(path), cache_root=str(cache_dir), config=config.model_dump(exclude_defaults=True),
                         detector=detector, tracker=TrackerSpec(reference_rate=1 / ANALYSIS_INTERVAL_S),
                         segmenter=segmenter, lanes=True, events=True, policy="standard")
    return make


def worker_timeout_s(info: VideoInfo) -> float:
    """Generous wall-clock limit: model loading plus 20x the video duration (6 h when unknown)."""
    return 900.0 + 20.0 * info.duration_s if info.duration_s else 6 * 3600.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AnalysisRefused(Exception):
    """The analysis cannot be queued; ``reasons`` are suitable for a user."""

    def __init__(self, reasons: list[str], status_code: int = 409) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons
        self.status_code = status_code


class AnalysisNotFound(LookupError):
    pass


class AnalysisService:
    def __init__(self, database: Database, videos: VideoRegistry, models: ModelRegistryCheck,
                 pressure: PressureReader, job_factory: JobFactory, run: WorkerRunner = run_in_worker,
                 timeout: Callable[[VideoInfo], float] = worker_timeout_s,
                 on_result: Optional[ResultHandler] = None) -> None:
        self.database = database
        self.videos = videos
        self.models = models
        self.pressure = pressure
        self.job_factory = job_factory
        self.run = run
        self.timeout = timeout
        self.on_result = on_result  # prepares evidence on this thread after a run with a result
        self.closing = False
        self.subscribers = 0  # open event streams, maintained on the event loop
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._tokens: dict[str, CancelToken] = {}
        self._thread: Optional[threading.Thread] = None

    # Lifecycle ---------------------------------------------------------------------------------

    def start(self) -> None:
        """Settle work left by a previous process, then start the runner thread."""
        self._interrupt("the server stopped while this analysis was running; start it again",
                        "the server stopped before this analysis started; start it again")
        self._thread = threading.Thread(target=self._loop, name="spectra-analysis", daemon=True)
        self._thread.start()

    def close(self, timeout_s: float = 120.0) -> None:
        """Refuse new work, cancel the running analysis and wait for its worker to exit."""
        with self._lock:
            self.closing = True
            for token in self._tokens.values():
                token.cancel()
            self._wake.notify_all()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                log.error("analysis worker did not stop within %.0f s", timeout_s)
        self._interrupt(None, "the server stopped before this analysis started; start it again")

    def _interrupt(self, running_message: Optional[str], queued_message: str) -> None:
        with self.database.connect() as conn:
            if running_message is not None:
                conn.execute(
                    "UPDATE analyses SET state = 'failed', stop_reason = 'interrupted', message = ?, "
                    "finished_at = ?, revision = revision + 1 WHERE state = 'running'", (running_message, _now()))
            conn.execute(
                "UPDATE analyses SET state = 'cancelled', stop_reason = 'interrupted', message = ?, "
                "finished_at = ?, revision = revision + 1 WHERE state = 'queued'", (queued_message, _now()))

    # Requests ----------------------------------------------------------------------------------

    def submit(self, video_id: str) -> tuple[Analysis, bool]:
        """Queue an analysis, or return the video's queued or running one. Returns (analysis, created)."""
        if self.closing:
            raise AnalysisRefused(["the server is shutting down"], 503)
        active = self._active_for(video_id)
        if active is not None:
            return self.get(active), False
        video = self.videos.get(video_id)  # raises VideoNotFound
        if not video.analysis.eligible or video.media is None:
            raise AnalysisRefused(video.analysis.reasons or ["the video is not ready for analysis"])
        reasons = []
        models = self.models.status()
        if not models.available:
            reasons.append("The analysis models are missing or failed verification.")
        level = self.pressure()
        if level is not None and level >= CRITICAL_PRESSURE:
            reasons.append("System memory pressure is critical; close other applications and try again.")
        if reasons:
            raise AnalysisRefused(reasons, 503)
        analysis_id = uuid.uuid4().hex
        try:
            with self.database.connect() as conn:
                conn.execute("INSERT INTO analyses (id, video_id, source_sha256, state, created_at, revision) "
                             "VALUES (?, ?, ?, 'queued', ?, 1)", (analysis_id, video_id, video.media.sha256, _now()))
        except sqlite3.IntegrityError:
            active = self._active_for(video_id)  # a concurrent request queued it first
            if active is None:
                raise
            return self.get(active), False
        with self._lock:
            self._wake.notify_all()
        return self.get(analysis_id), True

    def _active_for(self, video_id: str) -> Optional[str]:
        with self.database.connect() as conn:
            row = conn.execute("SELECT id FROM analyses WHERE video_id = ? AND state IN ('queued', 'running')",
                               (video_id,)).fetchone()
        return row["id"] if row else None

    def cancel(self, analysis_id: str) -> Analysis:
        """Cancel a queued analysis at once, or ask the running worker to stop.

        Cancelling a cancelled analysis is a no-op; other finished analyses raise AnalysisRefused.
        """
        with self._lock:
            state = self.get(analysis_id).state
            if state == "queued":
                with self.database.connect() as conn:
                    conn.execute("UPDATE analyses SET state = 'cancelled', stop_reason = 'cancelled', "
                                 "cancel_requested = 1, message = 'cancelled before it started', finished_at = ?, "
                                 "revision = revision + 1 WHERE id = ? AND state = 'queued'", (_now(), analysis_id))
            elif state == "running":
                with self.database.connect() as conn:
                    conn.execute("UPDATE analyses SET cancel_requested = 1, revision = revision + 1 "
                                 "WHERE id = ? AND state = 'running' AND cancel_requested = 0", (analysis_id,))
                token = self._tokens.get(analysis_id)
                if token is not None:
                    token.cancel()
            elif state != "cancelled":
                raise AnalysisRefused([f"the analysis already finished ({state})"])
        return self.get(analysis_id)

    def get(self, analysis_id: str) -> Analysis:
        with self.database.connect() as conn:
            row = conn.execute(
                "SELECT a.*, v.path AS video_path, CASE WHEN a.state = 'queued' THEN (SELECT count(*) FROM analyses q "
                "WHERE q.state = 'queued' AND q.rowid <= a.rowid) END AS queue_position "
                "FROM analyses a JOIN videos v ON v.id = a.video_id WHERE a.id = ?", (analysis_id,)).fetchone()
        if row is None:
            raise AnalysisNotFound(analysis_id)
        return self._analysis(row)

    def list(self, video_id: Optional[str] = None) -> list[Analysis]:
        with self.database.connect() as conn:
            rows = conn.execute(
                "SELECT a.*, v.path AS video_path, CASE WHEN a.state = 'queued' THEN (SELECT count(*) FROM analyses q "
                "WHERE q.state = 'queued' AND q.rowid <= a.rowid) END AS queue_position "
                "FROM analyses a JOIN videos v ON v.id = a.video_id WHERE ? IS NULL OR a.video_id = ? "
                "ORDER BY a.rowid DESC", (video_id, video_id)).fetchall()
        return [self._analysis(row) for row in rows]

    @staticmethod
    def _analysis(row: sqlite3.Row) -> Analysis:
        result = None
        if row["state"] in ("completed", "incomplete"):
            result = AnalysisResultSummary(cache_hit=bool(row["cache_hit"]), perception_key=row["perception_key"],
                                           analysis_key=row["analysis_key"], proposals=row["proposals"] or 0,
                                           notes=json.loads(row["notes_json"] or "[]"))
        return Analysis(
            id=row["id"], video_id=row["video_id"], file_name=Path(row["video_path"]).name,
            source_sha256=row["source_sha256"], state=row["state"], revision=row["revision"],
            queue_position=row["queue_position"], cancel_requested=bool(row["cancel_requested"]),
            progress=AnalysisProgress(stage=row["stage"], frames_analysed=row["frames_analysed"],
                                      media_time_s=row["media_time_s"], fraction=row["fraction"]),
            result=result, evidence=EvidenceStatus(state=row["evidence_state"] or "not_applicable",
                                                   build_key=row["build_key"], message=row["evidence_message"]),
            stop_reason=row["stop_reason"], message=row["message"], created_at=row["created_at"],
            started_at=row["started_at"], finished_at=row["finished_at"])

    # Execution ---------------------------------------------------------------------------------

    def _update(self, analysis_id: str, only_running: bool = True, **fields) -> None:
        columns = ", ".join(f"{name} = ?" for name in fields)
        condition = " AND state = 'running'" if only_running else ""
        with self.database.connect() as conn:
            conn.execute(f"UPDATE analyses SET {columns}, revision = revision + 1 WHERE id = ?{condition}",
                         (*fields.values(), analysis_id))

    def _claim(self) -> Optional[tuple[str, CancelToken]]:
        with self._lock:
            while not self.closing:
                with self.database.connect() as conn:
                    row = conn.execute("SELECT id FROM analyses WHERE state = 'queued' ORDER BY rowid LIMIT 1").fetchone()
                    if row is not None:
                        conn.execute("UPDATE analyses SET state = 'running', started_at = ?, stage = 'verifying source', "
                                     "revision = revision + 1 WHERE id = ? AND state = 'queued'", (_now(), row["id"]))
                if row is not None:
                    token = CancelToken()
                    self._tokens[row["id"]] = token
                    return row["id"], token
                self._wake.wait(1.0)
        return None

    def _loop(self) -> None:
        while True:
            claimed = self._claim()
            if claimed is None:
                return
            analysis_id, token = claimed
            outcome = None
            try:
                fields, outcome = self._execute(analysis_id, token)
            except Exception:
                log.exception("analysis %s failed", analysis_id)
                fields = dict(state="failed", stop_reason="error", message="an internal error stopped the analysis")
            finally:
                with self._lock:
                    self._tokens.pop(analysis_id, None)
            if fields["state"] in ("completed", "incomplete") and self.on_result is not None:
                fields["evidence_state"] = "pending"
            self._update(analysis_id, finished_at=_now(), **fields)
            if fields.get("evidence_state") == "pending":
                self.on_result(analysis_id, outcome)
            outcome = None  # release the result before waiting for the next job

    def _cancelled(self) -> dict:
        if self.closing:
            return dict(state="cancelled", stop_reason="shutdown", message="cancelled because the server stopped")
        return dict(state="cancelled", stop_reason="cancelled", message="cancelled on request")

    def _execute(self, analysis_id: str, token: CancelToken) -> tuple[dict, Optional[RunOutcome]]:
        analysis = self.get(analysis_id)
        try:
            video, path, info = self.videos.source_for_analysis(analysis.video_id)
        except VideoNotFound:
            return dict(state="failed", stop_reason="video", message="the video is no longer registered"), None
        if info is None or info.sha256 != analysis.source_sha256:
            return dict(state="failed", stop_reason="video",
                        message="the video changed after the analysis was requested; register it again"), None
        if not video.analysis.eligible:
            return dict(state="failed", stop_reason="video", message="; ".join(video.analysis.reasons)), None
        if token.cancelled:
            return self._cancelled(), None
        check = check_source(path, info, verify_hash=True)  # full SHA-256 of the file about to be analysed
        if check.status is not SourceStatus.AVAILABLE:
            return dict(state="failed", stop_reason="video", message=check.message), None
        if token.cancelled:
            return self._cancelled(), None
        level = self.pressure()
        if level is not None and level >= CRITICAL_PRESSURE:
            return dict(state="resource_stopped", stop_reason="resources",
                        message=f"system memory pressure level {level} is critical; the analysis did not start"), None
        try:
            job = self.job_factory(path, info)
        except (OSError, ValueError) as exc:
            return dict(state="failed", stop_reason="models",
                        message=f"the analysis models are unavailable: {exc}"), None
        profile = dict(config=job.config, detector_sha256=job.detector and job.detector.sha256,
                       segmenter_sha256=job.segmenter and job.segmenter.sha256, tracking=job.tracker is not None,
                       lanes=job.lanes, events=job.events, policy=job.policy)
        self._update(analysis_id, stage="starting analysis worker", profile_json=json.dumps(profile))
        last = [0.0, None]

        def on_progress(p: Progress) -> None:
            now = time.monotonic()
            if p.stage == last[1] and now - last[0] < PROGRESS_INTERVAL_S:
                return
            last[0], last[1] = now, p.stage
            self._update(analysis_id, stage=p.stage or None, frames_analysed=p.frames_analysed,
                         media_time_s=p.t, fraction=p.fraction)

        outcome = self.run(job, on_progress=on_progress, cancel=token, timeout_s=self.timeout(info))
        return self._outcome_fields(outcome), outcome

    def _outcome_fields(self, outcome: RunOutcome) -> dict:
        common = dict(cache_hit=int(outcome.cache_hit), notes_json=json.dumps(list(outcome.notes)))
        if outcome.state is RunState.COMPLETED:
            result = outcome.result
            ended_early = result.perception.timing.ended_early
            fields = dict(common, perception_key=result.perception_key, analysis_key=result.analysis_key,
                          proposals=len(result.events), stage=None, result_json=result_record(result))
            if ended_early:
                return dict(fields, state="incomplete", message=f"decoding ended early: {ended_early}")
            return dict(fields, state="completed", message=None)
        if outcome.state is RunState.CANCELLED:
            return dict(common, **self._cancelled())
        if outcome.stop is StopReason.RESOURCES:
            return dict(common, state="resource_stopped", stop_reason="resources", message=outcome.error)
        reason = "video" if outcome.stop is StopReason.VIDEO else "error"
        return dict(common, state="failed", stop_reason=reason, message=outcome.error)
