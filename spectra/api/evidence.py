"""Evidence for analyses: timelines and packages in the EvidenceStore, overlays and clips on request.

A finished analysis with a result (``completed`` or ``incomplete``) stores the analysers' reports and
proposals with it. Evidence is built with ``from_outcome`` right after the run, on the analysis thread,
and written to the ``EvidenceStore``; the analysis records the deterministic ``build_key``. Evidence has
its own state (``pending``, ``ready``, ``failed``) so a completed analysis is never mistaken for
completed evidence. When a build is missing, failed or was interrupted, it is rebuilt on request from
the verified perception cache (``PerceptionCache.lookup``) plus the stored reports and proposals;
``AnalysisResult`` validation rejects any combination whose analysis key does not match. Nothing here
loads a model or decodes video, except clip extraction, which stream-copies packets.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from spectra.api.store import Database
from spectra.engine.cache import PerceptionCache
from spectra.engine.contracts import AnalysisResult, AnalyzerReport, EventProposal, PerceptionResult, Role
from spectra.engine.runner import CancelToken, RunOutcome, RunState, StopReason
from spectra.engine.video import VideoError
from spectra.evidence.build import Records, from_outcome
from spectra.evidence.contracts import EvidencePackage, Overlay, ReviewConfig, ReviewInterval, Timeline
from spectra.evidence.media import ClipFile, ClipCancelled, ClipUnsupported, SourceUnavailable
from spectra.evidence.overlays import overlay
from spectra.evidence.store import EvidenceStore

log = logging.getLogger(__name__)
RESULT_STATES = ("completed", "incomplete")
MAX_OVERLAY_SPAN_S = 120.0


class EvidenceUnavailable(Exception):
    """Evidence cannot be provided now; the message is suitable for a user. ``status_code`` is HTTP."""

    def __init__(self, message: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.status_code = status_code


class EvidenceNotFound(LookupError):
    pass


def result_record(result: AnalysisResult) -> str:
    """What the perception cache does not hold: the analysers' reports and proposals."""
    return json.dumps({"reports": [r.model_dump(mode="json") for r in result.reports],
                       "events": [e.model_dump(mode="json") for e in result.events]})


class _Perception:
    """One verified perception entry with its frame times, reused across overlay requests."""

    def __init__(self, key: str, perception: PerceptionResult) -> None:
        self.key = key
        self.perception = perception
        self.records = Records(perception.frames)

    def close(self) -> None:
        close = getattr(self.perception.frames, "close", None)
        if close is not None:
            close()


class EvidenceService:
    def __init__(self, database: Database, cache_dir: Path, evidence_dir: Path,
                 config: ReviewConfig = ReviewConfig()) -> None:
        self.database = database
        self.cache = PerceptionCache(cache_dir)
        self.store = EvidenceStore(evidence_dir)
        self.config = config
        self._build_lock = threading.Lock()  # one build at a time; concurrent requests wait, then reuse it
        self._read_lock = threading.Lock()  # stored frames share one file handle
        self._clip_lock = threading.Lock()  # one stream copy at a time
        self._loaded: Optional[_Perception] = None
        self._cancel = CancelToken()

    def close(self) -> None:
        """Stop a running clip extraction and release the open cache entry."""
        self._cancel.cancel()
        with self._read_lock:
            if self._loaded is not None:
                self._loaded.close()
                self._loaded = None

    def start(self) -> None:
        """Builds interrupted by a stopped server become failed; they are rebuilt on request."""
        with self.database.connect() as conn:
            conn.execute("UPDATE analyses SET evidence_state = 'failed', evidence_message = "
                         "'evidence preparation was interrupted; it is rebuilt when the timeline is requested', "
                         "revision = revision + 1 WHERE evidence_state = 'pending'")

    # Building ------------------------------------------------------------------------------------

    def build_from_outcome(self, analysis_id: str, outcome: RunOutcome) -> None:
        """Called on the analysis thread right after a run with a result was recorded."""
        try:
            with self._build_lock:
                timeline, packages = from_outcome(outcome, self.config)
                self._store(analysis_id, timeline, packages)
        except Exception as exc:  # recorded; the stored result allows a rebuild on request
            log.exception("evidence for analysis %s could not be prepared", analysis_id)
            self._set(analysis_id, "failed", None, f"evidence could not be prepared: {exc}")
        finally:
            close = getattr(outcome.result.perception.frames, "close", None) if outcome.result else None
            if close is not None:
                close()

    def _store(self, analysis_id: str, timeline: Timeline, packages: list[EvidencePackage]) -> None:
        self.store.write(timeline, packages)  # keeps an existing valid build with the same key
        self._set(analysis_id, "ready", timeline.build_key, None)

    def _set(self, analysis_id: str, state: str, build_key: Optional[str], message: Optional[str]) -> None:
        with self.database.connect() as conn:
            conn.execute("UPDATE analyses SET evidence_state = ?, build_key = coalesce(?, build_key), "
                         "evidence_message = ?, revision = revision + 1 WHERE id = ?",
                         (state, build_key, message, analysis_id))

    def _row(self, analysis_id: str):
        with self.database.connect() as conn:
            row = conn.execute("SELECT a.*, v.path AS video_path, v.sha256 AS video_sha256 FROM analyses a "
                               "JOIN videos v ON v.id = a.video_id WHERE a.id = ?", (analysis_id,)).fetchone()
        if row is None:
            raise EvidenceNotFound(analysis_id)
        return row

    def _reconstruct(self, row) -> AnalysisResult:
        if not row["result_json"]:
            raise EvidenceUnavailable("this analysis has no stored result to rebuild evidence from; analyse the "
                                      "video again")
        found = self.cache.lookup(row["perception_key"])
        if found.result is None:
            raise EvidenceUnavailable("the analysis records are no longer in the perception cache"
                                      + (f" ({found.reason})" if found.reason else "") + "; analyse the video again")
        stored = json.loads(row["result_json"])
        try:
            return AnalysisResult(
                analysis_key=row["analysis_key"], perception_key=row["perception_key"], perception=found.result,
                reports=tuple(AnalyzerReport.model_validate(r) for r in stored["reports"]),
                events=tuple(EventProposal.model_validate(e) for e in stored["events"]))
        except ValueError as exc:
            close = getattr(found.result.frames, "close", None)
            if close is not None:
                close()
            raise EvidenceUnavailable(f"the cached records do not match this analysis: {exc}") from None

    def build_key(self, analysis_id: str) -> str:
        """The verified build key for an analysis with a result, rebuilding the evidence when needed."""
        row = self._row(analysis_id)
        if row["state"] not in RESULT_STATES:
            raise EvidenceUnavailable("this analysis has no result")
        if row["build_key"] and self.store.load(row["build_key"], row["analysis_key"]).timeline is not None:
            if row["evidence_state"] != "ready":
                self._set(analysis_id, "ready", row["build_key"], None)
            return row["build_key"]
        with self._build_lock:
            row = self._row(analysis_id)  # another request may have rebuilt it meanwhile
            if row["build_key"] and self.store.load(row["build_key"], row["analysis_key"]).timeline is not None:
                return row["build_key"]
            try:
                result = self._reconstruct(row)
            except EvidenceUnavailable as exc:
                self._set(analysis_id, "failed", None, str(exc))
                raise
            try:
                timeline, packages = from_outcome(RunOutcome(RunState.COMPLETED, result), self.config)
                self._store(analysis_id, timeline, packages)
            finally:
                result.perception.frames.close()
            return timeline.build_key

    # Reading -------------------------------------------------------------------------------------

    def timeline(self, analysis_id: str) -> Timeline:
        """The stored timeline of a run with a result, or a result-free timeline that says why there is none."""
        row = self._row(analysis_id)
        state = row["state"]
        if state in ("queued", "running"):
            raise EvidenceUnavailable("the analysis has not finished")
        if state in RESULT_STATES:
            key = self.build_key(analysis_id)
            lookup = self.store.load(key, row["analysis_key"])
            if lookup.timeline is None:
                raise EvidenceUnavailable(f"the stored evidence failed verification: {lookup.reason}")
            return lookup.timeline
        if state == "cancelled":
            outcome = RunOutcome(RunState.CANCELLED)
        else:
            stop = StopReason.RESOURCES if state == "resource_stopped" else StopReason.ERROR
            outcome = RunOutcome(RunState.FAILED, error=row["message"] or state, stop=stop)
        timeline, _ = from_outcome(outcome, self.config)
        return timeline.model_copy(update={"message": row["message"] or timeline.message})

    def duration(self, analysis_id: str) -> Optional[float]:
        """Length of the analysed video in analysis time: the declared duration, else the analysed span's end."""
        timeline = self.timeline(analysis_id)
        if timeline.video is not None and timeline.video.duration_s is not None:
            return timeline.video.duration_s
        return timeline.analysed_end_s

    def package(self, analysis_id: str, event_id: str) -> EvidencePackage:
        key = self.build_key(analysis_id)
        try:
            return self.store.package(key, event_id)
        except FileNotFoundError:
            raise EvidenceNotFound(event_id) from None
        except LookupError as exc:
            if "not part of this build" in str(exc):
                raise EvidenceNotFound(event_id) from None
            raise EvidenceUnavailable(str(exc)) from None
        except (OSError, ValueError) as exc:
            raise EvidenceUnavailable(f"the stored evidence failed verification: {exc}") from None

    def _perception(self, key: str) -> _Perception:
        """Called with the read lock held."""
        if self._loaded is not None and self._loaded.key == key:
            return self._loaded
        found = self.cache.lookup(key)
        if found.result is None:
            raise EvidenceUnavailable("the analysis records are no longer in the perception cache"
                                      + (f" ({found.reason})" if found.reason else ""))
        if self._loaded is not None:
            self._loaded.close()
        self._loaded = _Perception(key, found.result)
        return self._loaded

    def overlay(self, analysis_id: str, event_id: str, start_s: Optional[float] = None,
                end_s: Optional[float] = None) -> tuple[Overlay, EvidencePackage, PerceptionResult]:
        """Recorded drawing data for the package's review interval, or a requested span of at most
        MAX_OVERLAY_SPAN_S inside the analysed span. Nothing is interpolated."""
        pk = self.package(analysis_id, event_id)
        if start_s is not None or end_s is not None:
            start = pk.review.start_s if start_s is None else start_s
            end = pk.review.end_s if end_s is None else end_s
            if not 0 <= start < end:
                raise EvidenceUnavailable("start_s must be before end_s", 422)
            if end - start > MAX_OVERLAY_SPAN_S:
                raise EvidenceUnavailable(f"request at most {MAX_OVERLAY_SPAN_S:g} s of overlay", 422)
            if end < pk.analysed_start_s or start > pk.analysed_end_s:
                raise EvidenceUnavailable("the requested span lies outside the analysed span", 422)
            pk = pk.model_copy(update={"review": ReviewInterval(start_s=start, end_s=end, evidence_start_s=start,
                                                                evidence_end_s=start)})
        with self._read_lock:
            loaded = self._perception(pk.perception_key)
            try:
                drawn = overlay(loaded.perception, pk, self.config, loaded.records)
            except ValueError as exc:
                raise EvidenceUnavailable(f"the cached records do not match this package: {exc}") from None
            return drawn, pk, loaded.perception

    def clip(self, analysis_id: str, event_id: str, source: Path) -> tuple[ClipFile, Path]:
        """Stream-copy the package's review interval from the registered source, or reuse a verified copy."""
        pk = self.package(analysis_id, event_id)
        with self._clip_lock:
            try:
                clip = self.store.clip(pk, source, cancel=self._cancel)
            except SourceUnavailable as exc:
                raise EvidenceUnavailable(f"the source video is unavailable: {exc}") from None
            except ClipUnsupported as exc:
                raise EvidenceUnavailable(f"a clip cannot be cut from this video: {exc}", 422) from None
            except ClipCancelled:
                raise EvidenceUnavailable("clip extraction was cancelled because the server is stopping", 503) from None
            except VideoError as exc:
                raise EvidenceUnavailable(f"the clip could not be verified against the source: {exc}", 422) from None
            return clip, self.store.clip_path(pk, clip)

    def stored_clip(self, analysis_id: str, event_id: str) -> tuple[ClipFile, Path]:
        """A clip written earlier, checked by its record and size; the full hash was verified when written."""
        pk = self.package(analysis_id, event_id)
        directory = self.store.entry(pk.build_key) / "clips"
        try:
            clip = ClipFile.model_validate_json((directory / f"{event_id}.json").read_text())
        except (OSError, ValueError):
            raise EvidenceNotFound(event_id) from None
        path = directory / clip.file
        if path.resolve().parent != self.store.root.resolve() / pk.build_key / "clips":
            raise EvidenceNotFound(event_id)
        if (clip.source_sha256, clip.start_s, clip.end_s) != (pk.video.sha256, pk.review.start_s, pk.review.end_s) \
                or not path.is_file() or path.stat().st_size != clip.size_bytes:
            raise EvidenceNotFound(event_id)
        return clip, path


def layers(perception: PerceptionResult) -> dict[str, bool]:
    """Which overlay layers the run produced at all; a missing layer was not configured, not empty."""
    roles = {c.role for c in perception.provenance.components}
    return {"tracks": Role.TRACKER in roles, "lane": Role.EGO_LANE in roles}
