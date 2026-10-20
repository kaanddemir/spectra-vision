"""Registered source videos: registration, background probing, change detection and eligibility.

A registration stores the real path of a library file. Probing (full SHA-256, container, codecs,
timestamps) runs on one background thread through an isolated helper process, never on a request
path. Afterwards every use compares the file's size, modification time and inode with the probed
values: a cheap check that catches edits, replacement and deletion. The full hash is verified again
at registration and before analysis. Source files are only ever opened for reading.
"""
from __future__ import annotations

import json
import logging
import os
import stat
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from sqlite3 import Row
from typing import Callable, Iterable, Optional

from spectra.api.library import LibraryFolder, locate, resolve_video_reference
from spectra.api.playback import assess
from spectra.api.probe import ProbeReport, timeout_for
from spectra.api.schemas import (BrowserCheck, Eligibility, MediaInfo, PlaybackInfo, ProbeStatus, SourceState,
                                 Video)
from spectra.api.store import Database
from spectra.engine.contracts import VideoInfo

log = logging.getLogger(__name__)
ProbeFunction = Callable[[Path, float], ProbeReport]
RETRYABLE_PROBE_ERRORS = ("changed", "not_found", "timeout", "failed")


class VideoNotFound(LookupError):
    pass


class VideoUnavailable(RuntimeError):
    """The video cannot be used now; the message is suitable for a user."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, timezone.utc).isoformat(timespec="seconds")


class VideoRegistry:
    def __init__(self, database: Database, folders: Iterable[LibraryFolder], data_dir: Path,
                 probe: ProbeFunction) -> None:
        self.database = database
        self.folders = folders  # iterated on each use: folders added or removed apply at once
        self.data_dir = data_dir
        self._probe = probe
        self._executor: Optional[ThreadPoolExecutor] = None

    def start(self) -> None:
        """Start the probe thread and resume probes interrupted by a previous shutdown."""
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spectra-probe")
        with self.database.connect() as conn:
            untracked = conn.execute(
                "SELECT id FROM videos WHERE probe_state = 'pending' AND probe_token IS NULL").fetchall()
            for row in untracked:
                conn.execute("UPDATE videos SET probe_token = ? WHERE id = ?", (uuid.uuid4().hex, row["id"]))
            pending = conn.execute("SELECT id, probe_token FROM videos WHERE probe_state = 'pending' "
                                   "ORDER BY added_at").fetchall()
        for row in pending:
            self._submit(row["id"], row["probe_token"])

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    # Registration -------------------------------------------------------------------------------

    def register(self, folder_id: str, relative_path: str) -> tuple[Video, bool]:
        """Register a library file, or return its existing registration. Returns (video, created).

        An unchanged registration is returned as is. A changed file, or one whose previous probe
        failed for a reason that may be temporary, is probed again.
        """
        path = resolve_video_reference(self.folders, folder_id, relative_path, exclude=[self.data_dir])
        info = path.stat()
        current = (info.st_size, info.st_mtime_ns, info.st_ino)
        token = uuid.uuid4().hex
        with self.database.connect() as conn:
            row = conn.execute("SELECT * FROM videos WHERE path = ?", (str(path),)).fetchone()
            created = row is None
            if created:
                video_id = uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO videos (id, path, size_bytes, mtime_ns, inode, added_at, probe_state, probe_token) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?) ON CONFLICT (path) DO NOTHING",
                    (video_id, str(path), *current, _now(), token))
                row = conn.execute("SELECT * FROM videos WHERE path = ?", (str(path),)).fetchone()
                created = row["id"] == video_id
                if not created:
                    return self._video(row), False
            else:
                video_id = row["id"]
                unchanged = (row["size_bytes"], row["mtime_ns"], row["inode"]) == current
                retry = row["probe_state"] == "failed" and row["probe_error"] in RETRYABLE_PROBE_ERRORS
                if unchanged and not retry:
                    return self._video(row), False
                conn.execute(
                    "UPDATE videos SET size_bytes = ?, mtime_ns = ?, inode = ?, sha256 = NULL, info_json = NULL, "
                    "audio_codecs_json = NULL, playback_check_json = NULL, probe_state = 'pending', "
                    "probe_token = ?, probe_error = NULL, probe_message = NULL, probed_at = NULL WHERE id = ?",
                    (*current, token, video_id))
        self._submit(video_id, token)
        return self.get(video_id), created

    def _submit(self, video_id: str, token: str) -> None:
        if self._executor is not None:
            self._executor.submit(self._run_probe, video_id, token)

    def _run_probe(self, video_id: str, token: str) -> None:
        try:
            with self.database.connect() as conn:
                row = conn.execute("SELECT path, size_bytes FROM videos WHERE id = ? AND probe_token = ?",
                                   (video_id, token)).fetchone()
            if row is None:
                return  # removed or registered again since this probe was queued
            report = self._probe(Path(row["path"]), timeout_for(row["size_bytes"]))
        except Exception:
            log.exception("probe of video %s failed", video_id)
            report = ProbeReport(error="failed", message="the video could not be read")
        if report.error == "cancelled":
            return  # stays pending and resumes when the server starts again
        with self.database.connect() as conn:
            if report.info is not None and report.stat is not None:
                conn.execute(
                    "UPDATE videos SET probe_state = 'ready', sha256 = ?, info_json = ?, audio_codecs_json = ?, "
                    "size_bytes = ?, mtime_ns = ?, inode = ?, probed_at = ? WHERE id = ? AND probe_token = ?",
                    (report.info.sha256, report.info.model_dump_json(), json.dumps(report.audio_codecs),
                     report.stat.size_bytes, report.stat.mtime_ns, report.stat.inode, _now(), video_id, token))
            else:
                conn.execute(
                    "UPDATE videos SET probe_state = 'failed', probe_error = ?, probe_message = ?, probed_at = ? "
                    "WHERE id = ? AND probe_token = ?",
                    (report.error or "failed", report.message, _now(), video_id, token))

    # Queries ------------------------------------------------------------------------------------

    def _row(self, video_id: str) -> Row:
        with self.database.connect() as conn:
            row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        if row is None:
            raise VideoNotFound(video_id)
        return row

    def get(self, video_id: str) -> Video:
        return self._video(self._row(video_id))

    def list(self) -> list[Video]:
        with self.database.connect() as conn:
            rows = conn.execute("SELECT * FROM videos ORDER BY added_at DESC, id").fetchall()
        return [self._video(row) for row in rows]

    def registered_paths(self) -> dict[str, str]:
        with self.database.connect() as conn:
            return {row["path"]: row["id"] for row in conn.execute("SELECT id, path FROM videos")}

    def remove(self, video_id: str) -> None:
        """Forget a registration. The source file is not touched."""
        with self.database.connect() as conn:
            if conn.execute("SELECT 1 FROM analyses WHERE video_id = ? LIMIT 1", (video_id,)).fetchone():
                raise VideoUnavailable("this video has analyses and cannot be removed")
            if conn.execute("DELETE FROM videos WHERE id = ?", (video_id,)).rowcount == 0:
                raise VideoNotFound(video_id)

    # Source checks and playback -------------------------------------------------------------------

    def _source(self, row: Row) -> tuple[SourceState, Optional[os.stat_result]]:
        path = Path(row["path"])
        if locate(self.folders, path) is None:
            return SourceState(status="outside_library",
                               message="the video is not inside a library folder"), None
        try:
            real = path.resolve(strict=True)
            info = os.stat(path)
        except (OSError, RuntimeError):
            return SourceState(status="missing", message=f"{path.name} was moved or deleted"), None
        if real != path or not stat.S_ISREG(info.st_mode):
            return SourceState(status="changed", message=f"{path.name} was replaced"), None
        if (info.st_size, info.st_mtime_ns, info.st_ino) != (row["size_bytes"], row["mtime_ns"], row["inode"]):
            return SourceState(status="changed",
                               message=f"{path.name} changed after it was registered; register it again"), None
        return SourceState(status="available"), info

    def _video(self, row: Row) -> Video:
        path = Path(row["path"])
        source, _ = self._source(row)
        located = locate(self.folders, path)
        probe = ProbeStatus(state=row["probe_state"], error=row["probe_error"], message=row["probe_message"],
                            probed_at=row["probed_at"])
        media, playback = None, PlaybackInfo(assessment="unknown", reasons=[])
        if row["probe_state"] == "ready":
            info = VideoInfo.model_validate_json(row["info_json"])
            audio = json.loads(row["audio_codecs_json"])
            media = MediaInfo(sha256=info.sha256, container=info.container, codec=info.codec, audio_codecs=audio,
                              width=info.width, height=info.height, rotation=info.rotation,
                              duration_s=info.duration_s, nominal_fps=info.nominal_fps, start_s=info.start_s,
                              timestamp_source=info.timestamp_source.value)
            assessment = assess(info, audio, path.suffix)
            check = None
            if row["playback_check_json"]:
                stored = json.loads(row["playback_check_json"])
                if stored.pop("sha256", None) == info.sha256:
                    check = BrowserCheck(**stored)
            playback = PlaybackInfo(assessment=assessment.assessment, reasons=assessment.reasons,
                                    mime_type=assessment.mime_type, browser_check=check)
        servable = source.status == "available" and playback.assessment in ("likely", "uncertain")
        return Video(id=row["id"], file_name=path.name, folder_id=located[0].id if located else None,
                     path=located[1] if located else None, size_bytes=row["size_bytes"],
                     modified_at=_iso(row["mtime_ns"]), registered_at=row["added_at"], probe=probe, media=media,
                     source=source, playback=playback, analysis=self._eligibility(probe, source, playback),
                     media_url=f"/api/videos/{row['id']}/media" if servable else None)

    @staticmethod
    def _eligibility(probe: ProbeStatus, source: SourceState, playback: PlaybackInfo) -> Eligibility:
        reasons = []
        if source.status != "available":
            reasons.append(source.message or "the video file is unavailable")
        if probe.state == "pending":
            reasons.append("Video details are still being read.")
        elif probe.state == "failed":
            reasons.append(probe.message or "The video could not be read.")
        elif playback.assessment == "unsupported":
            reasons.extend(playback.reasons)
        elif playback.browser_check is None:
            reasons.append("Play the video in this browser to confirm it can be reviewed.")
        elif not playback.browser_check.playable:
            reasons.append("This browser could not play the video, so it cannot be reviewed here.")
        return Eligibility(eligible=not reasons, reasons=reasons)

    def source_for_analysis(self, video_id: str) -> tuple[Video, Path, Optional[VideoInfo]]:
        """The current video state, its real path and the probed metadata (None until probed)."""
        row = self._row(video_id)
        info = VideoInfo.model_validate_json(row["info_json"]) if row["probe_state"] == "ready" else None
        return self._video(row), Path(row["path"]), info

    def media(self, video_id: str) -> tuple[Path, os.stat_result, str]:
        """Path, verified stat and MIME type of a video that may be loaded for playback."""
        row = self._row(video_id)
        video = self._video(row)
        if video.media_url is None:
            reason = (video.source.message if video.source.status != "available"
                      else "the video is not ready for playback" if video.probe.state != "ready"
                      else "; ".join(video.playback.reasons) or "the video cannot be played in a browser")
            raise VideoUnavailable(reason)
        source, info = self._source(row)
        if info is None:
            raise VideoUnavailable(source.message or "the video file is unavailable")
        return Path(row["path"]), info, video.playback.mime_type or "application/octet-stream"

    def record_browser_check(self, video_id: str, playable: bool, detail: Optional[str],
                             user_agent: Optional[str], loaded_sha256: Optional[str] = None) -> Video:
        """Store the latest in-browser load result, bound to the probed file contents. ``loaded_sha256``
        names the contents the browser loaded; a result for other contents is refused."""
        row = self._row(video_id)
        video = self._video(row)
        if video.media_url is None or video.media is None:
            raise VideoUnavailable("only a readable, available video with a playable format can be checked")
        if loaded_sha256 is not None and loaded_sha256 != video.media.sha256:
            raise VideoUnavailable("the video changed after it was loaded; check it again")
        check = {"playable": playable, "checked_at": _now(), "user_agent": (user_agent or None) and user_agent[:300],
                 "detail": detail, "sha256": video.media.sha256}
        with self.database.connect() as conn:
            conn.execute("UPDATE videos SET playback_check_json = ? WHERE id = ? AND sha256 = ?",
                         (json.dumps(check), video_id, video.media.sha256))
        return self.get(video_id)
