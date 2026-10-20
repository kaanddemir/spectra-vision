"""One JSON document per analysis: the machine output, its evidence and the reviewer's work, kept apart.

The export reads only what is already persisted and verified: the analysis row, its stored evidence build
(timeline and packages through ``EvidenceStore``, which checks every file against the build manifest), and the
review decisions and edits, all database rows read in one read transaction so they form one snapshot. It never
writes, never rebuilds evidence and never touches the perception cache, models or the source video; when the
stored evidence is missing or fails verification the export is refused with the reason.

Machine output is copied unchanged from the evidence contracts. Human work is in a separate ``review`` object
per event; edited times never replace detected ones.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from spectra.api import API_VERSION
from spectra.api.evidence import RESULT_STATES
from spectra.api.store import Database
from spectra.evidence.store import EvidenceStore

FORMAT = "spectra.analysis-export"
FORMAT_VERSION = "1.0"
DISCLAIMER = ("Events are machine proposals for human review. 'machine' objects are Spectra's unchanged output: "
              "evidence items are marked observed, estimated or context; status 'supported' means only that an "
              "analyser's own conditions held, and rank orders proposals but is not a probability. 'review' "
              "objects hold a person's decisions, adjusted times and notes. An accepted proposal is a reviewer's "
              "judgement; it does not confirm that an incident occurred or assign responsibility. Missing "
              "evidence is not evidence that nothing happened.")
UNITS = {"time": "seconds of analysis time: the video's presentation time minus video.start_s",
         "fields": "names ending in _s, and t, t_start, t_peak and t_end, are times in seconds",
         "size_bytes": "bytes"}
# Absolute local paths (POSIX home or temporary folders, Windows drives) must never leave the computer.
_LOCAL_PATH = re.compile(r"(^|[\s\"'=])(/(Users|home|private|var|tmp|Volumes)/|[A-Za-z]:\\)")


class ExportNotFound(LookupError):
    pass


class ExportUnavailable(Exception):
    """The analysis cannot be exported now; the message is suitable for a user."""


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


class ExportService:
    def __init__(self, database: Database, store: EvidenceStore) -> None:
        self.database = database
        self.store = store

    def _snapshot(self, analysis_id: str):
        with self.database.connect() as conn:
            conn.execute("BEGIN")  # one read snapshot of the analysis, its decisions and its edits
            row = conn.execute("SELECT a.*, v.path AS video_path FROM analyses a JOIN videos v ON v.id = a.video_id "
                               "WHERE a.id = ?", (analysis_id,)).fetchone()
            if row is None:
                raise ExportNotFound(analysis_id)
            decisions = {r["event_id"]: r for r in conn.execute(
                "SELECT * FROM review_decisions WHERE analysis_id = ?", (analysis_id,))}
            edits = {r["event_id"]: r for r in conn.execute(
                "SELECT * FROM event_edits WHERE analysis_id = ?", (analysis_id,))}
        return row, decisions, edits

    def export(self, analysis_id: str) -> tuple[dict, str]:
        """The export document and the video's file name."""
        row, decisions, edits = self._snapshot(analysis_id)
        if row["state"] not in RESULT_STATES or not row["result_json"]:
            raise ExportUnavailable("only a completed analysis with results can be exported")
        if not row["build_key"] or row["evidence_state"] != "ready":
            raise ExportUnavailable("the evidence for this analysis is not ready; open its review to prepare it, "
                                    "then export again")
        lookup = self.store.load(row["build_key"], row["analysis_key"])
        if lookup.timeline is None:
            raise ExportUnavailable(f"the stored evidence failed verification ({lookup.reason}); open the review "
                                    f"to prepare it again, then export")
        timeline = lookup.timeline
        stored = [e["event_id"] for e in json.loads(row["result_json"])["events"]]
        if sorted(stored) != sorted(e.event_id for e in timeline.entries):
            raise ExportUnavailable("the stored evidence does not list this analysis's proposals")

        events = []
        for entry in timeline.entries:  # review order: chronological, as built
            try:
                package = self.store.package(timeline.build_key, entry.event_id)
            except (OSError, ValueError, LookupError) as exc:
                raise ExportUnavailable(f"the evidence for proposal {entry.event_id} failed verification ({exc})") \
                    from None
            if package.analysis_key != timeline.analysis_key or package.video.sha256 != timeline.video.sha256:
                raise ExportUnavailable(f"the evidence for proposal {entry.event_id} belongs to another analysis")
            evidence = package.model_dump(mode="json", exclude={"proposal", "video", "evidence_version", "analysis_key",
                                                                 "perception_key", "build_key"})
            decision, edit = decisions.get(entry.event_id), edits.get(entry.event_id)
            events.append({
                "event_id": entry.event_id,
                "machine": {
                    "title": entry.title,
                    "title_key": entry.title_key,
                    "proposal": package.proposal.model_dump(mode="json"),
                    "evidence": evidence,
                },
                "review": {
                    "decision": decision["decision"] if decision else "not_reviewed",
                    "decision_updated_at": decision["updated_at"] if decision else None,
                    "adjusted_t_start_s": edit["t_start"] if edit else None,
                    "adjusted_t_end_s": edit["t_end"] if edit else None,
                    "description": edit["description"] if edit else None,
                    "edit_updated_at": edit["updated_at"] if edit else None,
                },
            })

        video = timeline.video.model_dump(mode="json")
        document = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "application": {"name": "Spectra", "version": API_VERSION},
            "disclaimer": DISCLAIMER,
            "units": UNITS,
            "video": {k: video[k] for k in ("file_name", "size_bytes", "sha256", "container", "codec", "width",
                                            "height", "rotation", "duration_s", "nominal_fps", "start_s",
                                            "timestamp_source")},
            "analysis": {
                "id": row["id"],
                "state": row["state"],
                "created_at": row["created_at"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "analysis_key": timeline.analysis_key,
                "perception_key": timeline.perception_key,
                "evidence_build_key": timeline.build_key,
                "evidence_version": timeline.evidence_version,
                "processing": json.loads(row["profile_json"]) if row["profile_json"] else None,
                "notes": json.loads(row["notes_json"] or "[]"),
                "analysed_start_s": timeline.analysed_start_s,
                "analysed_end_s": timeline.analysed_end_s,
                "review_config": timeline.config.model_dump(mode="json"),
                "analyzers": [a.model_dump(mode="json") for a in timeline.analyzers],
                "gaps": [g.model_dump(mode="json") for g in timeline.gaps],
                "warnings": [w.model_dump(mode="json") for w in timeline.warnings],
            },
            "events": events,
        }
        # The reviewer's own notes are theirs to write; everything else is checked.
        checked = {**document, "events": [{**e, "review": {**e["review"], "description": None}} for e in events]}
        for text in _strings(checked):
            if row["video_path"] and row["video_path"] in text or _LOCAL_PATH.search(text):
                raise ExportUnavailable("the export would contain a local file path, so it was not created")
        return document, timeline.video.file_name


def render(document: dict) -> bytes:
    """UTF-8 JSON; NaN and Infinity are refused rather than written."""
    return (json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def file_name(video_name: str, analysis_id: str) -> str:
    """``<video stem>_<first 8 of the analysis id>_spectra.json``, restricted to safe characters."""
    base = re.split(r"[/\\]", video_name)[-1]
    stem = base.rsplit(".", 1)[0] if "." in base.strip(".") else base
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:80] or "video"
    return f"{safe}_{analysis_id[:8]}_spectra.json"
