"""A person's corrections to event proposals: adjusted start and end times and a plain-text description.

Edits belong to ``(analysis_id, event_id)`` like review decisions, and are validated against the proposals
stored with that analysis's result. They never change the proposal, its evidence, the analyser's status or
the review decision; the original times stay in the timeline and packages. Times are analysis times in
seconds, stored exactly as sent (no rounding), and must satisfy ``0 <= t_start < t_end <= duration``, where
the duration is the analysed video's (the timeline's ``video.duration_s``, or the end of the analysed span
when the container declares none). A description is stored as written; an empty one means none.

Changes use the same compare-and-set revisions as decisions: a change names the revision it was based on and
is refused when another change was saved since. Clearing keeps the row, so revisions never go back.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Callable, Optional

from spectra.api.reviews import RESULT_STATES, ReviewNotFound, ReviewRefused
from spectra.api.schemas import EventEdit, EventEdits
from spectra.api.store import Database

DurationLookup = Callable[[str], Optional[float]]


class EditInvalid(ValueError):
    """The edit is well-formed but not acceptable; the message is suitable for a user."""


class EditConflict(Exception):
    def __init__(self, current: EventEdit) -> None:
        super().__init__("this event was edited elsewhere; the latest edit is shown")
        self.current = current


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _edit(row: sqlite3.Row) -> EventEdit:
    return EventEdit(event_id=row["event_id"], t_start=row["t_start"], t_end=row["t_end"],
                     description=row["description"], revision=row["revision"], updated_at=row["updated_at"])


class EditService:
    def __init__(self, database: Database, duration: DurationLookup) -> None:
        self.database = database
        self.duration = duration

    def list(self, analysis_id: str) -> EventEdits:
        """Recorded edits of an analysis; proposals without one keep their proposed times and have no description."""
        with self.database.connect() as conn:
            if conn.execute("SELECT 1 FROM analyses WHERE id = ?", (analysis_id,)).fetchone() is None:
                raise ReviewNotFound("analysis not found")
            rows = conn.execute("SELECT * FROM event_edits WHERE analysis_id = ? ORDER BY event_id",
                                (analysis_id,)).fetchall()
        return EventEdits(analysis_id=analysis_id, edits=[_edit(r) for r in rows])

    @staticmethod
    def _proposals(conn: sqlite3.Connection, analysis_id: str) -> set[str]:
        analysis = conn.execute("SELECT state, result_json FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        if analysis is None:
            raise ReviewNotFound("analysis not found")
        if analysis["state"] not in RESULT_STATES or not analysis["result_json"]:
            raise ReviewRefused("this analysis has no event proposals to edit")
        return {e["event_id"] for e in json.loads(analysis["result_json"])["events"]}

    def set(self, analysis_id: str, event_id: str, t_start: Optional[float], t_end: Optional[float],
            description: Optional[str], expected_revision: int) -> EventEdit:
        description = description or None
        with self.database.connect() as conn:
            if event_id not in self._proposals(conn, analysis_id):
                raise ReviewNotFound("this analysis has no such event proposal")
        if (t_start is None) != (t_end is None):
            raise EditInvalid("give both a start and an end time, or neither to keep the proposed times")
        if t_start is not None:
            if t_start < 0:
                raise EditInvalid("the start time cannot be before the start of the video")
            if t_start >= t_end:
                raise EditInvalid("the start time must be before the end time")
            # Read before the write transaction: an evidence rebuild writes to the database itself.
            duration = self.duration(analysis_id)
            if duration is not None and t_end > duration:
                raise EditInvalid(f"the end time cannot be after the end of the video ({duration:g} s)")
        with self.database.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")  # the revision check and the write are atomic
            if event_id not in self._proposals(conn, analysis_id):
                raise ReviewNotFound("this analysis has no such event proposal")
            row = conn.execute("SELECT * FROM event_edits WHERE analysis_id = ? AND event_id = ?",
                               (analysis_id, event_id)).fetchone()
            current = _edit(row) if row else EventEdit(event_id=event_id, revision=0)
            if (current.t_start, current.t_end, current.description) == (t_start, t_end, description):
                return current  # already stored; no record for a proposal that was never edited
            if expected_revision != current.revision:
                raise EditConflict(current)
            updated = EventEdit(event_id=event_id, t_start=t_start, t_end=t_end, description=description,
                                revision=current.revision + 1, updated_at=_now())
            conn.execute(
                "INSERT INTO event_edits (analysis_id, event_id, t_start, t_end, description, revision, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (analysis_id, event_id) DO UPDATE SET "
                "t_start = excluded.t_start, t_end = excluded.t_end, description = excluded.description, "
                "revision = excluded.revision, updated_at = excluded.updated_at",
                (analysis_id, event_id, t_start, t_end, description, updated.revision, updated.updated_at))
            return updated
