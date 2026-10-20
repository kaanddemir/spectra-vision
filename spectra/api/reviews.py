"""Human review decisions on event proposals.

A decision belongs to one proposal of one analysis, ``(analysis_id, event_id)``; the event id must be one of
the proposals stored with that analysis's result, so a decision can never attach to another analysis, and a
new analysis of the same video starts with no decisions even when its proposals have the same ids. Decisions
are a person's judgement of a machine proposal: they never change the analysis row, its stored result, its
evidence or the analyser's own status, and accepting a proposal does not establish that anything happened.

Every change increases the proposal's ``revision``. A change names the revision it was based on and is
refused when another change was saved since (compare-and-set in one write transaction), so a late request
never overwrites a newer decision. Asking for the decision that is already stored changes nothing.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from spectra.api.schemas import ReviewDecision, ReviewDecisionName, ReviewDecisions
from spectra.api.store import Database

RESULT_STATES = ("completed", "incomplete")


class ReviewNotFound(LookupError):
    """The analysis or the proposal does not exist; the message names which."""


class ReviewRefused(Exception):
    """The analysis has no proposals to review."""


class ReviewConflict(Exception):
    """Another change was saved since the revision the request was based on."""

    def __init__(self, current: ReviewDecision) -> None:
        super().__init__("this decision was changed elsewhere; the latest decision is shown")
        self.current = current


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _decision(row: sqlite3.Row) -> ReviewDecision:
    return ReviewDecision(event_id=row["event_id"], decision=row["decision"], revision=row["revision"],
                          updated_at=row["updated_at"])


class ReviewService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def list(self, analysis_id: str) -> ReviewDecisions:
        """Recorded decisions of an analysis; proposals without one are not reviewed."""
        with self.database.connect() as conn:
            if conn.execute("SELECT 1 FROM analyses WHERE id = ?", (analysis_id,)).fetchone() is None:
                raise ReviewNotFound("analysis not found")
            rows = conn.execute("SELECT * FROM review_decisions WHERE analysis_id = ? ORDER BY event_id",
                                (analysis_id,)).fetchall()
        return ReviewDecisions(analysis_id=analysis_id, decisions=[_decision(r) for r in rows])

    def set(self, analysis_id: str, event_id: str, decision: ReviewDecisionName,
            expected_revision: int) -> ReviewDecision:
        """Record a decision for one proposal, based on ``expected_revision`` (0: none recorded yet)."""
        with self.database.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")  # one writer: the revision check and the write are atomic
            analysis = conn.execute("SELECT state, result_json FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
            if analysis is None:
                raise ReviewNotFound("analysis not found")
            if analysis["state"] not in RESULT_STATES or not analysis["result_json"]:
                raise ReviewRefused("this analysis has no event proposals to review")
            proposals = {e["event_id"] for e in json.loads(analysis["result_json"])["events"]}
            if event_id not in proposals:
                raise ReviewNotFound("this analysis has no such event proposal")
            row = conn.execute("SELECT * FROM review_decisions WHERE analysis_id = ? AND event_id = ?",
                               (analysis_id, event_id)).fetchone()
            current = _decision(row) if row else ReviewDecision(event_id=event_id, decision="not_reviewed",
                                                                 revision=0, updated_at=None)
            if decision == current.decision:
                return current  # already stored: nothing to change, and no record for an unreviewed proposal
            if expected_revision != current.revision:
                raise ReviewConflict(current)
            updated = ReviewDecision(event_id=event_id, decision=decision, revision=current.revision + 1,
                                     updated_at=_now())
            conn.execute(
                "INSERT INTO review_decisions (analysis_id, event_id, decision, revision, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (analysis_id, event_id) DO UPDATE SET "
                "decision = excluded.decision, revision = excluded.revision, updated_at = excluded.updated_at",
                (analysis_id, event_id, updated.decision, updated.revision, updated.updated_at))
            return updated
