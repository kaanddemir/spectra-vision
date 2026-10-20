"""SQLite application state: one schema, created atomically, and short-lived connections.

The database indexes registered source videos and analysis jobs. Analysis results stay in the
perception cache and evidence store; human review decisions and edits live in their own tables and never
change an analysis row, its stored result or its evidence.

There are no upgrades: a database that is not exactly this schema is refused and left unchanged.
"""
from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE videos (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    mtime_ns INTEGER NOT NULL,
    sha256 TEXT CHECK (sha256 IS NULL OR length(sha256) = 64),
    info_json TEXT,
    added_at TEXT NOT NULL,
    inode INTEGER,
    probe_state TEXT NOT NULL DEFAULT 'pending'
        CHECK (probe_state IN ('pending', 'ready', 'failed')),
    probe_token TEXT,
    probe_error TEXT,
    probe_message TEXT,
    probed_at TEXT,
    audio_codecs_json TEXT,
    playback_check_json TEXT
);
CREATE TABLE analyses (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos (id) ON DELETE RESTRICT,
    source_sha256 TEXT NOT NULL CHECK (length(source_sha256) = 64),
    state TEXT NOT NULL,
    message TEXT,
    perception_key TEXT,
    analysis_key TEXT,
    build_key TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    profile_json TEXT,
    stage TEXT,
    frames_analysed INTEGER NOT NULL DEFAULT 0,
    media_time_s REAL,
    fraction REAL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    stop_reason TEXT,
    cache_hit INTEGER,
    proposals INTEGER,
    notes_json TEXT,
    revision INTEGER NOT NULL DEFAULT 0,
    result_json TEXT,
    evidence_state TEXT
        CHECK (evidence_state IS NULL OR evidence_state IN ('pending', 'ready', 'failed')),
    evidence_message TEXT
);
CREATE INDEX analyses_by_video ON analyses (video_id, created_at);
CREATE INDEX analyses_by_state ON analyses (state);
CREATE UNIQUE INDEX one_active_analysis_per_video ON analyses (video_id)
    WHERE state IN ('queued', 'running');
CREATE TABLE library_folders (
    id TEXT PRIMARY KEY CHECK (length(id) = 16),
    path TEXT NOT NULL UNIQUE,
    added_at TEXT NOT NULL
);
-- Human decisions on event proposals. A row exists only once a person has decided; a proposal without one
-- is not reviewed. Clearing a decision keeps the row as 'not_reviewed', so its revision never goes back.
CREATE TABLE review_decisions (
    analysis_id TEXT NOT NULL REFERENCES analyses (id) ON DELETE RESTRICT,
    event_id TEXT NOT NULL CHECK (length(event_id) = 16),
    decision TEXT NOT NULL CHECK (decision IN ('not_reviewed', 'accepted', 'rejected')),
    revision INTEGER NOT NULL CHECK (revision >= 1),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (analysis_id, event_id)
);
-- A person's corrections to a proposal: times (both or neither; null keeps the proposed times) and a plain-text
-- description. The proposal and its evidence are never changed. Clearing keeps the row, as for decisions.
CREATE TABLE event_edits (
    analysis_id TEXT NOT NULL REFERENCES analyses (id) ON DELETE RESTRICT,
    event_id TEXT NOT NULL CHECK (length(event_id) = 16),
    t_start REAL CHECK (t_start IS NULL OR t_start >= 0),
    t_end REAL,
    description TEXT CHECK (description IS NULL OR length(description) BETWEEN 1 AND 2000),
    revision INTEGER NOT NULL CHECK (revision >= 1),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (analysis_id, event_id),
    CHECK ((t_start IS NULL) = (t_end IS NULL)),
    CHECK (t_start IS NULL OR t_start < t_end)
);
"""


class DatabaseError(RuntimeError):
    """The database cannot be used by this version of the application."""


@cache
def expected_structure() -> dict[str, str]:
    """Every table and index the schema creates, by name, with its definition (whitespace normalized)."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(SCHEMA)
        return _structure(conn)
    finally:
        conn.close()


def _structure(conn: sqlite3.Connection) -> dict[str, str]:
    """Definitions are compared whole, so columns, defaults, constraints, keys and index conditions all count."""
    rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type IN ('table', 'index') AND sql IS NOT NULL")
    return {name: re.sub(r"\s*([(),])\s*", r"\1", " ".join(sql.split())) for name, sql in rows}


class Database:
    """Opens one connection per unit of work, so callers on different threads never share one."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection; commit on success, roll back on any exception."""
        conn = sqlite3.connect(self.path, timeout=5.0)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> int:
        """Create the schema atomically in an empty database, then check it and return the version.

        A database that already has tables is never changed: it is used only if it is exactly this schema.
        """
        with self.connect() as conn:
            if conn.execute("PRAGMA user_version").fetchone()[0] == 0:
                if conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]:
                    raise DatabaseError("database has no schema version but already contains tables; "
                                        "the file was left unchanged")
                conn.execute("PRAGMA journal_mode = WAL")
                conn.executescript(f"BEGIN IMMEDIATE;\n{SCHEMA}\nPRAGMA user_version = {SCHEMA_VERSION};\nCOMMIT;")
        return self.check()

    def check(self) -> int:
        """Return the schema version, or raise if the database is unreadable or not exactly this schema."""
        with self.connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            structure = _structure(conn) if version == SCHEMA_VERSION else None
        if version != SCHEMA_VERSION:
            raise DatabaseError(f"database schema version {version} is not supported "
                                f"(expected {SCHEMA_VERSION}); the file was left unchanged")
        expected = expected_structure()
        missing = sorted(name for name in expected if name not in structure)
        if missing:
            raise DatabaseError(f"database is missing {', '.join(missing)}; the file was left unchanged")
        different = sorted(name for name in expected if structure[name] != expected[name])
        if different:
            raise DatabaseError(f"database does not match the schema: {', '.join(different)}; "
                                f"the file was left unchanged")
        return version
