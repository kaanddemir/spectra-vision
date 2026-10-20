import sqlite3

import pytest

from spectra.api.store import SCHEMA, SCHEMA_VERSION, Database, DatabaseError

SHA = "0" * 64
SQL = "\n".join(line for line in SCHEMA.splitlines() if not line.startswith("--"))  # the schema without comments
ALL_TABLES = {"videos", "analyses", "library_folders", "review_decisions", "event_edits"}


def tables(database):
    with database.connect() as conn:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def columns(database, table):
    with database.connect() as conn:
        return [row["name"] for row in conn.execute(f"PRAGMA table_info({table})")]


def add_video(conn, video_id="v", path="/a.mp4"):
    conn.execute("INSERT INTO videos (id, path, size_bytes, mtime_ns, added_at) VALUES (?, ?, 1, 1, 'x')",
                 (video_id, path))


def add_analysis(conn, analysis_id, state, video_id="v"):
    conn.execute("INSERT INTO analyses (id, video_id, source_sha256, state, created_at) VALUES (?, ?, ?, ?, 'x')",
                 (analysis_id, video_id, SHA, state))


def test_fresh_database_starts_at_the_current_schema(tmp_path):
    database = Database(tmp_path / "state.sqlite3")
    assert SCHEMA_VERSION == 1
    assert database.initialize() == 1
    assert database.initialize() == 1
    assert database.check() == 1
    assert tables(database) == ALL_TABLES
    assert columns(database, "library_folders") == ["id", "path", "added_at"]
    with database.connect() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        indexes = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL")}
    assert indexes == {"analyses_by_video", "analyses_by_state", "one_active_analysis_per_video"}


def test_schema_contains_every_column(tmp_path):
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    assert columns(database, "videos") == [
        "id", "path", "size_bytes", "mtime_ns", "sha256", "info_json", "added_at", "inode", "probe_state",
        "probe_token", "probe_error", "probe_message", "probed_at", "audio_codecs_json", "playback_check_json"]
    assert columns(database, "analyses") == [
        "id", "video_id", "source_sha256", "state", "message", "perception_key", "analysis_key", "build_key",
        "created_at", "started_at", "finished_at", "profile_json", "stage", "frames_analysed", "media_time_s",
        "fraction", "cancel_requested", "stop_reason", "cache_hit", "proposals", "notes_json", "revision",
        "result_json", "evidence_state", "evidence_message"]
    assert columns(database, "review_decisions") == ["analysis_id", "event_id", "decision", "revision", "updated_at"]
    assert columns(database, "event_edits") == ["analysis_id", "event_id", "t_start", "t_end", "description",
                                                "revision", "updated_at"]


def test_defaults_apply_to_new_rows(tmp_path):
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    with database.connect() as conn:
        add_video(conn)
        add_analysis(conn, "a", "completed")
        video = conn.execute("SELECT probe_state, inode FROM videos").fetchone()
        analysis = conn.execute("SELECT frames_analysed, cancel_requested, revision, evidence_state FROM analyses").fetchone()
    assert tuple(video) == ("pending", None)
    assert tuple(analysis) == (0, 0, 0, None)


@pytest.mark.parametrize("statement", [
    "INSERT INTO videos (id, path, size_bytes, mtime_ns, added_at) VALUES ('w', '/a.mp4', 1, 1, 'x')",
    "INSERT INTO videos (id, path, size_bytes, mtime_ns, added_at) VALUES ('w', '/b.mp4', -1, 1, 'x')",
    "UPDATE videos SET sha256 = 'short'",
    "UPDATE videos SET probe_state = 'unknown'",
    "UPDATE analyses SET source_sha256 = 'short'",
    "UPDATE analyses SET evidence_state = 'done'",
    "INSERT INTO analyses (id, video_id, source_sha256, state, created_at) VALUES ('b', 'missing', '" + SHA + "', 'queued', 'x')",
    "DELETE FROM videos",
    "DELETE FROM analyses",
    "INSERT INTO review_decisions VALUES ('missing', '0123456789abcdef', 'accepted', 1, 'x')",
    "INSERT INTO review_decisions VALUES ('a', 'short', 'accepted', 1, 'x')",
    "INSERT INTO review_decisions VALUES ('a', '0123456789abcdef', 'confirmed', 1, 'x')",
    "INSERT INTO review_decisions VALUES ('a', '0123456789abcdef', 'accepted', 0, 'x')",
    "INSERT INTO review_decisions VALUES ('a', 'fedcba9876543210', 'rejected', 1, 'x')",
    "INSERT INTO event_edits VALUES ('missing', '0123456789abcdef', 1, 2, NULL, 1, 'x')",
    "INSERT INTO event_edits VALUES ('a', '0123456789abcdef', 1, NULL, NULL, 1, 'x')",
    "INSERT INTO event_edits VALUES ('a', '0123456789abcdef', 2, 1, NULL, 1, 'x')",
    "INSERT INTO event_edits VALUES ('a', '0123456789abcdef', -1, 1, NULL, 1, 'x')",
    "INSERT INTO event_edits VALUES ('a', '0123456789abcdef', NULL, NULL, '', 1, 'x')",
    "INSERT INTO event_edits VALUES ('a', '0123456789abcdef', NULL, NULL, 'note', 0, 'x')",
])
def test_constraints_are_enforced(tmp_path, statement):
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    with database.connect() as conn:
        add_video(conn)
        add_analysis(conn, "a", "completed")
        conn.execute("INSERT INTO review_decisions VALUES ('a', 'fedcba9876543210', 'accepted', 1, 'x')")
    with pytest.raises(sqlite3.IntegrityError):
        with database.connect() as conn:
            conn.execute(statement)


def test_one_active_analysis_per_video(tmp_path):
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    with database.connect() as conn:
        add_video(conn)
        for state in ("completed", "cancelled", "queued"):
            add_analysis(conn, state, state)
    with pytest.raises(sqlite3.IntegrityError):
        with database.connect() as conn:
            add_analysis(conn, "second", "running")


def test_failed_unit_of_work_is_rolled_back(tmp_path):
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    with pytest.raises(RuntimeError):
        with database.connect() as conn:
            add_video(conn)
            raise RuntimeError("interrupted")
    with database.connect() as conn:
        assert conn.execute("SELECT count(*) FROM videos").fetchone()[0] == 0


def test_records_persist_across_restart(tmp_path):
    path = tmp_path / "state.sqlite3"
    first = Database(path)
    first.initialize()
    with first.connect() as conn:
        add_video(conn)
        add_analysis(conn, "a", "completed")
    second = Database(path)
    assert second.initialize() == SCHEMA_VERSION
    with second.connect() as conn:
        assert conn.execute("SELECT count(*) FROM videos").fetchone()[0] == 1
        assert conn.execute("SELECT state FROM analyses").fetchone()[0] == "completed"


@pytest.mark.parametrize("version", [2, 3, 4])
def test_other_schema_versions_are_refused_unchanged(tmp_path, version):
    path = tmp_path / "state.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE videos (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO videos VALUES ('kept')")
        conn.execute(f"PRAGMA user_version = {version}")
    before = path.read_bytes()
    with pytest.raises(DatabaseError, match="not supported"):
        Database(path).initialize()
    with pytest.raises(DatabaseError, match="not supported"):
        Database(path).check()
    assert path.read_bytes() == before


def test_unversioned_database_with_tables_is_refused_unchanged(tmp_path):
    path = tmp_path / "state.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE other (id TEXT)")
    before = path.read_bytes()
    with pytest.raises(DatabaseError, match="already contains tables"):
        Database(path).initialize()
    assert path.read_bytes() == before
    assert "videos" not in tables(Database(path))


def test_library_folder_constraints(tmp_path):
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    with database.connect() as conn:
        conn.execute("INSERT INTO library_folders VALUES ('0123456789abcdef', '/videos', 'x')")
    for statement in ("INSERT INTO library_folders VALUES ('fedcba9876543210', '/videos', 'x')",
                      "INSERT INTO library_folders VALUES ('short', '/other', 'x')",
                      "INSERT INTO library_folders VALUES ('fedcba9876543210', NULL, 'x')",
                      "INSERT INTO library_folders VALUES ('0123456789abcdef', '/other', 'x')"):
        with pytest.raises(sqlite3.IntegrityError):
            with database.connect() as conn:
                conn.execute(statement)


def schema_without(name):
    """The schema with one table or index statement removed."""
    statements = [s for s in SQL.split(";") if s.strip()]
    kept = [s for s in statements if f" {name} " not in s.split("(")[0] + " "]
    assert len(kept) == len(statements) - 1
    return ";".join(kept) + ";"


@pytest.mark.parametrize("name", ["library_folders", "event_edits", "one_active_analysis_per_video"])
def test_version_1_database_missing_a_table_or_index_is_refused_unchanged(tmp_path, name):
    path = tmp_path / "state.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript(schema_without(name) + "PRAGMA user_version = 1;")
    conn.close()
    before = path.read_bytes()
    with pytest.raises(DatabaseError, match=f"missing {name}"):
        Database(path).initialize()
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.sqlite3"]  # nothing was backed up or added


@pytest.mark.parametrize("old, new", [
    ("description TEXT CHECK (description IS NULL OR length(description) BETWEEN 1 AND 2000)", "description TEXT"),
    ("WHERE state IN ('queued', 'running')", "WHERE state = 'queued'"),
    ("    evidence_message TEXT\n", "    evidence_message TEXT,\n    extra TEXT\n"),
])
def test_version_1_database_with_a_different_definition_is_refused_unchanged(tmp_path, old, new):
    assert old in SCHEMA
    path = tmp_path / "state.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA.replace(old, new) + "PRAGMA user_version = 1;")
    conn.close()
    before = path.read_bytes()
    with pytest.raises(DatabaseError, match="does not match the schema"):
        Database(path).initialize()
    assert path.read_bytes() == before


def test_existing_version_1_database_written_elsewhere_is_accepted_without_changes(tmp_path):
    """Formatting of the stored definitions does not matter; their content does."""
    path = tmp_path / "state.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript(" ".join(SQL.split()).replace(" (", "(") + "PRAGMA user_version = 1;")
    conn.close()
    before = path.read_bytes()
    assert Database(path).initialize() == 1
    assert path.read_bytes() == before


def test_review_and_edit_records_persist_across_restart(tmp_path):
    path = tmp_path / "state.sqlite3"
    first = Database(path)
    first.initialize()
    with first.connect() as conn:
        add_video(conn)
        add_analysis(conn, "a", "completed")
        conn.execute("INSERT INTO review_decisions VALUES ('a', '0123456789abcdef', 'rejected', 3, 't1')")
        conn.execute("INSERT INTO event_edits VALUES ('a', '0123456789abcdef', 1.5, 2.5, 'note', 2, 't2')")
    second = Database(path)
    assert second.initialize() == 1
    with second.connect() as conn:
        assert tuple(conn.execute("SELECT * FROM review_decisions").fetchone()) == (
            "a", "0123456789abcdef", "rejected", 3, "t1")
        assert tuple(conn.execute("SELECT * FROM event_edits").fetchone()) == (
            "a", "0123456789abcdef", 1.5, 2.5, "note", 2, "t2")


def test_fresh_database_passes_sqlite_integrity_checks(tmp_path):
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    with database.connect() as conn:
        add_video(conn)
        add_analysis(conn, "a", "completed")
        conn.execute("INSERT INTO library_folders VALUES ('0123456789abcdef', '/videos', 'x')")
        conn.execute("INSERT INTO review_decisions VALUES ('a', '0123456789abcdef', 'accepted', 1, 'x')")
        conn.execute("INSERT INTO event_edits VALUES ('a', '0123456789abcdef', NULL, NULL, 'note', 1, 'x')")
        assert [tuple(r) for r in conn.execute("PRAGMA integrity_check")] == [("ok",)]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
