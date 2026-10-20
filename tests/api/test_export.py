"""JSON export: schema, identity, machine output kept apart from human review, evidence completeness, refusals,
download headers, privacy and that exporting changes nothing."""
import json
import shutil

import pytest

from spectra.api.evidence import result_record
from spectra.api.export import DISCLAIMER, FORMAT, FORMAT_VERSION, file_name
from spectra.evidence.build import build
from tests.api.test_evidence import analysed, app, library  # noqa: F401 (fixtures)
from tests.evidence import synth

ANALYSIS, VIDEO = f"{7:032x}", f"{8:032x}"


def three_types():
    return synth.result([synth.approach(12.0, 14.5), synth.cut_in(30.0, 31.2, entry_from=28.0),
                         synth.contact(44.0, 44.8)],
                        analyzers=(synth.APPROACH, synth.CUT_IN, synth.IMPACT, synth.CONTACT),
                        times=[round(0.1 * i, 3) for i in range(625)], info=synth.video(62.5))


def store(client, tmp_path, result=None, state="completed", evidence_state="ready", analysis_id=ANALYSIS):
    """A finished analysis with its evidence written by the real builder and store, as after a run."""
    result = result or three_types()
    timeline, packages = build(result)
    client.app.state.evidence.store.write(timeline, packages)
    with client.app.state.database.connect() as conn:
        if not conn.execute("SELECT 1 FROM videos WHERE id = ?", (VIDEO,)).fetchone():
            conn.execute("INSERT INTO videos (id, path, size_bytes, mtime_ns, added_at) VALUES (?, ?, 1, 1, 'x')",
                         (VIDEO, str(tmp_path / "library" / "secret folder" / "drive.mp4")))
        conn.execute(
            "INSERT INTO analyses (id, video_id, source_sha256, state, created_at, started_at, finished_at, "
            "perception_key, analysis_key, build_key, result_json, evidence_state, profile_json, notes_json) "
            "VALUES (?, ?, ?, ?, '2026-10-18T10:00:00+00:00', '2026-10-18T10:00:01+00:00', "
            "'2026-10-18T10:02:00+00:00', ?, ?, ?, ?, ?, ?, ?)",
            (analysis_id, VIDEO, result.perception.video.sha256, state, result.perception_key, result.analysis_key,
             timeline.build_key, result_record(result), evidence_state,
             json.dumps({"config": {"interval_s": 0.1}, "tracking": True}), json.dumps(["a note from the run"])))
    return timeline, packages


def export(client, analysis_id=ANALYSIS, **kwargs):
    return client.get(f"/api/analyses/{analysis_id}/export", **kwargs)


def strict_json(response):
    def refuse(value):
        raise ValueError(f"non-standard JSON value {value}")
    return json.loads(response.content.decode("utf-8"), parse_constant=refuse)


def state_of(client, tmp_path):
    """Every database row and every evidence file, to show an export changes nothing."""
    with client.app.state.database.connect() as conn:
        rows = {t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY 1, 2")]
                for t in ("videos", "analyses", "review_decisions", "event_edits")}
    evidence = tmp_path / "var" / "evidence"
    files = sorted((str(p.relative_to(evidence)), p.read_bytes()) for p in evidence.rglob("*") if p.is_file())
    return rows, files


@pytest.fixture
def client(make_client):
    return make_client()


# Document --------------------------------------------------------------------------------------

def test_download_headers_and_schema(client, tmp_path):
    timeline, _ = store(client, tmp_path)
    response = export(client)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert response.headers["content-disposition"] == \
        f'attachment; filename="{file_name(timeline.video.file_name, ANALYSIS)}"'
    assert response.headers["content-disposition"].endswith(f'_{ANALYSIS[:8]}_spectra.json"')
    assert response.headers["cache-control"] == "no-store"
    doc = strict_json(response)
    assert list(doc) == ["format", "format_version", "exported_at", "application", "disclaimer", "units", "video",
                         "analysis", "events"]
    assert (doc["format"], doc["format_version"]) == (FORMAT, FORMAT_VERSION) == ("spectra.analysis-export", "1.0")
    assert doc["disclaimer"] == DISCLAIMER and "does not confirm that an incident occurred" in DISCLAIMER
    assert "seconds" in doc["units"]["time"]


def test_video_and_analysis_identity(client, tmp_path):
    timeline, _ = store(client, tmp_path)
    doc = strict_json(export(client))
    assert doc["video"]["file_name"] == timeline.video.file_name
    assert doc["video"]["sha256"] == timeline.video.sha256 and doc["video"]["duration_s"] == 62.5
    analysis = doc["analysis"]
    assert analysis["id"] == ANALYSIS and analysis["state"] == "completed"
    assert (analysis["analysis_key"], analysis["perception_key"], analysis["evidence_build_key"]) == \
        (timeline.analysis_key, timeline.perception_key, timeline.build_key)
    assert analysis["processing"] == {"config": {"interval_s": 0.1}, "tracking": True}
    assert analysis["notes"] == ["a note from the run"]
    assert [a["name"] for a in analysis["analyzers"]] == [a.name for a in timeline.analyzers]


def test_every_proposal_with_its_unchanged_proposal_and_evidence(client, tmp_path):
    timeline, packages = store(client, tmp_path)
    doc = strict_json(export(client))
    assert [e["event_id"] for e in doc["events"]] == [e.event_id for e in timeline.entries]
    assert {e["machine"]["proposal"]["type"] for e in doc["events"]} == {"rapid_approach", "cut_in", "impact"}
    by_id = {p.proposal.event_id: p for p in packages}
    for event in doc["events"]:
        package = by_id[event["event_id"]]
        assert event["machine"]["proposal"] == package.proposal.model_dump(mode="json")
        evidence = event["machine"]["evidence"]
        whole = package.model_dump(mode="json")
        for key in ("tracks", "lanes", "gaps", "warnings", "unresolved", "related", "notes", "review", "analyzer",
                    "analysed_start_s", "analysed_end_s"):
            assert evidence[key] == whole[key]
        # Proposal-level evidence only: no per-frame records or overlay drawing data.
        assert not {"box", "valid_until", "boundaries"} & set(json.dumps(evidence).replace('"', " ").split())
        assert len(json.dumps(evidence)) < 50_000
    uncertain = [e for e in doc["events"] if e["machine"]["proposal"]["state"] == "uncertain"]
    assert uncertain and all(e["machine"]["proposal"]["reasons"] for e in uncertain)
    items = [i for e in doc["events"] for i in e["machine"]["proposal"]["evidence"]]
    assert {i["basis"] for i in items} >= {"observed"}


def test_review_work_is_separate_and_never_replaces_detected_times(client, tmp_path):
    timeline, _ = store(client, tmp_path)
    first, second, third = (e.event_id for e in timeline.entries)
    base = f"/api/analyses/{ANALYSIS}"
    note = "Kamyon fren yaptı — 雨の道路 🚗\nsecond line \"quoted\" </script>"
    assert client.put(f"{base}/reviews/{first}", json={"decision": "accepted", "expected_revision": 0}).status_code == 200
    assert client.put(f"{base}/reviews/{second}", json={"decision": "rejected", "expected_revision": 0}).status_code == 200
    assert client.put(f"{base}/edits/{first}", json={"t_start": 11.24, "t_end": 15.0, "description": note,
                                                     "expected_revision": 0}).status_code == 200
    assert client.put(f"{base}/edits/{third}", json={"t_start": None, "t_end": None, "description": "only a note",
                                                     "expected_revision": 0}).status_code == 200
    response = export(client)
    assert note.encode("utf-8") in response.content or json.dumps(note, ensure_ascii=False).encode() in response.content
    events = {e["event_id"]: e for e in strict_json(response)["events"]}
    a, b, c = events[first], events[second], events[third]
    assert a["machine"]["proposal"]["t_start"] == 12.0 or a["machine"]["proposal"]["t_start"] == timeline.entries[0].t_start
    assert (a["review"]["decision"], a["review"]["adjusted_t_start_s"], a["review"]["adjusted_t_end_s"]) == \
        ("accepted", 11.24, 15.0)
    assert a["review"]["description"] == note
    assert (b["review"]["decision"], b["review"]["adjusted_t_start_s"], b["review"]["description"]) == \
        ("rejected", None, None)
    assert (c["review"]["decision"], c["review"]["decision_updated_at"]) == ("not_reviewed", None)
    assert (c["review"]["adjusted_t_start_s"], c["review"]["description"]) == (None, "only a note")
    assert list(a["review"]) == ["decision", "decision_updated_at", "adjusted_t_start_s", "adjusted_t_end_s",
                                 "description", "edit_updated_at"]
    for event in events.values():  # the analyser's own status is untouched by any decision
        assert event["machine"]["proposal"]["state"] in ("supported", "uncertain")


def test_analysis_without_review_work_exports_every_proposal_as_not_reviewed(client, tmp_path):
    store(client, tmp_path)
    events = strict_json(export(client))["events"]
    assert len(events) == 3
    assert all(e["review"] == {"decision": "not_reviewed", "decision_updated_at": None, "adjusted_t_start_s": None,
                               "adjusted_t_end_s": None, "description": None, "edit_updated_at": None}
               for e in events)


def test_zero_proposals_is_a_valid_export(client, tmp_path):
    store(client, tmp_path, result=synth.result([]))
    doc = strict_json(export(client))
    assert doc["events"] == [] and doc["analysis"]["analyzers"]


def test_incomplete_analysis_keeps_its_state(client, tmp_path):
    store(client, tmp_path, state="incomplete")
    assert strict_json(export(client))["analysis"]["state"] == "incomplete"


def test_output_is_deterministic_apart_from_the_export_time(client, tmp_path):
    store(client, tmp_path)
    first, second = strict_json(export(client)), strict_json(export(client))
    first.pop("exported_at"), second.pop("exported_at")
    assert first == second
    assert json.dumps(first) == json.dumps(second)  # same key order as well


# Refusals --------------------------------------------------------------------------------------

def test_unknown_and_malformed_analyses(client):
    assert export(client, f"{99:032x}").status_code == 404
    assert export(client, "not-an-id").status_code == 422


@pytest.mark.parametrize("state", ["running", "queued", "failed", "cancelled", "resource_stopped"])
def test_analyses_without_results_are_refused(client, tmp_path, state):
    store(client, tmp_path, state=state, evidence_state=None)
    response = export(client)
    assert response.status_code == 409 and response.json()["detail"] == \
        "only a completed analysis with results can be exported"


@pytest.mark.parametrize("evidence_state", ["pending", "failed", None])
def test_evidence_that_is_not_ready_is_refused_without_a_rebuild(client, tmp_path, evidence_state):
    store(client, tmp_path, evidence_state=evidence_state)
    before = state_of(client, tmp_path)
    response = export(client)
    assert response.status_code == 409 and "not ready" in response.json()["detail"]
    assert state_of(client, tmp_path) == before


def test_missing_or_tampered_evidence_is_refused_and_left_as_it_is(client, tmp_path):
    timeline, packages = store(client, tmp_path)
    build_dir = tmp_path / "var" / "evidence" / timeline.build_key
    package_file = sorted((build_dir / "packages").iterdir())[0]
    package_file.write_bytes(package_file.read_bytes() + b" ")
    before = state_of(client, tmp_path)
    response = export(client)
    assert response.status_code == 409 and "failed verification" in response.json()["detail"]
    assert state_of(client, tmp_path) == before
    shutil.rmtree(build_dir)
    response = export(client)
    assert response.status_code == 409 and "failed verification" in response.json()["detail"]
    assert not build_dir.exists()  # nothing was rebuilt


def test_foreign_pages_cannot_export(client, tmp_path):
    store(client, tmp_path)
    assert export(client, headers={"origin": "https://evil.example"}).status_code == 403
    assert export(client, headers={"sec-fetch-site": "cross-site"}).status_code == 403
    assert export(client, headers={"host": "evil.example"}).status_code == 400


# Privacy and preservation ----------------------------------------------------------------------

def test_no_local_paths_but_the_reviewers_own_note_is_kept(client, tmp_path):
    timeline, _ = store(client, tmp_path)
    event_id = timeline.entries[0].event_id
    client.put(f"/api/analyses/{ANALYSIS}/edits/{event_id}",
               json={"t_start": None, "t_end": None, "description": "copy saved in /Users/me/Desktop",
                     "expected_revision": 0})
    body = export(client).content.decode("utf-8")
    assert str(tmp_path) not in body and "secret folder" not in body
    assert "copy saved in /Users/me/Desktop" in body


def test_export_changes_nothing(client, tmp_path):
    timeline, _ = store(client, tmp_path)
    client.put(f"/api/analyses/{ANALYSIS}/reviews/{timeline.entries[0].event_id}",
               json={"decision": "accepted", "expected_revision": 0})
    before = state_of(client, tmp_path)
    for _ in range(3):
        assert export(client).status_code == 200
    assert state_of(client, tmp_path) == before


def test_real_analysis_exports_its_proposal(app, tmp_path):  # noqa: F811
    client = app()
    analysis = analysed(client)
    timeline = client.get(f"/api/analyses/{analysis['id']}/timeline").json()
    doc = strict_json(client.get(f"/api/analyses/{analysis['id']}/export"))
    assert [e["event_id"] for e in doc["events"]] == [e["event_id"] for e in timeline["entries"]]
    assert doc["video"]["file_name"] == "drive.mp4" and doc["analysis"]["id"] == analysis["id"]
    assert str(tmp_path) not in json.dumps(doc)


@pytest.mark.parametrize("video, expected", [
    ("drive.mp4", "drive_0123abcd_spectra.json"),
    ("Kamyon yolu 2026-10.mov", "Kamyon_yolu_2026-10_0123abcd_spectra.json"),
    ("../../etc/passwd", "passwd_0123abcd_spectra.json"),
    ('a"b;c.mp4', "a_b_c_0123abcd_spectra.json"),
    ("雨.mp4", "video_0123abcd_spectra.json"),
    (".hidden", "hidden_0123abcd_spectra.json"),
])
def test_file_names_are_safe_and_predictable(video, expected):
    assert file_name(video, "0123abcd" + "0" * 24) == expected
