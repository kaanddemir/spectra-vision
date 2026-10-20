"""Event edits: adjusted times and descriptions with validation, compare-and-set revisions, persistence,
independence from review decisions and preservation of proposals and evidence."""
import threading

import pytest

from spectra.api.edits import EditConflict, EditService
from tests.api.conftest import shutdown
from tests.api.test_evidence import analysed, app, library  # noqa: F401 (fixtures)
from tests.api.test_reviews import A1, A2, E1, E2, E3, EMPTY, FAILED, RUNNING, analysis_rows, decisions, put, seed

DURATION = 62.5


@pytest.fixture
def client(make_client):
    client = make_client()
    seed(client.app.state.database)
    client.app.state.edits.duration = lambda analysis_id: DURATION  # seeded analyses have no evidence
    return client


def edit(client, analysis_id, event_id, expected_revision, t_start=None, t_end=None, description=None, **kwargs):
    return client.put(f"/api/analyses/{analysis_id}/edits/{event_id}",
                      json={"t_start": t_start, "t_end": t_end, "description": description,
                            "expected_revision": expected_revision}, **kwargs)


def edits(client, analysis_id):
    response = client.get(f"/api/analyses/{analysis_id}/edits")
    assert response.status_code == 200
    return {e["event_id"]: (e["t_start"], e["t_end"], e["description"], e["revision"]) for e in response.json()["edits"]}


# Edits -------------------------------------------------------------------------------------------

def test_proposals_start_unedited(client):
    assert client.get(f"/api/analyses/{A1}/edits").json() == {"analysis_id": A1, "edits": []}


def test_create_update_and_clear(client):
    first = edit(client, A1, E1, 0, 11.2, 15.0, "Truck brakes hard")
    assert first.status_code == 200
    assert {k: first.json()[k] for k in ("t_start", "t_end", "description", "revision")} == \
        {"t_start": 11.2, "t_end": 15.0, "description": "Truck brakes hard", "revision": 1}
    assert edit(client, A1, E1, 1, 11.2, 15.0).json()["revision"] == 2  # description cleared, times kept
    assert edits(client, A1) == {E1: (11.2, 15.0, None, 2)}
    assert edit(client, A1, E1, 2, description="Only a note").json()["revision"] == 3  # proposed times restored
    assert edits(client, A1) == {E1: (None, None, "Only a note", 3)}
    assert edit(client, A1, E1, 3, description="").json()["revision"] == 4  # an empty description is none
    assert edits(client, A1) == {E1: (None, None, None, 4)}


def test_times_and_text_are_stored_exactly(client):
    text = "  Two lines,\nkept as written <b>not html</b> **not markdown**  "
    edit(client, A1, E1, 0, 0.040000001, 62.5, text)
    assert edits(client, A1) == {E1: (0.040000001, 62.5, text, 1)}
    edit(client, A1, E2, 0, 0.0, 0.001)
    assert edits(client, A1)[E2][:2] == (0.0, 0.001)


def test_repeating_the_stored_edit_changes_nothing(client):
    first = edit(client, A1, E1, 0, 1.0, 2.0, "x").json()
    assert edit(client, A1, E1, 0, 1.0, 2.0, "x").json() == first
    restored = edit(client, A1, E2, 0)
    assert restored.json()["revision"] == 0 and E2 not in edits(client, A1)  # no record for an unedited proposal


def test_stale_edit_is_refused_with_the_latest_edit(client):
    edit(client, A1, E1, 0, 1.0, 2.0)
    edit(client, A1, E1, 1, 1.5, 2.0)
    stale = edit(client, A1, E1, 1, 3.0, 4.0)
    assert stale.status_code == 409
    detail = stale.json()["detail"]
    assert "edited elsewhere" in detail["message"] and detail["current"]["t_start"] == 1.5
    assert edits(client, A1) == {E1: (1.5, 2.0, None, 2)}


def test_concurrent_edits_from_the_same_revision_save_exactly_one(client):
    service: EditService = client.app.state.edits
    barrier = threading.Barrier(6)
    outcomes = []

    def change(i):
        barrier.wait()
        try:
            outcomes.append(service.set(A1, E1, float(i), float(i) + 1, None, 0).revision)
        except EditConflict as exc:
            outcomes.append(("conflict", exc.current.revision))

    threads = [threading.Thread(target=change, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count(1) == 1 and outcomes.count(("conflict", 1)) == 5
    assert edits(client, A1)[E1][3] == 1


# Validation --------------------------------------------------------------------------------------

@pytest.mark.parametrize("t_start, t_end, message", [
    (5.0, 5.0, "start time must be before the end time"),
    (6.0, 5.0, "start time must be before the end time"),
    (-0.1, 5.0, "before the start of the video"),
    (1.0, 62.6, "after the end of the video (62.5 s)"),
    (1.0, None, "both a start and an end time"),
    (None, 2.0, "both a start and an end time"),
])
def test_invalid_times_are_explained(client, t_start, t_end, message):
    response = edit(client, A1, E1, 0, t_start, t_end)
    assert response.status_code == 422 and message in response.json()["detail"]["message"]
    assert edits(client, A1) == {}


@pytest.mark.parametrize("body", [
    {"t_start": 1, "t_end": 2, "description": None},
    {"t_start": "1:00", "t_end": 2, "description": None, "expected_revision": 0},
    {"t_start": "NaN", "t_end": 2, "description": None, "expected_revision": 0},
    {"t_start": 1, "t_end": "Infinity", "description": None, "expected_revision": 0},
    {"t_start": None, "t_end": None, "description": 5, "expected_revision": 0},
    {"t_start": None, "t_end": None, "description": "x" * 2001, "expected_revision": 0},
    {"t_start": None, "t_end": None, "description": "a\x00b", "expected_revision": 0},
    {"t_start": None, "t_end": None, "description": None, "expected_revision": -1},
    {"t_start": None, "t_end": None, "description": None, "expected_revision": 0, "event_type": "impact"},
])
def test_malformed_bodies_are_refused(client, body):
    assert client.put(f"/api/analyses/{A1}/edits/{E1}", json=body).status_code == 422
    assert edits(client, A1) == {}


def test_description_at_the_length_limit_is_accepted(client):
    assert edit(client, A1, E1, 0, description="x" * 2000).status_code == 200


def test_unknown_analysis_unrelated_proposals_and_analyses_without_results(client):
    missing = f"{99:032x}"
    assert client.get(f"/api/analyses/{missing}/edits").status_code == 404
    assert edit(client, missing, E1, 0, 1.0, 2.0).status_code == 404
    assert edit(client, A1, E3, 0, 1.0, 2.0).json()["detail"] == "this analysis has no such event proposal"
    assert edit(client, EMPTY, E1, 0, description="x").status_code == 404
    for analysis_id in (FAILED, RUNNING):
        response = edit(client, analysis_id, E1, 0, description="x")
        assert response.status_code == 409 and response.json()["detail"] == "this analysis has no event proposals to edit"
    assert edits(client, EMPTY) == {}


def test_foreign_pages_cannot_edit(client):
    assert edit(client, A1, E1, 0, 1.0, 2.0, headers={"origin": "https://evil.example"}).status_code == 403
    assert edits(client, A1) == {}


# Independence, isolation, preservation and persistence -------------------------------------------

def test_edits_and_decisions_are_independent(client):
    put(client, A1, E1, "accepted", 0)
    edit(client, A1, E1, 0, 1.0, 2.0, "note")
    assert decisions(client, A1) == {E1: ("accepted", 1)}  # editing does not decide
    put(client, A1, E1, "rejected", 1)
    put(client, A1, E1, "not_reviewed", 2)
    assert edits(client, A1) == {E1: (1.0, 2.0, "note", 1)}  # deciding keeps edits
    edit(client, A1, E1, 1)  # resetting edits keeps the decision
    assert decisions(client, A1) == {E1: ("not_reviewed", 3)}
    edit(client, A1, E2, 0, description="unreviewed but edited")
    assert E2 not in decisions(client, A1)


def test_edits_belong_to_one_analysis_and_never_change_analyses(client):
    database = client.app.state.database
    before = analysis_rows(database)
    edit(client, A1, E1, 0, 1.0, 2.0)
    assert edits(client, A2) == {}
    edit(client, A2, E1, 0, 3.0, 4.0)
    assert edits(client, A1)[E1][:2] == (1.0, 2.0) and edits(client, A2)[E1][:2] == (3.0, 4.0)
    assert analysis_rows(database) == before


def test_edits_survive_a_restart(make_client):
    first = make_client()
    seed(first.app.state.database)
    first.app.state.edits.duration = lambda analysis_id: DURATION
    edit(first, A1, E1, 0, 1.25, 2.5, "kept")
    shutdown(first)
    second = make_client()
    assert edits(second, A1) == {E1: (1.25, 2.5, "kept", 1)}


def test_real_analysis_bounds_times_by_the_video_and_keeps_evidence_unchanged(app, tmp_path):  # noqa: F811
    client = app()
    first = analysed(client)
    timeline_url = f"/api/analyses/{first['id']}/timeline"
    timeline = client.get(timeline_url).json()
    entry = timeline["entries"][0]
    package_url = f"/api/analyses/{first['id']}/packages/{entry['event_id']}"
    package = client.get(package_url).json()
    evidence = tmp_path / "var" / "evidence"
    stored = sorted((p.relative_to(evidence), p.read_bytes()) for p in evidence.rglob("*") if p.is_file())
    duration = timeline["video"]["duration_s"] or timeline["analysed_end_s"]

    too_long = edit(client, first["id"], entry["event_id"], 0, 0.0, duration + 0.001)
    assert too_long.status_code == 422 and "after the end of the video" in too_long.json()["detail"]["message"]
    saved = edit(client, first["id"], entry["event_id"], 0, entry["t_start"] + 0.04, duration, "adjusted")
    assert saved.status_code == 200

    second = analysed(client)
    assert client.get(f"/api/analyses/{second['id']}/edits").json()["edits"] == []
    assert client.get(timeline_url).json() == timeline
    assert client.get(package_url).json() == package
    assert sorted((p.relative_to(evidence), p.read_bytes()) for p in evidence.rglob("*") if p.is_file()) == stored
    assert client.get(f"/api/analyses/{first['id']}").json()["revision"] == first["revision"]
