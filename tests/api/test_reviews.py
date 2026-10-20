"""Human review decisions: validation, compare-and-set revisions, persistence, isolation between analyses
and preservation of analysis results and evidence."""
import json
import threading

import pytest

from spectra.api.reviews import ReviewConflict, ReviewService

from tests.api.test_evidence import analysed, app, library  # noqa: F401 (fixtures)

SHA = "0" * 64
A1, A2, FAILED, EMPTY, RUNNING = (f"{n:032x}" for n in (1, 2, 3, 4, 5))
E1, E2, E3 = "00000000000000e1", "00000000000000e2", "00000000000000e3"


def result_json(*event_ids):
    return json.dumps({"reports": [], "events": [{"event_id": e} for e in event_ids]})


def seed(database):
    """One video with analyses in every relevant state; A1 and A2 share a proposal id, as a reanalysis does."""
    with database.connect() as conn:
        conn.execute("INSERT INTO videos (id, path, size_bytes, mtime_ns, added_at) VALUES ('v', '/a.mp4', 1, 1, 'x')")
        for analysis_id, state, result in [(A1, "completed", result_json(E1, E2)), (A2, "incomplete", result_json(E1, E3)),
                                           (FAILED, "failed", None), (EMPTY, "completed", result_json()),
                                           (RUNNING, "running", None)]:
            conn.execute("INSERT INTO analyses (id, video_id, source_sha256, state, created_at, result_json, revision) "
                         "VALUES (?, 'v', ?, ?, 'x', ?, 3)", (analysis_id, SHA, state, result))


@pytest.fixture
def client(make_client):
    client = make_client()
    seed(client.app.state.database)
    return client


def put(client, analysis_id, event_id, decision, expected_revision, **kwargs):
    return client.put(f"/api/analyses/{analysis_id}/reviews/{event_id}",
                      json={"decision": decision, "expected_revision": expected_revision}, **kwargs)


def decisions(client, analysis_id):
    response = client.get(f"/api/analyses/{analysis_id}/reviews")
    assert response.status_code == 200
    return {d["event_id"]: (d["decision"], d["revision"]) for d in response.json()["decisions"]}


def analysis_rows(database):
    with database.connect() as conn:
        return [tuple(r) for r in conn.execute("SELECT * FROM analyses ORDER BY id")]


# Decisions ---------------------------------------------------------------------------------------

def test_proposals_start_not_reviewed_without_records(client):
    assert client.get(f"/api/analyses/{A1}/reviews").json() == {"analysis_id": A1, "decisions": []}


def test_accept_reject_and_clear(client):
    accepted = put(client, A1, E1, "accepted", 0)
    assert accepted.status_code == 200
    body = accepted.json()
    assert (body["event_id"], body["decision"], body["revision"]) == (E1, "accepted", 1) and body["updated_at"]
    assert decisions(client, A1) == {E1: ("accepted", 1)}
    assert put(client, A1, E1, "rejected", 1).json()["revision"] == 2
    assert decisions(client, A1) == {E1: ("rejected", 2)}
    # Clearing keeps the record so later changes still need the newest revision.
    assert put(client, A1, E1, "not_reviewed", 2).json()["decision"] == "not_reviewed"
    assert decisions(client, A1) == {E1: ("not_reviewed", 3)}
    assert put(client, A1, E1, "accepted", 3).json()["revision"] == 4
    assert put(client, A1, E2, "rejected", 0).json()["revision"] == 1
    assert decisions(client, A1) == {E1: ("accepted", 4), E2: ("rejected", 1)}


def test_repeating_the_stored_decision_changes_nothing(client):
    first = put(client, A1, E1, "accepted", 0).json()
    for expected in (0, 1, 5):
        again = put(client, A1, E1, "accepted", expected)
        assert again.status_code == 200 and again.json() == first
    assert decisions(client, A1) == {E1: ("accepted", 1)}


def test_clearing_an_unreviewed_proposal_creates_no_record(client):
    response = put(client, A1, E1, "not_reviewed", 0)
    assert response.json() == {"event_id": E1, "decision": "not_reviewed", "revision": 0, "updated_at": None}
    assert decisions(client, A1) == {}


def test_stale_change_is_refused_with_the_latest_decision(client):
    put(client, A1, E1, "accepted", 0)
    put(client, A1, E1, "rejected", 1)
    stale = put(client, A1, E1, "accepted", 1)
    assert stale.status_code == 409
    detail = stale.json()["detail"]
    assert "changed elsewhere" in detail["message"]
    assert (detail["current"]["decision"], detail["current"]["revision"]) == ("rejected", 2)
    assert put(client, A1, E1, "not_reviewed", 0).status_code == 409
    assert decisions(client, A1) == {E1: ("rejected", 2)}


def test_concurrent_changes_from_the_same_revision_save_exactly_one(client):
    service: ReviewService = client.app.state.reviews
    barrier = threading.Barrier(8)
    outcomes = []

    def change(decision):
        barrier.wait()
        try:
            outcomes.append(service.set(A1, E1, decision, 0).decision)
        except ReviewConflict as exc:
            outcomes.append(("conflict", exc.current.revision))

    threads = [threading.Thread(target=change, args=("accepted" if i % 2 else "rejected",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    saved = [o for o in outcomes if isinstance(o, str)]
    stored, revision = decisions(client, A1)[E1]
    assert revision == 1 and stored in saved
    # Requests asking for the stored decision succeed without a change; the others are refused.
    assert all(o == stored or o == ("conflict", 1) for o in outcomes) and len(outcomes) == 8


# Validation --------------------------------------------------------------------------------------

@pytest.mark.parametrize("body", [
    {"decision": "confirmed", "expected_revision": 0},
    {"decision": "Accepted", "expected_revision": 0},
    {"decision": "fault", "expected_revision": 0},
    {"decision": None, "expected_revision": 0},
    {"decision": "accepted"},
    {"decision": "accepted", "expected_revision": -1},
    {"decision": "accepted", "expected_revision": "one"},
    {"decision": "accepted", "expected_revision": 0, "note": "x"},
    {"expected_revision": 0},
])
def test_invalid_bodies_are_refused(client, body):
    assert client.put(f"/api/analyses/{A1}/reviews/{E1}", json=body).status_code == 422
    assert decisions(client, A1) == {}


@pytest.mark.parametrize("analysis_id, event_id", [("x" * 32, E1), (A1, "E1"), (A1, "0" * 17), (A1[:-1], E1)])
def test_malformed_identifiers_are_refused(client, analysis_id, event_id):
    assert put(client, analysis_id, event_id, "accepted", 0).status_code in (404, 422)


def test_unknown_analysis_and_unrelated_proposals_are_not_found(client):
    missing = f"{99:032x}"
    assert client.get(f"/api/analyses/{missing}/reviews").status_code == 404
    response = put(client, missing, E1, "accepted", 0)
    assert response.status_code == 404 and response.json()["detail"] == "analysis not found"
    # E3 belongs to the other analysis of the same video.
    unrelated = put(client, A1, E3, "accepted", 0)
    assert unrelated.status_code == 404 and unrelated.json()["detail"] == "this analysis has no such event proposal"
    assert put(client, A1, "ffffffffffffffff", "accepted", 0).status_code == 404
    assert decisions(client, A1) == {} and decisions(client, A2) == {}


@pytest.mark.parametrize("analysis_id", [FAILED, RUNNING])
def test_analyses_without_a_result_have_nothing_to_review(client, analysis_id):
    assert decisions(client, analysis_id) == {}
    response = put(client, analysis_id, E1, "accepted", 0)
    assert response.status_code == 409 and response.json()["detail"] == "this analysis has no event proposals to review"


def test_zero_proposal_analysis_is_valid_without_records(client):
    assert client.get(f"/api/analyses/{EMPTY}/reviews").json() == {"analysis_id": EMPTY, "decisions": []}
    assert put(client, EMPTY, E1, "accepted", 0).status_code == 404
    with client.app.state.database.connect() as conn:
        assert conn.execute("SELECT count(*) FROM review_decisions").fetchone()[0] == 0


def test_foreign_pages_cannot_change_decisions(client):
    assert put(client, A1, E1, "accepted", 0, headers={"origin": "https://evil.example"}).status_code == 403
    assert put(client, A1, E1, "accepted", 0, headers={"sec-fetch-site": "cross-site"}).status_code == 403
    assert decisions(client, A1) == {}


# Isolation, preservation and persistence ---------------------------------------------------------

def test_decisions_belong_to_one_analysis(client):
    put(client, A1, E1, "accepted", 0)
    assert decisions(client, A2) == {}
    assert put(client, A2, E1, "rejected", 0).json()["revision"] == 1
    assert decisions(client, A1) == {E1: ("accepted", 1)} and decisions(client, A2) == {E1: ("rejected", 1)}


def test_decisions_never_change_analyses(client):
    database = client.app.state.database
    before = analysis_rows(database)
    listed = client.get("/api/analyses").json()
    put(client, A1, E1, "accepted", 0)
    put(client, A1, E1, "rejected", 1)
    put(client, A2, E3, "accepted", 0)
    assert analysis_rows(database) == before
    assert client.get("/api/analyses").json() == listed


def test_decisions_survive_a_restart(make_client, tmp_path):
    first = make_client()
    seed(first.app.state.database)
    put(first, A1, E1, "accepted", 0)
    put(first, A1, E2, "rejected", 0)
    put(first, A1, E2, "not_reviewed", 1)
    from tests.api.conftest import shutdown
    shutdown(first)
    second = make_client()
    assert decisions(second, A1) == {E1: ("accepted", 1), E2: ("not_reviewed", 2)}
    assert put(second, A1, E1, "rejected", 0).status_code == 409


def test_reanalysis_has_independent_decisions_and_evidence_is_unchanged(app, tmp_path):  # noqa: F811
    client = app()
    first = analysed(client)
    event_id = client.get(f"/api/analyses/{first['id']}/timeline").json()["entries"][0]["event_id"]
    package_url = f"/api/analyses/{first['id']}/packages/{event_id}"
    package = client.get(package_url).json()
    timeline = client.get(f"/api/analyses/{first['id']}/timeline").json()
    stored = sorted((p.relative_to(tmp_path), p.read_bytes()) for p in (tmp_path / "var" / "evidence").rglob("*")
                    if p.is_file())

    assert put(client, first["id"], event_id, "accepted", 0).status_code == 200
    second = analysed(client)
    assert second["id"] != first["id"]
    # The same records give the same proposal ids, but the new analysis starts unreviewed.
    assert client.get(f"/api/analyses/{second['id']}/timeline").json()["entries"][0]["event_id"] == event_id
    assert decisions(client, second["id"]) == {}
    assert decisions(client, first["id"]) == {event_id: ("accepted", 1)}

    assert client.get(package_url).json() == package
    assert client.get(f"/api/analyses/{first['id']}/timeline").json() == timeline
    after = sorted((p.relative_to(tmp_path), p.read_bytes()) for p in (tmp_path / "var" / "evidence").rglob("*")
                   if p.is_file())
    assert after == stored
    assert client.get(f"/api/analyses/{first['id']}").json()["revision"] == first["revision"]
