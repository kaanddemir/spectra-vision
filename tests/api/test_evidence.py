"""Evidence, timelines, overlays and clips of analyses, built from real perception caches written by an
in-process AnalysisRun with fake perception components (no models) on synthetic videos."""
import hashlib
import json
import shutil
import sqlite3
import threading

import pytest

from spectra.api.library import library_folders
from spectra.engine.cache import PerceptionCache
from spectra.engine.contracts import RunConfig
from spectra.engine.runner import AnalysisRun, RunOutcome, RunState, StopReason
from spectra.evidence.contracts import OverlayFrame
from spectra.evidence.overlays import frame_at
from tests.api.conftest import FakeProbe, wait_for_probe
from tests.api.overlay_cases import CASES_PATH, build_cases
from tests.api.test_analyses import finished, wait_until
from tests.engine.fakes import FakeAnalyzer, FakeDetector, FakeEgoLane, FakeSegmenter, FakeTracker, calm
from tests.engine.media import make_video
from tests.evidence import synth

VFR = [round(0.3 + i * 0.04, 3) for i in range(30)] + [round(2.0 + i * 0.1, 3) for i in range(20)]
STREAM_START_S = VFR[0]  # the container's first timestamp; the engine measures times from it
TIMES = [round(t - STREAM_START_S, 6) for t in VFR]


def fake_engine(job, on_progress=None, cancel=None, timeout_s=None):
    """The real AnalysisRun and cache with fake perception components, in this process."""
    return AnalysisRun(job.video, RunConfig(**job.config), cache=PerceptionCache(job.cache_root), guard=calm(),
                       detector=FakeDetector(), tracker=FakeTracker(), segmenter=FakeSegmenter(),
                       ego_lane=FakeEgoLane(), analyzers=[FakeAnalyzer()], on_progress=on_progress,
                       cancel=cancel).run()


def fixed(outcome):
    return lambda job, **kwargs: outcome


@pytest.fixture
def library(tmp_path):
    root = (tmp_path / "library").resolve()
    root.mkdir()
    make_video(root / "drive.mp4", VFR, gop=8, fmt="mp4")
    return root


@pytest.fixture
def app(make_client, library, tmp_path):
    from spectra.engine.worker import WorkerJob

    def make(run_worker=fake_engine):
        job = lambda path, info: WorkerJob(video=str(path), cache_root=str(tmp_path / "var" / "cache"))
        client = make_client(probe=FakeProbe(), library_dirs=(library,), run_worker=run_worker, job_factory=job)
        client.folder_id = library_folders([library])[0].id
        return client
    return make


def analysed(client, name="drive.mp4"):
    video = client.post("/api/videos", json={"folder_id": client.folder_id, "path": name}).json()
    wait_for_probe(client, video["id"])
    client.post(f"/api/videos/{video['id']}/playback-check", json={"playable": True})
    analysis = client.post("/api/analyses", json={"video_id": video["id"]}).json()
    return wait_until(client, analysis["id"], lambda a: finished(a) and a["evidence"]["state"] != "pending")


def builds(tmp_path):
    root = tmp_path / "var" / "evidence"
    return sorted(p.name for p in root.iterdir() if len(p.name) == 64) if root.exists() else []


# Lifecycle -------------------------------------------------------------------------------------

def test_completed_analysis_stores_evidence_and_records_its_build_key(app, tmp_path):
    client = app()
    analysis = analysed(client)
    assert analysis["state"] == "completed"
    assert analysis["evidence"]["state"] == "ready" and analysis["evidence"]["message"] is None
    timeline = client.get(f"/api/analyses/{analysis['id']}/timeline").json()
    assert timeline["state"] == "complete"
    assert timeline["build_key"] == analysis["evidence"]["build_key"]
    assert timeline["analysis_key"] == analysis["result"]["analysis_key"]
    assert timeline["perception_key"] == analysis["result"]["perception_key"]
    assert builds(tmp_path) == [timeline["build_key"]]
    [entry] = timeline["entries"]
    assert entry["title"] and entry["state"] == "supported" and entry["review"]["start_s"] <= entry["t_start"]
    assert [a["coverage"] for a in timeline["analyzers"]] == ["proposed"]
    assert (timeline["analysed_start_s"], timeline["analysed_end_s"]) == pytest.approx((TIMES[0], TIMES[-1]))
    assert timeline["video"]["start_s"] == pytest.approx(STREAM_START_S)
    package = client.get(f"/api/analyses/{analysis['id']}/packages/{entry['event_id']}").json()
    assert package["proposal"]["event_id"] == entry["event_id"]
    assert package["build_key"] == timeline["build_key"]
    assert package["review"] == entry["review"]
    assert package["video"]["sha256"] == analysis["source_sha256"]


def test_no_proposals_is_a_timeline_with_coverage_not_a_finding(app):
    client = app(run_worker=fixed(RunOutcome(RunState.COMPLETED, synth.result([]))))
    analysis = analysed(client)
    timeline = client.get(f"/api/analyses/{analysis['id']}/timeline").json()
    assert (timeline["state"], timeline["entries"], timeline["segments"]) == ("complete", [], [])
    assert {a["coverage"] for a in timeline["analyzers"]} == {"no_proposal"}
    assert analysis["result"]["proposals"] == 0


def test_incomplete_run_keeps_its_state_in_analysis_and_timeline(app):
    result = synth.result([synth.approach(2.0, 3.0)], ended_early="demuxing stopped: corrupt packet")
    client = app(run_worker=fixed(RunOutcome(RunState.COMPLETED, result)))
    analysis = analysed(client)
    assert analysis["state"] == "incomplete" and analysis["evidence"]["state"] == "ready"
    timeline = client.get(f"/api/analyses/{analysis['id']}/timeline").json()
    assert timeline["state"] == "incomplete"
    assert "corrupt packet" in timeline["message"]
    assert len(timeline["entries"]) == 1


@pytest.mark.parametrize("outcome, state, timeline_state", [
    (RunOutcome(RunState.CANCELLED), "cancelled", "cancelled"),
    (RunOutcome(RunState.FAILED, error="pressure level 4", stop=StopReason.RESOURCES), "resource_stopped",
     "resource_stopped"),
    (RunOutcome(RunState.FAILED, error="worker crashed", stop=StopReason.ERROR), "failed", "failed"),
])
def test_runs_without_a_result_have_no_evidence(app, tmp_path, outcome, state, timeline_state):
    client = app(run_worker=fixed(outcome))
    analysis = analysed(client)
    assert analysis["state"] == state
    assert analysis["evidence"] == {"state": "not_applicable", "build_key": None, "message": None}
    timeline = client.get(f"/api/analyses/{analysis['id']}/timeline").json()
    assert (timeline["state"], timeline["entries"], timeline["build_key"]) == (timeline_state, [], None)
    assert timeline["message"] == analysis["message"]
    assert timeline["video"] is None and timeline["analyzers"] == []
    assert client.get(f"/api/analyses/{analysis['id']}/packages/{'0' * 16}").status_code == 409
    assert builds(tmp_path) == []


def test_unfinished_analysis_has_no_timeline(app):
    from tests.api.test_analyses import ScriptedRunner
    runner = ScriptedRunner()
    client = app(run_worker=runner)
    video = client.post("/api/videos", json={"folder_id": client.folder_id, "path": "drive.mp4"}).json()
    wait_for_probe(client, video["id"])
    client.post(f"/api/videos/{video['id']}/playback-check", json={"playable": True})
    analysis = client.post("/api/analyses", json={"video_id": video["id"]}).json()
    response = client.get(f"/api/analyses/{analysis['id']}/timeline")
    assert response.status_code == 409 and "not finished" in response.json()["detail"]
    runner.release()


def test_failed_evidence_build_leaves_the_analysis_completed_and_is_rebuilt_on_request(app, monkeypatch):
    import spectra.api.evidence as evidence
    original = evidence.from_outcome
    calls = []

    def flaky(outcome, config):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("disk full")
        return original(outcome, config)
    monkeypatch.setattr(evidence, "from_outcome", flaky)
    client = app()
    analysis = analysed(client)
    assert analysis["state"] == "completed"
    assert analysis["evidence"]["state"] == "failed" and "disk full" in analysis["evidence"]["message"]
    timeline = client.get(f"/api/analyses/{analysis['id']}/timeline").json()
    assert timeline["state"] == "complete" and len(timeline["entries"]) == 1
    after = client.get(f"/api/analyses/{analysis['id']}").json()
    assert after["evidence"]["state"] == "ready" and after["evidence"]["build_key"] == timeline["build_key"]


def test_identical_analyses_reuse_one_build(app, tmp_path):
    client = app()
    first = analysed(client)
    second = analysed(client)
    assert second["result"]["cache_hit"] is True
    assert second["evidence"]["build_key"] == first["evidence"]["build_key"]
    assert builds(tmp_path) == [first["evidence"]["build_key"]]


# Recovery --------------------------------------------------------------------------------------

def test_restart_rebuilds_missing_evidence_from_the_perception_cache(app, tmp_path):
    client = app()
    analysis = analysed(client)
    url = f"/api/analyses/{analysis['id']}/timeline"
    before = client.get(url).json()
    shutil.rmtree(tmp_path / "var" / "evidence" / before["build_key"])
    restarted = app()
    assert restarted.get(url).json() == before
    entry = before["entries"][0]["event_id"]
    assert restarted.get(f"/api/analyses/{analysis['id']}/packages/{entry}").status_code == 200


def test_interrupted_build_is_marked_and_recovered(app, tmp_path):
    client = app()
    analysis = analysed(client)
    with sqlite3.connect(tmp_path / "var" / "spectra.sqlite3") as conn:
        conn.execute("UPDATE analyses SET evidence_state = 'pending', build_key = NULL WHERE id = ?",
                     (analysis["id"],))
    restarted = app()
    marked = restarted.get(f"/api/analyses/{analysis['id']}").json()
    assert marked["evidence"]["state"] == "failed" and "interrupted" in marked["evidence"]["message"]
    assert marked["state"] == "completed"
    assert restarted.get(f"/api/analyses/{analysis['id']}/timeline").status_code == 200
    assert restarted.get(f"/api/analyses/{analysis['id']}").json()["evidence"]["state"] == "ready"


def test_missing_cache_is_reported_and_nothing_is_rerun(app, tmp_path):
    runs = []

    def counting(job, **kwargs):
        runs.append(job.video)
        return fake_engine(job, **kwargs)
    client = app(run_worker=counting)
    analysis = analysed(client)
    shutil.rmtree(tmp_path / "var" / "evidence" / analysis["evidence"]["build_key"])
    shutil.rmtree(tmp_path / "var" / "cache" / analysis["result"]["perception_key"])
    response = client.get(f"/api/analyses/{analysis['id']}/timeline")
    assert response.status_code == 409
    assert "no longer in the perception cache" in response.json()["detail"]
    assert client.get(f"/api/analyses/{analysis['id']}").json()["evidence"]["state"] == "failed"
    assert len(runs) == 1


def test_tampered_cache_is_rejected(app, tmp_path):
    client = app()
    analysis = analysed(client)
    shutil.rmtree(tmp_path / "var" / "evidence" / analysis["evidence"]["build_key"])
    frames = tmp_path / "var" / "cache" / analysis["result"]["perception_key"] / "frames.jsonl"
    frames.write_bytes(frames.read_bytes() + b"\n")
    response = client.get(f"/api/analyses/{analysis['id']}/timeline")
    assert response.status_code == 409 and "does not match its recorded hash" in response.json()["detail"]


def test_mismatched_stored_result_is_rejected(app, tmp_path):
    client = app()
    analysis = analysed(client)
    shutil.rmtree(tmp_path / "var" / "evidence" / analysis["evidence"]["build_key"])
    with sqlite3.connect(tmp_path / "var" / "spectra.sqlite3") as conn:
        stored = json.loads(conn.execute("SELECT result_json FROM analyses").fetchone()[0])
        stored["reports"] = []
        conn.execute("UPDATE analyses SET result_json = ?", (json.dumps(stored),))
    response = client.get(f"/api/analyses/{analysis['id']}/timeline")
    assert response.status_code == 409 and "do not match this analysis" in response.json()["detail"]


def test_analysis_without_stored_result_explains_how_to_recover(app, tmp_path):
    client = app()
    analysis = analysed(client)
    shutil.rmtree(tmp_path / "var" / "evidence" / analysis["evidence"]["build_key"])
    with sqlite3.connect(tmp_path / "var" / "spectra.sqlite3") as conn:
        conn.execute("UPDATE analyses SET result_json = NULL")
    response = client.get(f"/api/analyses/{analysis['id']}/timeline")
    assert response.status_code == 409 and "analyse the video again" in response.json()["detail"]


def test_concurrent_requests_rebuild_once(app, tmp_path, monkeypatch):
    client = app()
    analysis = analysed(client)
    shutil.rmtree(tmp_path / "var" / "evidence" / analysis["evidence"]["build_key"])
    store = client.app.state.evidence.store
    writes = []
    original = store.write
    monkeypatch.setattr(store, "write", lambda *a: writes.append(1) or original(*a))
    results = []
    threads = [threading.Thread(target=lambda: results.append(
        client.get(f"/api/analyses/{analysis['id']}/timeline").json())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(writes) == 1
    assert all(r == results[0] for r in results) and results[0]["state"] == "complete"


# Validation ------------------------------------------------------------------------------------

def test_identifiers_are_validated(app):
    client = app()
    analysis = analysed(client)
    base = f"/api/analyses/{analysis['id']}"
    assert client.get(f"{base}/packages/{'0' * 16}").status_code == 404
    assert client.get(f"{base}/packages/not-an-event").status_code == 422
    assert client.get(f"{base}/packages/{'0' * 16}/overlay").status_code == 404
    assert client.get("/api/analyses/" + "0" * 32 + "/timeline").status_code == 404
    assert client.get("/api/analyses/bad/timeline").status_code == 422
    assert client.get(f"{base}/timeline", headers={"origin": "https://evil.example"}).status_code == 403


# Overlays --------------------------------------------------------------------------------------

def overlay_for(client, analysis, **params):
    timeline = client.get(f"/api/analyses/{analysis['id']}/timeline").json()
    entry = timeline["entries"][0]
    return entry, client.get(f"/api/analyses/{analysis['id']}/packages/{entry['event_id']}/overlay", params=params)


def test_overlay_frames_are_the_recorded_frames_with_validity_windows(app):
    client = app()
    analysis = analysed(client)
    entry, response = overlay_for(client, analysis)
    body = response.json()
    frames = body["overlay"]["frames"]
    review = entry["review"]
    assert [f["t"] for f in frames] == pytest.approx([t for t in TIMES if review["start_s"] <= t <= review["end_s"]])
    for k, f in enumerate(frames):
        assert f["t"] < f["valid_until"] <= f["t"] + 0.25 + 1e-6  # valid_until is rounded to 1 µs
        if k + 1 < len(frames):
            assert f["valid_until"] <= frames[k + 1]["t"] + 1e-6
        assert f["tracks"] is not None and f["lane"] is not None
    assert body["layers"] == {"tracks": True, "lane": True}
    timing = body["timing"]
    assert timing["stream_start_s"] == pytest.approx(STREAM_START_S) and timing["max_age_s"] == 0.25
    assert (timing["analysed_start_s"], timing["analysed_end_s"]) == pytest.approx((TIMES[0], TIMES[-1]))
    assert timing["timestamp_source"] == "container_pts"
    parsed = [OverlayFrame.model_validate(f) for f in frames]
    assert frame_at(parsed, 1.45) is None  # 1.16 + 0.25 < 1.45 < 1.7: not analysed here, nothing drawn
    assert frame_at(parsed, 1.3).t == pytest.approx(1.16)
    assert frame_at(parsed, 1.75).t == pytest.approx(1.7)


def test_overlay_for_a_requested_span(app):
    client = app()
    analysis = analysed(client)
    _, response = overlay_for(client, analysis, start_s=1.0, end_s=2.25)
    frames = response.json()["overlay"]["frames"]
    assert [f["t"] for f in frames] == pytest.approx([t for t in TIMES if 1.0 - 1e-6 <= t <= 2.25 + 1e-6])
    assert response.json()["overlay"]["start_s"] == 1.0


@pytest.mark.parametrize("params", [{"start_s": 3, "end_s": 2}, {"start_s": 0, "end_s": 500},
                                    {"start_s": 100, "end_s": 110}, {"start_s": -1}])
def test_invalid_overlay_spans_are_refused(app, params):
    client = app()
    _, response = overlay_for(client, analysed(client), **params)
    assert response.status_code == 422


def test_overlay_reports_missing_layers(app, tmp_path):

    def no_lanes(job, on_progress=None, cancel=None, timeout_s=None):
        return AnalysisRun(job.video, RunConfig(), cache=PerceptionCache(job.cache_root), guard=calm(),
                           detector=FakeDetector(), tracker=FakeTracker(), analyzers=[FakeAnalyzer()]).run()
    client = app(run_worker=no_lanes)
    body = overlay_for(client, analysed(client))[1].json()
    assert body["layers"] == {"tracks": True, "lane": False}
    assert all(f["lane"] is None for f in body["overlay"]["frames"])


def test_overlay_lookup_cases_are_current():
    """The shared reference for other implementations must match frame_at exactly."""
    assert json.loads(CASES_PATH.read_text()) == json.loads(json.dumps(build_cases()))


# Clips -----------------------------------------------------------------------------------------

def test_clip_is_stream_copied_reused_and_served(app, library, tmp_path):
    client = app()
    analysis = analysed(client)
    source = library / "drive.mp4"
    before = (hashlib.sha256(source.read_bytes()).hexdigest(), source.stat().st_mtime_ns)
    entry = client.get(f"/api/analyses/{analysis['id']}/timeline").json()["entries"][0]
    url = f"/api/analyses/{analysis['id']}/packages/{entry['event_id']}/clip"
    assert client.get(url).status_code == 404
    created = client.post(url)
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["media_url"] == url and body["mime_type"] == "video/mp4" and body["browser_playback"] == "unverified"
    clip = body["clip"]
    assert clip["source_sha256"] == analysis["source_sha256"] and clip["codec"] == "mpeg4"
    assert clip["origin_s"] <= entry["review"]["start_s"] <= clip["first_s"] + 1e-6
    path = tmp_path / "var" / "evidence" / analysis["evidence"]["build_key"] / "clips" / clip["file"]
    mtime = path.stat().st_mtime_ns
    assert client.post(url).json()["clip"] == clip and path.stat().st_mtime_ns == mtime
    served = client.get(url)
    assert served.status_code == 200 and hashlib.sha256(served.content).hexdigest() == clip["sha256"]
    part = client.get(url, headers={"range": "bytes=0-9"})
    assert part.status_code == 206 and len(part.content) == 10
    assert (hashlib.sha256(source.read_bytes()).hexdigest(), source.stat().st_mtime_ns) == before


def test_clip_of_a_changed_source_is_refused_but_review_continues(app, library):
    client = app()
    analysis = analysed(client)
    entry = client.get(f"/api/analyses/{analysis['id']}/timeline").json()["entries"][0]
    with open(library / "drive.mp4", "ab") as f:
        f.write(b"\0")
    response = client.post(f"/api/analyses/{analysis['id']}/packages/{entry['event_id']}/clip")
    assert response.status_code == 409 and "unavailable" in response.json()["detail"]
    assert client.get(f"/api/analyses/{analysis['id']}/timeline").status_code == 200
    assert client.get(f"/api/analyses/{analysis['id']}/packages/{entry['event_id']}/overlay").status_code == 200


@pytest.mark.parametrize("error, status", [("unsupported", 422), ("cancelled", 503)])
def test_clip_failures_leave_no_partial_files(app, tmp_path, monkeypatch, error, status):
    import spectra.evidence.media as media
    client = app()
    analysis = analysed(client)
    entry = client.get(f"/api/analyses/{analysis['id']}/timeline").json()["entries"][0]
    original = media._verify

    def fail(*args):
        if error == "cancelled":
            raise media.ClipCancelled("stop")
        raise media.VideoError("the written clip's frame times do not match the source")
    monkeypatch.setattr(media, "_verify", fail)
    response = client.post(f"/api/analyses/{analysis['id']}/packages/{entry['event_id']}/clip")
    assert response.status_code == status
    clips = tmp_path / "var" / "evidence" / analysis["evidence"]["build_key"] / "clips"
    assert list(clips.iterdir()) == []
    monkeypatch.setattr(media, "_verify", original)
    assert client.post(f"/api/analyses/{analysis['id']}/packages/{entry['event_id']}/clip").status_code == 200
