"""Analysis jobs against scripted runners and the real worker with decode-only (model-free) jobs."""
import hashlib
import json
import multiprocessing
import sqlite3
import threading
import time

import pytest

from spectra.api.library import library_folders
from spectra.engine.runner import Progress, RunOutcome, RunState, StopReason
from spectra.engine.worker import WorkerJob
from tests.api.conftest import FakeProbe, shutdown, wait_for_probe
from tests.engine.media import cfr_times, make_video
from tests.evidence import synth


def decode_only(cache_dir):
    """A real worker job without models: the worker decodes and caches records, nothing more."""
    return lambda path, info: WorkerJob(video=str(path), cache_root=str(cache_dir), config={"interval_s": 0.1})


def fake_result(ended_early=None, proposals=0):
    """A valid synthetic AnalysisResult; its perception records are in memory, not in a cache."""
    drafts = [synth.approach(2.0 + 5 * i, 3.0 + 5 * i) for i in range(proposals)]
    return synth.result(drafts, ended_early=ended_early)


class ScriptedRunner:
    """Stands in for run_in_worker: reports progress, then waits for release or cancellation."""

    def __init__(self, outcome=None):
        self.outcome = outcome or RunOutcome(RunState.COMPLETED, fake_result(), cache_hit=False, notes=("note",))
        self.calls, self.active, self.max_active = [], 0, 0
        self.gate = threading.Semaphore(0)
        self.lock = threading.Lock()

    def release(self, n=1):
        for _ in range(n):
            self.gate.release()

    def __call__(self, job, on_progress=None, cancel=None, timeout_s=None):
        with self.lock:
            self.calls.append(job.video)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            on_progress(Progress(RunState.RUNNING, 0, None, None, "loading models"))
            on_progress(Progress(RunState.RUNNING, 12, 1.2, None, "analysing frames"))
            while not self.gate.acquire(timeout=0.01):
                if cancel.cancelled:
                    return RunOutcome(RunState.CANCELLED)
            return self.outcome(job) if callable(self.outcome) else self.outcome
        finally:
            with self.lock:
                self.active -= 1


def wait_until(client, analysis_id, predicate, timeout_s=20.0):
    deadline = time.monotonic() + timeout_s
    while True:
        analysis = client.get(f"/api/analyses/{analysis_id}").json()
        if predicate(analysis) or time.monotonic() > deadline:
            return analysis
        time.sleep(0.01)


def finished(analysis):
    return analysis["state"] not in ("queued", "running")


@pytest.fixture
def library(tmp_path):
    root = (tmp_path / "library").resolve()
    root.mkdir()
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        make_video(root / name, cfr_times(20, 10.0), fmt="mp4")
    return root


@pytest.fixture
def app(make_client, library, tmp_path):
    """Factory for clients whose videos the fake probe declares as H.264 MP4."""
    folder_id = library_folders([library])[0].id

    def make(run_worker=None, **overrides):
        overrides.setdefault("job_factory", decode_only(tmp_path / "var" / "cache"))
        client = make_client(probe=FakeProbe(), library_dirs=(library,), run_worker=run_worker, **overrides)
        client.folder_id = folder_id
        return client
    return make


def eligible(client, name="a.mp4", playable=True):
    video = client.post("/api/videos", json={"folder_id": client.folder_id, "path": name}).json()
    wait_for_probe(client, video["id"])
    return client.post(f"/api/videos/{video['id']}/playback-check", json={"playable": playable}).json()


def submit(client, video_id):
    return client.post("/api/analyses", json={"video_id": video_id})


# Request validation and eligibility ------------------------------------------------------------

@pytest.mark.parametrize("body, status", [({}, 422), ({"video_id": "x"}, 422), ({"video_id": "0" * 32}, 404),
                                          ({"video_id": "0" * 32, "interval_s": 1}, 422)])
def test_request_validation(app, body, status):
    assert app().post("/api/analyses", json=body).status_code == status


def test_cross_site_requests_are_refused(app):
    client = app(run_worker=ScriptedRunner())
    video = eligible(client)
    assert client.post("/api/analyses", json={"video_id": video["id"]},
                       headers={"origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/analyses").json() == []


def test_analysis_requires_a_successful_browser_check(app):
    runner = ScriptedRunner()
    client = app(run_worker=runner)
    video = client.post("/api/videos", json={"folder_id": client.folder_id, "path": "a.mp4"}).json()
    wait_for_probe(client, video["id"])
    response = submit(client, video["id"])
    assert response.status_code == 409
    assert response.json()["detail"]["reasons"] == ["Play the video in this browser to confirm it can be reviewed."]
    client.post(f"/api/videos/{video['id']}/playback-check", json={"playable": False})
    assert "could not play" in submit(client, video["id"]).json()["detail"]["reasons"][0]
    assert runner.calls == []


def test_unsupported_and_unprobed_videos_are_refused(make_client, library, tmp_path):
    probe = FakeProbe({"a.mp4": ("avi", "mpeg4", ()), "b.mp4": "corrupt"})
    client = make_client(probe=probe, library_dirs=(library,), run_worker=ScriptedRunner())
    folder_id = library_folders([library])[0].id
    for name in ("a.mp4", "b.mp4"):
        video = client.post("/api/videos", json={"folder_id": folder_id, "path": name}).json()
        wait_for_probe(client, video["id"])
        assert submit(client, video["id"]).status_code == 409


def test_stale_playback_check_never_authorises_a_changed_video(app, library):
    client = app(run_worker=ScriptedRunner())
    video = eligible(client)
    with open(library / "a.mp4", "ab") as f:
        f.write(b"\0")
    assert "changed" in submit(client, video["id"]).json()["detail"]["reasons"][0]
    client.post("/api/videos", json={"folder_id": client.folder_id, "path": "a.mp4"})
    reprobed = wait_for_probe(client, video["id"])
    assert reprobed["playback"]["browser_check"] is None
    assert submit(client, video["id"]).status_code == 409


def test_change_while_queued_fails_at_verification(app, library):
    runner = ScriptedRunner()
    client = app(run_worker=runner)
    first, second = eligible(client, "a.mp4"), eligible(client, "b.mp4")
    running = submit(client, first["id"]).json()
    queued = submit(client, second["id"]).json()
    with open(library / "b.mp4", "ab") as f:
        f.write(b"\0")
    runner.release()
    result = wait_until(client, queued["id"], finished)
    assert (result["state"], result["stop_reason"]) == ("failed", "video")
    assert runner.calls == [str(library / "a.mp4")]
    assert wait_until(client, running["id"], finished)["state"] == "completed"


def test_same_size_rewrite_while_queued_is_caught_by_full_hash(app, library, monkeypatch):
    """Size, mtime and inode can match after a careful rewrite; check_source re-hashes the file."""
    runner = ScriptedRunner()
    client = app(run_worker=runner)
    video = eligible(client)
    import spectra.api.jobs as jobs
    from spectra.evidence.media import SourceCheck, SourceStatus
    seen = []

    def check(path, info, verify_hash=True):
        seen.append(verify_hash)
        return SourceCheck(status=SourceStatus.CHANGED, message="a.mp4 is not the analysed video")
    monkeypatch.setattr(jobs, "check_source", check)
    result = wait_until(client, submit(client, video["id"]).json()["id"], finished)
    assert (result["state"], result["message"]) == ("failed", "a.mp4 is not the analysed video")
    assert seen == [True] and runner.calls == []


@pytest.mark.parametrize("problem", ["models", "pressure", "closing"])
def test_unavailable_resources_refuse_new_analyses(app, models_dir, problem):
    client = app(run_worker=ScriptedRunner(), pressure=(lambda: 4) if problem == "pressure" else (lambda: 1))
    video = eligible(client)
    if problem == "models":
        (models_dir / "detector.onnx").unlink()
    if problem == "closing":
        client.app.state.analyses.closing = True
    response = submit(client, video["id"])
    assert response.status_code == 503
    assert response.json()["detail"]["reasons"]


def test_models_failing_at_start_fail_the_job_without_spawning(app):
    runner = ScriptedRunner()

    def factory(path, info):
        raise FileNotFoundError("missing detector model")
    client = app(run_worker=runner, job_factory=factory)
    result = wait_until(client, submit(client, eligible(client)["id"]).json()["id"], finished)
    assert (result["state"], result["stop_reason"]) == ("failed", "models")
    assert "missing detector model" in result["message"]
    assert runner.calls == []


def test_critical_pressure_at_start_stops_without_spawning(app):
    level = [1]
    runner = ScriptedRunner()
    client = app(run_worker=runner, pressure=lambda: level[0])
    video = eligible(client)
    blocker = ScriptedRunner()
    client.app.state.analyses.run = blocker
    first = submit(client, eligible(client, "b.mp4")["id"]).json()
    queued = submit(client, video["id"]).json()
    wait_until(client, first["id"], lambda a: a["progress"]["stage"] == "analysing frames")
    level[0] = 4
    blocker.release()
    result = wait_until(client, queued["id"], finished)
    assert (result["state"], result["stop_reason"]) == ("resource_stopped", "resources")
    assert wait_until(client, first["id"], finished)["state"] == "completed"
    assert len(blocker.calls) == 1 and runner.calls == []


# Queue, idempotency and single execution ------------------------------------------------------

def test_duplicate_requests_return_the_active_analysis(app):
    runner = ScriptedRunner()
    client = app(run_worker=runner)
    video = eligible(client)
    first = submit(client, video["id"])
    assert first.status_code == 201
    for _ in range(3):
        again = submit(client, video["id"])
        assert again.status_code == 200 and again.json()["id"] == first.json()["id"]
    runner.release()
    wait_until(client, first.json()["id"], finished)
    later = submit(client, video["id"])
    assert later.status_code == 201 and later.json()["id"] != first.json()["id"]
    runner.release()
    wait_until(client, later.json()["id"], finished)
    assert len(runner.calls) == 2


def test_concurrent_duplicate_requests_create_one_analysis(app):
    runner = ScriptedRunner()
    client = app(run_worker=runner)
    video = eligible(client)
    results = []
    threads = [threading.Thread(target=lambda: results.append(submit(client, video["id"]))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({r.json()["id"] for r in results}) == 1
    assert sorted(r.status_code for r in results) == [200] * 7 + [201]
    runner.release()


def test_fifo_queue_runs_one_worker_at_a_time(app, library):
    runner = ScriptedRunner()
    client = app(run_worker=runner)
    ids = [submit(client, eligible(client, name)["id"]).json()["id"] for name in ("a.mp4", "b.mp4", "c.mp4")]
    wait_until(client, ids[0], lambda a: a["state"] == "running")
    states = [client.get(f"/api/analyses/{i}").json() for i in ids]
    assert [s["state"] for s in states] == ["running", "queued", "queued"]
    assert [s["queue_position"] for s in states] == [None, 1, 2]
    time.sleep(0.2)
    assert len(runner.calls) == 1
    runner.release(3)
    for analysis_id in ids:
        assert wait_until(client, analysis_id, finished)["state"] == "completed"
    assert runner.calls == [str(library / n) for n in ("a.mp4", "b.mp4", "c.mp4")]
    assert runner.max_active == 1
    listed = client.get("/api/analyses").json()
    assert [a["id"] for a in listed] == ids[::-1]
    assert [a["id"] for a in client.get("/api/analyses", params={"video_id": listed[0]["video_id"]}).json()] == [ids[2]]


# Outcomes ------------------------------------------------------------------------------------

@pytest.mark.parametrize("outcome, state, stop_reason, message", [
    (RunOutcome(RunState.FAILED, error="system memory pressure level 4", stop=StopReason.RESOURCES),
     "resource_stopped", "resources", "system memory pressure level 4"),
    (RunOutcome(RunState.FAILED, error="a.mp4 contains no video stream", stop=StopReason.VIDEO),
     "failed", "video", "a.mp4 contains no video stream"),
    (RunOutcome(RunState.FAILED, error="analysis worker timed out after 9 s", stop=StopReason.ERROR),
     "failed", "error", "analysis worker timed out after 9 s"),
    (RunOutcome(RunState.FAILED, error="analysis worker exited with code -9 without a result", stop=StopReason.ERROR),
     "failed", "error", "analysis worker exited with code -9 without a result"),
    (RunOutcome(RunState.CANCELLED), "cancelled", "cancelled", "cancelled on request"),
    (RunOutcome(RunState.COMPLETED, fake_result(ended_early="demuxing stopped"), notes=("pressure 2",)),
     "incomplete", None, "decoding ended early: demuxing stopped"),
])
def test_outcomes_map_to_states_without_overstating_success(app, outcome, state, stop_reason, message):
    runner = ScriptedRunner(outcome)
    runner.release()
    client = app(run_worker=runner)
    result = wait_until(client, submit(client, eligible(client)["id"]).json()["id"], finished)
    assert (result["state"], result["stop_reason"], result["message"]) == (state, stop_reason, message)
    assert result["finished_at"] is not None
    assert (result["result"] is not None) == (state == "incomplete")
    if state == "incomplete":
        assert result["result"]["notes"] == ["pressure 2"]


def test_completed_summary_counts_proposals(app):
    runner = ScriptedRunner(RunOutcome(RunState.COMPLETED, fake_result(proposals=2), cache_hit=True))
    runner.release()
    client = app(run_worker=runner)
    result = wait_until(client, submit(client, eligible(client)["id"]).json()["id"], finished)
    assert result["state"] == "completed" and result["stop_reason"] is None
    expected = fake_result(proposals=2)
    assert result["result"] == {"cache_hit": True, "perception_key": expected.perception_key,
                                "analysis_key": expected.analysis_key, "proposals": 2, "notes": []}
    assert result["progress"]["frames_analysed"] == 12 and result["progress"]["media_time_s"] == 1.2
    assert result["progress"]["fraction"] is None


def test_runner_exception_fails_only_that_analysis(app):
    calls = []

    def crash(job, **kwargs):
        calls.append(job.video)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return RunOutcome(RunState.COMPLETED, fake_result())
    client = app(run_worker=crash)
    first = submit(client, eligible(client, "a.mp4")["id"]).json()
    second = submit(client, eligible(client, "b.mp4")["id"]).json()
    assert wait_until(client, first["id"], finished)["state"] == "failed"
    assert wait_until(client, first["id"], finished)["message"] == "an internal error stopped the analysis"
    assert wait_until(client, second["id"], finished)["state"] == "completed"


# Cancellation --------------------------------------------------------------------------------

def test_cancel_queued_analysis_never_starts_it(app):
    runner = ScriptedRunner()
    client = app(run_worker=runner)
    running = submit(client, eligible(client, "a.mp4")["id"]).json()
    queued = submit(client, eligible(client, "b.mp4")["id"]).json()
    cancelled = client.post(f"/api/analyses/{queued['id']}/cancel").json()
    assert (cancelled["state"], cancelled["stop_reason"], cancelled["cancel_requested"]) == ("cancelled", "cancelled", True)
    assert client.post(f"/api/analyses/{queued['id']}/cancel").json()["state"] == "cancelled"
    runner.release(2)
    assert wait_until(client, running["id"], finished)["state"] == "completed"
    time.sleep(0.1)
    assert len(runner.calls) == 1
    assert client.post(f"/api/analyses/{running['id']}/cancel").status_code == 409
    assert client.post("/api/analyses/" + "0" * 32 + "/cancel").status_code == 404


def test_cancel_running_analysis_uses_the_cancel_token(app):
    runner = ScriptedRunner()
    client = app(run_worker=runner)
    analysis = submit(client, eligible(client)["id"]).json()
    wait_until(client, analysis["id"], lambda a: a["progress"]["stage"] == "analysing frames")
    requested = client.post(f"/api/analyses/{analysis['id']}/cancel").json()
    assert requested["cancel_requested"] is True
    result = wait_until(client, analysis["id"], finished)
    assert (result["state"], result["stop_reason"], result["result"]) == ("cancelled", "cancelled", None)


# The real worker -------------------------------------------------------------------------------

def test_real_worker_completes_with_deterministic_cache_identity(app, library):
    client = app()
    video = eligible(client)
    source = library / "a.mp4"
    before = (hashlib.sha256(source.read_bytes()).hexdigest(), source.stat().st_mtime_ns)
    first = wait_until(client, submit(client, video["id"]).json()["id"], finished)
    assert first["state"] == "completed", first
    assert first["result"]["cache_hit"] is False and first["result"]["proposals"] == 0
    assert first["progress"]["frames_analysed"] == 20 and first["progress"]["fraction"] == 1.0
    second = wait_until(client, submit(client, video["id"]).json()["id"], finished)
    assert second["result"]["cache_hit"] is True
    assert second["result"]["perception_key"] == first["result"]["perception_key"]
    assert second["result"]["analysis_key"] == first["result"]["analysis_key"]
    assert (hashlib.sha256(source.read_bytes()).hexdigest(), source.stat().st_mtime_ns) == before
    assert multiprocessing.active_children() == []


def test_real_worker_cancellation_keeps_the_cache_usable(app):
    """The token is set as the worker starts, so the spawned process itself observes the cancellation."""
    from spectra.engine.worker import run_in_worker
    attempts = []

    def cancel_first(job, on_progress=None, cancel=None, timeout_s=None):
        if not attempts:
            cancel.cancel()
        attempts.append(job.video)
        return run_in_worker(job, on_progress=on_progress, cancel=cancel, timeout_s=timeout_s)
    client = app(run_worker=cancel_first)
    video = eligible(client)
    result = wait_until(client, submit(client, video["id"]).json()["id"], finished)
    assert (result["state"], result["stop_reason"], result["result"]) == ("cancelled", "cancelled", None)
    assert multiprocessing.active_children() == []
    again = wait_until(client, submit(client, video["id"]).json()["id"], finished)
    assert again["state"] == "completed"
    assert again["result"]["cache_hit"] is False
    assert len(attempts) == 2


# Event stream and polling ----------------------------------------------------------------------

def read_events(response, until_end=True):
    """Parse Server-Sent Events from one shared line iterator per response."""
    if not hasattr(response, "spectra_lines"):
        response.spectra_lines = response.iter_lines()
    events, current = [], {}
    for line in response.spectra_lines:
        if not line:
            if current:
                events.append(current)
                if not until_end or current.get("event") == "end":
                    return events
            current = {}
        elif not line.startswith(":"):
            key, _, value = line.partition(": ")
            current[key] = value
    return events


@pytest.fixture
def live(tmp_path, models_dir, library):
    """A real uvicorn server on a free loopback port, for true streaming and disconnects."""
    import socket

    import httpx
    import uvicorn

    from spectra.api.app import create_app
    from spectra.api.settings import Settings
    runner = ScriptedRunner()
    app_ = create_app(Settings(data_dir=tmp_path / "var", models_dir=models_dir, library_dirs=(library,)),
                      pressure=lambda: 1, probe=FakeProbe(), run_worker=runner,
                      job_factory=decode_only(tmp_path / "var" / "cache"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app_, host="127.0.0.1", port=port, log_level="warning", ws="none"))
    thread = threading.Thread(target=server.run)
    thread.start()
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10)
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    client.folder_id = library_folders([library])[0].id
    try:
        yield client, app_, runner
    finally:
        runner.release(10)
        client.close()
        server.should_exit = True
        thread.join(30)


def test_event_stream_follows_progress_to_the_final_state(live):
    client, app_, runner = live
    analysis = submit(client, eligible(client)["id"]).json()
    wait_until(client, analysis["id"], lambda a: a["progress"]["stage"] == "analysing frames")
    with client.stream("GET", f"/api/analyses/{analysis['id']}/events") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        first = read_events(response, until_end=False) + read_events(response, until_end=False)
        runner.release()
        events = first + read_events(response)
    assert events[0] == {"retry": "2000"}
    updates = [e for e in events if e.get("event") == "analysis"]
    assert json.loads(updates[0]["data"])["progress"] == {"stage": "analysing frames", "frames_analysed": 12,
                                                          "media_time_s": 1.2, "fraction": None}
    ids = [int(e["id"]) for e in updates]
    assert ids == sorted(set(ids))
    final = json.loads(updates[-1]["data"])
    assert final["state"] == "completed"
    assert events[-1]["event"] == "end"
    assert final == client.get(f"/api/analyses/{analysis['id']}").json()
    assert app_.state.analyses.subscribers == 0


def test_event_stream_reconnect_sends_only_newer_states(live):
    client, _, runner = live
    runner.release()
    analysis = wait_until(client, submit(client, eligible(client)["id"]).json()["id"], finished)
    url = f"/api/analyses/{analysis['id']}/events"
    with client.stream("GET", url, headers={"last-event-id": str(analysis["revision"])}) as response:
        events = read_events(response)
    assert [e.get("event") for e in events] == [None, "end"]
    with client.stream("GET", url, headers={"last-event-id": "1"}) as response:
        events = read_events(response)
    assert [e.get("event") for e in events] == [None, "analysis", "end"]
    assert json.loads(events[1]["data"]) == client.get(f"/api/analyses/{analysis['id']}").json()
    assert client.get("/api/analyses/" + "0" * 32 + "/events").status_code == 404
    assert client.get(url, headers={"origin": "https://evil.example"}).status_code == 403


def test_disconnected_event_stream_is_released(live):
    client, app_, runner = live
    analysis = submit(client, eligible(client)["id"]).json()
    service = app_.state.analyses
    with client.stream("GET", f"/api/analyses/{analysis['id']}/events") as response:
        read_events(response, until_end=False)
        read_events(response, until_end=False)
        assert service.subscribers == 1
    deadline = time.monotonic() + 5
    while service.subscribers and time.monotonic() < deadline:
        time.sleep(0.02)
    assert service.subscribers == 0
    runner.release()
    assert wait_until(client, analysis["id"], finished)["state"] == "completed"


def test_open_event_stream_does_not_block_shutdown(live):
    client, app_, runner = live
    analysis = submit(client, eligible(client)["id"]).json()
    with client.stream("GET", f"/api/analyses/{analysis['id']}/events") as response:
        read_events(response, until_end=False)
        app_.state.analyses.close()  # what shutdown does first; open streams end on their next poll
        read_events(response, until_end=True)  # returns once the server closes the stream
    assert app_.state.analyses.subscribers == 0
    final = client.get(f"/api/analyses/{analysis['id']}").json()
    assert (final["state"], final["stop_reason"]) == ("cancelled", "shutdown")


def test_stop_signal_with_an_open_event_stream_cancels_the_analysis_and_exits(tmp_path, models_dir, library):
    """uvicorn waits for open responses before the application's shutdown; the configured grace period
    ends that wait, so a followed analysis cannot keep a stopping server alive."""
    import socket

    import httpx
    import uvicorn

    from spectra.api.__main__ import GRACEFUL_SHUTDOWN_S, server_options
    from spectra.api.app import create_app
    from spectra.api.settings import Settings
    runner = ScriptedRunner()  # never released: the analysis would run until cancelled
    app_ = create_app(Settings(data_dir=tmp_path / "var", models_dir=models_dir, library_dirs=(library,)),
                      pressure=lambda: 1, probe=FakeProbe(), run_worker=runner,
                      job_factory=decode_only(tmp_path / "var" / "cache"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app_, **server_options(port, log_level="warning")))
    thread = threading.Thread(target=server.run)
    thread.start()
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10)
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        client.folder_id = library_folders([library])[0].id
        analysis = submit(client, eligible(client)["id"]).json()
        wait_until(client, analysis["id"], lambda a: a["state"] == "running")
        with client.stream("GET", f"/api/analyses/{analysis['id']}/events") as response:
            read_events(response, until_end=False)
            started = time.monotonic()
            server.should_exit = True  # what SIGINT or SIGTERM sets
            thread.join(GRACEFUL_SHUTDOWN_S + 10)
            assert not thread.is_alive()
            assert time.monotonic() - started < GRACEFUL_SHUTDOWN_S + 5
    finally:
        runner.release(10)
        client.close()
        server.should_exit = True
        thread.join(30)
    with sqlite3.connect(tmp_path / "var" / "spectra.sqlite3") as conn:
        state, reason = conn.execute("SELECT state, stop_reason FROM analyses WHERE id = ?", (analysis["id"],)).fetchone()
    assert (state, reason) == ("cancelled", "shutdown")
    assert app_.state.analyses.subscribers == 0


# Restart and shutdown --------------------------------------------------------------------------

def test_restart_settles_interrupted_work_without_resuming_it(app, tmp_path):
    client = app(run_worker=ScriptedRunner())
    a, b = eligible(client, "a.mp4"), eligible(client, "b.mp4")
    with sqlite3.connect(tmp_path / "var" / "spectra.sqlite3") as conn:  # left behind by a crashed server
        for analysis_id, video, state in (("1" * 32, a, "running"), ("2" * 32, b, "queued")):
            conn.execute("INSERT INTO analyses (id, video_id, source_sha256, state, created_at, revision) "
                         "VALUES (?, ?, ?, ?, 'then', 3)", (analysis_id, video["id"], video["media"]["sha256"], state))
    runner = ScriptedRunner()
    restarted = app(run_worker=runner)
    running, queued = (restarted.get(f"/api/analyses/{i}").json() for i in ("1" * 32, "2" * 32))
    assert (running["state"], running["stop_reason"]) == ("failed", "interrupted")
    assert (queued["state"], queued["stop_reason"]) == ("cancelled", "interrupted")
    assert running["revision"] == 4 and running["finished_at"] is not None
    time.sleep(0.1)
    assert runner.calls == []


def test_shutdown_cancels_the_running_worker_and_settles_the_queue(app):
    runner = ScriptedRunner()
    client = app(run_worker=runner)
    running = submit(client, eligible(client, "a.mp4")["id"]).json()
    queued = submit(client, eligible(client, "b.mp4")["id"]).json()
    wait_until(client, running["id"], lambda a: a["state"] == "running")
    database = client.app.state.database
    shutdown(client)
    with database.connect() as conn:
        rows = dict(conn.execute("SELECT id, state || '/' || stop_reason FROM analyses").fetchall())
    assert rows == {running["id"]: "cancelled/shutdown", queued["id"]: "cancelled/interrupted"}
    assert len(runner.calls) == 1


def test_real_worker_is_stopped_on_shutdown(app, library):
    make_video(library / "long.mp4", cfr_times(3000, 50.0), fmt="mp4")
    client = app(job_factory=lambda path, info: WorkerJob(video=str(path), cache_root=str(library.parent / "c")))
    analysis = submit(client, eligible(client, "long.mp4")["id"]).json()
    wait_until(client, analysis["id"], lambda a: a["progress"]["stage"] == "analysing frames" or finished(a))
    database = client.app.state.database
    shutdown(client)
    with database.connect() as conn:
        state = conn.execute("SELECT state || '/' || coalesce(stop_reason, '') FROM analyses WHERE id = ?",
                             (analysis["id"],)).fetchone()[0]
    assert state in ("cancelled/shutdown", "completed/")
    assert multiprocessing.active_children() == []
