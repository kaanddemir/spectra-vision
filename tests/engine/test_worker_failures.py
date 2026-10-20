import json
import os
from pathlib import Path
import subprocess
import sys

from spectra.engine.runner import RunState, StopReason
from spectra.engine.worker import WorkerJob, run_in_worker


def test_invalid_config_fails_before_spawning(tmp_path, monkeypatch):
    import spectra.engine.worker as worker
    video = tmp_path / "exists.mp4"
    video.touch()
    def unexpected(*a, **k):
        raise AssertionError("invalid config must not create a multiprocessing context")
    monkeypatch.setattr(worker.mp, "get_context", unexpected)
    outcome = run_in_worker(WorkerJob(str(video), str(tmp_path / "cache"), config={"unknown": 1}))
    assert outcome.state is RunState.FAILED and outcome.stop is StopReason.ERROR
    assert "unknown" in outcome.error


def test_import_unsafe_script_reports_worker_exit(tmp_path):
    video = tmp_path / "exists.mp4"
    video.touch()
    script = tmp_path / "unsafe.py"
    script.write_text("from spectra.engine.worker import WorkerJob, run_in_worker\n"
                      "import json\n"
                      f"r = run_in_worker(WorkerJob({str(video)!r}, {str(tmp_path / 'cache')!r}), timeout_s=20)\n"
                      "print(json.dumps({'state': r.state.value, 'stop': r.stop.value, 'error': r.error}))\n")
    env = os.environ | {"PYTHONPATH": str(Path(__file__).resolve().parents[2]), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-B", str(script)], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome["state"] == "failed" and outcome["stop"] == "error"
    assert "exited with code" in outcome["error"]
