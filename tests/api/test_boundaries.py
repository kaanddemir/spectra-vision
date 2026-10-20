import ast
import subprocess
import sys
from importlib.util import resolve_name
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Perception and model runtimes belong in analysis workers; the API only orchestrates them.
FORBIDDEN = ("evaluation", "onnxruntime", "cv2", "torch", "supervision", "trackers",
             "spectra.engine.detection", "spectra.engine.tracking")


def imported_modules(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            parent = ".".join(path.relative_to(ROOT).parts[:-1])
            base = resolve_name("." * node.level + (node.module or ""), parent) if node.level else node.module
            yield base
            yield from (f"{base}.{alias.name}" for alias in node.names)


def test_api_imports_only_orchestration_dependencies():
    violations = [(str(path), module) for path in (ROOT / "spectra" / "api").rglob("*.py")
                  for module in imported_modules(path)
                  if any(module == p or module.startswith(p + ".") for p in FORBIDDEN)]
    assert not violations


def test_evaluation_never_imports_api():
    tracked = subprocess.run(["git", "ls-files", "evaluation/*.py", "scripts/*.py"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    assert tracked
    violations = [(path, module) for path in tracked for module in imported_modules(ROOT / path)
                  if module == "spectra.api" or module.startswith("spectra.api.")]
    assert not violations


def test_serving_and_analysing_load_no_model_or_decoder_runtime(tmp_path):
    """A long-lived API process must not load runtimes: RSS budgets use the lifetime peak.

    Probing and analysis run in child processes; the parent only orchestrates them.
    """
    from tests.engine.media import cfr_times, make_video
    library = tmp_path / "library"
    library.mkdir()
    make_video(library / "clip.mp4", cfr_times(10, 10.0), fmt="mp4")
    script = f"""
import sys, time
from pathlib import Path
from fastapi.testclient import TestClient
from spectra.api.app import create_app
from spectra.api.library import library_folders
from spectra.api.settings import Settings
from spectra.engine.worker import WorkerJob
from tests.api.conftest import FakeProbe, write_registry, wait_for_probe

def main():
    root = Path({str(tmp_path)!r})
    library = (root / "library").resolve()
    settings = Settings(data_dir=root / "var", models_dir=write_registry(root / "models"), library_dirs=(library,))
    job = lambda path, info: WorkerJob(video=str(path), cache_root=str(root / "var" / "cache"), config={{"interval_s": 0.1}})
    app = create_app(settings, pressure=lambda: 1, probe=FakeProbe(), job_factory=job)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        assert client.get("/api/health").json()["models"]["available"]
        folder = library_folders([library])[0].id
        video = client.post("/api/videos", json={{"folder_id": folder, "path": "clip.mp4"}}).json()
        wait_for_probe(client, video["id"])
        client.post(f"/api/videos/{{video['id']}}/playback-check", json={{"playable": True}})
        analysis = client.post("/api/analyses", json={{"video_id": video["id"]}}).json()
        deadline = time.monotonic() + 60
        while client.get(f"/api/analyses/{{analysis['id']}}").json()["evidence"]["state"] in ("pending",) or \
                client.get(f"/api/analyses/{{analysis['id']}}").json()["state"] in ("queued", "running"):
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert client.get(f"/api/analyses/{{analysis['id']}}").json()["state"] == "completed"
        timeline = client.get(f"/api/analyses/{{analysis['id']}}/timeline").json()
        assert timeline["state"] == "complete" and timeline["entries"] == []
    loaded = sorted(m for m in ("onnxruntime", "cv2", "av", "torch", "supervision", "trackers") if m in sys.modules)
    assert not loaded, loaded

if __name__ == "__main__":
    main()
"""
    path = tmp_path / "serve_and_analyse.py"
    path.write_text(script)
    result = subprocess.run([sys.executable, "-B", str(path)], cwd=ROOT, capture_output=True, text=True, timeout=120,
                            env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    assert result.returncode == 0, result.stderr
