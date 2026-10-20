import hashlib
import json

import pytest

BASE_URL = "http://127.0.0.1:8765"


def write_registry(root):
    """Two tiny files with matching hashes: a valid registry that never needs real weights."""
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for role, shape in [("detector", [1, 3, 640, 640]), ("segmenter", [1, 3, 384, 512])]:
        path = root / f"{role}.onnx"
        path.write_bytes(role.encode())
        rows.append(dict(role=role, file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         input=shape, licence="test", export_script="test.py", source_checkpoint="test"))
    (root / "models.json").write_text(json.dumps({"models": rows}))
    return root


@pytest.fixture
def models_dir(tmp_path):
    return write_registry(tmp_path / "models")


@pytest.fixture
def make_client(tmp_path, models_dir):
    """Build a started TestClient for an app whose state lives under tmp_path."""
    from fastapi.testclient import TestClient

    from spectra.api.app import create_app
    from spectra.api.settings import Settings

    clients = []

    def make(pressure=lambda: 1, probe=None, job_factory=None, run_worker=None, pick_folder=None, **overrides):
        options = dict(data_dir=tmp_path / "var", models_dir=models_dir) | overrides
        app = create_app(Settings(**options), pressure=pressure, probe=probe, job_factory=job_factory,
                         run_worker=run_worker, pick_folder=pick_folder or no_dialog)
        client = TestClient(app, base_url=BASE_URL)
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for client in clients:
        shutdown(client)


def no_dialog():
    raise AssertionError("tests never open the native folder dialog")


def shutdown(client):
    """Run the application's shutdown once, also when a test already did."""
    if not getattr(client, "shut_down", False):
        client.shut_down = True
        client.__exit__(None, None, None)


MP4 = "mov,mp4,m4a,3gp,3g2,mj2"
MATROSKA = "matroska,webm"


class FakeProbe:
    """Stands in for the helper process: real stat and hash, declared container and codec.

    ``formats`` maps file names to (container, codec, audio codecs) or to a probe error kind.
    """

    def __init__(self, formats=None):
        self.formats = formats or {}
        self.calls = []

    def __call__(self, path, timeout_s):
        from spectra.api.probe import ProbeReport, file_stat
        from spectra.engine.contracts import TimestampSource, VideoInfo
        from spectra.engine.video import file_sha256
        self.calls.append(path)
        spec = self.formats.get(path.name, (MP4, "h264", ()))
        if isinstance(spec, str):
            return ProbeReport(error=spec, message=f"{path.name}: simulated {spec}")
        container, codec, audio = spec
        stat = file_stat(path)
        info = VideoInfo(file_name=path.name, size_bytes=stat.size_bytes, sha256=file_sha256(path),
                         container=container, codec=codec, width=64, height=48, coded_width=64, coded_height=48,
                         rotation=0, duration_s=2.0, nominal_fps=25.0,
                         timestamp_source=TimestampSource.CONTAINER)
        return ProbeReport(info=info, audio_codecs=list(audio), stat=stat)


@pytest.fixture
def library_root(tmp_path):
    root = (tmp_path / "library").resolve()
    (root / "trips" / "day").mkdir(parents=True)
    (root / "drive.mp4").write_bytes(b"0123456789" * 10)
    (root / "trips" / "dash.webm").write_bytes(b"webm-bytes")
    (root / "trips" / "day" / "old.mov").write_bytes(b"mov-bytes")
    (root / "notes.txt").write_text("not a video")
    (root / ".hidden.mp4").write_bytes(b"hidden")
    (tmp_path / "outside.mp4").write_bytes(b"outside")
    (root / "escape.mp4").symlink_to(tmp_path / "outside.mp4")
    return root


def wait_for_probe(client, video_id, timeout_s=10.0):
    import time
    deadline = time.monotonic() + timeout_s
    while True:
        video = client.get(f"/api/videos/{video_id}").json()
        if video["probe"]["state"] != "pending" or time.monotonic() > deadline:
            return video
        time.sleep(0.01)
