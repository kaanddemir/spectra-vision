"""The isolated media-probe helper, run as a real subprocess on small synthetic files."""
import hashlib
import os
import sys
import threading
import time

import pytest

from spectra.api.probe import ProbeRunner
from tests.api.conftest import MATROSKA, MP4
from tests.engine.media import H, W, cfr_times, make_video


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.parametrize("name, fmt, container", [("clip.mp4", "mp4", MP4), ("clip.mkv", "matroska", MATROSKA)])
def test_helper_reports_identity_and_format(tmp_path, name, fmt, container):
    path = make_video(tmp_path / name, cfr_times(10, 10.0), fmt=fmt)
    report = ProbeRunner().run(path, 60)
    assert report.error is None, report.message
    assert report.info.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (report.info.container, report.info.codec) == (container, "mpeg4")
    assert (report.info.width, report.info.height) == (W, H)
    assert report.audio_codecs == []
    assert report.stat.size_bytes == path.stat().st_size and report.stat.inode == path.stat().st_ino


@pytest.mark.parametrize("content, kinds", [
    (b"", {"corrupt", "unsupported"}),
    (b"\x00\x01garbage" * 500, {"corrupt", "unsupported"}),
])
def test_helper_reports_unreadable_media(tmp_path, content, kinds):
    path = tmp_path / "broken.mp4"
    path.write_bytes(content)
    report = ProbeRunner().run(path, 60)
    assert report.info is None
    assert report.error in kinds
    assert "broken.mp4" in report.message


def test_helper_handles_truncated_media(tmp_path):
    source = make_video(tmp_path / "full.mp4", cfr_times(20, 10.0), fmt="mp4")
    truncated = tmp_path / "truncated.mp4"
    truncated.write_bytes(source.read_bytes()[: source.stat().st_size // 2])
    report = ProbeRunner().run(truncated, 60)
    assert (report.info is None) == (report.error is not None)


def test_helper_reports_missing_file(tmp_path):
    report = ProbeRunner().run(tmp_path / "absent.mp4", 60)
    assert report.error == "not_found"


def test_helper_never_modifies_the_source(tmp_path):
    path = make_video(tmp_path / "clip.mp4", cfr_times(5, 10.0), fmt="mp4")
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    ProbeRunner().run(path, 60)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_crashing_helper_is_reported(tmp_path):
    runner = ProbeRunner([sys.executable, "-c", "import os; os._exit(3)"])
    report = runner.run(tmp_path / "x.mp4", 30)
    assert report.error == "failed" and "exit status 3" in report.message


def test_invalid_helper_output_is_reported(tmp_path):
    report = ProbeRunner([sys.executable, "-c", "print('not json')"]).run(tmp_path / "x.mp4", 30)
    assert report.error == "failed" and "invalid report" in report.message


HANGING_HELPER = """
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
with open(sys.argv[1], "w") as f:
    f.write(f"{os.getpid()} {child.pid}")
time.sleep(60)
"""


def wait_for_pids(path):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if path.exists() and len(path.read_text().split()) == 2:
            return [int(p) for p in path.read_text().split()]
        time.sleep(0.05)
    raise AssertionError("helper did not start")


def wait_until_dead(pids):
    deadline = time.monotonic() + 10
    while any(alive(p) for p in pids) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not any(alive(p) for p in pids)


def test_timeout_kills_the_helper_and_its_children(tmp_path):
    pid_file = tmp_path / "pids"
    report = ProbeRunner([sys.executable, "-c", HANGING_HELPER]).run(pid_file, 3)
    assert report.error == "timeout"
    assert wait_until_dead(wait_for_pids(pid_file))


def test_close_kills_a_running_helper_and_refuses_new_work(tmp_path):
    pid_file = tmp_path / "pids"
    runner = ProbeRunner([sys.executable, "-c", HANGING_HELPER])
    result = []
    thread = threading.Thread(target=lambda: result.append(runner.run(pid_file, 60)))
    thread.start()
    pids = wait_for_pids(pid_file)
    runner.close()
    thread.join(10)
    assert result[0].error == "cancelled"
    assert wait_until_dead(pids)
    assert runner.run(pid_file, 60).error == "cancelled"


def test_registration_uses_the_real_helper_end_to_end(make_client, tmp_path):
    from spectra.api.library import library_folders
    from tests.api.conftest import wait_for_probe
    library = tmp_path / "videos"
    library.mkdir()
    make_video(library / "clip.mp4", cfr_times(10, 10.0), fmt="mp4")
    (library / "broken.webm").write_bytes(b"not a webm file" * 100)
    client = make_client(library_dirs=(library,))
    folder_id = library_folders([library.resolve()])[0].id
    ok = client.post("/api/videos", json={"folder_id": folder_id, "path": "clip.mp4"}).json()
    broken = client.post("/api/videos", json={"folder_id": folder_id, "path": "broken.webm"}).json()
    ok, broken = wait_for_probe(client, ok["id"], 60), wait_for_probe(client, broken["id"], 60)
    assert ok["probe"]["state"] == "ready"
    assert (ok["media"]["container"], ok["media"]["codec"]) == (MP4, "mpeg4")
    assert ok["playback"]["assessment"] == "unsupported"
    assert broken["probe"]["state"] == "failed"
    assert broken["analysis"]["eligible"] is False
