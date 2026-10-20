import hashlib
import os

import pytest

from spectra.api.library import library_folders
from tests.api.conftest import MATROSKA, MP4, FakeProbe, wait_for_probe


@pytest.fixture
def setup(make_client, library_root):
    """Client factory over one library folder with a fake probe; returns (make, folder_id, probe)."""
    probe = FakeProbe({"dash.webm": (MATROSKA, "vp9", ("opus",)), "old.mov": (MP4, "mpeg4", ())})
    folder_id = library_folders([library_root])[0].id

    def make(**overrides):
        return make_client(probe=overrides.pop("probe", probe), library_dirs=(library_root,), **overrides)
    return make, folder_id, probe


def register(client, folder_id, path):
    return client.post("/api/videos", json={"folder_id": folder_id, "path": path})


def registered(client, folder_id, path):
    response = register(client, folder_id, path)
    assert response.status_code in (200, 201), response.text
    return wait_for_probe(client, response.json()["id"])


def test_registration_probes_in_background_and_reports_metadata(setup, library_root):
    make, folder_id, probe = setup
    client = make()
    response = register(client, folder_id, "drive.mp4")
    assert response.status_code == 201
    video = wait_for_probe(client, response.json()["id"])
    assert video["probe"]["state"] == "ready"
    assert video["file_name"] == "drive.mp4"
    assert (video["folder_id"], video["path"]) == (folder_id, "drive.mp4")
    assert video["size_bytes"] == 100
    media = video["media"]
    assert media["sha256"] == hashlib.sha256((library_root / "drive.mp4").read_bytes()).hexdigest()
    assert (media["container"], media["codec"], media["width"], media["height"]) == (MP4, "h264", 64, 48)
    assert media["duration_s"] == 2.0 and media["nominal_fps"] == 25.0
    assert video["source"] == {"status": "available", "message": None}
    assert video["playback"]["assessment"] == "likely"
    assert video["playback"]["mime_type"] == "video/mp4"
    assert video["media_url"] == f"/api/videos/{video['id']}/media"
    assert video["analysis"] == {"eligible": False, "reasons": [
        "Play the video in this browser to confirm it can be reviewed."]}
    assert str(library_root) not in str(video)
    assert client.get("/api/videos").json() == [video]


def test_duplicate_registration_returns_the_existing_video(setup, library_root):
    make, folder_id, probe = setup
    client = make()
    first = registered(client, folder_id, "drive.mp4")
    (library_root / "alias.mp4").symlink_to(library_root / "drive.mp4")
    for path in ("drive.mp4", "./drive.mp4", "alias.mp4"):
        response = register(client, folder_id, path)
        assert response.status_code == 200
        assert response.json()["id"] == first["id"]
    assert len(probe.calls) == 1
    assert len(client.get("/api/videos").json()) == 1


@pytest.mark.parametrize("path, status", [
    ("../outside.mp4", 400), ("escape.mp4", 400), ("/etc/hosts", 400), ("missing.mp4", 404),
    ("notes.txt", 415), ("trips", 400), ("", 422)])
def test_invalid_registrations_are_refused(setup, path, status):
    make, folder_id, probe = setup
    client = make()
    assert register(client, folder_id, path).status_code == status
    assert client.get("/api/videos").json() == []
    assert probe.calls == []


def test_unknown_or_malformed_folder_is_refused(setup):
    make, _, _ = setup
    client = make()
    assert register(client, "0" * 16, "drive.mp4").status_code == 404
    assert register(client, "../etc", "drive.mp4").status_code == 422
    assert client.post("/api/videos", json={"folder_id": "0" * 16, "path": "a.mp4", "extra": 1}).status_code == 422


def test_application_data_inside_a_library_cannot_be_registered(make_client, library_root):
    data = library_root / "state"
    client = make_client(probe=FakeProbe(), library_dirs=(library_root,), data_dir=data)
    (data / "evidence" / "clip.mp4").write_bytes(b"clip")
    folder_id = library_folders([library_root])[0].id
    assert register(client, folder_id, "state/evidence/clip.mp4").status_code == 400


def test_changed_file_is_refused_until_registered_again(setup, library_root):
    make, folder_id, probe = setup
    client = make()
    video = registered(client, folder_id, "drive.mp4")
    client.post(f"/api/videos/{video['id']}/playback-check", json={"playable": True})
    with open(library_root / "drive.mp4", "ab") as f:
        f.write(b"appended")
    changed = client.get(f"/api/videos/{video['id']}").json()
    assert changed["source"]["status"] == "changed"
    assert changed["media_url"] is None
    assert changed["analysis"]["eligible"] is False
    assert any("changed" in reason for reason in changed["analysis"]["reasons"])
    assert client.get(f"/api/videos/{video['id']}/media").status_code == 409
    again = register(client, folder_id, "drive.mp4")
    assert again.status_code == 200 and again.json()["id"] == video["id"]
    reprobed = wait_for_probe(client, video["id"])
    assert len(probe.calls) == 2
    assert reprobed["source"]["status"] == "available"
    assert reprobed["size_bytes"] == 108
    assert reprobed["media"]["sha256"] != video["media"]["sha256"]
    assert reprobed["playback"]["browser_check"] is None


def test_same_size_rewrite_is_detected(setup, library_root):
    make, folder_id, _ = setup
    client = make()
    video = registered(client, folder_id, "drive.mp4")
    source = library_root / "drive.mp4"
    stat = source.stat()
    replacement = library_root / "new.tmp"
    replacement.write_bytes(b"9876543210" * 10)
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    replacement.replace(source)  # same size and time, different inode
    assert client.get(f"/api/videos/{video['id']}").json()["source"]["status"] == "changed"


def test_deleted_or_moved_file_is_reported_missing(setup, library_root):
    make, folder_id, _ = setup
    client = make()
    video = registered(client, folder_id, "drive.mp4")
    (library_root / "drive.mp4").rename(library_root / "moved.mp4")
    missing = client.get(f"/api/videos/{video['id']}").json()
    assert missing["source"] == {"status": "missing", "message": "drive.mp4 was moved or deleted"}
    assert missing["analysis"]["eligible"] is False
    assert client.get(f"/api/videos/{video['id']}/media").status_code == 409


def test_replacing_a_file_with_a_link_is_detected(setup, library_root):
    make, folder_id, _ = setup
    client = make()
    video = registered(client, folder_id, "drive.mp4")
    (library_root / "drive.mp4").rename(library_root / "real.mp4")
    (library_root / "drive.mp4").symlink_to(library_root / "real.mp4")
    assert client.get(f"/api/videos/{video['id']}").json()["source"]["status"] == "changed"


def test_registrations_persist_and_respect_current_library_folders(setup, make_client, tmp_path):
    make, folder_id, probe = setup
    video = registered(make(), folder_id, "drive.mp4")
    restarted = make()
    assert restarted.get(f"/api/videos/{video['id']}").json() == video
    other = tmp_path / "other"
    other.mkdir()
    elsewhere = make_client(probe=probe, library_dirs=(other,))
    outside = elsewhere.get(f"/api/videos/{video['id']}").json()
    assert outside["source"]["status"] == "outside_library"
    assert outside["folder_id"] is None and outside["path"] is None
    assert elsewhere.get(f"/api/videos/{video['id']}/media").status_code == 409


def test_interrupted_probe_resumes_after_restart(setup):
    make, folder_id, probe = setup
    first = make(probe=FakeProbe({"drive.mp4": "cancelled"}))
    video_id = register(first, folder_id, "drive.mp4").json()["id"]
    assert wait_for_probe(first, video_id, timeout_s=0.5)["probe"]["state"] == "pending"
    resumed = wait_for_probe(make(), video_id)
    assert resumed["probe"]["state"] == "ready"


@pytest.mark.parametrize("error, retried", [("corrupt", False), ("unsupported", False), ("timeout", True),
                                            ("failed", True)])
def test_probe_failures_are_reported_and_retried_only_when_temporary(setup, error, retried):
    make, folder_id, _ = setup
    probe = FakeProbe({"drive.mp4": error})
    client = make(probe=probe)
    video = registered(client, folder_id, "drive.mp4")
    assert video["probe"]["state"] == "failed"
    assert video["probe"]["error"] == error
    assert video["media"] is None and video["media_url"] is None
    assert video["playback"]["assessment"] == "unknown"
    assert video["analysis"]["reasons"] == [f"drive.mp4: simulated {error}"]
    assert client.get(f"/api/videos/{video['id']}/media").status_code == 409
    register(client, folder_id, "drive.mp4")
    wait_for_probe(client, video["id"])
    assert len(probe.calls) == (2 if retried else 1)


def test_unsupported_codec_blocks_playback_and_analysis(setup):
    make, folder_id, _ = setup
    client = make()
    video = registered(client, folder_id, "trips/day/old.mov")
    assert video["probe"]["state"] == "ready"
    assert video["playback"]["assessment"] == "unsupported"
    assert video["media_url"] is None
    assert video["analysis"]["eligible"] is False
    assert "mpeg4" in video["analysis"]["reasons"][0]
    assert client.get(f"/api/videos/{video['id']}/media").status_code == 409
    assert client.post(f"/api/videos/{video['id']}/playback-check", json={"playable": True}).status_code == 409


def test_browser_check_decides_eligibility(setup):
    make, folder_id, _ = setup
    client = make()
    video = registered(client, folder_id, "trips/dash.webm")
    assert video["playback"]["assessment"] == "likely"
    assert video["playback"]["mime_type"] == "video/webm"
    url = f"/api/videos/{video['id']}/playback-check"
    failed = client.post(url, json={"playable": False, "detail": "MEDIA_ERR_SRC_NOT_SUPPORTED"},
                         headers={"user-agent": "TestBrowser/1"}).json()
    assert failed["playback"]["browser_check"]["playable"] is False
    assert failed["playback"]["browser_check"]["user_agent"] == "TestBrowser/1"
    assert failed["analysis"]["eligible"] is False
    assert "could not play" in failed["analysis"]["reasons"][0]
    passed = client.post(url, json={"playable": True}).json()
    assert passed["analysis"] == {"eligible": True, "reasons": []}
    assert client.post(url, json={"playable": True, "detail": "x" * 501}).status_code == 422
    assert client.post(url, json={}).status_code == 422


def test_browser_check_is_refused_for_contents_other_than_the_current_file(setup):
    make, folder_id, _ = setup
    client = make()
    video = registered(client, folder_id, "trips/dash.webm")
    url = f"/api/videos/{video['id']}/playback-check"
    stale = client.post(url, json={"playable": True, "sha256": "0" * 64})
    assert stale.status_code == 409
    assert "changed after it was loaded" in stale.json()["detail"]
    assert client.get(f"/api/videos/{video['id']}").json()["playback"]["browser_check"] is None
    assert client.post(url, json={"playable": True, "sha256": "not-a-hash"}).status_code == 422
    current = client.post(url, json={"playable": True, "sha256": video["media"]["sha256"]}).json()
    assert current["analysis"] == {"eligible": True, "reasons": []}


def test_media_supports_full_partial_and_head_requests(setup, library_root):
    make, folder_id, _ = setup
    client = make()
    video = registered(client, folder_id, "drive.mp4")
    source = library_root / "drive.mp4"
    before = (source.stat().st_mtime_ns, source.read_bytes())
    url = video["media_url"]
    full = client.get(url)
    assert full.status_code == 200
    assert full.content == before[1]
    assert full.headers["content-type"] == "video/mp4"
    assert full.headers["accept-ranges"] == "bytes"
    assert full.headers["x-content-type-options"] == "nosniff"
    assert full.headers["cache-control"] == "private, no-cache"
    part = client.get(url, headers={"range": "bytes=10-19"})
    assert part.status_code == 206
    assert part.content == before[1][10:20]
    assert part.headers["content-range"] == "bytes 10-19/100"
    tail = client.get(url, headers={"range": "bytes=-5"})
    assert (tail.status_code, tail.content) == (206, before[1][-5:])
    open_ended = client.get(url, headers={"range": "bytes=95-"})
    assert (open_ended.status_code, open_ended.content) == (206, before[1][95:])
    head = client.head(url)
    assert head.status_code == 200 and head.content == b"" and head.headers["content-length"] == "100"
    assert (source.stat().st_mtime_ns, source.read_bytes()) == before


@pytest.mark.parametrize("header, status", [("bytes=100-", 416), ("bytes=500-600", 416), ("items=0-1", 400),
                                            ("bytes=9-3", 400), ("bytes=", 400)])
def test_invalid_ranges_are_rejected(setup, header, status):
    make, folder_id, _ = setup
    client = make()
    video = registered(client, folder_id, "drive.mp4")
    response = client.get(video["media_url"], headers={"range": header})
    assert response.status_code == status
    if status == 416:
        assert response.headers["content-range"] == "bytes */100"


def test_media_for_unknown_or_malformed_ids(setup):
    make, _, _ = setup
    client = make()
    assert client.get("/api/videos/" + "0" * 32 + "/media").status_code == 404
    assert client.get("/api/videos/../../etc/passwd").status_code == 404
    assert client.get("/api/videos/not-an-id").status_code == 422


def test_removing_a_registration_never_touches_the_file(setup, make_client, library_root, tmp_path):
    make, folder_id, _ = setup
    client = make()
    video = registered(client, folder_id, "drive.mp4")
    assert client.delete(f"/api/videos/{video['id']}").status_code == 204
    assert client.get(f"/api/videos/{video['id']}").status_code == 404
    assert client.delete(f"/api/videos/{video['id']}").status_code == 404
    assert (library_root / "drive.mp4").read_bytes() == b"0123456789" * 10
    again = registered(client, folder_id, "drive.mp4")
    import sqlite3
    with sqlite3.connect(tmp_path / "var" / "spectra.sqlite3") as conn:
        conn.execute("INSERT INTO analyses (id, video_id, source_sha256, state, created_at) "
                     "VALUES ('a', ?, ?, 'completed', 'now')", (again["id"], again["media"]["sha256"]))
    assert client.delete(f"/api/videos/{again['id']}").status_code == 409


def test_cross_site_pages_cannot_register_or_read_media(setup):
    make, folder_id, _ = setup
    client = make()
    video = registered(client, folder_id, "drive.mp4")
    assert client.post("/api/videos", json={"folder_id": folder_id, "path": "drive.mp4"},
                       headers={"origin": "https://evil.example"}).status_code == 403
    assert client.get(video["media_url"], headers={"sec-fetch-site": "cross-site"}).status_code == 403


def test_pending_registration_without_probe_token_is_probed_on_start(setup, library_root, tmp_path):
    import sqlite3
    make, folder_id, probe = setup
    make()  # creates the database
    stat = (library_root / "drive.mp4").stat()
    with sqlite3.connect(tmp_path / "var" / "spectra.sqlite3") as conn:
        conn.execute("INSERT INTO videos (id, path, size_bytes, mtime_ns, inode, added_at) VALUES (?, ?, ?, ?, ?, ?)",
                     ("f" * 32, str(library_root / "drive.mp4"), stat.st_size, stat.st_mtime_ns, stat.st_ino, "then"))
    assert wait_for_probe(make(), "f" * 32)["probe"]["state"] == "ready"
