"""Library folders chosen in the application: the dialog, persistence, removal and safety."""
import sqlite3
import subprocess
from pathlib import Path

import pytest

from spectra.api.library import library_folders
from spectra.api.picker import PickerBusy, PickerUnavailable, choose_folder
from tests.api.conftest import FakeProbe, wait_for_probe

OWN_ORIGIN = "http://127.0.0.1:8765"


class Dialog:
    """Stands in for the native folder dialog: returns the queued choices in order (None = cancelled)."""

    def __init__(self, *choices):
        self.choices = list(choices)
        self.opened = 0

    def __call__(self):
        self.opened += 1
        choice = self.choices.pop(0)
        if isinstance(choice, Exception):
            raise choice
        return choice


@pytest.fixture
def videos_dir(tmp_path):
    root = (tmp_path / "Dashcam").resolve()
    (root / "trips").mkdir(parents=True)
    (root / "drive.mp4").write_bytes(b"drive")
    (root / "trips" / "night.mp4").write_bytes(b"night")
    return root


def add(client, **headers):
    return client.post("/api/library/folders", headers={"origin": OWN_ORIGIN, **headers})


def test_added_folder_is_browsable_at_once_and_remembered_across_restarts(make_client, videos_dir):
    dialog = Dialog(videos_dir)
    client = make_client(pick_folder=dialog, probe=FakeProbe())
    response = add(client)
    assert response.status_code == 201
    folder = response.json()["folder"]
    assert response.json()["created"] is True
    assert (folder["name"], folder["path"], folder["source"], folder["available"], folder["registered_videos"]) == (
        "Dashcam", str(videos_dir), "added", True, 0)
    listing = client.get(f"/api/library/{folder['id']}/entries").json()
    assert [(e["name"], e["kind"]) for e in listing["entries"]] == [("trips", "directory"), ("drive.mp4", "video")]
    video = client.post("/api/videos", json={"folder_id": folder["id"], "path": "trips/night.mp4"})
    assert video.status_code == 201
    wait_for_probe(client, video.json()["id"])
    assert (videos_dir / "trips" / "night.mp4").read_bytes() == b"night"

    restarted = make_client()
    assert [f["id"] for f in restarted.get("/api/library").json()] == [folder["id"]]
    assert restarted.get("/api/library").json()[0]["registered_videos"] == 1


def test_several_folders_and_duplicates(make_client, videos_dir, tmp_path):
    other = (tmp_path / "Archive").resolve()
    other.mkdir()
    client = make_client(pick_folder=Dialog(videos_dir, other, videos_dir / "trips" / "..", None))
    assert add(client).status_code == 201
    assert add(client).status_code == 201
    again = add(client)
    assert again.status_code == 200 and again.json()["created"] is False
    cancelled = add(client)
    assert cancelled.status_code == 200 and cancelled.json() == {"folder": None, "created": False}
    assert [f["name"] for f in client.get("/api/library").json()] == ["Dashcam", "Archive"]


def test_startup_folders_keep_working_and_are_not_removable_or_duplicated(make_client, videos_dir):
    client = make_client(library_dirs=(videos_dir,), pick_folder=Dialog(videos_dir))
    folder_id = library_folders([videos_dir])[0].id
    assert client.get("/api/library").json()[0]["source"] == "startup"
    duplicate = add(client)
    assert duplicate.status_code == 200 and duplicate.json()["folder"]["source"] == "startup"
    assert len(client.get("/api/library").json()) == 1
    assert client.delete(f"/api/library/folders/{folder_id}").status_code == 409
    assert client.get(f"/api/library/{folder_id}/entries").status_code == 200


@pytest.mark.parametrize("choice, status", [
    ("missing", 400), ("file", 400), ("root", 400), ("data", 400)])
def test_unsuitable_choices_are_refused(make_client, tmp_path, videos_dir, choice, status):
    paths = {"missing": tmp_path / "gone", "file": videos_dir / "drive.mp4", "root": Path("/"),
             "data": tmp_path / "var" / "cache"}
    client = make_client(pick_folder=Dialog(paths[choice]))
    assert add(client).status_code == status
    assert client.get("/api/library").json() == []


def test_dialog_problems_are_reported(make_client):
    client = make_client(pick_folder=Dialog(PickerUnavailable("no dialog"), PickerBusy("busy")))
    assert add(client).status_code == 501
    assert add(client).status_code == 409


def test_requests_carry_no_path_and_foreign_pages_are_refused(make_client, videos_dir):
    dialog = Dialog(videos_dir)
    client = make_client(pick_folder=dialog)
    assert add(client, origin="https://evil.example").status_code == 403
    assert client.post("/api/library/folders", headers={"sec-fetch-site": "cross-site"}).status_code == 403
    assert dialog.opened == 0
    # A body naming a path is ignored: the folder always comes from the dialog.
    response = client.post("/api/library/folders", json={"path": "/etc"}, headers={"origin": OWN_ORIGIN})
    assert response.json()["folder"]["path"] == str(videos_dir)
    assert client.delete("/api/library/folders/../etc").status_code == 404
    assert client.delete("/api/library/folders/not-an-id").status_code == 422


def test_removing_a_folder_keeps_files_registrations_and_analyses(make_client, videos_dir, tmp_path):
    client = make_client(pick_folder=Dialog(videos_dir, videos_dir), probe=FakeProbe())
    folder_id = add(client).json()["folder"]["id"]
    video = wait_for_probe(client, client.post("/api/videos", json={"folder_id": folder_id, "path": "drive.mp4"}).json()["id"])
    with sqlite3.connect(tmp_path / "var" / "spectra.sqlite3") as conn:
        conn.execute("INSERT INTO analyses (id, video_id, source_sha256, state, created_at) VALUES ('a', ?, ?, 'completed', 'x')",
                     (video["id"], "0" * 64))
    assert client.get("/api/library").json()[0]["registered_videos"] == 1

    assert client.delete(f"/api/library/folders/{folder_id}").status_code == 204
    assert client.get("/api/library").json() == []
    assert client.get(f"/api/library/{folder_id}/entries").status_code == 404
    kept = client.get(f"/api/videos/{video['id']}").json()
    assert kept["source"]["status"] == "outside_library"
    assert client.get("/api/analyses").json()[0]["id"] == "a"
    assert (videos_dir / "drive.mp4").read_bytes() == b"drive"
    assert client.delete(f"/api/library/folders/{folder_id}").status_code == 404

    # Adding the folder again makes the same registration available again.
    assert add(client).json()["folder"]["id"] == folder_id
    assert client.get(f"/api/videos/{video['id']}").json()["source"]["status"] == "available"


def test_missing_and_moved_folders_are_listed_unavailable(make_client, videos_dir, tmp_path):
    client = make_client(pick_folder=Dialog(videos_dir))
    folder_id = add(client).json()["folder"]["id"]
    moved = tmp_path / "Moved"
    videos_dir.rename(moved)
    folder = client.get("/api/library").json()[0]
    assert (folder["id"], folder["available"]) == (folder_id, False)
    assert client.get(f"/api/library/{folder_id}/entries").status_code == 404
    assert client.post("/api/videos", json={"folder_id": folder_id, "path": "drive.mp4"}).status_code == 404
    assert client.delete(f"/api/library/folders/{folder_id}").status_code == 204
    moved.rename(videos_dir)
    assert client.get("/api/library").json() == []


def test_symlinked_entries_cannot_escape_an_added_folder(make_client, videos_dir, tmp_path):
    (tmp_path / "secret.mp4").write_bytes(b"secret")
    (videos_dir / "link.mp4").symlink_to(tmp_path / "secret.mp4")
    client = make_client(pick_folder=Dialog(videos_dir))
    folder_id = add(client).json()["folder"]["id"]
    names = [e["name"] for e in client.get(f"/api/library/{folder_id}/entries").json()["entries"]]
    assert "link.mp4" not in names
    assert client.post("/api/videos", json={"folder_id": folder_id, "path": "link.mp4"}).status_code == 400


# The native dialog itself ---------------------------------------------------------------------

def completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_dialog_returns_the_chosen_folder_or_none_when_cancelled(monkeypatch):
    import spectra.api.picker as picker
    monkeypatch.setattr(picker, "_command", lambda: ["dialog"])
    assert choose_folder(run=lambda *a, **k: completed(stdout="/Users/me/Movies/\n")) == Path("/Users/me/Movies")
    assert choose_folder(run=lambda *a, **k: completed(1, stderr="User canceled. (-128)")) is None
    with pytest.raises(PickerUnavailable):
        choose_folder(run=lambda *a, **k: completed(1, stderr="no window server"))


def test_dialog_command_is_fixed_and_one_dialog_opens_at_a_time(monkeypatch):
    import spectra.api.picker as picker
    monkeypatch.setattr(picker.sys, "platform", "darwin")
    monkeypatch.setattr(picker.shutil, "which", lambda name: "/usr/bin/osascript" if name == "osascript" else None)
    command = picker._command()
    assert command[:2] == ["osascript", "-e"] and "choose folder" in command[2]
    assert picker._lock.acquire(blocking=False)
    try:
        with pytest.raises(PickerBusy):
            choose_folder(run=lambda *a, **k: completed(stdout="/x"))
    finally:
        picker._lock.release()


def test_no_supported_dialog_is_reported(monkeypatch):
    import spectra.api.picker as picker
    monkeypatch.setattr(picker.sys, "platform", "linux")
    monkeypatch.setattr(picker.shutil, "which", lambda name: None)
    with pytest.raises(PickerUnavailable, match="--library"):
        choose_folder()
