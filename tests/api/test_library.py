import os

import pytest

from spectra.api.library import LibraryPathError, library_folders, resolve_video_reference


@pytest.fixture
def library(tmp_path):
    root = (tmp_path / "library").resolve()
    (root / "trips").mkdir(parents=True)
    (root / "trips" / "drive.mp4").write_bytes(b"source")
    (tmp_path / "outside.mp4").write_bytes(b"secret")
    return root, library_folders([root])


def test_folder_ids_are_stable_and_distinct(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    first = library_folders([a, b])
    assert [f.id for f in first] == [f.id for f in reversed(library_folders([b, a]))]
    assert first[0].id != first[1].id
    assert first[0].name == "a"


def test_resolves_files_inside_the_folder_without_changing_them(library):
    root, folders = library
    source = root / "trips" / "drive.mp4"
    before = source.stat()
    assert resolve_video_reference(folders, folders[0].id, "trips/drive.mp4") == source
    assert resolve_video_reference(folders, folders[0].id, "./trips//drive.mp4") == source
    after = source.stat()
    assert (after.st_mtime_ns, after.st_size, source.read_bytes()) == (before.st_mtime_ns, 6, b"source")


def test_symlink_inside_the_folder_resolves_to_its_target(library):
    root, folders = library
    (root / "latest.mp4").symlink_to(root / "trips" / "drive.mp4")
    assert resolve_video_reference(folders, folders[0].id, "latest.mp4") == root / "trips" / "drive.mp4"


@pytest.mark.parametrize("reference, message", [
    ("", "required"), ("trips/\x00.mp4", "invalid character"), ("../outside.mp4", "inside"),
    ("trips/../../outside.mp4", "inside"), ("/etc/hosts", "inside"), (".", "required"),
    ("trips/absent.mp4", "not found"), ("trips", "regular file"),
])
def test_invalid_references_are_refused(library, reference, message):
    _, folders = library
    with pytest.raises(LibraryPathError, match=message):
        resolve_video_reference(folders, folders[0].id, reference)


def test_unknown_folder_is_refused(library):
    _, folders = library
    with pytest.raises(LibraryPathError, match="unknown library folder"):
        resolve_video_reference(folders, "0" * 16, "trips/drive.mp4")


def test_symlink_escaping_the_folder_is_refused(library, tmp_path):
    root, folders = library
    (root / "escape.mp4").symlink_to(tmp_path / "outside.mp4")
    (root / "escape_dir").symlink_to(tmp_path)
    for reference in ("escape.mp4", "escape_dir/outside.mp4"):
        with pytest.raises(LibraryPathError, match="inside"):
            resolve_video_reference(folders, folders[0].id, reference)


def test_fifo_is_refused_without_opening_it(library):
    root, folders = library
    os.mkfifo(root / "pipe.mp4")
    with pytest.raises(LibraryPathError, match="regular file"):
        resolve_video_reference(folders, folders[0].id, "pipe.mp4")


def test_application_data_is_never_a_source(library):
    root, folders = library
    data = root / "state"
    (data / "evidence").mkdir(parents=True)
    (data / "evidence" / "clip.mp4").write_bytes(b"clip")
    with pytest.raises(LibraryPathError, match="application data"):
        resolve_video_reference(folders, folders[0].id, "state/evidence/clip.mp4", exclude=[data])


def test_unreadable_file_is_refused(library):
    root, folders = library
    locked = root / "locked.mp4"
    locked.write_bytes(b"x")
    locked.chmod(0)
    try:
        if os.access(locked, os.R_OK):
            pytest.skip("running with privileges that ignore file permissions")
        with pytest.raises(LibraryPathError, match="not readable"):
            resolve_video_reference(folders, folders[0].id, "locked.mp4")
    finally:
        locked.chmod(0o600)


def test_library_endpoint_lists_configured_folders(make_client, library):
    root, folders = library
    body = make_client(library_dirs=(root,)).get("/api/library").json()
    assert body == [{"id": folders[0].id, "name": "library", "path": str(root), "source": "startup",
                     "available": True, "registered_videos": 0}]
    assert make_client().get("/api/library").json() == []


def browse(client, folder_id, path=None):
    return client.get(f"/api/library/{folder_id}/entries", params={} if path is None else {"path": path})


def test_browsing_lists_subfolders_and_video_candidates_only(make_client, library_root):
    from tests.api.conftest import FakeProbe
    folder_id = library_folders([library_root])[0].id
    client = make_client(probe=FakeProbe(), library_dirs=(library_root,))
    body = browse(client, folder_id).json()
    assert body["folder"]["id"] == folder_id
    assert (body["path"], body["parent"], body["truncated"]) == ("", None, False)
    assert [(e["name"], e["kind"], e["path"]) for e in body["entries"]] == [
        ("trips", "directory", "trips"), ("drive.mp4", "video", "drive.mp4")]
    video = body["entries"][1]
    assert video["size_bytes"] == 100 and video["modified_at"] and video["video_id"] is None
    nested = browse(client, folder_id, "trips/day").json()
    assert (nested["path"], nested["parent"]) == ("trips/day", "trips")
    assert [e["path"] for e in nested["entries"]] == ["trips/day/old.mov"]
    assert browse(client, folder_id, "trips").json()["parent"] == ""
    assert str(library_root) not in str(nested["entries"])


def test_browsing_marks_registered_videos(make_client, library_root):
    from tests.api.conftest import FakeProbe
    folder_id = library_folders([library_root])[0].id
    client = make_client(probe=FakeProbe(), library_dirs=(library_root,))
    video_id = client.post("/api/videos", json={"folder_id": folder_id, "path": "drive.mp4"}).json()["id"]
    entries = browse(client, folder_id).json()["entries"]
    assert entries[1]["video_id"] == video_id


@pytest.mark.parametrize("path, status", [("..", 400), ("trips/../..", 400), ("/etc", 400), ("absent", 404),
                                          ("drive.mp4", 400), ("x" * 5000, 422)])
def test_browsing_refuses_paths_outside_folders(make_client, library_root, path, status):
    folder_id = library_folders([library_root])[0].id
    client = make_client(library_dirs=(library_root,))
    assert browse(client, folder_id, path).status_code == status


def test_browsing_unknown_folder(make_client, library_root):
    client = make_client(library_dirs=(library_root,))
    assert browse(client, "0" * 16).status_code == 404
    assert browse(client, "not-a-folder").status_code == 422


def test_browsing_skips_escaping_directory_links_and_data_directory(make_client, library_root, tmp_path):
    (library_root / "linked").symlink_to(tmp_path)
    data = library_root / "state"
    folder_id = library_folders([library_root])[0].id
    client = make_client(library_dirs=(library_root,), data_dir=data)
    names = [e["name"] for e in browse(client, folder_id).json()["entries"]]
    assert "linked" not in names and "state" not in names and "escape.mp4" not in names
    assert browse(client, folder_id, "state").status_code == 400


def test_browsing_truncates_large_folders(make_client, library_root, monkeypatch):
    import spectra.api.library as library
    monkeypatch.setattr(library, "MAX_ENTRIES", 1)
    folder_id = library_folders([library_root])[0].id
    body = browse(make_client(library_dirs=(library_root,)), folder_id).json()
    assert body["truncated"] is True and len(body["entries"]) == 1


def test_unsupported_extension_is_refused(library):
    root, folders = library
    (root / "notes.txt").write_text("x")
    with pytest.raises(LibraryPathError, match="not a supported video file type") as error:
        resolve_video_reference(folders, folders[0].id, "notes.txt")
    assert error.value.status_code == 415
