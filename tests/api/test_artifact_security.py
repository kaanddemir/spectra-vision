"""Tampered local artifact references must never read files outside their entry."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from spectra.api.evidence import EvidenceNotFound, EvidenceService
from spectra.engine.cache import PerceptionCache
from spectra.engine.contracts import SegmentationRef
from spectra.evidence.media import ClipFile, load_clip
from spectra.evidence.store import EvidenceStore
from tests.engine.cache_fixtures import result, store
from tests.evidence import synth
from spectra.evidence.build import build


def clip_record(file):
    return dict(file=file, sha256=hashlib.sha256(b"private").hexdigest(), size_bytes=7,
                format="mp4", codec="h264", source_sha256="a" * 64, start_s=1.0, end_s=2.0,
                origin_s=1.0, first_s=1.0, last_s=2.0, frames=1)


@pytest.mark.parametrize("file", ["../private.txt", "/tmp/private.txt", "nested/clip.mp4",
                                  "..\\private.txt", "", ".", "..", "bad\x00.mp4"])
def test_clip_record_rejects_non_filenames(file):
    with pytest.raises(ValueError, match="single file name"):
        ClipFile(**clip_record(file))


@pytest.mark.parametrize("kind", ["traversal", "absolute", "symlink"])
def test_cache_manifest_rejects_external_files_before_hashing(tmp_path, monkeypatch, kind):
    cache = PerceptionCache(tmp_path / "cache")
    key = store(cache, result())
    outside = tmp_path / "private.txt"
    outside.write_bytes(b"private")
    entry = cache.entry(key)
    name = "../../private.txt" if kind == "traversal" else str(outside) if kind == "absolute" else "link"
    if kind == "symlink":
        (entry / name).symlink_to(outside)
    manifest_path = entry / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][name] = hashlib.sha256(b"private").hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    from spectra.engine import cache as module
    original = module.file_sha256

    def confined(path):
        assert path.resolve() != outside.resolve(), "external file was opened"
        return original(path)

    monkeypatch.setattr(module, "file_sha256", confined)
    found = cache.lookup(key)
    assert found.result is None and "escapes" in found.reason


def test_mask_symlink_is_refused_before_decode(tmp_path):
    cache = PerceptionCache(tmp_path)
    key = "a" * 64
    entry = cache.entry(key)
    entry.mkdir()
    outside = tmp_path / "private.txt"
    outside.write_bytes(b"private")
    (entry / "mask.png").symlink_to(outside)
    ref = SegmentationRef(path="mask.png", sha256=hashlib.sha256(b"private").hexdigest(),
                          width=1, height=1, classes=("road",))
    with pytest.raises(ValueError, match="escapes"):
        cache.load_mask(key, ref)


@pytest.mark.parametrize("kind", ["traversal", "absolute", "symlink"])
def test_evidence_manifest_refuses_external_file(tmp_path, kind):
    evidence = EvidenceStore(tmp_path / "evidence")
    timeline, packages = build(synth.result([synth.approach(5.0, 7.0)]))
    entry = evidence.write(timeline, packages)
    outside = tmp_path / "private.txt"
    outside.write_bytes(b"private")
    name = "../../private.txt" if kind == "traversal" else str(outside) if kind == "absolute" else "link"
    if kind == "symlink":
        (entry / name).symlink_to(outside)
    path = entry / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][name] = hashlib.sha256(b"private").hexdigest()
    path.write_text(json.dumps(manifest))
    lookup = evidence.load(timeline.build_key)
    assert lookup.timeline is None and "escapes" in lookup.reason


def test_clip_symlink_is_not_reused_or_served(tmp_path):
    outside = tmp_path / "private.txt"
    outside.write_bytes(b"private")
    key = "b" * 64
    event_id = "c" * 16
    clips = tmp_path / key / "clips"
    clips.mkdir(parents=True)
    (clips / "clip.mp4").symlink_to(outside)
    (clips / f"{event_id}.json").write_text(json.dumps(clip_record("clip.mp4")))
    interval = SimpleNamespace(start_s=1.0, end_s=2.0)
    assert load_clip(clips, event_id, "a" * 64, interval) is None
    service = EvidenceService(None, tmp_path / "cache", tmp_path)
    service.package = lambda *args: SimpleNamespace(build_key=key, review=interval,
                                                   video=SimpleNamespace(sha256="a" * 64))
    with pytest.raises(EvidenceNotFound):
        service.stored_clip("unused", event_id)
