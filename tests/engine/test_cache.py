import json
import os

import numpy as np
import pytest

from spectra.engine.cache import STAGING, PerceptionCache, perception_key
from spectra.engine.contracts import Provenance, RunConfig
from tests.engine.fakes import SHA, video

from tests.engine.cache_fixtures import DET, SEG, provenance, result, store


def test_key_is_stable_and_changes_with_every_input():
    r = result()
    key = perception_key(r.video, r.config, r.provenance)
    assert key == perception_key(r.video, r.config, r.provenance) and len(key) == 64
    variants = [
        (video(sha256="b" * 64), r.config, r.provenance),
        (r.video, RunConfig(interval_s=0.1), r.provenance),
        (r.video, RunConfig(interval_s=0.2, max_width=640), r.provenance),
        (r.video, r.config, provenance(DET.model_copy(update={"version": "2"}))),
        (r.video, r.config, provenance(DET.model_copy(update={"params": {"input_px": 512}}))),
        (r.video, r.config, provenance(DET.model_copy(update={"checkpoint_sha256": "c" * 64}))),
        (r.video, r.config, provenance(DET.model_copy(update={"interval_s": 0.5}))),
        (r.video, r.config, Provenance(engine_version="1", decoder={"pyav": "19.0.1"}, components=(DET,))),
        (r.video, r.config, Provenance(engine_version="0", decoder={"pyav": "20"}, components=(DET,))),
    ]
    keys = {perception_key(*v) for v in variants}
    assert key not in keys and len(keys) == len(variants)


def test_store_and_replay(tmp_path):
    cache = PerceptionCache(tmp_path)
    r = result()
    key = store(cache, r)
    found = cache.lookup(key)
    assert found.result == r and found.reason is None
    assert not any((tmp_path / STAGING).iterdir())


def test_serialisation_is_byte_identical_across_writes(tmp_path):
    a, b = PerceptionCache(tmp_path / "a"), PerceptionCache(tmp_path / "b")
    key = store(a, result())
    store(b, result())
    for name in ("perception.json", "manifest.json"):
        assert (a.entry(key) / name).read_bytes() == (b.entry(key) / name).read_bytes()


def test_absent_entry_is_a_plain_miss(tmp_path):
    found = PerceptionCache(tmp_path).lookup("0" * 64)
    assert found.result is None and found.reason is None


def test_interrupted_write_leaves_no_entry(tmp_path):
    cache = PerceptionCache(tmp_path)
    r = result()
    key = perception_key(r.video, r.config, r.provenance)
    writer = cache.begin(key)
    (writer.staging / "perception.json").write_text("{partial")  # crash before commit
    assert cache.lookup(key).result is None and not cache.entry(key).exists()
    writer.abort()
    assert not writer.staging.exists()


@pytest.mark.parametrize("damage,reason", [
    (lambda d: (d / "perception.json").write_bytes((d / "perception.json").read_bytes()[:-5]), "does not match"),
    (lambda d: (d / "manifest.json").write_text("{"), "unreadable manifest"),
    (lambda d: (d / "manifest.json").unlink(), "unreadable manifest"),
    (lambda d: (d / "perception.json").unlink(), "missing file"),
])
def test_corrupted_entries_are_rejected_with_a_reason(tmp_path, damage, reason):
    cache = PerceptionCache(tmp_path)
    key = store(cache, result())
    damage(cache.entry(key))
    found = cache.lookup(key)
    assert found.result is None and reason in found.reason


def _rewrite(entry, mutate):
    """Edit perception.json and update the manifest hash, as a stale-but-consistent entry would be."""
    import hashlib
    data = json.loads((entry / "perception.json").read_text())
    mutate(data)
    raw = json.dumps(data).encode()
    (entry / "perception.json").write_bytes(raw)
    manifest = json.loads((entry / "manifest.json").read_text())
    manifest["files"]["perception.json"] = hashlib.sha256(raw).hexdigest()
    (entry / "manifest.json").write_text(json.dumps(manifest))


def test_content_that_does_not_match_its_key_is_stale(tmp_path):
    cache = PerceptionCache(tmp_path)
    key = store(cache, result())
    _rewrite(cache.entry(key), lambda d: d["provenance"]["components"][0].update(version="2"))
    assert "does not hash to the entry key" in cache.lookup(key).reason


def test_incompatible_schema_versions_are_rejected_explicitly(tmp_path):
    cache = PerceptionCache(tmp_path)
    key = store(cache, result())
    path = cache.entry(key) / "manifest.json"
    manifest = json.loads(path.read_text())
    path.write_text(json.dumps({**manifest, "schema_version": "0.1"}))
    assert "incompatible schema version" in cache.lookup(key).reason
    path.write_text(json.dumps(manifest))
    _rewrite(cache.entry(key), lambda d: d.update(schema_version="0.1"))
    assert "fails validation" in cache.lookup(key).reason


def test_rejected_entry_is_replaced_atomically(tmp_path):
    cache = PerceptionCache(tmp_path)
    key = store(cache, result())
    (cache.entry(key) / "perception.json").write_text("garbage")
    assert cache.lookup(key).result is None
    store(cache, result())
    assert cache.lookup(key).result == result()
    assert not any((tmp_path / STAGING).iterdir())


def test_commit_refuses_a_result_for_another_key(tmp_path):
    writer = PerceptionCache(tmp_path).begin("f" * 64)
    with pytest.raises(ValueError, match="does not hash"):
        writer.commit(result())
    writer.abort()


def test_masks_round_trip_and_are_verified(tmp_path):
    cache = PerceptionCache(tmp_path)
    classes = ("road", "lane_marking", "ego_vehicle")
    masks = [np.random.default_rng(i).integers(0, 3, (24, 32), dtype=np.uint8) for i in range(3)]
    template = result((SEG,), refs=[_ref() for _ in range(3)])  # the key does not depend on frame content
    key = perception_key(template.video, template.config, template.provenance)
    writer = cache.begin(key)
    refs = [writer.write_mask(i, m, classes) for i, m in enumerate(masks)]
    writer.commit(result((SEG,), refs=refs))
    found = cache.lookup(key)
    assert found.result is not None
    for frame, mask in zip(found.result.frames, masks):
        assert np.array_equal(cache.load_mask(key, frame.segmentation), mask)
    path = cache.entry(key) / refs[1].path
    os.chmod(path, 0o644)
    path.write_bytes(path.read_bytes()[:-3] + b"xyz")
    assert "does not match" in cache.lookup(key).reason
    with pytest.raises(ValueError):
        cache.load_mask(key, refs[1])


def _ref():
    from spectra.engine.contracts import SegmentationRef
    return SegmentationRef(path="masks/x.png", sha256=SHA, width=32, height=24, classes=("road",))


def test_masks_with_unnamed_classes_or_wrong_type_are_refused(tmp_path):
    writer = PerceptionCache(tmp_path).begin("e" * 64)
    with pytest.raises(ValueError, match="without a name"):
        writer.write_mask(0, np.full((4, 4), 3, np.uint8), ("road", "lane_marking"))
    with pytest.raises(ValueError, match="uint8"):
        writer.write_mask(0, np.zeros((4, 4), np.int32), ("road",))
    writer.abort()


def test_keys_must_be_hex_digests(tmp_path):
    with pytest.raises(ValueError):
        PerceptionCache(tmp_path).entry("../escape")
