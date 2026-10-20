import json

import numpy as np
import pytest

from evaluation.perception_cache import ModelInfo, PerceptionCache


def make(**kw):
    base = dict(media_sha256="abc", frame_size=(100, 50), timestamps=[0.0, 0.1, 0.25],
                detections=[[0, 1, 1, 10, 10, 0.9, 2, 7], [2, 2, 2, 12, 12, 0.8, 2, 7], [1, 5, 5, 9, 9, 0.5, 0, -1]],
                motion=np.zeros((3, 6)), models=(ModelInfo("det", "1", "f00"),), config={"res": 640})
    base.update(kw)
    return PerceptionCache(**base)


def test_round_trip_preserves_content(tmp_path):
    c = make(path=np.ones((3, 4)), class_names={2: "car"})
    c.save(tmp_path)
    d = PerceptionCache.load(tmp_path)
    np.testing.assert_array_equal(d.timestamps, c.timestamps)
    np.testing.assert_array_equal(d.detections, c.detections)
    np.testing.assert_array_equal(d.path, c.path)
    assert d.models == c.models and d.config_hash == c.config_hash and d.class_names == {2: "car"}


def test_track_access_is_sorted_and_ignores_untracked():
    c = make()
    assert c.track_ids() == [7]
    assert list(c.track(7)[:, 0]) == [0, 2]


def test_tampered_arrays_are_detected(tmp_path):
    make().save(tmp_path)
    m = json.loads((tmp_path / "manifest.json").read_text())
    m["arrays_sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="hash"):
        PerceptionCache.load(tmp_path)


def test_unknown_schema_version_is_rejected(tmp_path):
    make().save(tmp_path)
    m = json.loads((tmp_path / "manifest.json").read_text())
    m["schema_version"] = "9"
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="schema"):
        PerceptionCache.load(tmp_path)


@pytest.mark.parametrize("kw, msg", [
    (dict(timestamps=[0.0, 0.2, 0.1]), "non-decreasing"),
    (dict(timestamps=[]), "no frames"),
    (dict(detections=[[3, 1, 1, 2, 2, 1, 0, 0]]), "out of range"),
    (dict(detections=[[0, 5, 1, 2, 2, 1, 0, 0]]), "x2 > x1"),
    (dict(motion=np.zeros((2, 6))), "shape"),
])
def test_invalid_caches_are_rejected(kw, msg):
    with pytest.raises(ValueError, match=msg):
        make(**kw)


def test_config_hash_is_order_independent():
    assert make(config={"a": 1, "b": 2}).config_hash == make(config={"b": 2, "a": 1}).config_hash
