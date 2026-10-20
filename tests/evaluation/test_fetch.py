import hashlib
import json

import pytest

from evaluation.fetch import fetch


def manifest(tmp_path, payload: bytes, sha1: str, size: int | None = None):
    src = tmp_path / "src.bin"
    src.write_bytes(payload)
    m = {"clips": [{"id": "a", "filename": "a.bin", "url": src.as_uri(), "bytes": size or len(payload), "sha1": sha1}]}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(m))
    return path


def test_verified_download_then_skip_when_present(tmp_path):
    data = b"clip" * 100
    m = manifest(tmp_path, data, hashlib.sha1(data).hexdigest())
    assert fetch(m, tmp_path / "out", max_mb=1)[0]["status"] == "downloaded"
    assert fetch(m, tmp_path / "out", max_mb=1)[0]["status"] == "present"


def test_hash_mismatch_removes_partial_file(tmp_path):
    m = manifest(tmp_path, b"clip", "0" * 40)
    with pytest.raises(ValueError, match="verification failed"):
        fetch(m, tmp_path / "out", max_mb=1)
    assert list((tmp_path / "out").iterdir()) == []


def test_budget_is_enforced_before_any_download(tmp_path):
    m = manifest(tmp_path, b"clip", "0" * 40, size=5_000_000)
    with pytest.raises(ValueError, match="budget"):
        fetch(m, tmp_path / "out", max_mb=1)
