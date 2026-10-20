import json

import pytest
from pydantic import ValidationError

from evaluation.registry import HERE, Status, Use, load


def test_shipped_registry_is_valid():
    reg = load()
    assert reg.datasets and reg.models


def test_no_model_is_marked_distributable_without_full_clearance():
    for m in load().models:
        if m.use is Use.DISTRIBUTABLE:
            assert all(t.status is Status.CONFIRMED for t in (m.code_licence, m.checkpoint_licence, m.training_data))


def test_distributable_with_unconfirmed_terms_is_rejected(tmp_path):
    data = json.loads((HERE / "registry.json").read_text())
    data["models"][0]["use"] = "distributable"
    (tmp_path / "r.json").write_text(json.dumps(data))
    with pytest.raises(ValidationError, match="distributable requires"):
        load(tmp_path / "r.json")


def test_duplicate_ids_are_rejected(tmp_path):
    data = json.loads((HERE / "registry.json").read_text())
    data["datasets"].append(data["datasets"][0])
    (tmp_path / "r.json").write_text(json.dumps(data))
    with pytest.raises(ValidationError, match="duplicate"):
        load(tmp_path / "r.json")
