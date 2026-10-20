import json
from pathlib import Path

MANIFESTS = Path(__file__).resolve().parents[2] / "evaluation" / "manifests"


def test_sample_manifests_are_complete_and_consistent():
    for path in MANIFESTS.glob("*.json"):
        m = json.loads(path.read_text())
        ids = [c["id"] for c in m["clips"]]
        assert len(ids) == len(set(ids)), path
        assert m["total_bytes"] == sum(c["bytes"] for c in m["clips"]), path
        for c in m["clips"]:
            assert c["licence"] and c["url"].startswith("https://") and c["bytes"] > 0, c["id"]
