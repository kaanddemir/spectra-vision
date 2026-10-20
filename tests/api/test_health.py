import pytest

from spectra.api.store import SCHEMA_VERSION


def test_healthy_server_reports_ready(make_client, tmp_path):
    response = make_client().get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == {"available": True, "schema_version": SCHEMA_VERSION, "message": None}
    assert body["models"]["available"] is True
    assert [m["role"] for m in body["models"]["models"]] == ["detector", "segmenter"]
    assert body["memory"] == {"pressure_level": 1, "state": "normal"}
    assert body["analysis"] == {"available": True, "reasons": []}
    assert body["library_folders"] == 0
    for name in ("cache", "evidence", "tmp", "spectra.sqlite3"):
        assert (tmp_path / "var" / name).exists()


def test_missing_model_degrades_health_without_failing_the_request(make_client, models_dir):
    (models_dir / "segmenter.onnx").unlink()
    body = make_client().get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["models"]["available"] is False
    assert "missing segmenter model" in body["models"]["message"]
    assert body["analysis"]["available"] is False
    assert any("models" in reason for reason in body["analysis"]["reasons"])


def test_changed_model_file_is_verified_again(make_client, models_dir):
    client = make_client()
    assert client.get("/api/health").json()["models"]["available"] is True
    (models_dir / "detector.onnx").write_bytes(b"replaced weights")
    body = client.get("/api/health").json()
    assert body["models"]["available"] is False
    assert "sha256 mismatch" in body["models"]["message"]


def test_unchanged_registry_is_hashed_once(make_client, monkeypatch):
    import spectra.api.system as system
    calls = []
    original = system.default_specs
    monkeypatch.setattr(system, "default_specs", lambda root: calls.append(root) or original(root))
    client = make_client()
    for _ in range(3):
        assert client.get("/api/health").json()["models"]["available"] is True
    assert len(calls) == 1


@pytest.mark.parametrize("level, state, ready", [
    (1, "normal", True), (2, "warning", True), (4, "critical", False), (None, "unavailable", True)])
def test_memory_pressure_states(make_client, level, state, ready):
    body = make_client(pressure=lambda: level).get("/api/health").json()
    assert body["memory"] == {"pressure_level": level, "state": state}
    assert body["analysis"]["available"] is ready
    assert body["status"] == ("ok" if ready else "degraded")


def test_unusable_database_degrades_health(make_client, tmp_path):
    import sqlite3
    client = make_client()
    with sqlite3.connect(tmp_path / "var" / "spectra.sqlite3") as conn:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    body = client.get("/api/health").json()
    assert body["database"]["available"] is False
    assert "not supported" in body["database"]["message"]
    assert body["status"] == "degraded"


def test_openapi_is_served_locally_without_external_documentation_pages(make_client):
    client = make_client()
    assert client.get("/api/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
