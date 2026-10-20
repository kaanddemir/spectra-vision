import time

import pytest

from evaluation import hardware

pytestmark = pytest.mark.slow


@pytest.fixture
def normal_pressure(monkeypatch):
    """These tests are about timeouts, crashes and reporting, not the machine's current memory pressure;
    the pressure check itself is covered by ``test_benchmark_is_not_started_under_memory_pressure``."""
    monkeypatch.setattr(hardware, "memory_pressure_level", lambda: 1)


def _sleepy(q, seconds):
    time.sleep(seconds)
    q.put({"ok": True})


def _crashes(q):
    raise RuntimeError("boom")


def test_probe_reports_core_fields():
    info = hardware.probe()
    for key in ("machine", "system", "logical_cpus", "memory_gb", "disk_free_gb"):
        assert key in info


def test_isolated_run_times_out_and_cleans_up(normal_pressure):
    r = hardware.run_isolated(_sleepy, (30,), timeout_s=1)
    assert r["ok"] is False and "timeout" in r["error"] and r["wall_s"] < 15


def test_isolated_run_reports_child_crash_as_failure(normal_pressure):
    r = hardware.run_isolated(_crashes, (), timeout_s=60)
    assert r["ok"] is False and "exited" in r["error"] and r["wall_s"] < 30


def test_oversized_model_is_skipped_before_loading(tmp_path):
    f = tmp_path / "big.onnx"
    f.write_bytes(b"\0" * (2 * 2**20))
    r = hardware.bench_onnx(f, ["cpu"], budget_mb=1)
    assert r[0]["ok"] is False and "skipped" in r[0]["error"]


def test_benchmark_is_not_started_under_memory_pressure(monkeypatch):
    monkeypatch.setattr(hardware, "memory_pressure_level", lambda: 2)
    r = hardware.run_isolated(_sleepy, (0,), timeout_s=5)
    assert r["ok"] is False and "pressure" in r["error"]


def test_successful_run_records_pressure_and_swap(normal_pressure):
    r = hardware.run_isolated(_sleepy, (0,), timeout_s=30)
    assert r["ok"] is True and "memory_pressure_after" in r
