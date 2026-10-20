import cv2
import numpy as np
import pytest

from evaluation import detector_bench, hardware
from evaluation.hardware import run_isolated

ADAPTER = "tests.evaluation.test_detector_bench:FakeDetector"


@pytest.fixture
def normal_pressure(monkeypatch):
    """Pins the pressure reading in this process only. A spawned child still reads the real value, so tests
    that need the child's pressure check to pass call ``_child`` in-process."""
    monkeypatch.setattr(hardware, "memory_pressure_level", lambda: 1)


def _in_process(*args) -> dict:
    import queue
    q = queue.Queue()
    detector_bench._child(q, *args)
    return q.get_nowait()


class FakeDetector:
    def __init__(self, fail: bool = False) -> None:
        self.name, self.fail = "fake", fail

    def load(self) -> None:
        if self.fail:
            raise RuntimeError("load failed")

    def predict(self, frame):
        return np.array([[0, 0, 10, 10, 0.9, 2.0]])


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(12):
        w.write(np.full((48, 64, 3), i * 10, np.uint8))
    w.release()
    return str(path)


def test_benchmark_reports_cold_and_warm_timings_separately(video, normal_pressure):
    r = _in_process(ADAPTER, {}, video, 5, 1, 2, 4096)
    assert r["ok"] and r["frames_timed"] == 5 and r["mean_detections"] == 1
    for key in ("load_s", "first_inference_s", "warm_median_ms", "peak_rss_mb", "cpu_utilisation_cores"):
        assert key in r


def test_benchmark_stops_when_memory_pressure_rises_during_the_run(video, monkeypatch):
    monkeypatch.setattr(hardware, "memory_pressure_level", lambda: 2)
    r = _in_process(ADAPTER, {}, video, 5, 1, 2, 4096)
    assert r["ok"] is False and "memory pressure level 2" in r["error"]


def test_adapter_failure_is_reported_not_raised(video, normal_pressure):
    r = run_isolated(detector_bench._child, ("tests.evaluation.test_detector_bench:FakeDetector", {"fail": True},
                                             video, 5, 1, 2, 4096), 60)
    assert r["ok"] is False and "load failed" in r["error"]


def test_unreadable_video_is_reported(tmp_path, normal_pressure):
    r = run_isolated(detector_bench._child, ("tests.evaluation.test_detector_bench:FakeDetector", {},
                                             str(tmp_path / "missing.mp4"), 5, 1, 2, 4096), 60)
    assert r["ok"] is False and "no frames" in r["error"]
