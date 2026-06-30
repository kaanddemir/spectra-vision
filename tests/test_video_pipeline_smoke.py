from types import SimpleNamespace

import numpy as np

import spectra.analysis.video as video
from spectra.analysis.risk import RiskEvent


_SPECS: list[dict] = []


def _make_event(frame_index: int, timestamp_sec: float, spec: dict) -> RiskEvent:
    return RiskEvent(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        state=spec["state"],
        ttc_sec=spec.get("ttc"),
        direction="center",
        lane="center",
        object_type="car",
        confidence=spec.get("confidence", 0.9),
        near_score=spec.get("near", 0.4),
        velocity_magnitude=0.1,
        closing_speed=spec.get("closing", 0.3),
        bbox=(1, 1, 3, 3),
        reason="",
        object_id=1,
        crossing_risk=spec.get("crossing", 0.7),
        brake_score=spec.get("brake", 0.0),
    )


class _FakeLoader:
    def __init__(self, source, max_frames=None, start_sec=0.0, end_sec=None, start_frame=0, end_frame=None):
        self.fps = 10.0
        self.frame_count = len(_SPECS)
        self.start_frame = start_frame
        self.end_frame = end_frame

    def frames(self):
        for index, spec in enumerate(_SPECS):
            yield video.VideoFrame(
                frame_index=index,
                timestamp_sec=spec["t"],
                bgr=np.zeros((4, 4, 3), dtype=np.uint8),
            )

    def close(self):
        pass


class _FakeAnalyzer:
    def __init__(self, **kwargs):
        self.processed_frames = 0
        self.performance_stats = video._empty_performance_stats()
        self.performance_sample_logs = []
        self.depth_refresh = {
            "runs": 1,
            "skips": 0,
            "initial_runs": 1,
            "periodic_runs": 0,
            "motion_triggered_runs": 0,
            "cooldown_frames": 3,
        }

    def process_frame(self, frame_bgr, frame_index, timestamp_sec):
        event = _make_event(frame_index, timestamp_sec, _SPECS[frame_index])
        self.processed_frames += 1
        return video.FrameAnalysis(
            primary_event=event,
            all_events=[event],
            primary_risk_score=video._risk_score(event),
            frame_bgr=frame_bgr,
            lane=SimpleNamespace(width=4, height=4),
        )


def _patch_pipeline(monkeypatch):
    monkeypatch.setattr(video, "VideoLoader", _FakeLoader)
    monkeypatch.setattr(video, "SpatialFrameAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr(video, "annotate_frame", lambda bgr, p, a, lane=None, traffic_light_state="none": bgr)
    monkeypatch.setattr(video, "_ensure_required_models", lambda: None)


def test_analyze_spatial_video_returns_client_ready_shape(monkeypatch):
    _SPECS[:] = [
        {"t": 0.0, "state": "SAFE", "ttc": None},
        {"t": 1.0, "state": "CAUTION", "ttc": 2.0},
    ]
    _patch_pipeline(monkeypatch)

    result = video.analyze_spatial_video("clip.mp4", max_processed_frames=100, max_saved_events=10, resize_max_side=256)

    assert set(result) >= {"fps", "frame_count", "processed_frames", "frames", "events", "peak_event", "performance_summary"}
    assert result["processed_frames"] == 2
    assert len(result["frames"]) == 2
    assert set(result["frames"][0]) >= {"frameIndex", "timestampSec", "state", "primary", "trafficLight", "objects"}


def test_saved_events_are_deduped_per_second_by_risk_score(monkeypatch):
    _SPECS[:] = [
        {"t": 0.0, "state": "DANGER", "ttc": 0.8, "near": 0.8, "closing": 0.8},
        {"t": 5.0, "state": "CAUTION", "ttc": 2.0, "near": 0.4, "closing": 0.3},
        {"t": 5.3, "state": "DANGER", "ttc": 0.5, "near": 0.9, "closing": 0.95},
    ]
    _patch_pipeline(monkeypatch)

    result = video.analyze_spatial_video("clip.mp4", max_processed_frames=100, max_saved_events=10, resize_max_side=256)

    assert sorted(round(e["timestamp_sec"], 2) for e in result["events"]) == [0.0, 5.3]


def test_saved_events_get_deferred_rgb_payloads(monkeypatch):
    _SPECS[:] = [{"t": 0.0, "state": "DANGER", "ttc": 0.5, "near": 0.8, "closing": 0.9}]
    _patch_pipeline(monkeypatch)

    result = video.analyze_spatial_video("clip.mp4", max_processed_frames=100, max_saved_events=10, resize_max_side=256)

    saved = result["events"][0]
    assert isinstance(saved["original_rgb"], np.ndarray)
    assert isinstance(saved["overlay_rgb"], np.ndarray)
