"""Faz 1 regression: the SpatialFrameAnalyzer extraction preserves the
analyze_spatial_video orchestration contract (v2 result shape, per-second
event dedup, deferred RGB render) without needing the real vision models."""

from types import SimpleNamespace

import numpy as np

import spectra.analysis.video as video
from spectra.analysis.risk import RiskEvent

# Per-frame script shared between the fake loader and fake analyzer.
_SPECS: list[dict] = []


def _make_event(frame_index: int, timestamp_sec: float, spec: dict) -> RiskEvent:
    return RiskEvent(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        state=spec["state"],
        ttc_sec=spec["ttc"],
        direction="center",
        lane="center",
        object_type="car",
        confidence=0.9,
        near_score=spec["near"],
        velocity_magnitude=0.1,
        closing_speed=spec["closing"],
        bbox=(1, 1, 3, 3),
        reason="",
        object_id=1,
        expansion_rate=0.1,
        lateral_velocity_norm=0.0,
        crossing_risk=0.1,
        lane_position=0.0,
        ttc_components=(),
    )


class _FakeLoader:
    def __init__(self, source, max_frames=None, start_sec=0.0, end_sec=None):
        self.fps = 10.0
        self.frame_count = len(_SPECS)
        self.max_frames = max_frames

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
        self.performance_sample_logs: list[str] = []
        self.depth_refresh = {
            "runs": 1, "skips": 0, "initial_runs": 1, "periodic_runs": 0,
            "motion_triggered_runs": 0, "cooldown_frames": 3,
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


def _patch(monkeypatch):
    monkeypatch.setattr(video, "VideoLoader", _FakeLoader)
    monkeypatch.setattr(video, "SpatialFrameAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr(
        video, "annotate_frame", lambda bgr, p, a, lane=None, traffic_light_state="none": bgr
    )
    monkeypatch.setattr(video, "_ensure_required_models", lambda: None)


def test_result_shape_and_processed_count(monkeypatch):
    _SPECS[:] = [
        {"t": 0.0, "state": "SAFE", "ttc": None, "near": 0.1, "closing": 0.0},
        {"t": 1.0, "state": "CAUTION", "ttc": 2.0, "near": 0.4, "closing": 0.3},
    ]
    _patch(monkeypatch)

    result = video.analyze_spatial_video(
        "clip.mp4", max_processed_frames=100, max_saved_events=10, resize_max_side=256
    )

    assert set(result) >= {
        "fps", "frame_count", "processed_frames", "frames", "events",
        "peak_event", "performance_summary", "performance_logs",
    }
    assert result["fps"] == 10.0
    assert result["frame_count"] == 2
    assert result["processed_frames"] == 2
    assert len(result["frames"]) == 2
    # Frame rows carry the v2 field names.
    assert set(result["frames"][0]) >= {
        "frameIndex", "timestampSec", "stabilizedRiskState", "primaryObjectId",
        "primaryRiskScore", "primaryLane", "objects",
    }


def test_per_second_dedup_keeps_higher_score(monkeypatch):
    _SPECS[:] = [
        {"t": 0.0, "state": "DANGER", "ttc": 0.5, "near": 0.8, "closing": 0.9},
        {"t": 5.0, "state": "CAUTION", "ttc": 2.0, "near": 0.4, "closing": 0.3},
        {"t": 5.3, "state": "DANGER", "ttc": 0.4, "near": 0.9, "closing": 0.95},
    ]
    _patch(monkeypatch)

    result = video.analyze_spatial_video(
        "clip.mp4", max_processed_frames=100, max_saved_events=10, resize_max_side=256
    )

    # t=5.0 and t=5.3 are within 1s -> the stronger (t=5.3) replaces it.
    times = sorted(round(e["timestamp_sec"], 2) for e in result["events"])
    assert times == [0.0, 5.3]


def test_saved_events_get_deferred_rgb(monkeypatch):
    _SPECS[:] = [
        {"t": 0.0, "state": "DANGER", "ttc": 0.5, "near": 0.8, "closing": 0.9},
    ]
    _patch(monkeypatch)

    result = video.analyze_spatial_video(
        "clip.mp4", max_processed_frames=100, max_saved_events=10, resize_max_side=256
    )

    assert len(result["events"]) == 1
    saved = result["events"][0]
    assert "original_rgb" in saved and "overlay_rgb" in saved
    assert isinstance(saved["original_rgb"], np.ndarray)


def test_smooth_lane_confidence_passthrough_on_first_frame():
    # No prior state -> the raw confidence passes through unchanged.
    assert video._smooth_lane_confidence(None, 0.7) == 0.7


def test_smooth_lane_confidence_rises_faster_than_it_falls():
    prev = 0.5
    rose = video._smooth_lane_confidence(prev, 1.0)
    fell = video._smooth_lane_confidence(prev, 0.0)
    # Both move toward the raw value but stay between prev and raw, and a rise
    # of the same magnitude covers more ground than a fall (asymmetric alpha).
    assert prev < rose < 1.0
    assert 0.0 < fell < prev
    assert (rose - prev) > (prev - fell)


def test_smooth_lane_confidence_clamps_to_unit_range():
    assert video._smooth_lane_confidence(0.5, 5.0) <= 1.0
    assert video._smooth_lane_confidence(0.5, -3.0) >= 0.0


def test_analyzer_init_sets_cross_frame_state(monkeypatch):
    monkeypatch.setattr(video, "_ensure_required_models", lambda: None)
    monkeypatch.setattr(video, "get_lanenet_model", lambda: object())
    monkeypatch.setattr(video, "get_detector", lambda: object())

    analyzer = video.SpatialFrameAnalyzer(resize_max_side=256, fps=30.0)

    assert analyzer.processed_frames == 0
    assert analyzer.previous_frame is None
    assert analyzer.last_depth is None and analyzer.last_flow is None
    assert analyzer.cached_road_roi is None
    assert analyzer.tracker is not None
    assert analyzer.lane_kalman is not None
    assert analyzer.stabilizer is not None
    assert analyzer.depth_refresh["runs"] == 0
