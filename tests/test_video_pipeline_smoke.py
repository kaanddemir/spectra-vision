from types import SimpleNamespace

import numpy as np

import spectra.analysis.video as video
from spectra.analysis.risk import RiskEvent, score_event


_SPECS: list[dict] = []


def _make_event(frame_index: int, timestamp_sec: float, spec: dict) -> RiskEvent:
    return RiskEvent(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        raw_state=spec["state"],
        collision_ttc_sec=spec.get("ttc"),
        direction="center",
        lane="center",
        object_type="car",
        risk_confidence=spec.get("confidence", 0.9),
        proximity_score=spec.get("near", 0.4),
        radial_flow_score=0.1,
        approach_score=spec.get("closing", 0.3),
        bbox=(1, 1, 3, 3),
        reason="",
        object_id=1,
        corridor_score=spec.get("crossing", 0.7),
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
        self.sensitivity = video.resolve_sensitivity(kwargs.get("sensitivity", "balanced"))
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
            raw_primary_score=score_event(event),
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
    assert set(result["frames"][0]) >= {
        "frame_index",
        "timestamp_sec",
        "stabilized_state",
        "primary",
        "traffic_light",
        "objects",
    }


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


def _corridor_event(*, lane_position, distance_m, bbox):
    # Minimal RiskEvent for exercising _is_near_in_corridor's geometry escape.
    return RiskEvent(
        frame_index=0,
        timestamp_sec=0.0,
        raw_state="CAUTION",
        collision_ttc_sec=1.8,
        direction="left",
        lane="left",
        object_type="car",
        risk_confidence=0.9,
        proximity_score=1.0,
        radial_flow_score=0.1,
        approach_score=3.1,
        bbox=bbox,
        reason="",
        object_id=1,
        lane_position=lane_position,
        distance_m=distance_m,
        closing_mps=3.1,
    )


def test_wide_bottom_box_counts_as_near_in_corridor_despite_off_center_position():
    # A close, wide, bottom-anchored lead whose lane_position snapped to the
    # -1.5 clamp still counts as an in-corridor threat via the geometry escape.
    frame_shape = (100, 100)  # H, W
    lead = _corridor_event(lane_position=-1.5, distance_m=6.5, bbox=(0, 73, 41, 99))
    assert video._is_near_in_corridor(lead, frame_shape)
    # Without frame geometry, the clamp position alone still reads off-corridor.
    assert not video._is_near_in_corridor(lead)


def test_narrow_edge_box_stays_off_corridor():
    # A close but narrow side car near the frame edge is NOT promoted — the
    # width test rejects it, so genuine side traffic does not over-alarm.
    frame_shape = (100, 100)
    side = _corridor_event(lane_position=1.5, distance_m=7.1, bbox=(90, 71, 100, 99))
    assert not video._is_near_in_corridor(side, frame_shape)


def _passing_event(*, corridor_score):
    # A close, strongly-closing object (like a vehicle being passed in an
    # adjacent stopped lane) with a real depth TTC.
    from spectra.analysis.risk import TtcComponent

    return RiskEvent(
        frame_index=0,
        timestamp_sec=0.0,
        raw_state="SAFE",
        collision_ttc_sec=0.4,
        direction="left",
        lane="left",
        object_type="truck",
        risk_confidence=0.6,
        proximity_score=1.0,
        radial_flow_score=0.1,
        approach_score=0.5,
        bbox=(1, 1, 3, 3),
        reason="",
        object_id=4,
        corridor_score=corridor_score,
        distance_m=7.2,
        closing_mps=16.4,
        ttc_components=(TtcComponent("depth", 0.4, 0.9),),
    )


def test_off_corridor_object_withholds_collision_ttc_and_pressure():
    # A passing (off-corridor, low crossing) vehicle must not surface a collision
    # TTC or TTC-pressure — it reads SAFE, so "0.4s / 85%" would be contradictory.
    passing = _passing_event(corridor_score=0.05)
    ttc = video._ttc_metric(passing)
    assert ttc["display"] == "—"
    assert ttc["collision_ttc_sec"] is None
    assert video._risk_metric(passing)["factors"]["ttc_score"] == 0.0


def test_in_corridor_object_still_surfaces_collision_ttc():
    # A genuine in-path closing object keeps its collision TTC and score.
    approaching = _passing_event(corridor_score=0.7)
    ttc = video._ttc_metric(approaching)
    assert ttc["display"] == "0.4s"
    assert ttc["collision_ttc_sec"] == 0.4
    assert video._risk_metric(approaching)["factors"]["ttc_score"] > 0.0
