from spectra.analysis.risk import RiskEvent
from spectra.analysis.video import _lane_metric, _normalized_bbox, _object_metric
from spectra.app import _serialize_result


def test_serialize_result_includes_performance_logs():
    summary = {
        "processed_frames": 12,
        "elapsed_sec": 0.123,
        "effective_fps": 97.56,
    }
    result = {
        "fps": 30.0,
        "frame_count": 12,
        "processed_frames": 12,
        "frames": [],
        "events": [],
        "peak_event": None,
        "performance_summary": summary,
        "performance_logs": ["[FRAME    0] preprocess=1ms"],
    }

    serialized = _serialize_result(result, elapsed_sec=0.1234, source_name="sample.mp4")

    assert serialized["payload"]["performance_summary"] == summary
    assert serialized["payload"]["performance_logs"] == ["[FRAME    0] preprocess=1ms"]


def test_serialize_result_exposes_frame_dimensions():
    result = {
        "fps": 30.0,
        "frame_count": 12,
        "processed_frames": 12,
        "frame_width": 512,
        "frame_height": 288,
        "frames": [],
        "events": [],
        "peak_event": None,
        "performance_summary": {},
        "performance_logs": [],
    }

    meta = _serialize_result(result, elapsed_sec=0.1, source_name="s.mp4")["payload"]["metadata"]

    assert meta["frameWidth"] == 512
    assert meta["frameHeight"] == 288


def _sample_event() -> RiskEvent:
    return RiskEvent(
        frame_index=3,
        timestamp_sec=0.1,
        state="DANGER",
        ttc_sec=1.2,
        direction="center",
        lane="center",
        object_type="car",
        confidence=0.9,
        near_score=0.5,
        velocity_magnitude=0.3,
        closing_speed=0.4,
        bbox=(100, 50, 300, 250),
        reason="",
        object_id=7,
    )


def test_normalized_bbox_scales_to_unit_square():
    event = _sample_event()
    assert _normalized_bbox(event, 400, 200) == [0.25, 0.25, 0.75, 1.25]
    # No bbox or zero dims yields None rather than a divide error.
    assert _normalized_bbox(event, 0, 0) is None


def test_object_metric_carries_normalized_bbox():
    metric = _object_metric(_sample_event(), 400, 200)
    assert metric["bbox"] == [0.25, 0.25, 0.75, 1.25]


def test_lane_metric_emits_normalized_corridor():
    lane = _lane_metric(None, 400, 200)
    assert lane["detected"] is False
    assert len(lane["corridor"]) == 4
    for x, y in lane["corridor"]:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0
