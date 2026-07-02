import asyncio
import io

import numpy as np
from fastapi import UploadFile

import spectra.app as app_module
from spectra.analysis.risk import RiskEvent
from spectra.analysis.video import _lane_metric, _object_metric
from spectra.app import _serialize_result


def _client_object(**overrides):
    obj = {
        "display_id": 7,
        "object_type": "car",
        "raw_state": "CAUTION",
        "risk": {
            "risk_score": 0.42,
            "factors": {
                "ttc_score": 0.2,
                "proximity_score": 0.4,
                "approach_score": 0.3,
                "corridor_score": 0.2,
                "brake_score": 0.0,
            },
        },
        "eta": {
            "collision_ttc_sec": 2.2,
            "display": "2.2s",
            "ttc_agreement": 0.9,
            "sources": {"depth": {"ttc_sec": 2.2, "confidence": 0.64}},
        },
        "motion": {"distance_m": 18.5, "closing_mps": 8.4},
        "lane": {"lane": "center", "lane_position": 0.0, "corridor_score": 0.2},
        "confidence": {
            "risk_confidence": 0.67,
            "detection_confidence": 0.72,
            "lane_confidence": 0.8,
            "depth_confidence": 0.64,
            "flow_confidence": 0.5,
            "expansion_confidence": 0.55,
        },
        "object_id": 7,
        "bbox": [0.1, 0.1, 0.5, 0.5],
    }
    obj.update(overrides)
    return obj


def _result_with_event(event):
    return {
        "fps": 30.0,
        "frame_count": 3,
        "processed_frames": 3,
        "frames": [
            {
                "frame_index": event["frame_index"],
                "timestamp_sec": event["timestamp_sec"],
                "stabilized_state": event["stabilized_state"],
                "primary": event["primary"],
                "traffic_light": event["traffic_light"],
                "lane_geometry": event["lane_geometry"],
                "objects": event["objects"],
            }
        ],
        "events": [event],
        "peak_event": event,
        "performance_summary": {"processed_frames": 3},
        "performance_logs": ["frame 0"],
    }


def _assert_keys_absent(value, forbidden: set[str]) -> None:
    if isinstance(value, dict):
        assert not (set(value) & forbidden)
        for child in value.values():
            _assert_keys_absent(child, forbidden)
    elif isinstance(value, list):
        for child in value:
            _assert_keys_absent(child, forbidden)


def test_serialize_result_keeps_v6_shape_and_strips_internal_fields():
    event = {
        "frame_index": 1,
        "timestamp_sec": 0.1,
        "raw_primary_score": 0.42,
        "risk_score": 0.41,
        "stabilized_state": "CAUTION",
        "primary": {"object_id": 7, "raw_primary_score": 0.42, "lane": "center"},
        "traffic_light": {"state": "red", "confidence": 0.8},
        "lane_geometry": {"detected": False, "confidence": 0.25, "corridor": []},
        "objects": [_client_object()],
    }

    payload = _serialize_result(_result_with_event(event), elapsed_sec=0.01, source_name="sample.mp4")["payload"]

    assert payload["schema_version"] == 6
    assert set(payload) >= {"frames", "events", "peak_event", "images", "performance"}
    assert set(payload["metadata"]) >= {
        "source_name",
        "frame_count",
        "processed_frames",
        "frame_width",
        "frame_height",
        "elapsed_sec",
    }
    assert payload["performance"]["summary"] == {"processed_frames": 3}
    assert payload["frames"][0]["primary"] == {"object_id": 7, "raw_primary_score": 0.42, "lane": "center"}
    assert payload["peak_event"]["objects"][0]["object_id"] == 7
    assert payload["peak_event"]["primary"]["raw_primary_score"] == payload["peak_event"]["objects"][0]["risk"]["risk_score"]
    assert payload["peak_event"]["objects"][0]["eta"]["collision_ttc_sec"] == 2.2
    assert payload["peak_event"]["objects"][0]["risk"]["factors"]["ttc_score"] == 0.2
    assert "timelineRows" not in payload
    assert "raw_primary_score" not in payload["peak_event"]
    assert "risk_score" not in payload["peak_event"]
    assert "frameIndex" not in payload
    assert "timestampSec" not in payload["peak_event"]
    assert "peakEvent" not in payload
    assert "collisionSec" not in payload["peak_event"]["objects"][0]["eta"]
    assert "etaPressure" not in payload["peak_event"]["objects"][0]["risk"]["factors"]
    assert "trackId" not in payload["peak_event"]["primary"]
    _assert_keys_absent(
        payload,
        {"frameIndex", "timestampSec", "peakEvent", "collisionSec", "etaPressure", "trackId"},
    )


def test_serialize_result_extracts_images_to_top_level_image_ref():
    event = {
        "frame_index": 42,
        "timestamp_sec": 1.4,
        "stabilized_state": "DANGER",
        "primary": {"object_id": 1, "raw_primary_score": 0.9, "lane": "center"},
        "traffic_light": {"state": "none", "confidence": 0.0},
        "lane_geometry": {"detected": False, "confidence": 0.25, "corridor": []},
        "objects": [_client_object(display_id=1, object_id=1, raw_state="DANGER")],
        "original_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "overlay_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
    }

    payload = _serialize_result(_result_with_event(event), elapsed_sec=0.0, source_name="x.mp4")["payload"]

    assert payload["peak_event"]["image_ref"] == "f42"
    assert set(payload["images"]["f42"]) == {"original", "blend"}
    assert "original_rgb" not in payload["peak_event"]
    assert "overlay_rgb" not in payload["peak_event"]


def test_object_metric_exposes_stable_client_fields():
    event = RiskEvent(
        frame_index=3,
        timestamp_sec=0.1,
        raw_state="DANGER",
        collision_ttc_sec=1.2,
        direction="center",
        lane="center",
        object_type="car",
        risk_confidence=0.9,
        proximity_score=0.5,
        radial_flow_score=0.3,
        approach_score=0.4,
        bbox=(100, 50, 300, 250),
        reason="",
        object_id=7,
        lane_confidence=0.7,
        flow_confidence=0.5,
        ttc_agreement=0.85,
    )

    metric = _object_metric(event, 400, 200)

    assert metric["object_id"] == 7
    assert metric["bbox"] == [0.25, 0.25, 0.75, 1.25]
    assert set(metric["confidence"]) == {
        "risk_confidence",
        "detection_confidence",
        "lane_confidence",
        "depth_confidence",
        "flow_confidence",
        "expansion_confidence",
    }
    assert metric["confidence"]["lane_confidence"] == 0.7
    assert metric["eta"]["ttc_agreement"] == 0.85


def test_lane_metric_default_corridor_is_normalized():
    lane = _lane_metric(None, 400, 200)

    assert lane["detected"] is False
    assert len(lane["corridor"]) == 4
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in lane["corridor"])


def test_analyze_endpoint_forwards_clamped_analysis_settings(monkeypatch):
    captured = {}

    def fake_analyze_spatial_video(**kwargs):
        captured.update(kwargs)
        return {
            "fps": 30.0,
            "frame_count": 0,
            "processed_frames": 0,
            "frames": [],
            "events": [],
            "peak_event": None,
        }

    monkeypatch.setattr(app_module, "analyze_spatial_video", fake_analyze_spatial_video)

    async def call_endpoint(max_processed_frames=0):
        upload = UploadFile(filename="clip.mp4", file=io.BytesIO(b"fake video bytes"))
        return await app_module.analyze_endpoint(
            file=upload,
            mode="video",
            max_processed_frames=max_processed_frames,
            max_saved_events=99,
            resize_max_side=9999,
            depth_every=0,
            adaptive_depth="0",
            detect_every=999,
            lane_every=999,
            flow_every=999,
            start_sec=2.5,
            end_sec=0.0,
            start_frame=12,
            end_frame=90,
            session_id="",
        )

    response = asyncio.run(call_endpoint())

    assert response["payload"]["schema_version"] == 6
    assert captured["max_processed_frames"] == 1
    assert captured["max_saved_events"] == 50
    assert captured["resize_max_side"] == 1024
    assert captured["depth_every"] == 1
    assert captured["adaptive_depth"] is False
    assert captured["detect_every"] == 10
    assert captured["lane_every"] == 10
    assert captured["flow_every"] == 10
    assert captured["start_sec"] == 2.5
    assert captured["end_sec"] is None
    assert captured["start_frame"] == 12
    assert captured["end_frame"] == 90
    assert captured["progress_callback"] is None

    asyncio.run(call_endpoint(max_processed_frames=9000))
    assert captured["max_processed_frames"] == 9000
