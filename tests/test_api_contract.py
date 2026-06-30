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
        "id": 7,
        "label": "Car",
        "class": "car",
        "state": "CAUTION",
        "risk": {
            "score": 0.42,
            "factors": {
                "etaPressure": 0.2,
                "proximity": 0.4,
                "approach": 0.3,
                "crossing": 0.2,
                "brake": 0.0,
            },
        },
        "eta": {
            "collisionSec": 2.2,
            "display": "2.2s",
            "agreement": 0.9,
            "sources": {"depth": {"etaSec": 2.2, "confidence": 0.64}},
        },
        "motion": {"distanceM": 18.5, "closingSpeedMps": 8.4},
        "lane": {"bucket": "center", "position": 0.0, "crossing": 0.2},
        "confidence": {
            "overall": 0.67,
            "detection": 0.72,
            "lane": 0.8,
            "depth": 0.64,
            "flow": 0.5,
            "expansion": 0.55,
        },
        "tracking": {"trackId": 7},
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
                "frameIndex": event["frameIndex"],
                "timestampSec": event["timestampSec"],
                "state": event["state"],
                "primary": event["primary"],
                "trafficLight": event["trafficLight"],
                "objects": event["objects"],
            }
        ],
        "events": [event],
        "peak_event": event,
        "performance_summary": {"processed_frames": 3},
        "performance_logs": ["frame 0"],
    }


def test_serialize_result_keeps_v5_shape_and_strips_internal_fields():
    event = {
        "frame_index": 1,
        "timestamp_sec": 0.1,
        "frameIndex": 1,
        "timestampSec": 0.1,
        "state": "CAUTION",
        "primary": {"trackId": 7, "score": 0.42, "lane": "center"},
        "trafficLight": {"state": "red", "confidence": 0.8},
        "laneGeometry": {"detected": False, "confidence": 0.25, "corridor": []},
        "objects": [_client_object()],
    }

    payload = _serialize_result(_result_with_event(event), elapsed_sec=0.01, source_name="sample.mp4")["payload"]

    assert payload["schemaVersion"] == 5
    assert set(payload) >= {"frames", "events", "peakEvent", "images", "performance"}
    assert payload["performance"]["summary"] == {"processed_frames": 3}
    assert payload["frames"][0]["primary"] == {"trackId": 7, "score": 0.42, "lane": "center"}
    assert payload["peakEvent"]["objects"][0]["tracking"]["trackId"] == 7
    assert "timelineRows" not in payload
    assert "frame_index" not in payload["peakEvent"]
    assert "timestamp_sec" not in payload["peakEvent"]


def test_serialize_result_extracts_images_to_top_level_imageref():
    event = {
        "frame_index": 42,
        "timestamp_sec": 1.4,
        "frameIndex": 42,
        "timestampSec": 1.4,
        "state": "DANGER",
        "primary": {"trackId": 1, "score": 0.9, "lane": "center"},
        "trafficLight": {"state": "none", "confidence": 0.0},
        "laneGeometry": {"detected": False, "confidence": 0.25, "corridor": []},
        "objects": [_client_object(id=1, tracking={"trackId": 1}, state="DANGER")],
        "original_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "overlay_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
    }

    payload = _serialize_result(_result_with_event(event), elapsed_sec=0.0, source_name="x.mp4")["payload"]

    assert payload["peakEvent"]["imageRef"] == "f42"
    assert set(payload["images"]["f42"]) == {"original", "blend"}
    assert "original_rgb" not in payload["peakEvent"]
    assert "overlay_rgb" not in payload["peakEvent"]


def test_object_metric_exposes_stable_client_fields():
    event = RiskEvent(
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
        lane_confidence=0.7,
        flow_confidence=0.5,
        ttc_agreement=0.85,
    )

    metric = _object_metric(event, 400, 200)

    assert metric["tracking"]["trackId"] == 7
    assert metric["bbox"] == [0.25, 0.25, 0.75, 1.25]
    assert set(metric["confidence"]) == {"overall", "detection", "lane", "depth", "flow", "expansion"}
    assert metric["confidence"]["lane"] == 0.7
    assert metric["eta"]["agreement"] == 0.85


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

    async def call_endpoint():
        upload = UploadFile(filename="clip.mp4", file=io.BytesIO(b"fake video bytes"))
        return await app_module.analyze_endpoint(
            file=upload,
            mode="video",
            max_processed_frames=0,
            max_saved_events=99,
            resize_max_side=9999,
            depth_every=0,
            adaptive_depth="0",
            detect_every=999,
            lane_every=999,
            flow_every=999,
            start_sec=2.5,
            end_sec=0.0,
            session_id="",
        )

    response = asyncio.run(call_endpoint())

    assert response["payload"]["schemaVersion"] == 5
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
    assert captured["progress_callback"] is None
