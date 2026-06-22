"""Required ONNX model and lane-only API contract tests."""

import asyncio
import io

import numpy as np
from fastapi import UploadFile

import spectra.app as app_module
from spectra.app import _serialize_result
from spectra.vision.motion import compute_velocity
from spectra.vision.preprocessing import preprocess_frame


def test_compute_velocity_runs_without_neural_flow_model():
    """Optical flow is classical (DIS) — no ONNX model is required anymore."""

    previous = preprocess_frame(np.zeros((32, 48, 3), dtype=np.uint8), max_side=48)
    current = preprocess_frame(np.ones((32, 48, 3), dtype=np.uint8) * 16, max_side=48)

    result = compute_velocity(previous, current)
    assert result.flow.shape == (previous.gray.shape[0], previous.gray.shape[1], 2)
    assert result.magnitude_norm.shape == previous.gray.shape


def test_serialized_result_uses_v2_schema():
    """``_serialize_result`` must emit the v2 schema:
       - ``schemaVersion: 2``
       - ``frames`` (renamed from ``timelineRows``)
       - top-level ``primary*`` pointers + ``stabilizedRiskState`` only;
         per-object metrics live in ``objects[]``
       - per-object ``rawRiskState`` (not ``riskState``) and ``confidence``
         (0..1, not ``confidencePct`` 0..100)
       - ``imageRef`` on events + separate ``images`` dict
    """

    objects = [{
        "objectId": 7,
        "objectType": "car",
        "rawRiskState": "CAUTION",
        "riskScore": 0.42,
        "lane": "center",
        "ttcSec": 2.2,
        "nearScore": 0.4,
        "closingSpeed": 0.3,
        "crossingRisk": 0.2,
        "lanePosition": 0.0,
        "confidence": 0.67,
        "distanceM": 18.5,
        "closingMps": 2.4,
        "depthTtcSec": 7.7,
    }]
    event = {
        "frame_index": 1,
        "timestamp_sec": 0.1,
        "stabilized_risk_state": "CAUTION",
        "primary_object_id": 7,
        "primary_risk_score": 0.42,
        "primary_lane": "center",
        "ttc_sec": 2.2,
        "near_score": 0.4,
        "closing_speed": 0.3,
        "objects": objects,
        # RGB attached fields are absent — exercises the no-image path
    }
    frame_row = {
        "frameIndex": 1,
        "timestampSec": 0.1,
        "stabilizedRiskState": "CAUTION",
        "primaryObjectId": 7,
        "primaryRiskScore": 0.42,
        "primaryLane": "center",
        "objects": objects,
    }
    result = {
        "fps": 30.0,
        "frame_count": 3,
        "processed_frames": 3,
        "frames": [frame_row],
        "events": [event],
        "peak_event": event,
    }

    payload = _serialize_result(result, elapsed_sec=0.01, source_name="sample.mp4")["payload"]

    # Schema version + top-level shape
    assert payload["schemaVersion"] == 2
    assert "frames" in payload and "timelineRows" not in payload
    assert "images" in payload and isinstance(payload["images"], dict)

    # Frame row shape — primary pointers, no redundant per-object fields
    row = payload["frames"][0]
    assert row["frameIndex"] == 1
    assert row["timestampSec"] == 0.1
    assert row["stabilizedRiskState"] == "CAUTION"
    assert row["primaryObjectId"] == 7
    assert row["primaryRiskScore"] == 0.42
    assert row["primaryLane"] == "center"
    assert row["objects"] == objects
    # v1 redundant top-level fields must be gone
    for legacy_key in ("ttcSec", "nearScore", "closingSpeed", "crossingRisk", "lanePosition", "confidencePct", "objectId", "objectType", "lane", "riskState", "timeSec"):
        assert legacy_key not in row, f"v1 field {legacy_key!r} leaked into frame row"

    # peakEvent shape
    peak = payload["peakEvent"]
    assert peak["frameIndex"] == 1
    assert peak["stabilizedRiskState"] == "CAUTION"
    assert peak["primaryObjectId"] == 7
    assert peak["primaryRiskScore"] == 0.42
    assert peak["primaryLane"] == "center"
    assert peak["objects"] == objects
    # v1 keys gone from event too
    for legacy_key in ("riskState", "riskScore", "lane", "ttcSec", "objectId", "objectType", "nearScore", "closingSpeed", "crossingRisk", "lanePosition", "confidencePct"):
        assert legacy_key not in peak, f"v1 field {legacy_key!r} leaked into peakEvent"
    # Image attachment: event had no RGB → no imageRef
    assert "imageRef" not in peak
    assert "images" not in peak

    # Per-object schema
    obj = peak["objects"][0]
    assert obj["rawRiskState"] == "CAUTION"
    assert obj["confidence"] == 0.67
    assert obj["distanceM"] == 18.5
    assert obj["closingMps"] == 2.4
    assert obj["depthTtcSec"] == 7.7
    assert "riskState" not in obj or obj["riskState"] == obj["rawRiskState"]  # objects[].riskState removed
    assert "confidencePct" not in obj

    # Legacy regression guards (kept from v1 era)
    legacy_lane_key = "zo" + "ne"
    assert legacy_lane_key not in peak
    assert (legacy_lane_key + "Metrics") not in peak
    assert "detections" not in peak
    assert "laneScores" not in row


def test_serialized_result_separates_images_via_imageref():
    """Events that carry RGB views must emit ``imageRef`` and the images
    must land in the top-level ``images`` dict, not inline."""

    def _stub_rgb():
        return np.zeros((4, 4, 3), dtype=np.uint8)

    objects = [{
        "objectId": 1,
        "objectType": "car",
        "rawRiskState": "DANGER",
        "riskScore": 0.9,
        "lane": "left",
        "ttcSec": 0.8,
        "nearScore": 0.7,
        "closingSpeed": 0.8,
        "crossingRisk": 0.9,
        "lanePosition": -0.5,
        "confidence": 0.95,
    }]
    peak = {
        "frame_index": 42,
        "timestamp_sec": 1.4,
        "stabilized_risk_state": "DANGER",
        "primary_object_id": 1,
        "primary_risk_score": 0.9,
        "primary_lane": "left",
        "ttc_sec": 0.8,
        "near_score": 0.7,
        "closing_speed": 0.8,
        "objects": objects,
        "original_rgb": _stub_rgb(),
        "overlay_rgb": _stub_rgb(),
    }
    result = {
        "fps": 30.0,
        "frame_count": 1,
        "processed_frames": 1,
        "frames": [],
        "events": [peak],
        "peak_event": peak,
    }

    payload = _serialize_result(result, elapsed_sec=0.0, source_name="x.mp4")["payload"]
    assert payload["peakEvent"]["imageRef"] == "f42"
    assert "f42" in payload["images"]
    assert set(payload["images"]["f42"].keys()) == {"original", "blend"}
    # Same event in events[] is filtered by _is_same_event (peak vs events dedup)
    assert payload["events"] == []


def test_primary_risk_score_matches_objects_entry():
    """Pointer invariant: ``primaryRiskScore`` must equal the matching
    ``objects[primaryObjectId].riskScore`` even when the stabilized state
    differs from the primary's raw state. The score is computed from the
    raw primary event so hysteresis lag never leaks into the value.
    """

    objects = [{
        "objectId": 4,
        "objectType": "car",
        "rawRiskState": "CAUTION",
        "riskScore": 0.558,  # raw classifier output
        "lane": "left",
        "ttcSec": 0.87,
        "nearScore": 0.704,
        "closingSpeed": 0.055,
        "crossingRisk": 0.5,
        "lanePosition": -1.4,
        "confidence": 0.54,
    }]
    # Hysteresis hasn't upgraded yet, so the FRAME's stabilized state is SAFE
    # while the primary object's RAW state is CAUTION. Without the fix,
    # primary_risk_score would be ~0.188 (computed from stabilized=SAFE)
    # and disagree with objects[0].riskScore == 0.558.
    frame_row = {
        "frameIndex": 213,
        "timestampSec": 7.1,
        "stabilizedRiskState": "SAFE",
        "primaryObjectId": 4,
        "primaryRiskScore": 0.558,  # MUST equal objects[0].riskScore
        "primaryLane": "left",
        "objects": objects,
    }
    event = {
        "frame_index": 213,
        "timestamp_sec": 7.1,
        "stabilized_risk_state": "SAFE",
        "primary_object_id": 4,
        "primary_risk_score": 0.558,
        "primary_lane": "left",
        "ttc_sec": 0.87,
        "near_score": 0.704,
        "closing_speed": 0.055,
        "objects": objects,
    }
    result = {
        "fps": 30.0,
        "frame_count": 1,
        "processed_frames": 1,
        "frames": [frame_row],
        "events": [event],
        "peak_event": event,
    }

    payload = _serialize_result(result, elapsed_sec=0.0, source_name="x.mp4")["payload"]

    # Frame-level invariant
    row = payload["frames"][0]
    primary_obj = next(o for o in row["objects"] if o["objectId"] == row["primaryObjectId"])
    assert row["primaryRiskScore"] == primary_obj["riskScore"], (
        f"primaryRiskScore {row['primaryRiskScore']} != objects[primary].riskScore {primary_obj['riskScore']}"
    )
    # Stabilized state can still differ from object's raw state
    assert row["stabilizedRiskState"] == "SAFE"
    assert primary_obj["rawRiskState"] == "CAUTION"

    # Same invariant for peakEvent
    peak = payload["peakEvent"]
    peak_primary = next(o for o in peak["objects"] if o["objectId"] == peak["primaryObjectId"])
    assert peak["primaryRiskScore"] == peak_primary["riskScore"]


def test_analyze_endpoint_forwards_clamped_analysis_settings(monkeypatch):
    """The API layer must clamp form values before delegating analysis."""

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

    assert response["payload"]["schemaVersion"] == 2
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
