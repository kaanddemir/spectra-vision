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


def _v5_object(**overrides):
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
            "sources": {
                "depth": {"etaSec": 2.2, "confidence": 0.64},
                "flow": {"etaSec": 2.4, "confidence": 0.5},
                "expansion": {"etaSec": 2.1, "confidence": 0.55},
            },
        },
        "motion": {
            "distanceM": 18.5,
            "closingSpeedMps": 8.4,
            "expansionScore": 0.1,
            "radialScore": 0.2,
        },
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


def test_serialized_result_uses_v5_schema():
    """``_serialize_result`` must emit the v5 schema:
       - ``schemaVersion: 5``
       - ``frames`` timeline rows with a ``primary`` pointer + ``state``
       - object-centric per-object metrics under ``objects[]`` (``risk``,
         ``eta``, ``motion``, ``lane``, ``confidence``, ``tracking``)
       - frame-level ``trafficLight`` as ``{state, confidence}``
       - ``imageRef`` on events + separate ``images`` dict
       - internal snake_case diagnostics stripped from the client payload
    """

    objects = [_v5_object()]
    event = {
        # internal diagnostics used for dedup / image keying
        "frame_index": 1,
        "timestamp_sec": 0.1,
        # v5 client-facing row (as produced by _frame_row / _event_payload_base)
        "frameIndex": 1,
        "timestampSec": 0.1,
        "state": "CAUTION",
        "primary": {"trackId": 7, "score": 0.42, "lane": "center"},
        "trafficLight": {"state": "red", "confidence": 0.8},
        "laneGeometry": {"detected": False, "confidence": 0.25, "corridor": []},
        "objects": objects,
        # RGB attached fields are absent — exercises the no-image path
    }
    frame_row = {
        "frameIndex": 1,
        "timestampSec": 0.1,
        "state": "CAUTION",
        "primary": {"trackId": 7, "score": 0.42, "lane": "center"},
        "trafficLight": {"state": "red", "confidence": 0.8},
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
    assert payload["schemaVersion"] == 5
    assert "frames" in payload and "timelineRows" not in payload
    assert "images" in payload and isinstance(payload["images"], dict)
    assert set(payload["performance"]) == {"summary", "logs"}

    # Frame row shape — primary pointer, no redundant per-object fields
    row = payload["frames"][0]
    assert row["frameIndex"] == 1
    assert row["timestampSec"] == 0.1
    assert row["state"] == "CAUTION"
    assert row["primary"] == {"trackId": 7, "score": 0.42, "lane": "center"}
    assert row["objects"] == objects
    # legacy flat fields must be gone
    for legacy_key in ("stabilizedRiskState", "primaryObjectId", "primaryRiskScore", "primaryLane", "ttcSec", "lanePosition", "objectId", "objectType", "lane", "riskState", "timeSec"):
        assert legacy_key not in row, f"legacy field {legacy_key!r} leaked into frame row"

    # peakEvent shape
    peak = payload["peakEvent"]
    assert peak["frameIndex"] == 1
    assert peak["state"] == "CAUTION"
    assert peak["primary"] == {"trackId": 7, "score": 0.42, "lane": "center"}
    assert peak["trafficLight"] == {"state": "red", "confidence": 0.8}
    assert peak["objects"] == objects
    # internal diagnostics must not leak to the client
    for internal_key in ("frame_index", "timestamp_sec", "risk_score", "primary_risk_score", "stabilizedRiskState", "primaryObjectId"):
        assert internal_key not in peak, f"internal field {internal_key!r} leaked into peakEvent"
    # Image attachment: event had no RGB → no imageRef
    assert "imageRef" not in peak
    assert "images" not in peak

    # Per-object schema (object-centric v5 shape)
    obj = peak["objects"][0]
    assert obj["id"] == 7
    assert obj["label"] == "Car"
    assert obj["class"] == "car"
    assert obj["state"] == "CAUTION"
    assert obj["tracking"]["trackId"] == 7
    assert obj["risk"]["score"] == 0.42
    assert set(obj["risk"]["factors"]) == {"etaPressure", "proximity", "approach", "crossing", "brake"}
    for value in obj["risk"]["factors"].values():
        assert 0.0 <= value <= 1.0
    assert obj["eta"]["collisionSec"] == 2.2
    assert obj["eta"]["display"] == "2.2s"
    assert set(obj["eta"]["sources"]) == {"depth", "flow", "expansion"}
    assert obj["eta"]["sources"]["depth"] == {"etaSec": 2.2, "confidence": 0.64}
    assert obj["motion"]["distanceM"] == 18.5
    assert obj["motion"]["closingSpeedMps"] == 8.4
    assert obj["lane"] == {"bucket": "center", "position": 0.0, "crossing": 0.2}
    assert obj["confidence"]["overall"] == 0.67
    assert set(obj["confidence"]) == {"overall", "detection", "lane", "depth", "flow", "expansion"}
    # legacy flat per-object keys gone
    for removed in ("objectId", "displayId", "objectType", "rawRiskState", "riskScore", "riskFactors", "collisionEta", "kinematics", "evidence", "overallConfidence", "ttcAgreement", "lanePosition"):
        assert removed not in obj, f"legacy per-object field {removed!r} leaked"

    # Legacy regression guards
    assert "detections" not in peak
    assert "laneScores" not in row


def test_serialized_result_separates_images_via_imageref():
    """Events that carry RGB views must emit ``imageRef`` and the images
    must land in the top-level ``images`` dict, not inline."""

    def _stub_rgb():
        return np.zeros((4, 4, 3), dtype=np.uint8)

    objects = [_v5_object(
        id=1,
        state="DANGER",
        tracking={"trackId": 1},
    )]
    peak = {
        "frame_index": 42,
        "timestamp_sec": 1.4,
        "frameIndex": 42,
        "timestampSec": 1.4,
        "state": "DANGER",
        "primary": {"trackId": 1, "score": 0.9, "lane": "left"},
        "trafficLight": {"state": "none", "confidence": 0.0},
        "laneGeometry": {"detected": False, "confidence": 0.25, "corridor": []},
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
    """Pointer invariant: ``primary.score`` must equal the matching
    ``objects[trackId].risk.score`` even when the stabilized frame ``state``
    differs from the primary's raw object ``state``. The score is computed
    from the raw primary event so hysteresis lag never leaks into the value.
    """

    objects = [_v5_object(
        id=4,
        state="CAUTION",
        risk={
            "score": 0.558,  # raw classifier output
            "factors": {"etaPressure": 0.5, "proximity": 0.704, "approach": 0.055, "crossing": 0.5, "brake": 0.0},
        },
        tracking={"trackId": 4},
    )]
    # Hysteresis hasn't upgraded yet, so the FRAME's stabilized state is SAFE
    # while the primary object's RAW state is CAUTION. primary.score MUST still
    # equal objects[0].risk.score == 0.558.
    frame_row = {
        "frameIndex": 213,
        "timestampSec": 7.1,
        "state": "SAFE",
        "primary": {"trackId": 4, "score": 0.558, "lane": "left"},
        "trafficLight": {"state": "none", "confidence": 0.0},
        "objects": objects,
    }
    event = {
        "frame_index": 213,
        "timestamp_sec": 7.1,
        "frameIndex": 213,
        "timestampSec": 7.1,
        "state": "SAFE",
        "primary": {"trackId": 4, "score": 0.558, "lane": "left"},
        "trafficLight": {"state": "none", "confidence": 0.0},
        "laneGeometry": {"detected": False, "confidence": 0.25, "corridor": []},
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
    primary_obj = next(o for o in row["objects"] if o["tracking"]["trackId"] == row["primary"]["trackId"])
    assert row["primary"]["score"] == primary_obj["risk"]["score"], (
        f"primary.score {row['primary']['score']} != objects[primary].risk.score {primary_obj['risk']['score']}"
    )
    # Stabilized frame state can still differ from object's raw state
    assert row["state"] == "SAFE"
    assert primary_obj["state"] == "CAUTION"

    # Same invariant for peakEvent
    peak = payload["peakEvent"]
    peak_primary = next(o for o in peak["objects"] if o["tracking"]["trackId"] == peak["primary"]["trackId"])
    assert peak["primary"]["score"] == peak_primary["risk"]["score"]


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
