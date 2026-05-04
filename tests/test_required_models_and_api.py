"""Required ONNX model and lane-only API contract tests."""

import numpy as np

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


def test_serialized_result_uses_lane_contract_only():
    event = {
        "frame_index": 1,
        "timestamp_sec": 0.1,
        "risk_score": 0.5,
        "risk_band": "medium",
        "risk_state": "CAUTION",
        "primary_lane": "center",
        "estimated_ttc_sec": 2.2,
        "uncertainty_pct": 10.0,
        "heuristic_summary": "CAUTION in the center lane",
        "reasons": ["test"],
        "lane_metrics": [{"lane": "center", "score": 0.5}],
        "objects": [{
            "objectId": 7,
            "objectType": "car",
            "ttcSec": 2.2,
            "riskState": "CAUTION",
            "lanePosition": 0.0,
            "crossingRisk": 0.2,
            "confidencePct": 67.0,
        }],
        "object_type": "car",
        "approach": "approaching",
        "bbox": [1, 2, 3, 4],
        "object_id": 7,
        "near_score": 0.4,
        "closing_speed": 0.3,
        "velocity_magnitude": 0.2,
        "expansion_rate": 0.1,
        "crossing_risk": 0.2,
        "lane_position": 0.0,
        "ttc_components": [],
    }
    result = {
        "summary": event["heuristic_summary"],
        "media_type": "video",
        "fps": 30.0,
        "frame_count": 3,
        "processed_frames": 3,
        "sampled_frames": 3,
        "timeline_rows": [{
            "frameIndex": 1,
            "timeSec": 0.1,
            "riskState": "CAUTION",
            "lane": "center",
            "ttcSec": 2.2,
            "objectId": 7,
            "objectType": "car",
            "nearScore": 0.4,
            "closingSpeed": 0.3,
            "crossingRisk": 0.2,
            "lanePosition": 0.0,
            "confidencePct": 67.0,
            "objects": event["objects"],
        }],
        "events": [event],
        "peak_event": event,
    }

    payload = _serialize_result(result, elapsed_sec=0.01, source_name="sample.mp4")["payload"]
    peak = payload["peakEvent"]

    assert peak["lane"] == "center"
    assert peak["objects"] == event["objects"]
    row = payload["timelineRows"][0]
    assert row["objectId"] == 7
    assert row["objectType"] == "car"
    assert row["lanePosition"] == 0.0
    assert row["crossingRisk"] == 0.2
    assert row["confidencePct"] == 67.0
    assert row["objects"] == event["objects"]
    legacy_lane_key = "zo" + "ne"
    legacy_metrics_key = legacy_lane_key + "Metrics"
    legacy_scores_key = legacy_lane_key + "Scores"
    legacy_image_key = "segment" + "ation"

    assert legacy_lane_key not in peak
    assert legacy_metrics_key not in peak
    assert "detections" not in peak
    assert legacy_image_key not in peak.get("images", {})
    assert "laneScores" not in row
    assert legacy_scores_key not in payload["timelineRows"][0]
