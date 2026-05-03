"""Required ONNX model and lane-only API contract tests."""

import numpy as np
import pytest

from spectra.app import _serialize_result
from spectra.vision import models
from spectra.vision.motion import compute_velocity
from spectra.vision.preprocessing import preprocess_frame


def test_compute_velocity_errors_when_neuflow_missing(monkeypatch):
    monkeypatch.setattr(models, "get_flow_model", lambda: None)
    previous = preprocess_frame(np.zeros((32, 48, 3), dtype=np.uint8), max_side=48)
    current = preprocess_frame(np.ones((32, 48, 3), dtype=np.uint8) * 16, max_side=48)

    with pytest.raises(RuntimeError, match="NeuFlow ONNX model missing"):
        compute_velocity(previous, current)


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
        "timeline_rows": [{"frameIndex": 1, "lane": "center", "laneScores": {"center": 0.5}}],
        "events": [event],
        "peak_event": event,
    }

    payload = _serialize_result(result, elapsed_sec=0.01, source_name="sample.mp4")["payload"]
    peak = payload["peakEvent"]

    assert peak["lane"] == "center"
    assert peak["laneMetrics"] == [{"lane": "center", "score": 0.5}]
    legacy_lane_key = "zo" + "ne"
    legacy_metrics_key = legacy_lane_key + "Metrics"
    legacy_scores_key = legacy_lane_key + "Scores"
    legacy_image_key = "segment" + "ation"

    assert legacy_lane_key not in peak
    assert legacy_metrics_key not in peak
    assert legacy_image_key not in peak.get("images", {})
    assert "laneScores" in payload["timelineRows"][0]
    assert legacy_scores_key not in payload["timelineRows"][0]
