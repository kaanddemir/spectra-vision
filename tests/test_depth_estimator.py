"""Unit tests for AI depth calibration."""

import numpy as np

from zone_risk.pipeline.risk_calculator import calculate_region_risk
from zone_risk.vision.depth_estimator import _calibrate_model_near_map


class TestModelNearMapCalibration:
    def test_suppresses_smooth_perspective_plane(self):
        height, width = 80, 120
        vertical_near = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(height, 1)
        near_map = np.repeat(vertical_near, width, axis=1)

        calibrated = _calibrate_model_near_map(near_map)

        assert calibrated.dtype == np.float32
        assert float(calibrated.max()) <= 0.35

    def test_preserves_row_relative_obstacle(self):
        height, width = 80, 120
        vertical_near = np.linspace(0.0, 0.75, height, dtype=np.float32).reshape(height, 1)
        near_map = np.repeat(vertical_near, width, axis=1)
        near_map[42:62, 48:72] = np.clip(near_map[42:62, 48:72] + 0.35, 0.0, 1.0)

        calibrated = _calibrate_model_near_map(near_map)

        obstacle_score = float(np.percentile(calibrated[42:62, 48:72], 80))
        road_score_same_rows = float(np.percentile(calibrated[42:62, :24], 80))
        assert obstacle_score > 0.80
        assert road_score_same_rows < 0.25

    def test_road_plane_with_strong_flow_is_not_danger(self):
        height, width = 80, 120
        vertical_near = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(height, 1)
        near_map = _calibrate_model_near_map(np.repeat(vertical_near, width, axis=1))
        high_motion = np.ones((height, width), dtype=np.float32)
        flow = np.zeros((height, width, 2), dtype=np.float32)

        event = calculate_region_risk(
            frame_index=0,
            timestamp_sec=0.0,
            bbox=(0, 0, width, height),
            object_type="road plane",
            near_map=near_map,
            magnitude_norm=high_motion,
            divergence_norm=high_motion,
            flow=flow,
            roi_mask=np.ones((height, width), dtype=bool),
        )

        assert event.near_score < 0.35
        assert event.state != "DANGER"
