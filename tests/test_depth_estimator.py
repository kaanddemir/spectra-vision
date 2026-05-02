"""Unit tests for depth estimation and AI-depth calibration."""

import numpy as np
import pytest

from zone_risk.vision import depth_model
from zone_risk.vision.depth_estimator import DepthResult, _calibrate_model_near_map, estimate_frame_depth
from zone_risk.vision.preprocess import preprocess_frame


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


class TestDepthEstimator:
    def test_estimate_frame_depth_uses_required_onnx_model(self, monkeypatch):
        class FakeDepthModel:
            def predict(self, rgb):
                height, width = rgb.shape[:2]
                values = np.linspace(0.0, 1.0, height * width, dtype=np.float32)
                return values.reshape(height, width)

        monkeypatch.setattr(depth_model, "get_model", lambda: FakeDepthModel())
        frame_bgr = np.zeros((64, 96, 3), dtype=np.uint8)
        frame_bgr[:, :, 0] = np.linspace(20, 180, 96, dtype=np.uint8)
        frame_bgr[:, :, 1] = np.linspace(40, 200, 64, dtype=np.uint8).reshape(64, 1)
        frame = preprocess_frame(frame_bgr, max_side=96)

        result = estimate_frame_depth(frame)

        assert isinstance(result, DepthResult)
        assert result.depth_map.shape == frame.gray.shape
        assert result.near_map.shape == frame.gray.shape
        assert result.depth_map.dtype == np.uint8
        assert result.near_map.dtype == np.float32
        assert 0.0 <= float(result.near_map.min()) <= float(result.near_map.max()) <= 1.0

    def test_estimate_frame_depth_errors_when_onnx_missing(self, monkeypatch):
        monkeypatch.setattr(depth_model, "get_model", lambda: None)
        frame = preprocess_frame(np.zeros((32, 48, 3), dtype=np.uint8), max_side=48)

        with pytest.raises(RuntimeError, match="Depth Anything ONNX model missing"):
            estimate_frame_depth(frame)
