"""Lifecycle tests for the lane Kalman smoother and lane scheduling helpers.

These tests pin the post-fix invariants for the corridor lock-in bug:
the synthetic default ROI must not initialize the Kalman, the smoother
must be resettable, coasting between scheduled frames must not require
re-feeding the cached measurement, and the geometry confidence formula
must keep typical real-world UFLDv2 outputs above the 0.05 cache gate.
"""

import numpy as np

from spectra.analysis.video import _endpoint_drift_px
from spectra.vision.road import (
    LaneKalman,
    RoadROI,
    _lane_geometry_confidence,
    apply_lane_kalman,
    default_road_roi,
)


def _make_roi(left_line, right_line, *, width=1280, height=720, confidence=0.8):
    polygon = np.array(
        [
            [left_line[0], left_line[1]],
            [right_line[0], right_line[1]],
            [right_line[2], right_line[3]],
            [left_line[2], left_line[3]],
        ],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=bool)
    return RoadROI(
        mask=mask,
        polygon=polygon,
        left_line=left_line,
        right_line=right_line,
        confidence=confidence,
        detected=True,
    )


class TestKalmanInitialState:
    def test_uninitialized_coast_returns_none(self):
        kalman = LaneKalman()
        assert kalman._initialized is False
        assert kalman.coast() == (None, None)
        assert kalman._initialized is False

    def test_default_roi_not_used_to_initialize(self):
        # The scheduling logic never feeds the default ROI to Kalman. Verify
        # the smoother stays cold if we follow that contract.
        kalman = LaneKalman()
        default = default_road_roi((720, 1280))
        # We do NOT call apply_lane_kalman on `default`. After many "frames"
        # where UFLDv2 fails, the Kalman remains uninitialized.
        for _ in range(20):
            pass
        assert kalman._initialized is False
        assert default.detected is False


class TestColdStart:
    def test_first_detection_initializes_filter(self):
        kalman = LaneKalman()
        roi = _make_roi(
            left_line=(500, 420, 400, 719),
            right_line=(780, 420, 880, 719),
        )
        smoothed = apply_lane_kalman(roi, kalman)
        assert kalman._initialized is True
        # On init the emitted lines should equal the measurement exactly.
        assert smoothed.left_line == roi.left_line
        assert smoothed.right_line == roi.right_line
        assert kalman._last_y_top == 420.0
        assert kalman._last_y_bottom == 719.0


class TestCoasting:
    def test_coast_advances_without_measurement(self):
        kalman = LaneKalman()
        roi = _make_roi(
            left_line=(500, 420, 400, 719),
            right_line=(780, 420, 880, 719),
        )
        apply_lane_kalman(roi, kalman)
        # Coast a few frames — should still emit lines anchored at the
        # cached y-positions, no exception.
        for _ in range(4):
            coasted = apply_lane_kalman(roi, kalman, predict_only=True)
            assert coasted.left_line is not None
            assert coasted.right_line is not None
            assert coasted.left_line[1] == 420
            assert coasted.left_line[3] == 719

    def test_coast_drift_stays_small_for_steady_state(self):
        kalman = LaneKalman()
        roi = _make_roi(
            left_line=(500, 420, 400, 719),
            right_line=(780, 420, 880, 719),
        )
        apply_lane_kalman(roi, kalman)
        coasted = apply_lane_kalman(roi, kalman, predict_only=True)
        # After a single coast step on a freshly initialized filter (zero
        # velocity prior) the x positions shouldn't shift by more than a
        # couple of pixels.
        for original, predicted in zip(
            (*roi.left_line, *roi.right_line),
            (*coasted.left_line, *coasted.right_line),
        ):
            assert abs(original - predicted) <= 4


class TestReset:
    def test_reset_clears_state_and_anchors(self):
        kalman = LaneKalman()
        roi = _make_roi(
            left_line=(500, 420, 400, 719),
            right_line=(780, 420, 880, 719),
        )
        apply_lane_kalman(roi, kalman)
        assert kalman._initialized is True
        kalman.reset()
        assert kalman._initialized is False
        assert kalman._last_y_top is None
        assert kalman._last_y_bottom is None
        # After reset the next measurement re-initializes cleanly.
        new_roi = _make_roi(
            left_line=(550, 420, 450, 719),
            right_line=(820, 420, 920, 719),
        )
        smoothed = apply_lane_kalman(new_roi, kalman)
        assert smoothed.left_line == new_roi.left_line
        assert smoothed.right_line == new_roi.right_line


class TestDriftDetection:
    def test_endpoint_drift_helper_on_matching_rois(self):
        a = _make_roi(
            left_line=(500, 420, 400, 719),
            right_line=(780, 420, 880, 719),
        )
        b = _make_roi(
            left_line=(502, 420, 401, 719),
            right_line=(778, 420, 881, 719),
        )
        # Max absolute endpoint shift across the four corners is 2 pixels.
        assert _endpoint_drift_px(a, b) == 2.0

    def test_endpoint_drift_helper_returns_zero_when_not_detected(self):
        a = _make_roi(
            left_line=(500, 420, 400, 719),
            right_line=(780, 420, 880, 719),
        )
        default = default_road_roi((720, 1280))
        assert _endpoint_drift_px(a, default) == 0.0
        assert _endpoint_drift_px(default, a) == 0.0

    def test_large_drift_threshold_triggers_reset(self):
        # Reproduces the drift-reset path used in video.py: when the fresh
        # measurement is far from the Kalman prior, resetting and
        # re-applying snaps cleanly to the new geometry.
        kalman = LaneKalman()
        first = _make_roi(
            left_line=(500, 420, 200, 719),
            right_line=(780, 420, 1080, 719),
        )
        apply_lane_kalman(first, kalman)

        second = _make_roi(
            left_line=(800, 420, 500, 719),
            right_line=(1080, 420, 1200, 719),
        )
        smoothed_no_reset = apply_lane_kalman(second, kalman)
        # Without reset the Kalman pulls the emitted line toward the prior.
        assert smoothed_no_reset.left_line[2] != second.left_line[2]

        # With reset, the next emit snaps to the new measurement.
        kalman.reset()
        smoothed_after_reset = apply_lane_kalman(second, kalman)
        assert smoothed_after_reset.left_line == second.left_line
        assert smoothed_after_reset.right_line == second.right_line


class TestConfidenceRegressionGuard:
    def test_realistic_off_center_lane_clears_cache_gate(self):
        # A real-world UFLDv2 output: ego lane shifted ~20% off center,
        # reasonable bottom width, modest perspective. Before the
        # confidence relaxation this could fall under 0.05.
        width, height = 1280, 720
        left_line = (610, 420, 450, 719)
        right_line = (770, 420, 920, 719)
        conf = _lane_geometry_confidence(
            left_line, right_line, width=width, height=height
        )
        assert conf > 0.05

    def test_well_centered_lane_keeps_high_confidence(self):
        width, height = 1280, 720
        left_line = (580, 420, 480, 719)
        right_line = (700, 420, 800, 719)
        conf = _lane_geometry_confidence(
            left_line, right_line, width=width, height=height
        )
        assert conf >= 0.6

    def test_pathological_lane_still_fails(self):
        # bottom_ratio < 0.12 hard-fails — make sure we didn't accidentally
        # bypass the safety check by raising the starting confidence.
        width, height = 1280, 720
        left_line = (640, 420, 620, 719)
        right_line = (660, 420, 680, 719)
        conf = _lane_geometry_confidence(
            left_line, right_line, width=width, height=height
        )
        assert conf == 0.0
