"""Unit tests for risk_calculator.py — pure functions and StateStabilizer."""

import pytest
import numpy as np

from zone_risk.pipeline.risk_calculator import (
    zone_from_bbox,
    direction_from_flow,
    compute_ttc,
    classify_state,
    calculate_region_risk,
    is_clear_safe_event,
    is_imminent_danger,
    MetricEmaSmoother,
    score_event,
    select_primary_event,
    StateStabilizer,
    RiskEvent,
    stabilized_event_state,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_event(state="SAFE", near_score=0.1, closing_speed=0.05, ttc_sec=10.0, object_id=None):
    return RiskEvent(
        frame_index=0,
        timestamp_sec=0.0,
        state=state,
        ttc_sec=ttc_sec,
        direction="center",
        zone="center",
        object_type="test",
        confidence=0.5,
        near_score=near_score,
        velocity_magnitude=0.0,
        closing_speed=closing_speed,
        bbox=(0, 0, 100, 100),
        reason="test",
        object_id=object_id,
    )


# ── zone_from_bbox ────────────────────────────────────────────────────────────

class TestZoneFromBbox:
    def test_left_zone(self):
        assert zone_from_bbox((0, 0, 50, 100), width=300) == "left"

    def test_right_zone(self):
        assert zone_from_bbox((250, 0, 300, 100), width=300) == "right"

    def test_center_zone(self):
        assert zone_from_bbox((100, 0, 200, 100), width=300) == "center"

    def test_exact_left_boundary(self):
        # center_x = 100, width/3 = 100 → NOT less than → center
        assert zone_from_bbox((50, 0, 150, 100), width=300) == "center"

    def test_edge_bbox_at_left_edge(self):
        assert zone_from_bbox((0, 0, 10, 100), width=300) == "left"


# ── direction_from_flow ───────────────────────────────────────────────────────

class TestDirectionFromFlow:
    def test_center_when_near_zero(self):
        assert direction_from_flow(0.01) == "center"

    def test_left_for_negative_flow(self):
        assert direction_from_flow(-0.1) == "left"

    def test_right_for_positive_flow(self):
        assert direction_from_flow(0.1) == "right"

    def test_exact_threshold_is_right(self):
        # condition is strict < 0.015, so 0.015 is NOT center
        assert direction_from_flow(0.015) == "right"

    def test_just_below_threshold_is_center(self):
        assert direction_from_flow(0.0149) == "center"


# ── compute_ttc ───────────────────────────────────────────────────────────────

class TestComputeTtc:
    def test_none_when_not_closing(self):
        assert compute_ttc(near_score=0.9, closing_speed=0.0) is None

    def test_none_at_threshold(self):
        assert compute_ttc(near_score=0.5, closing_speed=1e-3) is None

    def test_value_when_closing(self):
        ttc = compute_ttc(near_score=0.5, closing_speed=0.5)
        # distance_proxy = 0.5, closing_speed = 0.5 → TTC = 1.0
        assert ttc == pytest.approx(1.0)

    def test_very_near_object(self):
        # Relative monocular depth can saturate, but the reported pseudo-TTC
        # should not collapse to a literal zero.
        ttc = compute_ttc(near_score=1.0, closing_speed=0.5)
        assert ttc == pytest.approx(0.36)

    def test_no_visible_object_returns_none(self):
        # near_score=0.0 means no object in view — TTC is meaningless even
        # if flow suggests motion, so it must be suppressed.
        assert compute_ttc(near_score=0.0, closing_speed=0.1) is None

    def test_clamps_far_objects_to_none(self):
        # near=0.20 (just at floor), closing=0.08 → distance_proxy=0.80,
        # TTC=10s — beyond the reported max, so suppressed instead of shown.
        assert compute_ttc(near_score=0.20, closing_speed=0.08) is None

    def test_low_closing_speed_returns_none(self):
        # Below the closing-speed floor: flow is too noisy to trust.
        assert compute_ttc(near_score=0.5, closing_speed=0.05) is None


# ── classify_state ────────────────────────────────────────────────────────────

class TestClassifyState:
    def test_danger_when_near_and_closing_fast(self):
        assert classify_state(near_score=0.4, closing_speed=0.5, ttc_sec=0.5) == "DANGER"

    def test_caution_via_ttc(self):
        # near >= 0.25, ttc < 3.0 → CAUTION
        assert classify_state(near_score=0.3, closing_speed=0.3, ttc_sec=2.0) == "CAUTION"

    def test_caution_via_high_near_and_speed(self):
        # near >= 0.72, closing >= 0.10, no ttc
        assert classify_state(near_score=0.75, closing_speed=0.15, ttc_sec=None) == "CAUTION"

    def test_safe_low_values(self):
        assert classify_state(near_score=0.1, closing_speed=0.05, ttc_sec=10.0) == "SAFE"

    def test_safe_when_no_ttc_and_low_near(self):
        assert classify_state(near_score=0.2, closing_speed=0.5, ttc_sec=None) == "SAFE"

    def test_low_ttc_drives_danger_regardless_of_near_gate(self):
        # If compute_ttc has produced a sub-1s TTC, the evidence gating has
        # already passed — classification must trust it and not silently
        # downgrade to SAFE because near_score is below an internal threshold.
        assert classify_state(near_score=0.22, closing_speed=0.9, ttc_sec=0.1) == "DANGER"

    def test_caution_via_ttc_without_high_near(self):
        # Sub-3s TTC must drive at least CAUTION even when near_score is low,
        # so the displayed state never contradicts the displayed TTC.
        assert classify_state(near_score=0.22, closing_speed=0.5, ttc_sec=1.8) == "CAUTION"


# ── calculate_region_risk ────────────────────────────────────────────────────

class TestCalculateRegionRisk:
    def test_lateral_motion_magnitude_does_not_count_as_closing(self):
        height, width = 80, 120
        near_map = np.full((height, width), 0.85, dtype=np.float32)
        magnitude_norm = np.ones((height, width), dtype=np.float32)
        divergence_norm = np.zeros((height, width), dtype=np.float32)
        flow = np.zeros((height, width, 2), dtype=np.float32)
        flow[..., 0] = 18.0

        event = calculate_region_risk(
            frame_index=0,
            timestamp_sec=0.0,
            bbox=(0, 0, width, height),
            object_type="lateral motion",
            near_map=near_map,
            magnitude_norm=magnitude_norm,
            divergence_norm=divergence_norm,
            flow=flow,
        )

        assert event.velocity_magnitude == pytest.approx(1.0)
        assert event.closing_speed == pytest.approx(0.0)
        assert event.state == "SAFE"

    def test_radial_expansion_with_divergence_counts_as_closing(self):
        height, width = 80, 120
        near_map = np.full((height, width), 0.85, dtype=np.float32)
        magnitude_norm = np.ones((height, width), dtype=np.float32)
        divergence_norm = np.full((height, width), 0.9, dtype=np.float32)
        y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)
        focus_x = (width - 1) / 2.0
        focus_y = height * 0.55
        flow = np.zeros((height, width, 2), dtype=np.float32)
        flow[..., 0] = (x_coords - focus_x) * 0.55
        flow[..., 1] = (y_coords - focus_y) * 0.55

        event = calculate_region_risk(
            frame_index=0,
            timestamp_sec=0.0,
            bbox=(0, 0, width, height),
            object_type="approaching object",
            near_map=near_map,
            magnitude_norm=magnitude_norm,
            divergence_norm=divergence_norm,
            flow=flow,
        )

        assert event.closing_speed > 0.35
        assert event.ttc_sec is not None and event.ttc_sec < 1.0
        assert event.state == "DANGER"

    def test_local_high_motion_near_object_can_override_zone_average(self):
        height, width = 80, 120
        near_map = np.full((height, width), 0.18, dtype=np.float32)
        magnitude_norm = np.full((height, width), 0.05, dtype=np.float32)
        divergence_norm = np.zeros((height, width), dtype=np.float32)
        flow = np.zeros((height, width, 2), dtype=np.float32)

        near_map[30:60, 45:75] = 0.55
        magnitude_norm[30:60, 45:75] = 1.0
        flow[30:60, 45:75, 1] = 14.0

        event = calculate_region_risk(
            frame_index=0,
            timestamp_sec=0.0,
            bbox=(0, 0, width, height),
            object_type="localized impact",
            near_map=near_map,
            magnitude_norm=magnitude_norm,
            divergence_norm=divergence_norm,
            flow=flow,
        )

        assert event.near_score > 0.45
        assert event.closing_speed > 0.35
        assert event.state == "DANGER"

    def test_edge_lane_expansion_is_not_immediate_collision(self):
        height, width = 80, 160
        near_map = np.full((height, width), 0.9, dtype=np.float32)
        magnitude_norm = np.ones((height, width), dtype=np.float32)
        divergence_norm = np.full((height, width), 0.9, dtype=np.float32)
        y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)
        focus_x = (width - 1) / 2.0
        focus_y = height * 0.55
        flow = np.zeros((height, width, 2), dtype=np.float32)
        flow[..., 0] = (x_coords - focus_x) * 0.55
        flow[..., 1] = (y_coords - focus_y) * 0.55

        event = calculate_region_risk(
            frame_index=0,
            timestamp_sec=0.0,
            bbox=(0, 0, int(width * 0.25), height),
            object_type="side-lane expansion",
            near_map=near_map,
            magnitude_norm=magnitude_norm,
            divergence_norm=divergence_norm,
            flow=flow,
        )

        assert event.state != "DANGER"
        assert event.ttc_sec is None or event.ttc_sec >= 1.0


# ── is_imminent_danger ────────────────────────────────────────────────────────

class TestIsImminentDanger:
    def test_true_for_low_ttc_danger(self):
        event = make_event(state="DANGER", near_score=0.6, closing_speed=0.12, ttc_sec=0.8)
        assert is_imminent_danger(event) is True

    def test_true_for_low_ttc_even_when_near_is_marginal(self):
        event = make_event(state="DANGER", near_score=0.22, closing_speed=0.9, ttc_sec=0.8)
        assert is_imminent_danger(event) is True

    def test_false_for_non_urgent_danger(self):
        event = make_event(state="DANGER", near_score=0.6, closing_speed=0.2, ttc_sec=2.0)
        assert is_imminent_danger(event) is False


class TestStabilizedEventState:
    def test_imminent_danger_bypasses_upgrade_delay(self):
        stabilizer = StateStabilizer(upgrade_frames=3, downgrade_frames=5)
        event = make_event(state="DANGER", near_score=0.6, closing_speed=0.12, ttc_sec=0.8)

        assert stabilized_event_state(stabilizer, event) == "DANGER"

    def test_clear_safe_resets_held_danger(self):
        stabilizer = StateStabilizer(upgrade_frames=3, downgrade_frames=5)
        stabilizer.current_state = "DANGER"
        stabilizer.pending_state = "DANGER"
        event = make_event(state="SAFE", near_score=0.2, closing_speed=0.05, ttc_sec=10.7)

        assert is_clear_safe_event(event) is True
        assert stabilized_event_state(stabilizer, event) == "SAFE"
        assert stabilizer.current_state == "SAFE"

    def test_held_state_does_not_outrank_current_metrics(self):
        stabilizer = StateStabilizer(upgrade_frames=3, downgrade_frames=5)
        stabilizer.current_state = "DANGER"
        stabilizer.pending_state = "DANGER"
        event = make_event(state="CAUTION", near_score=0.5, closing_speed=0.2, ttc_sec=2.2)

        assert stabilized_event_state(stabilizer, event) == "CAUTION"

    def test_low_ttc_caution_bypasses_safe_upgrade_delay(self):
        stabilizer = StateStabilizer(upgrade_frames=3, downgrade_frames=5)
        event = make_event(state="CAUTION", near_score=0.25, closing_speed=0.4, ttc_sec=2.0)

        assert stabilized_event_state(stabilizer, event) == "CAUTION"


class TestMetricEmaSmoother:
    def test_dropout_does_not_immediately_clear_ttc(self):
        smoother = MetricEmaSmoother()

        first = smoother.smooth_event(
            make_event(state="DANGER", near_score=0.85, closing_speed=0.3, ttc_sec=0.6, object_id=102)
        )
        second = smoother.smooth_event(
            make_event(state="SAFE", near_score=0.85, closing_speed=0.0, ttc_sec=None, object_id=102)
        )

        assert first.ttc_sec == pytest.approx(0.6)
        assert second.closing_speed == pytest.approx(0.21)
        assert second.ttc_sec == pytest.approx(0.86)
        assert second.state == "DANGER"

    def test_smoothing_is_scoped_per_zone_object(self):
        smoother = MetricEmaSmoother()

        smoother.smooth_event(
            make_event(state="DANGER", near_score=0.85, closing_speed=0.3, ttc_sec=0.6, object_id=101)
        )
        separate_zone = smoother.smooth_event(
            make_event(state="SAFE", near_score=0.85, closing_speed=0.0, ttc_sec=None, object_id=103)
        )

        assert separate_zone.closing_speed == pytest.approx(0.0)
        assert separate_zone.ttc_sec is None
        assert separate_zone.state == "SAFE"


# ── score_event ───────────────────────────────────────────────────────────────

class TestScoreEvent:
    def test_danger_scores_higher_than_safe(self):
        danger = make_event(state="DANGER", near_score=0.5, closing_speed=0.5, ttc_sec=0.5)
        safe = make_event(state="SAFE", near_score=0.1, closing_speed=0.05, ttc_sec=10.0)
        assert score_event(danger) > score_event(safe)

    def test_none_ttc_gives_zero_ttc_weight(self):
        e = make_event(state="CAUTION", near_score=0.3, closing_speed=0.2, ttc_sec=None)
        score = score_event(e)
        # state_weight=1.0, ttc_weight=0.0, near=0.3, closing=0.2
        assert score == pytest.approx(1.5)

    def test_low_ttc_increases_score(self):
        e_low = make_event(state="CAUTION", near_score=0.3, closing_speed=0.2, ttc_sec=0.5)
        e_high = make_event(state="CAUTION", near_score=0.3, closing_speed=0.2, ttc_sec=2.9)
        assert score_event(e_low) > score_event(e_high)


# ── select_primary_event ──────────────────────────────────────────────────────

class TestSelectPrimaryEvent:
    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError):
            select_primary_event([])

    def test_returns_highest_scored(self):
        danger = make_event(state="DANGER", near_score=0.5, closing_speed=0.5, ttc_sec=0.5)
        safe = make_event(state="SAFE", near_score=0.1, closing_speed=0.05, ttc_sec=10.0)
        assert select_primary_event([safe, danger]) is danger

    def test_single_event_returns_itself(self):
        e = make_event()
        assert select_primary_event([e]) is e


# ── StateStabilizer ───────────────────────────────────────────────────────────

class TestStateStabilizer:
    def test_starts_safe(self):
        s = StateStabilizer()
        assert s.current_state == "SAFE"

    def test_upgrade_requires_n_consecutive_frames(self):
        s = StateStabilizer(upgrade_frames=3, downgrade_frames=5)
        assert s.process("DANGER") == "SAFE"   # frame 1: not yet
        assert s.process("DANGER") == "SAFE"   # frame 2: not yet
        assert s.process("DANGER") == "DANGER" # frame 3: transition

    def test_downgrade_requires_n_consecutive_frames(self):
        s = StateStabilizer(upgrade_frames=3, downgrade_frames=5)
        for _ in range(3):
            s.process("DANGER")
        for _ in range(4):
            assert s.process("SAFE") == "DANGER"
        assert s.process("SAFE") == "SAFE"

    def test_reset_counter_on_state_change_mid_pending(self):
        s = StateStabilizer(upgrade_frames=3, downgrade_frames=5)
        s.process("DANGER")  # counter=1, pending=DANGER
        s.process("CAUTION") # pending changed → counter reset to 1
        s.process("DANGER")  # pending changed again → counter reset to 1
        # none reached 3 consecutive frames
        assert s.current_state == "SAFE"

    def test_same_state_resets_counter(self):
        s = StateStabilizer(upgrade_frames=3, downgrade_frames=5)
        s.process("DANGER")  # 1
        s.process("DANGER")  # 2
        s.process("SAFE")    # counter reset
        s.process("DANGER")  # 1
        s.process("DANGER")  # 2
        assert s.current_state == "SAFE"  # never reached 3 consecutive
