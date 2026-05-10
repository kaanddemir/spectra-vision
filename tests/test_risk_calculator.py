"""Unit tests for the object-centric fused TTC risk calculator."""

import numpy as np
import pytest

from spectra.analysis.risk import (
    ExpansionSmoother,
    RiskEvent,
    StateStabilizer,
    TtcComponent,
    calculate_track_risk,
    classify_state,
    direction_from_lateral,
    expansion_rate_from_track,
    fuse_ttc,
    is_imminent_danger,
    lane_crossing_risk,
    lane_lateral_velocity,
    score_event,
    stabilized_event_state,
    ttc_from_expansion,
    ttc_from_flow,
)
from spectra.analysis.tracking import IoUTracker, Track, TrackSample
from spectra.vision.detection import Detection
from spectra.vision.road import (
    LaneFrame,
    filter_relevant_detections,
    lane_corridor_relevance,
    lane_position,
)


def make_lane(detected=True):
    return LaneFrame(
        vanishing_point=(150.0, 80.0),
        left_line=(120, 100, 60, 199),
        right_line=(180, 100, 240, 199),
        left_x_at_bottom=60.0,
        right_x_at_bottom=240.0,
        lane_width_at_bottom=180.0,
        lane_center_x_at_bottom=150.0,
        confidence=0.85 if detected else 0.25,
        detected=detected,
        width=300,
        height=200,
    )


def make_event(state="SAFE", ttc_sec=None, near_score=0.1, closing_speed=0.05, crossing=0.2):
    return RiskEvent(
        frame_index=0,
        timestamp_sec=0.0,
        state=state,
        ttc_sec=ttc_sec,
        direction="center",
        lane="center",
        object_type="car",
        confidence=0.5,
        near_score=near_score,
        velocity_magnitude=0.1,
        closing_speed=closing_speed,
        bbox=(10, 10, 50, 50),
        reason="test",
        object_id=1,
        crossing_risk=crossing,
    )


def make_track(track_id, bbox, t, history=None):
    track = Track(
        track_id=track_id,
        class_name="car",
        confidence=0.9,
        bbox=bbox,
        frame_index=int(t * 30),
        timestamp_sec=float(t),
    )
    for sample_t, sample_bbox in history or []:
        track.history.append(
            TrackSample(
                frame_index=int(sample_t * 30),
                timestamp_sec=float(sample_t),
                bbox=sample_bbox,
            )
        )
    return track


class TestLaneGeometry:
    def test_lane_position_uses_bbox_bottom_y(self):
        lane = make_lane()
        lower = lane_position((200, 160, 220, 199), lane)
        upper = lane_position((200, 60, 220, 100), lane)

        assert lower == pytest.approx(0.667, abs=0.02)
        assert upper == pytest.approx(2.0, abs=0.02)

    def test_detection_filter_keeps_near_ego_lane_vehicle(self):
        lane = make_lane()
        detections = [
            Detection(bbox=(130, 145, 170, 199), class_name="car", confidence=0.9),
        ]

        assert filter_relevant_detections(detections, lane) == detections

    def test_detection_filter_keeps_close_partial_cut_in_vehicle(self):
        lane = make_lane()
        detections = [
            Detection(bbox=(0, 130, 70, 199), class_name="car", confidence=0.9),
        ]

        assert filter_relevant_detections(detections, lane) == detections
        assert lane_corridor_relevance(detections[0].bbox, lane) >= 0.7

    def test_detection_filter_keeps_distant_ego_corridor_vehicle(self):
        lane = make_lane()
        detections = [
            Detection(bbox=(145, 88, 155, 110), class_name="car", confidence=0.8),
        ]

        assert filter_relevant_detections(detections, lane) == detections

    def test_detection_filter_keeps_distant_watch_band_vehicle(self):
        lane = make_lane()
        detections = [
            Detection(bbox=(95, 88, 115, 110), class_name="car", confidence=0.8),
        ]

        assert filter_relevant_detections(detections, lane) == detections

    def test_detection_filter_rejects_distant_outer_lane_vehicle(self):
        lane = make_lane()
        detections = [
            Detection(bbox=(20, 88, 42, 110), class_name="car", confidence=0.9),
        ]

        assert filter_relevant_detections(detections, lane) == []

    def test_detection_filter_keeps_static_side_lane_vehicle_out_of_tracker(self):
        lane = make_lane()
        tracker = IoUTracker()
        detections = [
            Detection(bbox=(260, 130, 290, 190), class_name="car", confidence=0.9),
        ]

        tracks = tracker.update(
            filter_relevant_detections(detections, lane),
            frame_index=0,
            timestamp_sec=0.0,
        )

        assert tracks == []


class TestDirectionFromLateral:
    def test_center_when_small(self):
        assert direction_from_lateral(0.05) == "center"

    def test_left_for_negative(self):
        assert direction_from_lateral(-0.2) == "left"

    def test_right_for_positive(self):
        assert direction_from_lateral(0.2) == "right"


class TestTtcComponents:
    def test_expansion_ttc_value_when_growing(self):
        component = ttc_from_expansion(0.5, history_age=4)
        assert component.value == pytest.approx(2.0)
        assert component.confidence == pytest.approx(1.0)

    def test_expansion_ttc_none_when_stable(self):
        assert ttc_from_expansion(0.01, history_age=4).value is None

    def test_flow_ttc_uses_measured_frame_dt(self):
        h, w = 200, 300
        vp = (150.0, 80.0)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        radial_x = xs - vp[0]
        radial_y = ys - vp[1]
        dist = np.maximum(np.sqrt(radial_x * radial_x + radial_y * radial_y), 1.0)
        flow = np.stack((radial_x / dist, radial_y / dist), axis=-1).astype(np.float32) * 4.0

        component = ttc_from_flow((130, 120, 170, 160), flow, vp, flow_dt_sec=0.1)

        assert component.value == pytest.approx(1.5, abs=0.05)
        assert component.confidence > 0.5

    def test_weighted_median_ignores_low_confidence_outlier(self):
        fused, _ = fuse_ttc(
            [
                TtcComponent("expansion", 0.5, 0.2),
                TtcComponent("flow", 8.0, 0.1),
                TtcComponent("depth", 1.2, 0.8),
            ]
        )

        assert fused == pytest.approx(1.2)


class TestExpansionRateFromTrack:
    def test_zero_when_no_history(self):
        track = make_track(1, (10, 10, 50, 50), t=1.0)
        assert expansion_rate_from_track(track) == 0.0

    def test_positive_when_growing(self):
        track = make_track(
            1,
            (0, 0, 100, 100),
            t=1.0,
            history=[(0.5, (10, 10, 90, 90))],
        )
        assert expansion_rate_from_track(track) > 0.1

    def test_negative_when_shrinking(self):
        track = make_track(
            1,
            (10, 10, 90, 90),
            t=1.0,
            history=[(0.5, (0, 0, 100, 100))],
        )
        assert expansion_rate_from_track(track) < 0.0


class TestCrossing:
    def test_lateral_velocity_is_lane_relative(self):
        lane = make_lane()
        track = make_track(
            1,
            (140, 120, 180, 190),
            t=1.0,
            history=[(0.5, (100, 120, 140, 190))],
        )

        assert lane_lateral_velocity(track, lane) > 0.0

    def test_side_lane_motion_toward_center_increases_crossing_risk(self):
        lane = make_lane()
        static = make_track(1, (225, 130, 245, 190), t=1.0)
        moving_in = make_track(
            1,
            (225, 130, 245, 190),
            t=1.0,
            history=[(0.5, (260, 130, 280, 190))],
        )

        assert lane_crossing_risk(moving_in, lane, 2.0) > lane_crossing_risk(static, lane, 2.0)


class TestClassifyState:
    def test_safe_when_low_confidence(self):
        state = classify_state(
            fused_ttc=0.5,
            crossing=0.9,
            near_score=0.5,
            expansion_rate=0.5,
            lane_pos=0.0,
            confidence=0.1,
        )
        assert state == "SAFE"

    def test_danger_low_ttc_in_corridor(self):
        state = classify_state(
            fused_ttc=0.5,
            crossing=0.7,
            near_score=0.5,
            expansion_rate=0.5,
            lane_pos=0.0,
            confidence=0.8,
        )
        assert state == "DANGER"

    def test_caution_mid_ttc(self):
        state = classify_state(
            fused_ttc=2.0,
            crossing=0.4,
            near_score=0.3,
            expansion_rate=0.2,
            lane_pos=0.2,
            confidence=0.8,
        )
        assert state == "CAUTION"

    def test_safe_when_outside_corridor(self):
        state = classify_state(
            fused_ttc=0.5,
            crossing=0.05,
            near_score=0.5,
            expansion_rate=0.5,
            lane_pos=2.0,
            confidence=0.8,
        )
        assert state != "DANGER"


class TestCalculateTrackRisk:
    def test_strong_expansion_in_corridor_danger(self):
        height, width = 200, 300
        near_map = np.full((height, width), 0.6, dtype=np.float32)
        magnitude = np.full((height, width), 0.4, dtype=np.float32)
        flow = np.zeros((height, width, 2), dtype=np.float32)
        track = make_track(
            1,
            (130, 120, 180, 190),
            t=1.0,
            history=[(0.7, (140, 130, 170, 170))],
        )

        event = calculate_track_risk(
            track=track,
            near_map=near_map,
            flow=flow,
            magnitude_norm=magnitude,
            lane=make_lane(),
            expansion_rate=expansion_rate_from_track(track),
            depth_history={},
            flow_dt_sec=1.0 / 30.0,
            depth_is_fresh=True,
            frame_index=track.frame_index,
            timestamp_sec=track.timestamp_sec,
        )

        assert event.state == "DANGER"
        assert event.ttc_sec is not None
        assert event.bbox == track.bbox

    def test_no_expansion_safe(self):
        height, width = 200, 300
        near_map = np.full((height, width), 0.3, dtype=np.float32)
        magnitude = np.zeros((height, width), dtype=np.float32)
        flow = np.zeros((height, width, 2), dtype=np.float32)
        track = make_track(
            1,
            (130, 100, 170, 140),
            t=1.0,
            history=[(0.5, (130, 100, 170, 140))],
        )

        event = calculate_track_risk(
            track=track,
            near_map=near_map,
            flow=flow,
            magnitude_norm=magnitude,
            lane=make_lane(),
            expansion_rate=0.0,
            depth_history={},
            flow_dt_sec=1.0 / 30.0,
            depth_is_fresh=True,
            frame_index=track.frame_index,
            timestamp_sec=track.timestamp_sec,
        )

        assert event.state == "SAFE"
        assert event.ttc_sec is None

    def test_stale_depth_does_not_update_depth_history(self):
        height, width = 200, 300
        near_map = np.full((height, width), 0.4, dtype=np.float32)
        magnitude = np.zeros((height, width), dtype=np.float32)
        flow = np.zeros((height, width, 2), dtype=np.float32)
        track = make_track(1, (130, 120, 180, 190), t=1.0)
        history = {1: (0.0, 0.2)}

        event = calculate_track_risk(
            track=track,
            near_map=near_map,
            flow=flow,
            magnitude_norm=magnitude,
            lane=make_lane(),
            expansion_rate=0.0,
            depth_history=history,
            flow_dt_sec=1.0 / 30.0,
            depth_is_fresh=False,
            frame_index=track.frame_index,
            timestamp_sec=track.timestamp_sec,
        )

        assert history == {1: (0.0, 0.2)}
        assert next(c for c in event.ttc_components if c.name == "depth").value is None

    def test_side_lane_static_object_safe(self):
        height, width = 200, 300
        near_map = np.full((height, width), 0.2, dtype=np.float32)
        magnitude = np.zeros((height, width), dtype=np.float32)
        flow = np.zeros((height, width, 2), dtype=np.float32)
        track = make_track(1, (270, 130, 298, 190), t=1.0)

        event = calculate_track_risk(
            track=track,
            near_map=near_map,
            flow=flow,
            magnitude_norm=magnitude,
            lane=make_lane(),
            expansion_rate=0.0,
            depth_history={},
            flow_dt_sec=1.0 / 30.0,
            depth_is_fresh=True,
            frame_index=track.frame_index,
            timestamp_sec=track.timestamp_sec,
        )

        assert event.state == "SAFE"


class TestExpansionSmoother:
    def test_first_value_is_passthrough(self):
        smoother = ExpansionSmoother()
        assert smoother.update(1, 0.5) == pytest.approx(0.5)

    def test_smoothing_bounds_jitter(self):
        smoother = ExpansionSmoother()
        smoother.update(1, 0.0)
        assert smoother.update(1, 1.0) < 1.0

    def test_forget_drops_inactive_tracks(self):
        smoother = ExpansionSmoother()
        smoother.update(1, 0.5)
        smoother.update(2, 0.5)
        smoother.forget({1})
        assert 2 not in smoother._state


class TestStabilizer:
    def test_imminent_danger_bypasses_upgrade_delay(self):
        stabilizer = StateStabilizer(upgrade_frames=3, downgrade_frames=5)
        event = make_event(state="DANGER", ttc_sec=0.5)
        assert is_imminent_danger(event)
        assert stabilized_event_state(stabilizer, event) == "DANGER"

    def test_danger_held_through_single_caution_frame(self):
        stabilizer = StateStabilizer(upgrade_frames=3, downgrade_frames=5)
        stabilizer.current_state = "DANGER"
        stabilizer.pending_state = "DANGER"
        event = make_event(state="CAUTION", ttc_sec=2.0)
        assert stabilized_event_state(stabilizer, event) == "DANGER"

    def test_upgrade_requires_n_frames(self):
        stabilizer = StateStabilizer(upgrade_frames=3, downgrade_frames=5)
        for _ in range(2):
            assert stabilizer.process("CAUTION") == "SAFE"
        assert stabilizer.process("CAUTION") == "CAUTION"


class TestScoring:
    def test_danger_outranks_safe(self):
        danger = make_event(state="DANGER", ttc_sec=0.5, near_score=0.5, closing_speed=0.5)
        safe = make_event(state="SAFE")
        assert score_event(danger) > score_event(safe)

class TestIoUTracker:
    def test_assigns_new_ids_to_unmatched_detections(self):
        tracker = IoUTracker()
        dets = [
            Detection(bbox=(0, 0, 50, 50), class_name="car", confidence=0.9),
            Detection(bbox=(100, 100, 150, 150), class_name="car", confidence=0.9),
        ]
        tracks = tracker.update(dets, frame_index=0, timestamp_sec=0.0)
        assert len({t.track_id for t in tracks}) == 2

    def test_links_overlapping_detections_across_frames(self):
        tracker = IoUTracker(iou_threshold=0.2)
        first = [Detection(bbox=(0, 0, 50, 50), class_name="car", confidence=0.9)]
        tracks_t0 = tracker.update(first, frame_index=0, timestamp_sec=0.0)
        second = [Detection(bbox=(5, 5, 55, 55), class_name="car", confidence=0.9)]
        tracks_t1 = tracker.update(second, frame_index=1, timestamp_sec=0.1)

        assert tracks_t0[0].track_id == tracks_t1[0].track_id
        assert len(tracks_t1[0].history) == 1

    def test_does_not_link_across_classes(self):
        tracker = IoUTracker(iou_threshold=0.2)
        tracker.update(
            [Detection(bbox=(0, 0, 50, 50), class_name="car", confidence=0.9)],
            frame_index=0,
            timestamp_sec=0.0,
        )
        tracks = tracker.update(
            [Detection(bbox=(0, 0, 50, 50), class_name="person", confidence=0.9)],
            frame_index=1,
            timestamp_sec=0.1,
        )

        assert tracks[0].class_name == "person"
        assert len(tracks[0].history) == 0
