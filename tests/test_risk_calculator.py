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
    distance_m_for_bbox,
    direction_from_lateral,
    expansion_rate_from_track,
    fuse_ttc,
    is_imminent_danger,
    lane_crossing_risk,
    lane_lateral_velocity,
    near_score_from_distance,
    score_event,
    stabilized_event_state,
    ttc_from_depth_delta,
    ttc_from_expansion,
    ttc_from_flow,
)
from spectra.analysis.overlay import annotate_frame
from spectra.analysis.tracking import IoUTracker, Track, TrackSample
from spectra.vision.detection import Detection
from spectra.vision.road import (
    LaneFrame,
    estimate_road_roi_from_lanes,
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


def make_depth_m(height=200, width=300, distance=30.0):
    return np.full((height, width), distance, dtype=np.float32)


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
    def test_lane_roi_confidence_high_for_plausible_inner_lanes(self):
        lanes = [
            np.empty((0, 2), dtype=np.float32),
            np.array([[120, 116], [100, 150], [60, 199]], dtype=np.float32),
            np.array([[180, 116], [205, 150], [240, 199]], dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        ]

        roi = estimate_road_roi_from_lanes(lanes, width=300, height=200)

        assert roi.detected
        assert roi.confidence > 0.75

    def test_lane_roi_rejects_implausibly_wide_lane(self):
        lanes = [
            np.empty((0, 2), dtype=np.float32),
            np.array([[5, 116], [2, 150], [0, 199]], dtype=np.float32),
            np.array([[295, 116], [298, 150], [299, 199]], dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        ]

        roi = estimate_road_roi_from_lanes(lanes, width=300, height=200)

        assert not roi.detected
        assert roi.confidence == pytest.approx(0.25)

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

        # The filter admits the partially-visible side vehicle so the tracker
        # can build history before it fully intrudes.
        assert filter_relevant_detections(detections, lane) == detections

        # Its *static* corridor relevance is intentionally moderate, not a flat
        # high floor: a barely-clipping side-lane object must not read as
        # in-lane on geometry alone (a moving cut-in is caught dynamically via
        # lateral velocity in lane_crossing_risk). So it stays a positive
        # candidate but well below a fully in-lane vehicle.
        edge = lane_corridor_relevance(detections[0].bbox, lane)
        centered = lane_corridor_relevance((130, 145, 170, 199), lane)
        assert 0.0 < edge < centered

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

    def test_detection_filter_keeps_distant_outer_watch_vehicle(self):
        lane = make_lane()
        detections = [
            Detection(bbox=(55, 88, 75, 110), class_name="car", confidence=0.8),
        ]

        assert filter_relevant_detections(detections, lane) == detections

    def test_detection_filter_expands_watch_band_when_lane_confidence_is_low(self):
        lane = make_lane()
        low_conf_lane = LaneFrame(
            **{**lane.__dict__, "confidence": 0.25, "detected": False}
        )
        detections = [
            Detection(bbox=(55, 88, 75, 110), class_name="car", confidence=0.8),
        ]

        assert filter_relevant_detections(detections, low_conf_lane) == detections

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

    def test_overlay_handles_detected_and_low_confidence_lanes(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        event = make_event(state="SAFE")

        detected = annotate_frame(frame, event, [event], lane=make_lane())
        low_conf = annotate_frame(frame, event, [event], lane=make_lane(detected=False))

        assert detected.shape == frame.shape
        assert low_conf.shape == frame.shape
        assert detected.dtype == np.uint8
        assert low_conf.dtype == np.uint8

    def test_overlay_hides_safe_object_boxes_but_draws_actionable_boxes(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        lane = make_lane()

        safe = make_event(state="SAFE")
        safe_without_objects = annotate_frame(frame, safe, [], lane=lane)
        safe_with_object = annotate_frame(frame, safe, [safe], lane=lane)
        assert np.array_equal(safe_with_object, safe_without_objects)

        danger = make_event(state="DANGER", ttc_sec=0.8)
        danger_without_objects = annotate_frame(frame, danger, [], lane=lane)
        danger_with_object = annotate_frame(frame, danger, [danger], lane=lane)
        assert not np.array_equal(danger_with_object, danger_without_objects)


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

    def test_far_object_crossing_damped_vs_near(self):
        # Same rightward-to-center motion; one high in frame (far, near the
        # horizon), one low (near). The collision-cone distance reliability
        # damps the far object's predicted crossing.
        lane = make_lane()
        far = make_track(1, (225, 100, 245, 130), t=1.0, history=[(0.7, (255, 100, 275, 130))])
        near = make_track(2, (225, 160, 245, 190), t=1.0, history=[(0.7, (255, 160, 275, 190))])
        assert lane_crossing_risk(far, lane, 2.0) < lane_crossing_risk(near, lane, 2.0)


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


class TestMetricDepth:
    def test_distance_m_for_bbox_uses_lower_center_percentile(self):
        depth_m = np.full((100, 100), 40.0, dtype=np.float32)
        depth_m[50:90, 35:65] = 12.0
        depth_m[0:40, :] = 3.0
        assert distance_m_for_bbox(depth_m, (20, 20, 80, 90)) == pytest.approx(12.0)

    def test_near_score_from_metric_distance_thresholds(self):
        assert near_score_from_distance(8.0) == pytest.approx(1.0)
        assert near_score_from_distance(60.0) == pytest.approx(0.0)
        assert near_score_from_distance(34.0) == pytest.approx(0.5)
        assert near_score_from_distance(None) == pytest.approx(0.0)

    def test_metric_depth_ttc_uses_closing_distance_rate(self):
        history = {1: (0.0, 30.0)}
        depth_m = make_depth_m(distance=20.0)
        component, distance_m, closing_mps = ttc_from_depth_delta(
            1,
            (20, 20, 80, 90),
            depth_m,
            2.0,
            history,
            update_history=True,
        )

        assert distance_m == pytest.approx(20.0)
        assert closing_mps == pytest.approx(5.0)
        assert component.value == pytest.approx(4.0)
        assert component.confidence > 0.0

    def test_metric_depth_ttc_ignores_static_or_receding_distance(self):
        history = {1: (0.0, 20.0)}
        component, distance_m, closing_mps = ttc_from_depth_delta(
            1,
            (20, 20, 80, 90),
            make_depth_m(distance=22.0),
            1.0,
            history,
            update_history=True,
        )

        assert distance_m == pytest.approx(22.0)
        assert closing_mps == pytest.approx(-2.0)
        assert component.value is None


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
            depth_m=make_depth_m(distance=6.0),
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
            depth_m=make_depth_m(distance=40.0),
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
            depth_m=make_depth_m(distance=25.0),
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

    def test_metric_closing_mps_drives_approach_score(self):
        height, width = 200, 300
        near_map = np.full((height, width), 0.4, dtype=np.float32)
        magnitude = np.zeros((height, width), dtype=np.float32)
        flow = np.zeros((height, width, 2), dtype=np.float32)
        track = make_track(1, (130, 120, 180, 190), t=1.0)

        event = calculate_track_risk(
            track=track,
            depth_m=make_depth_m(distance=20.0),
            near_map=near_map,
            flow=flow,
            magnitude_norm=magnitude,
            lane=make_lane(),
            expansion_rate=0.0,
            depth_history={1: (0.0, 30.0)},
            flow_dt_sec=1.0 / 30.0,
            depth_is_fresh=True,
            frame_index=track.frame_index,
            timestamp_sec=track.timestamp_sec,
        )

        assert event.closing_mps == pytest.approx(10.0)
        assert event.closing_speed == pytest.approx(0.415, abs=0.001)

    def test_approach_falls_back_to_limited_visual_cues_without_metric_closing(self):
        height, width = 200, 300
        near_map = np.full((height, width), 0.4, dtype=np.float32)
        magnitude = np.ones((height, width), dtype=np.float32)
        flow = np.zeros((height, width, 2), dtype=np.float32)
        track = make_track(1, (130, 120, 180, 190), t=1.0)

        event = calculate_track_risk(
            track=track,
            depth_m=make_depth_m(distance=20.0),
            near_map=near_map,
            flow=flow,
            magnitude_norm=magnitude,
            lane=make_lane(),
            expansion_rate=0.6,
            depth_history={},
            flow_dt_sec=1.0 / 30.0,
            depth_is_fresh=True,
            frame_index=track.frame_index,
            timestamp_sec=track.timestamp_sec,
        )

        assert event.closing_mps is None
        assert event.closing_speed == pytest.approx(0.5)

    def test_side_lane_static_object_safe(self):
        height, width = 200, 300
        near_map = np.full((height, width), 0.2, dtype=np.float32)
        magnitude = np.zeros((height, width), dtype=np.float32)
        flow = np.zeros((height, width, 2), dtype=np.float32)
        track = make_track(1, (270, 130, 298, 190), t=1.0)

        event = calculate_track_risk(
            track=track,
            depth_m=make_depth_m(distance=50.0),
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

    def test_risk_event_keeps_track_id_and_exposes_display_id(self):
        height, width = 200, 300
        near_map = np.full((height, width), 0.3, dtype=np.float32)
        magnitude = np.zeros((height, width), dtype=np.float32)
        flow = np.zeros((height, width, 2), dtype=np.float32)
        track = make_track(17, (130, 100, 170, 140), t=1.0)
        track.display_id = 2

        event = calculate_track_risk(
            track=track,
            depth_m=make_depth_m(distance=35.0),
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

        assert event.object_id == 17
        assert event.display_id == 2


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
    def test_pending_tracks_are_hidden_on_first_detection(self):
        tracker = IoUTracker()
        dets = [
            Detection(bbox=(0, 0, 20, 20), class_name="car", confidence=0.6),
        ]
        tracks = tracker.update(dets, frame_index=0, timestamp_sec=0.0)

        assert tracks == []

    def test_links_and_confirms_overlapping_detections_across_frames(self):
        tracker = IoUTracker(iou_threshold=0.2)
        first = [Detection(bbox=(0, 0, 50, 50), class_name="car", confidence=0.6)]
        tracks_t0 = tracker.update(first, frame_index=0, timestamp_sec=0.0)
        second = [Detection(bbox=(5, 5, 55, 55), class_name="car", confidence=0.6)]
        tracks_t1 = tracker.update(second, frame_index=1, timestamp_sec=0.1)
        third = [Detection(bbox=(10, 10, 60, 60), class_name="car", confidence=0.6)]
        tracks_t2 = tracker.update(third, frame_index=2, timestamp_sec=0.2)

        assert tracks_t0 == []
        assert tracks_t1 == []
        assert tracks_t2[0].track_id == 1
        assert tracks_t2[0].confirmed
        assert tracks_t2[0].display_id == 1
        assert len(tracks_t2[0].history) == 2

    def test_fast_confirms_large_high_confidence_detection(self):
        tracker = IoUTracker()
        tracks = tracker.update(
            [Detection(bbox=(0, 0, 50, 50), class_name="car", confidence=0.9)],
            frame_index=0,
            timestamp_sec=0.0,
            frame_shape=(200, 300, 3),
        )

        assert len(tracks) == 1
        assert tracks[0].confirmed
        assert tracks[0].display_id == 1

    def test_links_low_iou_detection_by_center_and_scale(self):
        tracker = IoUTracker(iou_threshold=0.5)
        first = [Detection(bbox=(0, 0, 100, 100), class_name="car", confidence=0.9)]
        tracks_t0 = tracker.update(first, frame_index=0, timestamp_sec=0.0, frame_shape=(200, 300, 3))
        second = [Detection(bbox=(35, 10, 135, 110), class_name="car", confidence=0.8)]
        tracks_t1 = tracker.update(second, frame_index=1, timestamp_sec=0.1, frame_shape=(200, 300, 3))

        assert tracks_t0[0].track_id == 1
        assert tracks_t1[0].track_id == 1
        assert tracks_t1[0].display_id == 1
        assert len(tracks_t1[0].history) == 1

    def test_reconnects_after_short_detection_miss(self):
        tracker = IoUTracker(iou_threshold=0.5, max_misses=5)
        tracker.update(
            [Detection(bbox=(0, 0, 100, 100), class_name="car", confidence=0.9)],
            frame_index=0,
            timestamp_sec=0.0,
            frame_shape=(200, 300, 3),
        )
        tracker.update(
            [Detection(bbox=(10, 0, 110, 100), class_name="car", confidence=0.9)],
            frame_index=1,
            timestamp_sec=0.1,
            frame_shape=(200, 300, 3),
        )
        # A confirmed track coasts through a short detection miss: it keeps
        # emitting (with its last bbox) so a live threat does not vanish from
        # the active set on a single missed detection frame.
        coasting = tracker.update([], frame_index=2, timestamp_sec=0.2, frame_shape=(200, 300, 3))
        assert len(coasting) == 1
        assert coasting[0].track_id == 1
        assert coasting[0].misses == 1

        tracks = tracker.update(
            [Detection(bbox=(75, 0, 175, 100), class_name="car", confidence=0.9)],
            frame_index=3,
            timestamp_sec=0.3,
            frame_shape=(200, 300, 3),
        )

        assert len(tracks) == 1
        assert tracks[0].track_id == 1
        assert tracks[0].display_id == 1
        assert tracks[0].misses == 0

    def test_center_scale_gate_does_not_merge_adjacent_tracks(self):
        tracker = IoUTracker(iou_threshold=0.5)
        tracks_t0 = tracker.update(
            [
                Detection(bbox=(0, 0, 100, 100), class_name="car", confidence=0.9),
                Detection(bbox=(160, 0, 260, 100), class_name="car", confidence=0.9),
            ],
            frame_index=0,
            timestamp_sec=0.0,
            frame_shape=(200, 320, 3),
        )

        tracks_t1 = tracker.update(
            [
                Detection(bbox=(35, 0, 135, 100), class_name="car", confidence=0.9),
                Detection(bbox=(195, 0, 295, 100), class_name="car", confidence=0.9),
            ],
            frame_index=1,
            timestamp_sec=0.1,
            frame_shape=(200, 320, 3),
        )

        assert sorted(track.track_id for track in tracks_t0) == [1, 2]
        assert sorted(track.track_id for track in tracks_t1) == [1, 2]
        assert sorted(track.display_id for track in tracks_t1) == [1, 2]

    def test_display_ids_are_scoped_per_class(self):
        tracker = IoUTracker()
        tracks = tracker.update(
            [
                Detection(bbox=(0, 0, 80, 80), class_name="car", confidence=0.9),
                Detection(bbox=(120, 0, 200, 80), class_name="truck", confidence=0.9),
            ],
            frame_index=0,
            timestamp_sec=0.0,
            frame_shape=(200, 320, 3),
        )

        by_class = {track.class_name: track for track in tracks}
        assert by_class["car"].track_id == 1
        assert by_class["truck"].track_id == 2
        assert by_class["car"].display_id == 1
        assert by_class["truck"].display_id == 1

    def test_pending_false_positive_does_not_consume_display_id(self):
        tracker = IoUTracker()
        tracker.update(
            [Detection(bbox=(0, 0, 20, 20), class_name="car", confidence=0.6)],
            frame_index=0,
            timestamp_sec=0.0,
        )

        tracks = tracker.update(
            [Detection(bbox=(40, 40, 100, 100), class_name="car", confidence=0.9)],
            frame_index=1,
            timestamp_sec=0.1,
            frame_shape=(200, 300, 3),
        )

        assert len(tracks) == 1
        assert tracks[0].track_id == 2
        assert tracks[0].display_id == 1

    def test_pending_false_positive_does_not_propagate(self):
        tracker = IoUTracker()
        tracker.update(
            [Detection(bbox=(0, 0, 20, 20), class_name="car", confidence=0.6)],
            frame_index=0,
            timestamp_sec=0.0,
        )

        assert tracker.propagate() == []

    def test_does_not_link_across_classes(self):
        tracker = IoUTracker(iou_threshold=0.2)
        tracker.update(
            [Detection(bbox=(0, 0, 50, 50), class_name="car", confidence=0.6)],
            frame_index=0,
            timestamp_sec=0.0,
        )
        tracks = tracker.update(
            [Detection(bbox=(0, 0, 50, 50), class_name="person", confidence=0.6)],
            frame_index=1,
            timestamp_sec=0.1,
        )

        assert tracks == []
