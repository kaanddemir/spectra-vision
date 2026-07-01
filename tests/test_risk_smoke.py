import numpy as np
import pytest

from spectra.analysis.risk import (
    DepthDeltaSmoother,
    ExpansionSmoother,
    SpatialFields,
    StateStabilizer,
    build_object_events,
    calculate_track_risk,
    score_raw,
    state_from_score,
    ttc_from_depth_delta,
)
from spectra.analysis.tracking import Track, TrackSample
from spectra.vision.depth import DepthResult
from spectra.vision.motion import FlowResult
from spectra.vision.road import LaneFrame


def _lane():
    return LaneFrame(
        vanishing_point=(150.0, 80.0),
        left_line=(120, 100, 60, 199),
        right_line=(180, 100, 240, 199),
        left_x_at_bottom=60.0,
        right_x_at_bottom=240.0,
        lane_width_at_bottom=180.0,
        lane_center_x_at_bottom=150.0,
        confidence=0.85,
        detected=True,
        width=300,
        height=200,
    )


def _depth(distance=30.0, near=0.4, height=200, width=300):
    near_map = np.full((height, width), near, dtype=np.float32)
    depth_m = np.full((height, width), distance, dtype=np.float32)
    return DepthResult(depth_m=depth_m, near_map=near_map, depth_map=(near_map * 255).astype(np.uint8))


def _flow(height=200, width=300, magnitude=0.0):
    return FlowResult(
        flow=np.zeros((height, width, 2), dtype=np.float32),
        magnitude_norm=np.full((height, width), magnitude, dtype=np.float32),
        divergence_norm=np.zeros((height, width), dtype=np.float32),
    )


def _track(track_id, bbox, t=1.0, prior_bbox=None, prior_t=0.7):
    track = Track(
        track_id=track_id,
        class_name="car",
        confidence=0.9,
        bbox=bbox,
        frame_index=int(t * 30),
        timestamp_sec=float(t),
    )
    if prior_bbox is not None:
        track.history.append(
            TrackSample(frame_index=int(prior_t * 30), timestamp_sec=float(prior_t), bbox=prior_bbox)
        )
    return track


def test_stabilizer_fast_clear_downgrades_in_fewer_frames():
    # Normal downgrade needs the full anti-flicker window; a receding/cleared
    # scene (fast_clear) lets an elevated banner fall in fewer frames so a
    # passed threat's DANGER does not linger.
    slow = StateStabilizer(downgrade_frames=7, fast_downgrade_frames=3)
    slow.current_state = slow.pending_state = "DANGER"
    for _ in range(3):
        assert slow.process("SAFE") == "DANGER"  # 3 SAFE frames, still DANGER

    fast = StateStabilizer(downgrade_frames=7, fast_downgrade_frames=3)
    fast.current_state = fast.pending_state = "DANGER"
    assert fast.process("SAFE", fast_clear=True) == "DANGER"
    assert fast.process("SAFE", fast_clear=True) == "DANGER"
    assert fast.process("SAFE", fast_clear=True) == "SAFE"  # cleared after 3


def test_imminent_escalation_requires_proximity_or_cue_agreement():
    # A far, single-cue (depth-only, cues disagree) TTC<1s reading — the symptom
    # of a young depth-Kalman track that accepted a multi-metre jump — must NOT
    # snap to DANGER; it stays at its score band (CAUTION here).
    assert (
        state_from_score(
            0.41, 0.6, ttc_imminent_streak=3, confidence=0.6, distance_m=21.6, ttc_agreement=0.0
        )
        == "CAUTION"
    )
    # A genuinely close object escalates on proximity alone.
    assert (
        state_from_score(
            0.41, 0.6, ttc_imminent_streak=3, confidence=0.6, distance_m=5.0, ttc_agreement=0.0
        )
        == "DANGER"
    )
    # A far approach still escalates when the three TTC cues corroborate it.
    assert (
        state_from_score(
            0.41, 0.6, ttc_imminent_streak=3, confidence=0.6, distance_m=21.6, ttc_agreement=0.8
        )
        == "DANGER"
    )


def test_no_tracks_returns_single_safe_event():
    primary, events = build_object_events(
        frame_index=0,
        timestamp_sec=0.0,
        tracks=[],
        fields=SpatialFields(_depth(), _flow(), _lane(), flow_dt_sec=1.0 / 30.0, depth_is_fresh=True),
        expansion_smoother=ExpansionSmoother(),
        depth_smoother=DepthDeltaSmoother(),
    )

    assert primary.state == "SAFE"
    assert primary.bbox is None
    assert events == [primary]


def test_lane_relevance_gate_keeps_off_path_score_low():
    in_path = score_raw(None, near_score=0.8, closing_speed=0.8, crossing_risk=0.9, confidence=0.9)
    off_path = score_raw(None, near_score=0.8, closing_speed=0.8, crossing_risk=0.02, confidence=0.9)

    assert in_path > off_path
    assert off_path < 0.05


def test_static_side_lane_object_stays_safe():
    event = calculate_track_risk(
        track=_track(1, (270, 130, 298, 190)),
        depth_m=_depth(distance=50.0, near=0.2).depth_m,
        near_map=_depth(distance=50.0, near=0.2).near_map,
        flow=_flow().flow,
        magnitude_norm=_flow().magnitude_norm,
        lane=_lane(),
        expansion_rate=0.0,
        depth_history={},
        flow_dt_sec=1.0 / 30.0,
        depth_is_fresh=True,
        frame_index=30,
        timestamp_sec=1.0,
    )

    assert event.state == "SAFE"
    assert abs(event.lane_position) > 1.0


def test_metric_depth_closing_produces_physical_ttc():
    history = {}
    component = distance = closing_mps = None
    for i in range(8):
        t = (i + 1) * 0.2
        measured_distance = 30.0 - (5.0 * t)
        component, distance, closing_mps = ttc_from_depth_delta(
            1,
            (20, 20, 80, 90),
            _depth(distance=measured_distance).depth_m,
            t,
            history,
            update_history=True,
        )

    assert component.value is not None
    assert closing_mps == pytest.approx(5.0, abs=0.8)
    assert component.value == pytest.approx(distance / closing_mps, rel=0.20)


def test_build_object_events_selects_in_path_risk_over_static_side_object():
    primary, events = build_object_events(
        frame_index=30,
        timestamp_sec=1.0,
        tracks=[
            _track(1, (130, 120, 180, 190), prior_bbox=(140, 130, 170, 170)),
            _track(2, (270, 130, 298, 190)),
        ],
        fields=SpatialFields(_depth(distance=6.0, near=0.7), _flow(magnitude=0.4), _lane(), 1.0 / 30.0, True),
        expansion_smoother=ExpansionSmoother(),
        depth_smoother=DepthDeltaSmoother(),
    )

    assert len(events) == 2
    assert primary.object_id == 1
    assert primary.state in {"CAUTION", "DANGER"}
