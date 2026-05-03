"""Unit tests for risk event assembly and quick motion scoring."""

import numpy as np

from spectra.analysis.risk import (
    DepthDeltaSmoother,
    ExpansionSmoother,
    SpatialFields,
    build_object_events,
    compute_quick_risk,
)
from spectra.analysis.tracking import Track, TrackSample
from spectra.vision.depth import DepthResult
from spectra.vision.motion import FlowResult
from spectra.vision.road import LaneFrame


def make_lane():
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


def make_flow(height=200, width=300, magnitude=0.0, divergence=0.0):
    return FlowResult(
        flow=np.zeros((height, width, 2), dtype=np.float32),
        magnitude_norm=np.full((height, width), magnitude, dtype=np.float32),
        divergence_norm=np.full((height, width), divergence, dtype=np.float32),
    )


def make_depth(height=200, width=300, near=0.4):
    near_map = np.full((height, width), near, dtype=np.float32)
    return DepthResult(
        near_map=near_map,
        depth_map=(near_map * 255).astype(np.uint8),
    )


def make_fields(depth=None, flow=None, *, depth_is_fresh=True):
    return SpatialFields(
        depth=depth or make_depth(),
        flow=flow or make_flow(),
        lane=make_lane(),
        flow_dt_sec=1.0 / 30.0,
        depth_is_fresh=depth_is_fresh,
    )


def make_track(track_id, bbox, t, prior_bbox=None, prior_t=None):
    track = Track(
        track_id=track_id,
        class_name="car",
        confidence=0.9,
        bbox=bbox,
        frame_index=int(t * 30),
        timestamp_sec=float(t),
    )
    if prior_bbox is not None and prior_t is not None:
        track.history.append(
            TrackSample(
                frame_index=int(prior_t * 30),
                timestamp_sec=float(prior_t),
                bbox=prior_bbox,
            )
        )
    return track


class TestComputeQuickRisk:
    def test_zero_motion_low_risk(self):
        assert compute_quick_risk(make_flow(magnitude=0.0, divergence=0.0), width=300, height=200) < 0.1

    def test_high_motion_high_risk(self):
        assert compute_quick_risk(make_flow(magnitude=1.0, divergence=1.0), width=300, height=200) > 0.5


class TestBuildObjectEvents:
    def test_no_tracks_returns_safe(self):
        primary, events = build_object_events(
            frame_index=0,
            timestamp_sec=0.0,
            tracks=[],
            fields=make_fields(),
            expansion_smoother=ExpansionSmoother(),
            depth_smoother=DepthDeltaSmoother(),
        )

        assert primary.state == "SAFE"
        assert primary.bbox is None
        assert events == [primary]

    def test_expanding_track_in_corridor_dangerous(self):
        track = make_track(
            track_id=1,
            bbox=(130, 130, 180, 190),
            t=1.0,
            prior_bbox=(140, 140, 165, 170),
            prior_t=0.7,
        )
        primary, events = build_object_events(
            frame_index=30,
            timestamp_sec=1.0,
            tracks=[track],
            fields=make_fields(depth=make_depth(near=0.6), flow=make_flow(magnitude=0.4)),
            expansion_smoother=ExpansionSmoother(),
            depth_smoother=DepthDeltaSmoother(),
        )

        assert len(events) == 1
        assert primary.object_id == 1
        assert primary.state in {"CAUTION", "DANGER"}

    def test_static_track_safe(self):
        track = make_track(
            track_id=1,
            bbox=(130, 130, 180, 190),
            t=1.0,
            prior_bbox=(130, 130, 180, 190),
            prior_t=0.7,
        )
        primary, _ = build_object_events(
            frame_index=30,
            timestamp_sec=1.0,
            tracks=[track],
            fields=make_fields(depth=make_depth(near=0.2)),
            expansion_smoother=ExpansionSmoother(),
            depth_smoother=DepthDeltaSmoother(),
        )

        assert primary.state == "SAFE"

    def test_picks_worst_object_as_primary(self):
        expanding = make_track(
            track_id=1,
            bbox=(130, 130, 180, 190),
            t=1.0,
            prior_bbox=(140, 140, 165, 170),
            prior_t=0.7,
        )
        static = make_track(
            track_id=2,
            bbox=(20, 20, 40, 40),
            t=1.0,
            prior_bbox=(20, 20, 40, 40),
            prior_t=0.7,
        )

        primary, events = build_object_events(
            frame_index=30,
            timestamp_sec=1.0,
            tracks=[static, expanding],
            fields=make_fields(depth=make_depth(near=0.5), flow=make_flow(magnitude=0.3)),
            expansion_smoother=ExpansionSmoother(),
            depth_smoother=DepthDeltaSmoother(),
        )

        assert len(events) == 2
        assert primary.object_id == 1
