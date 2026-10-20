"""Lane geometry, ego-lane tracking, lane changes, hood masking and track-to-lane relations.

Geometry tests use masks rendered from a pinhole camera with known lanes (``lane_scenes``). They check the
method against exact ground truth; they say nothing about accuracy on real footage.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from spectra.engine.adapters import ComponentFailure, Segmentation
from spectra.engine.contracts import (Box, Detection, EgoLaneObservation, EgoLaneState, LaneBoundary, LaneReason,
                                      LaneRelation, ObjectCategory, TrackedObject)
from spectra.engine.lanes import (CLASSES, EGO, INPUT_SIZE, MARKING, ROAD, LaneGeometryEstimator, RoadFit,
                                  SegnetOnnxSegmenter, _x_at, observed_far,
                                  mark_ego_vehicle, parse_logits, relate_tracks)
from spectra.engine.video import Frame
from tests.engine.lane_scenes import H, W, render, truth_offset

MODEL = Path(__file__).resolve().parents[2] / "models/engine/comma10k_segnet_512x384.onnx"
MODEL_SHA256 = "334f58a45509fa13fe9e248ff87d7db2ecc6ac60f59a5f9f34241f639cbc69ff"


def frame(t: float, aspect: tuple[int, int] = (4, 3)) -> Frame:
    """A frame whose image has the given width:height; the estimator reads only its shape."""
    return Frame(None, t, None, np.zeros((aspect[1], aspect[0], 3), np.uint8))


def drive(scenes, times=None, estimator=None):
    """Feed (kwargs for ``render``) per frame at 5 Hz unless ``times`` is given; return all observations."""
    est = estimator or LaneGeometryEstimator()
    est.load()
    times = times or [0.2 * i for i in range(len(scenes))]
    out = []
    for i, (t, kw) in enumerate(zip(times, scenes)):
        mask = kw if isinstance(kw, np.ndarray) else render(phase=1.5 * i, **kw)
        out.append(est.estimate(frame(t), Segmentation(mask, CLASSES)))
    return out


def settled(n=12, **kw) -> EgoLaneObservation:
    return drive([kw] * n)[-1]


# --- segmentation adapter -------------------------------------------------------------------------


def test_segmenter_refuses_a_model_with_the_wrong_hash(tmp_path):
    fake = tmp_path / "m.onnx"
    fake.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="sha256"):
        SegnetOnnxSegmenter(fake, "0" * 64).load()
    with pytest.raises(FileNotFoundError):
        SegnetOnnxSegmenter(tmp_path / "absent.onnx", "0" * 64).load()


def test_segmenter_identity_records_checkpoint_classes_and_interval():
    s = SegnetOnnxSegmenter(MODEL, MODEL_SHA256, interval_s=0.2)
    assert s.info.checkpoint_sha256 == MODEL_SHA256 and s.info.interval_s == 0.2
    assert s.info.params["classes"] == ["road", "lane_marking", "undrivable", "movable", "ego_vehicle"]


def test_malformed_logits_are_a_recoverable_frame_failure():
    w, h = INPUT_SIZE
    good = np.zeros((1, 5, h, w), np.float32)
    good[0, EGO, :10] = 1
    mask = parse_logits(good)
    assert mask.shape == (h, w) and mask.dtype == np.uint8 and (mask[:10] == EGO).all()
    with pytest.raises(ComponentFailure, match="shape"):
        parse_logits(np.zeros((1, 4, h, w), np.float32))
    bad = good.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ComponentFailure, match="non-finite"):
        parse_logits(bad)


@pytest.mark.skipif(not MODEL.is_file(), reason="local segmentation model not present")
@pytest.mark.model
def test_real_segmenter_produces_a_full_size_class_mask():
    s = SegnetOnnxSegmenter(MODEL, MODEL_SHA256, threads=2)
    s.load()
    try:
        img = np.full((720, 1280, 3), 90, np.uint8)
        seg = s.segment(Frame(0, 0.0, 0, img))
        assert seg.mask.shape == (INPUT_SIZE[1], INPUT_SIZE[0]) and seg.classes == CLASSES
        assert int(seg.mask.max()) < len(CLASSES)
    finally:
        s.close()
    with pytest.raises(RuntimeError):
        s.segment(Frame(0, 0.0, 0, img))


@pytest.mark.model
def test_model_file_matches_the_recorded_hash():
    if not MODEL.is_file():
        pytest.skip("local segmentation model not present")
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == MODEL_SHA256


# --- camera geometry ------------------------------------------------------------------------------


@pytest.mark.parametrize("scene", [
    dict(), dict(camera_x=-1.2), dict(camera_x=0.9), dict(yaw_deg=4), dict(yaw_deg=-6, camera_x=-0.9),
    dict(pitch_deg=3), dict(pitch_deg=-2), dict(principal_x=200), dict(principal_x=330), dict(height=2.4),
    dict(focal=450)])
@pytest.mark.slow
def test_offset_matches_ground_truth_whatever_the_camera_orientation(scene):
    o = settled(**scene)
    assert o.state is EgoLaneState.ESTIMATED
    assert o.offset == pytest.approx(truth_offset(scene.get("camera_x", 0.0)), abs=0.08)
    assert o.width == pytest.approx(3.6 / scene.get("height", 1.3), rel=0.1)


@pytest.mark.parametrize("aspect", [(16, 9), (16, 5), (9, 16)])
@pytest.mark.slow
def test_lane_geometry_does_not_depend_on_the_frame_aspect_ratio(aspect, monkeypatch):
    """The segmenter squashes every frame to the 4:3 mask. A scene rendered at the frame's own aspect and
    resized the same way must give the lane width in camera heights and the offset of the 4:3 case."""
    import cv2
    import tests.engine.lane_scenes as scenes
    src_w = round(H * aspect[0] / aspect[1])
    monkeypatch.setattr(scenes, "W", src_w)
    est = LaneGeometryEstimator()
    est.load()
    for i in range(12):
        mask = scenes.render(phase=1.5 * i, camera_x=0.9, principal_x=src_w / 2)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        o = est.estimate(frame(0.2 * i, aspect), Segmentation(mask, CLASSES))
    assert o.state is EgoLaneState.ESTIMATED
    assert o.width == pytest.approx(3.6 / 1.3, rel=0.1)
    assert o.offset == pytest.approx(truth_offset(0.9), abs=0.08)


@pytest.mark.slow
def test_offset_does_not_depend_on_the_image_centre():
    """A camera off the windscreen centre or yawed moves the lanes in the image, not the camera's lane
    position; the image-centre rule this replaces reports a large offset here."""
    centred = settled()
    for kw in (dict(principal_x=180), dict(yaw_deg=6), dict(yaw_deg=-6)):
        assert settled(**kw).offset == pytest.approx(centred.offset, abs=0.06)
    o = settled(principal_x=180)  # camera mounted off the windscreen centre, or a cropped frame
    left, right = o.left.near[0], o.right.near[0]
    image_centre_rule = (0.5 - (left + right) / 2) / (right - left)
    assert abs(image_centre_rule - o.offset) > 0.1


@pytest.mark.slow
def test_small_roll_is_tolerated_and_recorded_as_an_assumption():
    o = settled(roll_deg=2)
    assert o.offset == pytest.approx(0.0, abs=0.1)
    assumptions = LaneGeometryEstimator().info.params["assumptions"]
    assert "roll negligible" in assumptions and "flat road" in assumptions


# --- curves, missing and occluded markings --------------------------------------------------------


@pytest.mark.parametrize("scene", [dict(curvature=0.004), dict(curvature=-0.006, camera_x=-0.6),
                                   dict(curvature=0.01, yaw_deg=3)])
@pytest.mark.slow
def test_curved_roads_keep_the_lane_position(scene):
    o = settled(**scene)
    assert o.offset == pytest.approx(truth_offset(scene.get("camera_x", 0.0)), abs=0.15)
    assert len(o.left.points) > 2  # curved boundaries are sampled, not reduced to a segment


@pytest.mark.slow
def test_boundaries_extend_only_a_bounded_distance_beyond_the_evidence():
    o = settled()
    for b in (o.left, o.right):
        assert b.evidence_y is not None and b.far[1] <= b.evidence_y
        assert b.far[1] > o.horizon_y


@pytest.mark.slow
def test_no_markings_is_unavailable_and_a_missing_ego_boundary_is_ambiguous():
    blank = render(boundaries=0)
    blank[blank == 1] = 0
    assert drive([blank] * 3)[-1].state is EgoLaneState.UNAVAILABLE
    one_side = settled(missing=(1,))
    assert one_side.state is EgoLaneState.AMBIGUOUS and one_side.offset is None


@pytest.mark.slow
def test_briefly_occluded_boundaries_are_carried_then_dropped():
    scenes = [{}] * 10 + [render(missing=(0, 1, 2, 3, 4))] * 8
    obs = drive(scenes)
    carried = obs[11]
    assert carried.state is EgoLaneState.ESTIMATED
    assert not carried.left.measured and carried.left.age_s > 0
    assert obs[-1].state is not EgoLaneState.ESTIMATED  # beyond max_age_s nothing is invented


@pytest.mark.slow
def test_recovers_after_unreliable_frames():
    noise = np.random.default_rng(0).integers(0, 5, (H, W)).astype(np.uint8)
    obs = drive([{}] * 6 + [noise] * 3 + [{}] * 12)
    assert obs[-1].state is EgoLaneState.ESTIMATED
    assert obs[-1].offset == pytest.approx(0.0, abs=0.08)


# --- lane changes ---------------------------------------------------------------------------------


def lane_change_scenes(start: float, end: float, n_move: int = 15, before: int = 8, after: int = 12, **kw):
    xs = [start] * before + list(np.linspace(start, end, n_move)) + [end] * after
    return [dict(camera_x=float(x), **kw) for x in xs]


def completed(obs):
    seen = []
    for o in obs:
        c = o.lane_change
        if c and c.phase == "completed" and (c.direction, c.t_start) not in seen:
            seen.append((c.direction, c.t_start))
    return seen


@pytest.mark.parametrize("start,end,direction", [(0.0, -3.6, "left"), (0.0, 3.6, "right")])
@pytest.mark.slow
def test_one_lane_change_is_reported_once_with_its_direction(start, end, direction):
    obs = drive(lane_change_scenes(start, end))
    changes = completed(obs)
    assert [d for d, _ in changes] == [direction]
    assert changes[0][1] <= 0.2 * 8 + 3 * 0.2  # began near the start of the movement
    assert any(o.lane_change and o.lane_change.phase == "crossing" for o in obs)


@pytest.mark.slow
def test_lane_change_on_a_curve_and_with_irregular_timestamps():
    scenes = lane_change_scenes(0.0, -3.6, curvature=0.004)
    rng = np.random.default_rng(1)
    times = list(np.cumsum(rng.uniform(0.1, 0.35, len(scenes))))
    assert [d for d, _ in completed(drive(scenes, times))] == ["left"]


@pytest.mark.slow
def test_a_curve_alone_is_not_a_lane_change():
    scenes = [dict(curvature=c) for c in np.r_[np.zeros(6), np.linspace(0, 0.01, 10), np.full(10, 0.01),
                                                np.linspace(0.01, -0.006, 10)]]
    assert completed(drive(scenes)) == []


@pytest.mark.slow
def test_drifting_towards_a_line_and_back_is_not_a_lane_change():
    xs = [0.0] * 6 + list(np.linspace(0, -1.7, 6)) + [-1.7] * 2 + list(np.linspace(-1.7, 0, 6)) + [0.0] * 6
    obs = drive([dict(camera_x=float(x)) for x in xs])
    assert completed(obs) == []
    assert obs[-1].lane_change is None


@pytest.mark.slow
def test_a_crossing_that_returns_is_reported_as_cancelled_on_one_frame():
    xs = [0.0] * 6 + list(np.linspace(0, -1.7, 6)) + [-1.7] * 2 + list(np.linspace(-1.7, 0, 6)) + [0.0] * 6
    obs = drive([dict(camera_x=float(x)) for x in xs])
    phases = [o.lane_change.phase if o.lane_change else None for o in obs]
    assert "crossing" in phases and phases.count("cancelled") == 1 and "completed" not in phases
    k = phases.index("cancelled")
    assert all(p == "crossing" for p in phases[phases.index("crossing"):k]) and set(phases[k + 1:]) == {None}
    crossing = next(o.lane_change for o in obs if o.lane_change)
    assert obs[k].lane_change.model_copy(update={"phase": "crossing"}) == crossing  # the same crossing


@pytest.mark.slow
def test_lane_reasons_are_codes():
    obs = drive(lane_change_scenes(0.0, -3.6))
    assert any(LaneReason.ON_BOUNDARY in o.reasons for o in obs)
    assert all(isinstance(r, LaneReason) for o in obs for r in o.reasons)


@pytest.mark.slow
def test_camera_shake_is_not_a_lane_change():
    rolls = [0.0] * 8 + [3.0, -3.0, 2.5] + [0.0] * 10
    pitches = [0.0] * 8 + [2.0, -2.0, 1.0] + [0.0] * 10
    obs = drive([dict(roll_deg=r, pitch_deg=p) for r, p in zip(rolls, pitches)])
    assert completed(obs) == []


@pytest.mark.slow
def test_estimator_state_resets_between_runs():
    est = LaneGeometryEstimator()
    drive(lane_change_scenes(0.0, -3.6), estimator=est)
    est.close()
    fresh = drive([{}] * 3, estimator=est)
    assert all(o.lane_change is None for o in fresh)


# --- hood masking ---------------------------------------------------------------------------------


def det(x1, y1, x2, y2, score=0.8):
    return Detection(box=Box(x1=x1, y1=y1, x2=x2, y2=y2), score=score, category=ObjectCategory.CAR)


@pytest.mark.slow
def test_hood_detections_are_flagged_not_deleted_and_nearby_cars_are_kept():
    ego = np.zeros((384, 512), bool)
    ego[300:] = True  # bonnet over the bottom 84 rows
    width, height = 1280, 720
    hood = det(100, 300 / 384 * 720 + 5, 1180, 720)
    close_car = det(500, 380, 800, 600)  # bottom edge overlaps the bonnet a little
    far_car = det(600, 300, 680, 360)
    out = mark_ego_vehicle([hood, close_car, far_car], ego, width, height)
    assert len(out) == 3
    assert out[0].ego_vehicle and out[0].ego_overlap > 0.9
    assert not out[1].ego_vehicle and 0 < out[1].ego_overlap < 0.5
    assert not out[2].ego_vehicle and out[2].ego_overlap == 0


@pytest.mark.slow
def test_a_vehicle_covering_the_hood_region_is_kept_when_the_mask_says_movable():
    """At impact a real vehicle fills the bottom of the image; the segmenter labels its pixels movable,
    so the ego mask is small there and the detection stays."""
    ego = np.zeros((384, 512), bool)
    ego[340:, :100] = ego[340:, 412:] = True  # bonnet visible only at the sides
    out = mark_ego_vehicle([det(200, 300, 1080, 720)], ego, 1280, 720)
    assert not out[0].ego_vehicle


# --- track-to-lane relations ----------------------------------------------------------------------


def track(i, cx, bottom, w=80, h=60):
    return TrackedObject(track_id=i, box=Box(x1=cx - w / 2, y1=bottom - h, x2=cx + w / 2, y2=bottom),
                         category=ObjectCategory.CAR)


@pytest.mark.slow
def test_tracks_are_related_to_the_ego_lane_and_its_neighbours():
    o = settled()
    width, height = 1280, 720
    yr = 0.75
    xl, xr = (np.interp(yr, *zip(*sorted((p[1], p[0]) for p in b.points))) for b in (o.left, o.right))
    lane = (xr - xl) * width
    mid = (xl + xr) / 2 * width
    rel = relate_tracks([track(0, mid, yr * height), track(1, mid - lane, yr * height),
                         track(2, mid + lane, yr * height), track(3, mid + lane / 2, yr * height),
                         track(4, mid, (o.horizon_y + 0.005) * height)], o, 1.0, width, height)
    by_id = {r.track_id: r for r in rel}
    assert by_id[0].relation is LaneRelation.EGO_LANE and by_id[0].lateral == pytest.approx(0, abs=0.05)
    assert by_id[1].relation is LaneRelation.LEFT_ADJACENT
    assert by_id[2].relation is LaneRelation.RIGHT_ADJACENT
    assert by_id[3].relation is LaneRelation.AMBIGUOUS  # straddling the boundary
    assert by_id[4].relation is LaneRelation.AMBIGUOUS and by_id[4].lateral is None  # beyond the model's reach
    assert not by_id[0].beyond_evidence


@pytest.mark.slow
def test_no_relations_without_a_reliable_ego_lane():
    t = [track(0, 640, 600)]
    assert relate_tracks(t, None, 0.0, 1280, 720) is None
    assert relate_tracks(t, EgoLaneObservation(state=EgoLaneState.AMBIGUOUS), 0.0, 1280, 720) is None
    bare = EgoLaneObservation(left=LaneBoundary(near=(0.3, 1), far=(0.45, 0.6)),
                              right=LaneBoundary(near=(0.7, 1), far=(0.55, 0.6)), offset=0.0)
    assert relate_tracks(t, bare, 0.0, 1280, 720) is None  # no sampled geometry to relate to


@pytest.mark.slow
def test_too_little_visible_road_is_ambiguous_and_starts_no_lane_change():
    """A low camera behind a long bonnet sees too few rows of road to separate lateral position from bend."""
    scenes = lane_change_scenes(0.0, -3.6, hood_rows=170)
    obs = drive(scenes)
    assert all(o.state is EgoLaneState.AMBIGUOUS and o.offset is None for o in obs[2:])
    assert obs[-1].reasons == (LaneReason.SHALLOW_ROAD,)
    assert completed(obs) == []


@pytest.mark.slow
def test_lane_change_decisions_pause_while_the_bend_exceeds_the_limit():
    """While the bend term moves the boundaries more than ``max_bend`` lane widths at the nearest row, the
    lane is still reported but no lane change completes, and the reason is recorded. (Synthetic roads obey
    the model exactly, so the limit is lowered here to exercise the mechanism.)"""
    scenes = lane_change_scenes(0.0, -3.6, curvature=0.01)
    obs = drive(scenes, estimator=LaneGeometryEstimator(max_bend=1e-4))
    assert any(LaneReason.SHARP_BEND in o.reasons for o in obs)
    assert completed(obs) == []
    assert sum(o.state is EgoLaneState.ESTIMATED for o in obs) > len(obs) // 3
    assert [d for d, _ in completed(drive(scenes))] == ["left"]  # default limit: this curve is gentle


def bend_drive(parts):
    """(camera_x from, to, frames, curvature) segments. With ``max_bend`` at 0.003, frames at curvature 0.02
    are sharp and straight frames are not, so decisions pause exactly on the curved segments."""
    scenes = [dict(camera_x=float(x), curvature=k) for x0, x1, n, k in parts for x in np.linspace(x0, x1, n)]
    return drive(scenes, estimator=LaneGeometryEstimator(max_bend=3e-3))


BEND = 0.02
LEFT_DURING_BEND = [(0, 0, 8, 0), (0, -3.6, 15, BEND), (-3.6, -3.6, 6, BEND), (-3.6, -3.6, 6, 0)]


@pytest.mark.slow
def test_a_side_reached_during_a_bend_pause_does_not_complete_a_lane_change_afterwards():
    # boundaries change sides while decisions are paused; the crossing back on straight road starts from a
    # side that was never observed, as on the mountain-road hairpin where the fit sweeps across and back
    obs = bend_drive(LEFT_DURING_BEND + [(-3.6, 0, 15, 0), (0, 0, 12, 0)])
    assert any(LaneReason.SHARP_BEND in o.reasons for o in obs)
    assert completed(obs) == []


@pytest.mark.slow
def test_fresh_evidence_after_the_bend_reports_lane_changes_again():
    obs = bend_drive(LEFT_DURING_BEND + [(-3.6, 0, 15, 0), (0, 0, 12, 0), (0, 3.6, 15, 0), (3.6, 3.6, 12, 0)])
    assert [d for d, _ in completed(obs)] == ["right"]


@pytest.mark.slow
def test_a_lane_change_just_after_a_bend_is_reported():
    obs = bend_drive([(0, 0, 8, 0), (0, 0, 10, BEND), (0, 0, 6, 0), (0, -3.6, 15, 0), (-3.6, -3.6, 12, 0)])
    assert [d for d, _ in completed(obs)] == ["left"]


@pytest.mark.slow
def test_a_crossing_observed_before_the_bend_completes_after_it():
    obs = bend_drive([(0, 0, 8, 0), (0, -1.8, 8, 0), (-1.8, -2.4, 6, BEND), (-2.4, -3.6, 8, 0), (-3.6, -3.6, 12, 0)])
    assert [d for d, _ in completed(obs)] == ["left"]


# --- evidence along observed paint ---------------------------------------------------------------

PREVIOUS = dict(evidence_min_matches=10 ** 9)  # evidence never moves beyond the fit: the version-2 behaviour
FH, FW, FHORIZON = 100, 200, 20.0


def straight_fit(far=60.0, laterals=(-1.2, 1.2)):
    # boundaries x = a * (y - 20) + 100, meeting at column 100 on the horizon
    return RoadFit(horizon=FHORIZON, b=100.0, c=0.0, laterals=list(laterals), support=[40] * len(laterals),
                   far=[far] * len(laterals), near_row=99.0, residual=1.0)


def paint(rows, laterals=(-1.2, 1.2), offset=0.0):
    mask = np.full((FH, FW), ROAD, np.uint8)
    for y in rows:
        for a in laterals:
            mask[y, int(round(a * (y - FHORIZON) + 100 + offset))] = MARKING
    return mask


@pytest.mark.slow
def test_evidence_follows_paint_on_the_fitted_curve_up_to_the_horizon_margin():
    assert observed_far(paint(range(23, 100)), straight_fit(), 1.0) == [23.0, 23.0]


@pytest.mark.slow
def test_dash_gaps_are_bridged_and_a_long_gap_ends_the_evidence():
    dashed = [y for y in range(30, 60) if y % 8 < 4]  # 4 painted rows, 4 empty
    assert observed_far(paint(dashed), straight_fit(), 1.0) == [32.0, 32.0]
    gap = list(range(50, 60)) + list(range(30, 40))  # 10 empty rows between them
    assert observed_far(paint(gap), straight_fit(), 1.0) == [50.0, 50.0]


@pytest.mark.slow
def test_too_few_painted_rows_leave_the_evidence_where_the_fit_put_it():
    assert observed_far(paint(range(60, 100)), straight_fit(), 1.0) == [60.0, 60.0]
    assert observed_far(paint([58, 57]), straight_fit(), 1.0) == [60.0, 60.0]  # 2 rows: fewer than 3


@pytest.mark.slow
def test_paint_off_the_curve_or_nearer_another_boundary_does_not_count():
    off = paint(range(30, 60), offset=15.0)  # another line, away from both boundaries
    assert observed_far(off, straight_fit(), 1.0) == [60.0, 60.0]
    middle = paint(range(30, 60), laterals=(0.0,))  # paint half-way between the two boundaries
    assert observed_far(middle, straight_fit(), 1.0) == [60.0, 60.0]
    lone = paint(range(30, 60), laterals=(-1.2,))
    assert observed_far(lone, straight_fit(laterals=(-1.2,)), 1.0) == [60.0]  # no spacing to judge by


@pytest.mark.slow
def test_evidence_parameters_are_part_of_the_estimator_identity():
    a, b = LaneGeometryEstimator().info, LaneGeometryEstimator(evidence_max_gap=3).info
    assert a.version == "5" and a.params["evidence"]["max_gap_rows"] == 6
    assert a.params != b.params


def ego_view(o):
    return (o.state, o.offset, o.width, o.confidence, o.lateral_rate, o.lane_change,
            None if o.left is None else o.left.lateral, None if o.right is None else o.right.lateral)


@pytest.mark.parametrize("scenes", [
    [{}] * 12,
    lane_change_scenes(0.0, -3.6),
    lane_change_scenes(0.0, 3.6, curvature=0.004),
    [dict(curvature=c) for c in np.r_[np.zeros(6), np.linspace(0, 0.01, 10), np.full(10, 0.01)]],
    [dict(roll_deg=r, pitch_deg=p) for r, p in zip([0.0] * 8 + [3.0, -3.0, 2.5] + [0.0] * 6,
                                                   [0.0] * 8 + [2.0, -2.0, 1.0] + [0.0] * 6)],
], ids=["straight", "lane-change-left", "lane-change-curve", "curve", "shake"])
@pytest.mark.slow
def test_extended_evidence_leaves_the_ego_lane_and_lane_changes_unchanged(scenes):
    new, old = drive(scenes), drive(scenes, estimator=LaneGeometryEstimator(**PREVIOUS))
    assert [ego_view(o) for o in new] == [ego_view(o) for o in old]
    assert completed(new) == completed(old)


def solid_pair():
    # continuous markings reach the horizon margin; the synthetic dashes beyond the fit are 11 rows apart
    return (drive([dict(dashed=False)] * 12)[-1],
            drive([dict(dashed=False)] * 12, estimator=LaneGeometryEstimator(**PREVIOUS))[-1])


@pytest.mark.slow
def test_a_dash_gap_longer_than_the_limit_leaves_the_evidence_unchanged():
    new, old = settled(), drive([{}] * 12, estimator=LaneGeometryEstimator(**PREVIOUS))[-1]
    assert (new.left.evidence_y, new.right.evidence_y) == (old.left.evidence_y, old.right.evidence_y)


@pytest.mark.slow
def test_observed_markings_extend_the_evidence_and_never_beyond_the_curve():
    new, old = solid_pair()
    for nb, ob in ((new.left, old.left), (new.right, old.right)):
        assert nb.evidence_y < ob.evidence_y  # farther up the image
        assert nb.far[1] <= nb.evidence_y and nb.far[1] > new.horizon_y
        y = ob.evidence_y
        assert _x_at(nb, y) == pytest.approx(_x_at(ob, y), abs=1e-3)  # the same fitted curve


@pytest.mark.slow
def test_existing_assignments_are_kept_and_only_newly_observed_rows_become_observable():
    new, old = solid_pair()
    width, height = 1280, 720
    ys = {"near": 0.75, "new": (new.left.evidence_y + old.left.evidence_y) / 2,
          "beyond": (new.horizon_y + min(new.left.evidence_y, new.right.evidence_y)) / 2}
    tracks = []
    for k, y in enumerate(ys.values()):
        ref = y if k < 2 else ys["new"]  # beyond the evidence the boundaries are not drawn: place by the last row
        xl, xr = _x_at(new.left, ref), _x_at(new.right, ref)
        mid, lane = (xl + xr) / 2 * width, (xr - xl) * width
        tracks += [track(10 * k, mid, y * height, w=8, h=6), track(10 * k + 1, mid - lane, y * height, w=8, h=6)]
    rn = {r.track_id: r for r in relate_tracks(tracks, new, 1.0, width, height)}
    ro = {r.track_id: r for r in relate_tracks(tracks, old, 1.0, width, height)}
    for i in (0, 1):  # near the camera: identical
        assert rn[i] == ro[i] and not rn[i].beyond_evidence
    for i in (10, 11):  # between the old and the new evidence: now observed, same lane as the curve gives
        assert ro[i].beyond_evidence and not rn[i].beyond_evidence
    assert rn[10].relation is LaneRelation.EGO_LANE and rn[11].relation is LaneRelation.LEFT_ADJACENT
    for i in (20, 21):  # beyond the new evidence: still not observed
        assert rn[i].beyond_evidence or rn[i].lateral is None


@pytest.mark.slow
def test_a_far_cut_in_is_observed_only_where_the_paint_is():
    from spectra.engine.contracts import FrameRecord, OutputStatus, Role
    from spectra.engine.events import CutInAnalyzer
    new, old = solid_pair()
    width, height = 1280, 720
    y = (new.left.evidence_y + old.left.evidence_y) / 2
    xl, xr = _x_at(new.left, y), _x_at(new.right, y)
    ts = [round(0.1 * k, 1) for k in range(61)]
    lateral = lambda t: float(np.interp(t, [0, 1.5, 3.0, 6.0], [-1.0, -1.0, 0.0, 0.0]))

    def records(lane):
        out = []
        for i, t in enumerate(ts):
            cx = ((xl + xr) / 2 + lateral(t) * (xr - xl)) * width
            tr = (track(7, cx, y * height, w=8, h=6).model_copy(update={"score": 0.9}),)
            out.append(FrameRecord(index=i, t=t, status={Role.DETECTOR: OutputStatus.OK, Role.TRACKER: OutputStatus.OK,
                                                         Role.EGO_LANE: OutputStatus.OK},
                                   detections=(), tracks=tr, ego_lane=lane,
                                   lane_relations=relate_tracks(tr, lane, t, width, height)))
        return out
    assert CutInAnalyzer().analyze(records(old), width, height).drafts == []  # entry beyond the old evidence
    found = CutInAnalyzer().analyze(records(new), width, height).drafts
    assert len(found) == 1 and found[0].track_ids == (7,)
