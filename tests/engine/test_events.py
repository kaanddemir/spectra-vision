"""Apparent-approach and cut-in analysers on small synthetic perception records.

Scenes come from a pinhole camera (focal length 1000 px, 1280x720) so the true time-to-contact and passing
offset are known. They check the analysers' logic and failure mechanisms, not real-world accuracy.
"""
import numpy as np
import pytest

from spectra.engine.adapters import AnalyzerUnavailable
from spectra.engine.contracts import (Basis, Box, CameraMotion, EgoLaneObservation, EgoLaneState, EventType,
                                      EvidenceState, FrameRecord, LaneBoundary, LaneChange, LaneRelation,
                                      ObjectCategory, OutputStatus, ReasonCode, Role, TrackedObject,
                                      TrackLaneRelation)
from spectra.engine.events import ApproachAnalyzer, CutInAnalyzer, theil_sen

W, H, F = 1280, 720, 1000.0
ROLES = (Role.DETECTOR, Role.TRACKER, Role.EGO_LANE)


def lane_obs(ids=(1, 2), change=None):
    def boundary(x, i):
        return LaneBoundary(near=(x, 1.0), far=(0.5, 0.5), points=((x, 1.0), (0.5, 0.5)), track_id=i,
                            evidence_y=0.5)
    return EgoLaneObservation(state=EgoLaneState.ESTIMATED, left=boundary(0.2, ids[0]), right=boundary(0.8, ids[1]),
                              offset=0.0, confidence=0.8, width=2.5, lane_change=change)


def lane(t):
    """The same estimated ego lane at every frame."""
    return lane_obs()


def box_at(x_m, z_m, width_m=1.8, height_m=1.5, cam_h=1.3):
    """Box of a vehicle ``x_m`` metres right of the camera and ``z_m`` ahead."""
    xc, w, h = W / 2 + F * x_m / z_m, F * width_m / z_m, F * height_m / z_m
    bottom = H / 2 + F * cam_h / z_m
    return xc - w / 2, bottom - h, xc + w / 2, bottom


def frames(times, objects, lanes=None, motion=None):
    """objects(t) -> [(track_id, (x1, y1, x2, y2), relation, lateral, beyond, score)]; lanes(t) -> lane or None."""
    out, prev = [], None
    for i, t in enumerate(times):
        current = lanes(t) if lanes else None
        tracks, rels = [], []
        for tid, (x1, y1, x2, y2), rel, lat, beyond, score in objects(t):
            tracks.append(TrackedObject(track_id=tid, box=Box(x1=x1, y1=y1, x2=x2, y2=y2), score=score,
                                        category=ObjectCategory.CAR))
            if rel is not None and current is not None:
                rels.append(TrackLaneRelation(track_id=tid, relation=rel, lateral=lat, lane_t=t, beyond_evidence=beyond))
        status = {Role.DETECTOR: OutputStatus.OK, Role.TRACKER: OutputStatus.OK,
                  Role.EGO_LANE: OutputStatus.OK if current is not None else OutputStatus.FAILED}
        cm = motion(t, prev) if motion and prev is not None else None
        out.append(FrameRecord(index=i, t=t, status=status, errors={} if current else {Role.EGO_LANE: "x"},
                               detections=(), tracks=tuple(tracks), ego_lane=current,
                               lane_relations=tuple(rels) if current is not None else None, camera_motion=cm))
        prev = t
    return out


def times(duration=4.0, step=0.1, start=0.0):
    return [round(start + k * step, 4) for k in range(int(duration / step) + 1)]


def approach(z0=40.0, speed=10.0, x_m=0.0, rel=LaneRelation.EGO_LANE, jitter=0.0, score=0.9, seed=0, tid=1):
    rng = np.random.default_rng(seed)

    def objects(t):
        z = z0 - speed * t
        x1, y1, x2, y2 = box_at(x_m, z)
        if jitter:
            w = x2 - x1
            x1, x2 = x1 + rng.normal(0, jitter * w), x2 + rng.normal(0, jitter * w)
        return [(tid, (x1, y1, x2, y2), rel, 0.0 if rel is LaneRelation.EGO_LANE else None, False, score)]
    return objects


def propose(analyzer, records, width, height):
    """The analyser's proposals; checks that its counts reconcile."""
    out = analyzer.analyze(records, width, height)
    assert all(n > 0 for n in out.rejected.values())
    return out.drafts


def measurement(c, name):
    return next(e.value for e in c.evidence if e.name == name)


def motion_estimate(since_t, scale=1.0, rotation=0.0, tx=0.0, ty=0.0, inliers=300):
    return CameraMotion(since_t=since_t, scale=scale, rotation=rotation, tx=tx, ty=ty, matches=400, inliers=inliers,
                        residual_px=0.3)


# --- Theil-Sen ------------------------------------------------------------------------------------


def test_theil_sen_matches_scipy_including_its_interval():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 1, 12))
    y = 0.4 * t + rng.normal(0, 0.02, 12)
    y[4] += 0.5  # an outlier
    ours = theil_sen(t, y, 0.9)
    ref = stats.theilslopes(y, t, alpha=0.9)
    assert ours == pytest.approx((ref.slope, ref.low_slope, ref.high_slope))


# --- apparent approach ----------------------------------------------------------------------------


def test_closing_object_in_the_ego_lane_gives_one_supported_proposal_with_approximate_tau():
    # 40 m at 10 m/s: true time to the camera plane is 4 s at t=0 and 1 s at t=3
    out = propose(ApproachAnalyzer(), frames(times(3.0), approach(), lane), W, H)
    assert len(out) == 1
    c = out[0]
    assert c.type is EventType.RAPID_APPROACH and c.state is EvidenceState.SUPPORTED and c.track_ids == (1,)
    assert c.t_start <= c.t_peak <= c.t_end and 0 <= c.rank <= 1 and c.reasons == ()
    true_tau = (40 - 10 * c.t_peak) / 10
    assert measurement(c, "apparent_tau_s") == pytest.approx(true_tau, rel=0.15)
    tau = next(e for e in c.evidence if e.name == "apparent_tau_s")
    assert tau.basis is Basis.ESTIMATED and tau.unit == "s" and "approximate and unverified" in tau.note
    assert next(e for e in c.evidence if e.name == "ego_lane_fraction").basis is Basis.OBSERVED
    assert next(e for e in c.evidence if e.name == "box_height_fraction").basis is Basis.CONTEXT
    assert not any(e.name == "image_log_scale_rate" for e in c.evidence)  # no camera motion recorded
    assert not any("distance" in e.name or "speed" in e.name for e in c.evidence)


@pytest.mark.parametrize("speed", [0.0, -5.0])
def test_following_at_constant_gap_or_receding_gives_nothing(speed):
    out = propose(ApproachAnalyzer(), frames(times(6.0), approach(z0=20, speed=speed, jitter=0.01), lane), W, H)
    assert out == []


def test_slow_closing_of_ordinary_following_gives_nothing():
    # 30 m closing at 2 m/s: tau 15 s, larger than the review threshold throughout
    assert propose(ApproachAnalyzer(), frames(times(6.0), approach(z0=30, speed=2, jitter=0.01), lane), W, H) == []


def test_irregular_timestamps_give_the_same_tau():
    rng = np.random.default_rng(1)
    ts = list(np.round(np.cumsum(rng.uniform(0.04, 0.2, 40)), 4))
    ts = [t for t in ts if t <= 3.0]
    out = propose(ApproachAnalyzer(), frames(ts, approach(), lane), W, H)
    assert len(out) == 1
    assert measurement(out[0], "apparent_tau_s") == pytest.approx((40 - 10 * out[0].t_peak) / 10, rel=0.15)


def test_box_jitter_widens_the_interval_instead_of_triggering():
    # tau 3.5-6 s with 5% width noise: the point estimate may dip below 3 s, the conservative bound does not
    ts = times(2.5)
    out = propose(ApproachAnalyzer(), frames(ts, approach(z0=60, speed=10, jitter=0.05, seed=7), lane), W, H)
    assert out == []
    # the same closing without noise is still not below the threshold: the jitter is what could have fired
    clean = propose(ApproachAnalyzer(), frames(ts, approach(z0=60, speed=10), lane), W, H)
    assert clean == []


def test_identity_switch_to_a_nearer_object_is_not_an_approach():
    def objects(t):
        z = 40.0 if t < 1.5 else 15.0  # one track id, two different parked vehicles
        return [(1, box_at(0.0, z), LaneRelation.EGO_LANE, 0.0, False, 0.9)]
    assert propose(ApproachAnalyzer(), frames(times(3.0), objects, lane), W, H) == []


def test_interrupted_observations_still_give_one_episode():
    ts = [t for t in times(3.0) if not 1.2 <= t <= 1.5]  # 0.4 s without the object
    out = propose(ApproachAnalyzer(), frames(ts, approach(), lane), W, H)
    assert len(out) == 1


def test_low_score_predicted_and_edge_touching_boxes_are_not_size_evidence():
    assert propose(ApproachAnalyzer(), frames(times(3.0), approach(score=0.3), lane), W, H) == []

    def at_edge(t):
        x1, y1, x2, y2 = box_at(0.0, 40 - 10 * t)
        return [(1, (0.0, y1, x2 - x1, y2), LaneRelation.EGO_LANE, 0.0, False, 0.9)]
    assert propose(ApproachAnalyzer(), frames(times(3.0), at_edge, lane), W, H) == []


def test_camera_motion_is_reported_as_context_and_does_not_create_or_cancel_proposals():
    def zoom(t, prev):  # the whole image expanding by 30% per second
        return motion_estimate(prev, scale=float(np.exp(0.3 * (t - prev))))
    still = frames(times(3.0), approach(z0=20, speed=0.0), lane, motion=zoom)
    assert propose(ApproachAnalyzer(), still, W, H) == []
    out = propose(ApproachAnalyzer(), frames(times(3.0), approach(), lane, motion=zoom), W, H)
    assert len(out) == 1 and measurement(out[0], "image_log_scale_rate") == pytest.approx(0.3, abs=1e-3)
    assert next(e for e in out[0].evidence if e.name == "image_log_scale_rate").basis is Basis.CONTEXT


def test_camera_shake_moving_boxes_without_resizing_them_gives_nothing():
    def shaking(t):
        x1, y1, x2, y2 = box_at(0.0, 20.0)
        dx, dy = 40 * np.sin(17 * t), 30 * np.sin(23 * t)
        return [(1, (x1 + dx, y1 + dy, x2 + dx, y2 + dy), LaneRelation.EGO_LANE, 0.0, False, 0.9)]
    assert propose(ApproachAnalyzer(), frames(times(4.0), shaking, lane), W, H) == []


def test_unknown_lane_relation_gives_an_uncertain_proposal_not_a_negative():
    out = propose(ApproachAnalyzer(), frames(times(3.0), approach(rel=None)), W, H)
    assert len(out) == 1 and out[0].state is EvidenceState.UNCERTAIN
    assert out[0].reasons == (ReasonCode.LANE_NOT_ESTABLISHED,)
    beyond = approach(rel=LaneRelation.EGO_LANE)

    def beyond_evidence(t):
        return [o[:4] + (True,) + o[5:] for o in beyond(t)]
    out = propose(ApproachAnalyzer(), frames(times(3.0), beyond_evidence, lane), W, H)
    assert len(out) == 1 and out[0].state is EvidenceState.UNCERTAIN


def test_passing_offset_measures_lateral_offset_in_object_widths():
    out = propose(ApproachAnalyzer(), frames(times(3.0), approach(x_m=0.9, rel=None)), W, H)
    assert len(out) == 1 and measurement(out[0], "passing_offset_widths") == pytest.approx(0.5, abs=0.05)


def test_object_passing_beside_or_observed_in_another_lane_is_left_out():
    # 4 m to the left (2.2 widths) and closing fast: reaches the camera plane but not the camera's path
    assert propose(ApproachAnalyzer(), frames(times(3.0), approach(x_m=-4.0, rel=None)), W, H) == []
    adjacent = approach(rel=LaneRelation.LEFT_ADJACENT)
    assert propose(ApproachAnalyzer(), frames(times(3.0), adjacent, lane), W, H) == []


def test_growth_in_width_only_is_not_looming():
    # a vehicle emerging from behind another, or a track sliding onto a neighbour: wider, same height
    def emerging(t):
        x1, y1, x2, y2 = box_at(0.0, 25.0)
        return [(1, (x1 - 60 * t, y1, x2, y2), None, None, False, 0.9)]
    assert propose(ApproachAnalyzer(), frames(times(3.0), emerging), W, H) == []


def test_genuine_looming_survives_height_noise():
    rng = np.random.default_rng(5)
    base = approach()

    def noisy(t):
        tid, (x1, y1, x2, y2), *rest = base(t)[0]
        return [(tid, (x1, y1 + rng.normal(0, 0.02) * (y2 - y1), x2, y2), *rest)]
    out = propose(ApproachAnalyzer(), frames(times(3.0), noisy, lane), W, H)
    assert len(out) == 1 and out[0].state is EvidenceState.SUPPORTED
    assert measurement(out[0], "apparent_tau_height_s") == pytest.approx((40 - 10 * out[0].t_peak) / 10, rel=0.2)


def test_noisy_oncoming_vehicle_well_off_the_path_is_left_out_from_the_whole_episode():
    # oncoming at 25 m/s, 6 m to the left (3.3 widths): early boxes are small and noisy
    objects = approach(z0=60, speed=25, x_m=-6.0, rel=None, jitter=0.03, seed=2)
    assert propose(ApproachAnalyzer(), frames(times(1.8), objects), W, H) == []


def test_visible_side_of_an_off_axis_vehicle_shrinks_the_offset_so_it_stays_uncertain():
    """Known limitation: the box includes the vehicle's side, which pulls its centre towards the image
    centre. The fitted offset is too small (never too large), so the proposal is kept as uncertain."""
    def oncoming_with_side(t):
        z, x_m, half, length = 50 - 25 * t, -4.0, 0.9, 4.5
        x1 = W / 2 + F * (x_m - half) / z
        x2 = W / 2 + max(F * (x_m + half) / (z + length), F * (x_m + half) / z)
        y2 = H / 2 + F * 1.3 / z
        return [(1, (x1, y2 - F * 1.5 / z, x2, y2), None, None, False, 0.9)]
    out = propose(ApproachAnalyzer(), frames(times(1.5), oncoming_with_side), W, H)
    assert len(out) == 1 and out[0].state is EvidenceState.UNCERTAIN
    assert -1.5 <= measurement(out[0], "passing_offset_widths") < 0  # true offset: -2.2 widths


def test_no_tracker_output_is_reported_as_unavailable_not_as_no_events():
    recs = [FrameRecord(index=i, t=t, status={Role.DETECTOR: OutputStatus.OK}, detections=())
            for i, t in enumerate(times(1.0))]
    with pytest.raises(AnalyzerUnavailable, match="no tracker output"):
        propose(ApproachAnalyzer(), recs, W, H)


# --- cut-in ---------------------------------------------------------------------------------------


def lateral_path(points):
    """Piecewise-linear lateral position (lane widths, + right) of a vehicle 20 m ahead over time."""
    ts, ls = zip(*points)
    return lambda t: float(np.interp(t, ts, ls))


def relation_of(lat):
    d = abs(lat)
    if d <= 0.4:
        return LaneRelation.EGO_LANE
    if d < 0.6 or 1.4 < d < 1.6:
        return LaneRelation.AMBIGUOUS
    if d <= 1.4:
        return LaneRelation.LEFT_ADJACENT if lat < 0 else LaneRelation.RIGHT_ADJACENT
    return LaneRelation.BEYOND


def vehicle(path, tid=7, beyond=False, score=0.9, alive=lambda t: True):
    def objects(t):
        if not alive(t):
            return []
        lat = path(t)
        return [(tid, box_at(lat * 3.5, 20.0), relation_of(lat), round(lat, 4), beyond, score)]
    return objects


ENTRY = lateral_path([(0, -1.0), (1.5, -1.0), (3.0, 0.0), (6.0, 0.0)])


def test_sustained_entry_from_the_adjacent_lane_gives_one_supported_cut_in():
    out = propose(CutInAnalyzer(), frames(times(6.0), vehicle(ENTRY), lane), W, H)
    assert len(out) == 1
    c = out[0]
    assert c.type is EventType.CUT_IN and c.state is EvidenceState.SUPPORTED and c.track_ids == (7,)
    assert 2.0 < c.t_start < c.t_peak < c.t_end < 3.5  # crossing the band between 0.6 and 0.4 lane widths
    assert c.t_end - c.t_peak == pytest.approx(0.4, abs=0.11)
    assert measurement(c, "lane_lateral_before") < -0.5 and abs(measurement(c, "lane_lateral_after")) <= 0.4
    assert [e.value for e in c.evidence if e.name == "lane_relation"] == ["left_adjacent", "ego_lane"]
    assert not any("danger" in e.note or "fault" in e.note for e in c.evidence)


def test_vehicle_already_in_the_ego_lane_or_staying_adjacent_gives_nothing():
    for path in (lateral_path([(0, 0.0), (6, 0.1)]), lateral_path([(0, -1.0), (6, -0.8)])):
        assert propose(CutInAnalyzer(), frames(times(6.0), vehicle(path), lane), W, H) == []


def test_ego_lane_change_shifts_relations_without_a_cut_in():
    # the camera moves left into the vehicle's lane: the vehicle stays put, the camera's lane changes identity
    shifted = lateral_path([(0, -1.0), (2.0, -1.0), (2.1, 0.0), (6.0, 0.0)])

    def lanes(t):
        if 1.6 <= t < 2.1:
            return None  # camera on the boundary: no relations
        change = LaneChange(direction="left", phase="completed", t_start=1.0, boundary_id=1) if 2.1 <= t <= 3 else None
        return lane_obs((1, 2) if t < 2.0 else (5, 1), change)
    assert propose(CutInAnalyzer(), frames(times(6.0), vehicle(shifted), lanes), W, H) == []


def test_boundary_re_identification_without_a_lane_change_is_uncertain():
    def lanes(t):
        return lane_obs((1, 2) if t < 2.5 else (8, 9))
    out = propose(CutInAnalyzer(), frames(times(6.0), vehicle(ENTRY), lanes), W, H)
    assert len(out) == 1 and out[0].state is EvidenceState.UNCERTAIN
    assert out[0].reasons == (ReasonCode.BOUNDARIES_REIDENTIFIED,)


def test_single_frame_relation_flicker_is_not_a_cut_in():
    flicker = lateral_path([(0, -1.0), (6, -1.0)])

    def objects(t):
        o = vehicle(flicker)(t)
        if abs(t - 3.0) < 1e-6:
            o = [o[0][:2] + (LaneRelation.EGO_LANE, 0.1) + o[0][4:]]
        return o
    assert propose(CutInAnalyzer(), frames(times(6.0), objects, lane), W, H) == []


def test_unobserved_entry_with_unchanged_lane_is_uncertain():
    def lanes(t):
        return None if 1.6 <= t <= 4.8 else lane_obs()
    slow = lateral_path([(0, -1.0), (1.5, -1.0), (4.0, 0.0), (7.0, 0.0)])
    out = propose(CutInAnalyzer(), frames(times(7.0), vehicle(slow), lanes), W, H)
    assert len(out) == 1 and out[0].state is EvidenceState.UNCERTAIN
    assert out[0].reasons == (ReasonCode.ENTRY_NOT_OBSERVED,)


def test_without_lane_relations_cut_ins_are_unavailable_not_absent():
    with pytest.raises(AnalyzerUnavailable, match="no lane relations"):
        propose(CutInAnalyzer(), frames(times(3.0), vehicle(ENTRY)), W, H)


def test_relations_without_a_position_or_beyond_the_lane_evidence_are_not_evidence():
    def objects(t):
        return [o[:3] + (None,) + o[4:] if o[2] is LaneRelation.AMBIGUOUS else o for o in vehicle(ENTRY)(t)]
    assert len(propose(CutInAnalyzer(), frames(times(6.0), objects, lane), W, H)) == 1
    # on a bend the far boundaries are extrapolated: relations there say nothing about the vehicle
    out = propose(CutInAnalyzer(), frames(times(6.0), vehicle(ENTRY, beyond=True), lane), W, H)
    assert out == []


def test_track_identity_interruption_gives_no_invented_cut_in():
    before = vehicle(ENTRY, tid=7, alive=lambda t: t < 2.2)
    after = vehicle(ENTRY, tid=8, alive=lambda t: t >= 2.2)
    out = propose(CutInAnalyzer(), frames(times(6.0), lambda t: before(t) + after(t), lane), W, H)
    assert out == []


def test_low_score_observations_are_not_lane_evidence():
    assert propose(CutInAnalyzer(), frames(times(6.0), vehicle(ENTRY, score=0.3), lane), W, H) == []


def test_one_proposal_per_entry_and_a_new_one_after_leaving_and_returning():
    out = propose(CutInAnalyzer(), frames(times(10.0), vehicle(ENTRY), lane), W, H)
    assert len(out) == 1
    twice = lateral_path([(0, -1.0), (1.5, -1.0), (3.0, 0.0), (4.5, 0.0), (6.0, 1.0), (7.5, 1.0), (9.0, 0.0),
                          (11.0, 0.0)])
    out = propose(CutInAnalyzer(), frames(times(11.0), vehicle(twice), lane), W, H)
    assert [c.evidence[0].value for c in out] == ["left_adjacent", "right_adjacent"]
    assert len({c.t_start for c in out}) == 2


def test_only_a_completed_ego_lane_change_suppresses_a_cut_in():
    # the camera starts to cross, returns (cancelled), and the lane tracker re-identifies its boundaries: the
    # camera never left its lane, so a vehicle entering it is still a proposal, but an uncertain one
    def lanes(t, phase):
        change = LaneChange(direction="left", phase=phase, t_start=1.8, boundary_id=1) if 2.0 <= t <= 2.6 else None
        if phase == "cancelled" and change is not None and t < 2.6:
            change = change.model_copy(update={"phase": "crossing"})
        return lane_obs((1, 2) if t < 2.5 else (8, 9), change)
    out = propose(CutInAnalyzer(), frames(times(6.0), vehicle(ENTRY), lambda t: lanes(t, "cancelled")), W, H)
    assert len(out) == 1 and out[0].reasons == (ReasonCode.BOUNDARIES_REIDENTIFIED,)
    # a crossing that was never resolved within the records is not a completed change either
    out = propose(CutInAnalyzer(), frames(times(6.0), vehicle(ENTRY), lambda t: lanes(t, "crossing")), W, H)
    assert len(out) == 1 and out[0].state is EvidenceState.UNCERTAIN
    result = CutInAnalyzer().analyze(frames(times(6.0), vehicle(ENTRY), lambda t: lanes(t, "completed")), W, H)
    assert result.drafts == [] and result.rejected == {ReasonCode.EGO_LANE_CHANGE: 1}


def switching(t_switch=1.6, z_a=(40.0, 10.0), z_b=(10.0, 5.0), tid=1):
    """One track that follows vehicle A closing at ``z_a`` (start m, m/s), then from ``t_switch`` a nearer
    vehicle B closing at ``z_b``: the tracker kept the id across two objects."""
    def objects(t):
        z = z_a[0] - z_a[1] * t if t < t_switch else z_b[0] - z_b[1] * (t - t_switch)
        return [(tid, box_at(0.0, z), LaneRelation.EGO_LANE, 0.0, False, 0.9)]
    return objects


def test_identity_switch_to_a_larger_vehicle_inside_an_episode_is_uncertain():
    # A at 24 m when the id moves to B at 10 m: the box height steps by 2.4x in one observation
    out = propose(ApproachAnalyzer(), frames(times(2.6), switching(z_b=(10.0, 5.0)), lane), W, H)
    assert len(out) == 1 and out[0].reasons == (ReasonCode.IDENTITY_JUMP,)
    step = next(e for e in out[0].evidence if e.name == "height_step_ratio")
    assert step.basis is Basis.OBSERVED and step.t < 1.6 <= step.t_end and step.value > 2


@pytest.mark.parametrize("z0,speed,duration", [(40.0, 10.0, 3.0), (12.0, 10.0, 0.9), (30.0, 25.0, 1.0)])
def test_steady_approaches_including_fast_final_ones_are_not_identity_jumps(z0, speed, duration):
    # the 12 m at 10 m/s case doubles its box within 0.5 s: a fixed-ratio rule would flag it
    out = propose(ApproachAnalyzer(), frames(times(duration), approach(z0=z0, speed=speed), lane), W, H)
    assert len(out) == 1 and out[0].state is EvidenceState.SUPPORTED, out[0].reasons if out else out
    assert not any(e.name == "height_step_ratio" for e in out[0].evidence)


def test_rejected_patterns_are_counted_by_reason():
    other = ApproachAnalyzer().analyze(frames(times(3.0), approach(rel=LaneRelation.LEFT_ADJACENT), lane), W, H)
    assert other.drafts == [] and other.rejected == {ReasonCode.OTHER_LANE: 1}
    beside = ApproachAnalyzer().analyze(frames(times(3.0), approach(x_m=-4.0, rel=None)), W, H)
    assert beside.drafts == [] and beside.rejected == {ReasonCode.PASSES_BESIDE: 1}
    kept = ApproachAnalyzer().analyze(frames(times(3.0), approach(), lane), W, H)
    assert len(kept.drafts) == 1 and not kept.rejected


# --- impact involving the camera vehicle ------------------------------------------------------------

from spectra.engine.contracts import EventSubtype, UnresolvedPattern  # noqa: E402
from spectra.engine.events import ImpactAnalyzer  # noqa: E402


def shake(jolt_at=None, jolt=0.05, unreliable=(), missing=(), start=0.0, seed=0):
    """Camera motion: small random shake, an abrupt rotation of ``jolt`` rad at ``jolt_at``; estimates
    in ``unreliable`` (intervals) have few inliers, those in ``missing`` are absent, none before ``start``."""
    rng = np.random.default_rng(seed)

    def motion(t, prev):
        noise = rng.normal(0, 1, 4)
        if t < start or any(a <= t <= b for a, b in missing):
            return None
        rot = 0.0005 * noise[0] + (jolt if jolt_at is not None and abs(t - jolt_at) < 1e-6 else 0.0)
        inliers = 5 if any(a <= t <= b for a, b in unreliable) else 300
        return motion_estimate(prev, scale=float(np.exp(0.002 * noise[1])), rotation=rot, tx=0.5 * noise[2],
                               ty=0.5 * noise[3], inliers=inliers)
    return motion


def closing(z_far=25.0, z_near=6.0, t0=6.0, speed=10.0, tid=1, rel=LaneRelation.EGO_LANE, until=None,
            x_m=0.0):
    """A vehicle at ``z_far`` until ``t0``, then closing at ``speed`` to ``z_near``, where it stays (or
    disappears at ``until``)."""
    def objects(t):
        if until is not None and t >= until:
            return []
        z = max(z_near, z_far - speed * max(0.0, t - t0))
        return [(tid, box_at(x_m, z), rel, 0.0 if rel is LaneRelation.EGO_LANE else None, False, 0.9)]
    return objects


T_STOP = 6.0 + (25.0 - 6.0) / 10.0  # 7.9 s: the vehicle reaches 6 m


def impact(objects, motion, duration=11.0, lanes=lane, **kw):
    return ImpactAnalyzer(**kw).analyze(frames(times(duration), objects, lanes, motion=motion), W, H)


def test_close_approach_with_a_jolt_gives_a_supported_camera_vehicle_proposal():
    out = impact(closing(), shake(jolt_at=8.0))
    assert len(out.drafts) == 1 and not out.rejected and not out.unresolved
    d = out.drafts[0]
    assert (d.type, d.subtype, d.state, d.track_ids) == (EventType.IMPACT, EventSubtype.CAMERA_VEHICLE,
                                                         EvidenceState.SUPPORTED, (1,))
    assert d.t_peak == pytest.approx(8.0) and d.t_start < T_STOP <= d.t_end + 0.3
    names = {e.name: e for e in d.evidence}
    assert names["camera_jolt_z"].basis is Basis.ESTIMATED and names["camera_jolt_z"].value >= 10
    assert names["close_box_height"].basis is Basis.OBSERVED and names["close_box_height"].value >= 0.25
    assert names["apparent_tau_s"].basis is Basis.ESTIMATED and "approach_episode" in names
    assert names["motion_estimates_judged"].value == 1.0 and names["motion_estimates_judged"].basis is Basis.CONTEXT


def test_a_jolt_alone_is_not_an_impact():
    following = closing(z_far=12.0, z_near=12.0)  # a vehicle at a constant gap; the camera jolts
    out = impact(following, shake(jolt_at=8.0))
    assert out.drafts == [] and not out.rejected  # no approach episode, nothing to consider
    assert impact(lambda t: [], shake(jolt_at=8.0)).drafts == []


def test_a_close_approach_without_a_jolt_is_not_an_impact():
    out = impact(closing(), shake())
    assert out.drafts == [] and out.rejected == {ReasonCode.NO_JOLT: 1} and not out.unresolved


def test_a_jolt_after_an_approach_that_ended_far_away_is_not_an_impact():
    out = impact(closing(z_near=15.0, t0=6.0), shake(jolt_at=7.1))
    assert out.drafts == [] and out.rejected == {ReasonCode.NOT_CLOSE: 1}


def test_lost_while_large_is_close():
    # closing to 10 m (box height 0.21 of the frame) and then gone: hidden under the bonnet or out of view
    out = impact(closing(z_near=10.0, until=7.6), shake(jolt_at=7.7))
    assert len(out.drafts) == 1
    assert {e.name for e in out.drafts[0].evidence} >= {"close_track_lost", "camera_jolt_z"}


def test_missing_motion_is_recorded_and_never_replaces_the_jolt():
    # the estimates fail exactly when the jolt would be measured: no proposal, the gap is listed
    out = impact(closing(), shake(missing=[(7.5, 8.5)]))
    assert out.drafts == [] and out.rejected == {ReasonCode.MOTION_ESTIMATE_MISSING: 1}
    (u,) = out.unresolved
    assert u.reason is ReasonCode.MOTION_ESTIMATE_MISSING and u.track_ids == (1,) and u.t_start < T_STOP < u.t_end
    # an abrupt change measured by an unreliable fit (few inliers) is not a jolt either
    out = impact(closing(), shake(jolt_at=8.0, unreliable=[(7.95, 8.05)]))
    assert out.drafts == [] and out.rejected == {ReasonCode.MOTION_ESTIMATE_MISSING: 1}
    # but a reliable jolt still counts when other estimates in the window are missing
    out = impact(closing(), shake(jolt_at=8.0, missing=[(8.45, 8.55)]))
    assert len(out.drafts) == 1 and measurement(out.drafts[0], "motion_estimates_judged") < 1.0


def test_without_an_earlier_baseline_the_jolt_cannot_be_judged():
    out = impact(closing(), shake(jolt_at=8.0, start=7.2))
    assert out.drafts == [] and out.rejected == {ReasonCode.MOTION_BASELINE_INSUFFICIENT: 1}
    assert out.unresolved[0].reason is ReasonCode.MOTION_BASELINE_INSUFFICIENT


def test_an_approach_whose_lane_is_not_established_gives_an_uncertain_impact():
    out = impact(closing(rel=None), shake(jolt_at=8.0), lanes=None)
    assert len(out.drafts) == 1
    assert out.drafts[0].state is EvidenceState.UNCERTAIN
    assert out.drafts[0].reasons == (ReasonCode.LANE_NOT_ESTABLISHED,)


def test_overlapping_proposals_merge_into_one():
    def two(t):
        return closing(tid=1)(t) + closing(tid=2, x_m=0.3)(t)
    out = impact(two, shake(jolt_at=8.0))
    assert len(out.drafts) == 1 and out.drafts[0].track_ids == (1, 2) and out.rejected == {ReasonCode.MERGED: 1}


def test_impacts_need_tracks_and_camera_motion():
    with pytest.raises(AnalyzerUnavailable, match="no camera-motion estimates"):
        impact(closing(), None)
    recs = [FrameRecord(index=i, t=t, status={Role.DETECTOR: OutputStatus.OK}, detections=())
            for i, t in enumerate(times(1.0))]
    with pytest.raises(AnalyzerUnavailable, match="no tracker output"):
        ImpactAnalyzer().analyze(recs, W, H)


def test_unresolved_patterns_take_only_missing_signal_reasons():
    with pytest.raises(ValueError, match="not a missing-signal reason"):
        UnresolvedPattern(reason=ReasonCode.NO_JOLT, t_start=1.0, t_end=2.0)


# --- contact between other road users (experimental) ------------------------------------------------

from spectra.engine.events import RoadUserContactAnalyzer  # noqa: E402


def pair(z_a=20.0, z_b=20.0, meet_at=3.0, deflect=True, closing_mps=1.5):
    """Two 1.8 m wide vehicles ahead at depths ``z_a`` and ``z_b`` (m). A is centred 1.8 m left; B closes on
    it sideways at ``closing_mps`` until their sides overlap by 0.3 m, from ``meet_at``. With ``deflect``,
    A is then pushed 1.5 m right within 0.2 s."""
    def objects(t):
        xb = max(-0.3, closing_mps * (meet_at - t))  # B's centre; its left side meets A's right side at 0
        xa = -1.8 + (1.5 * min(1.0, (t - meet_at) / 0.2) if deflect and t > meet_at else 0.0)
        return [(1, box_at(xa, z_a), None, None, False, 0.9), (2, box_at(xb, z_b), None, None, False, 0.9)]
    return objects


def contact(objects, motion=None, duration=5.0):
    return RoadUserContactAnalyzer().analyze(frames(times(duration), objects, motion=motion or shake()), W, H)


def test_two_road_users_meeting_with_an_abrupt_change_give_an_experimental_uncertain_proposal():
    out = contact(pair())
    assert len(out.drafts) == 1
    d = out.drafts[0]
    assert (d.type, d.subtype, d.state, d.reasons, d.experimental, d.track_ids) == (
        EventType.IMPACT, EventSubtype.ROAD_USERS, EvidenceState.UNCERTAIN, (ReasonCode.NOT_VALIDATED,), True,
        (1, 2))
    assert measurement(d, "velocity_change_z") >= 10


def test_meeting_without_an_abrupt_change_or_at_different_depths_is_not_contact():
    out = contact(pair(deflect=False))
    assert out.drafts == [] and out.rejected == {ReasonCode.NO_JOLT: 1}
    far = contact(pair(z_b=35.0))  # overlapping in the image, one far behind the other
    assert far.drafts == [] and not far.rejected


def test_road_user_contact_is_not_configured_by_default():
    import inspect

    from spectra.engine import worker
    assert "RoadUserContactAnalyzer" not in inspect.getsource(worker)


def test_theil_sen_matches_scipy_on_seeded_untied_data():
    scipy_stats = pytest.importorskip('scipy.stats')
    rng = np.random.default_rng(912)
    for _ in range(30):
        t = np.cumsum(rng.uniform(0.05, 0.5, 25))
        y = 0.7 * t + rng.normal(size=len(t))
        slope, low, high = theil_sen(t, y, 0.9)
        expected = scipy_stats.theilslopes(y, t, 0.9)
        assert np.allclose([slope, low, high], [expected.slope, expected.low_slope, expected.high_slope])
