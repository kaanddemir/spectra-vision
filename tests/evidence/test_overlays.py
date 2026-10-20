import pytest

from spectra.engine.contracts import EgoLaneObservation, EgoLaneState, LaneRelation, OutputStatus, Role
from spectra.evidence.build import build
from spectra.evidence.contracts import ReviewConfig
from spectra.evidence.overlays import frame_at, overlay
from tests.evidence import synth as S


def make(times=None, **kw):
    res = S.result([S.approach(5.0, 7.0)], times=times, **kw)
    _, [pk] = build(res)
    return res, pk


def test_overlay_frames_are_the_analysed_frames_of_the_interval():
    res, pk = make(tracks=lambda t: [(1, False), (7, False)])
    o = overlay(res.perception, pk)
    expected = [f.t for f in res.perception.frames if pk.review.start_s <= f.t <= pk.review.end_s]
    assert [f.t for f in o.frames] == expected  # nothing interpolated, nothing added
    first = o.frames[0]
    assert [(t.track_id, t.related) for t in first.tracks] == [(1, True), (7, False)]
    assert first.lane is not None and first.lane.left.evidence_y == 0.6
    assert (o.width, o.height) == (S.W, S.H)


def test_validity_ends_at_the_next_frame_or_the_maximum_age():
    times = [round(0.1 * i, 3) for i in range(60)] + [round(9 + 0.1 * i, 3) for i in range(60)]
    res, pk = make(times, info=S.video(duration=15.0))
    o = overlay(res.perception, pk, ReviewConfig(overlay_max_age_s=0.25))
    by_t = {f.t: f for f in o.frames}
    assert by_t[5.0].valid_until == 5.1
    assert by_t[5.9].valid_until == 6.15  # a gap follows: shown at most 0.25 s, then nothing
    assert frame_at(o.frames, 5.05).t == 5.0
    assert frame_at(o.frames, 7.0) is None  # inside the gap: no records, no drawing
    assert frame_at(o.frames, 0.0) is None or frame_at(o.frames, 0.0).t >= o.start_s


def test_relations_that_were_not_computed_stay_none():
    res, pk = make(relation=lambda t, i: None if t > 6 else LaneRelation.EGO_LANE)
    o = overlay(res.perception, pk)
    rel = {f.t: f.tracks[0].relation for f in o.frames}
    assert rel[5.0] is LaneRelation.EGO_LANE and rel[6.5] is None


def test_predicted_boxes_are_flagged():
    res, pk = make(tracks=lambda t: [(1, 6.0 <= t <= 6.2)])
    o = overlay(res.perception, pk)
    assert [f.t for f in o.frames if f.tracks[0].predicted] == [6.0, 6.1, 6.2]


def test_overlays_need_the_records_the_package_was_built_from():
    res, pk = make()
    other = S.result(info=S.video(sha256="d" * 64))
    with pytest.raises(ValueError):
        overlay(other.perception, pk)


# --- lane estimation at a coarser interval than tracking ------------------------------------------------

def alternate(perception, scheduled=lambda i, f: i % 2 == 0, change=lambda f: f):
    """Lane estimation scheduled on every other frame only, as with a segment interval of twice the interval;
    ``change`` alters the scheduled frames' records."""
    frames = []
    for i, f in enumerate(perception.frames):
        if scheduled(i, f):
            frames.append(change(f))
        else:
            status = {**f.status, Role.EGO_LANE: OutputStatus.NOT_SCHEDULED}
            frames.append(f.model_copy(update={"status": status, "ego_lane": None, "lane_relations": ()}))
    return perception.model_copy(update={"frames": tuple(frames)})


def lanes_by_t(o):
    return {f.t: (None if f.lane is None else f.lane.observed_t) for f in o.frames}


def test_unscheduled_frames_show_the_previous_lane_with_its_own_time():
    res, pk = make()
    p = alternate(res.perception)
    o = overlay(p, pk, ReviewConfig(overlay_max_age_s=0.25))
    lanes = lanes_by_t(o)
    scheduled = {f.t for f in p.frames if f.status[Role.EGO_LANE] is OutputStatus.OK}
    for f in o.frames:
        if f.t in scheduled:
            assert lanes[f.t] == f.t  # an observation of its own
        else:
            prev = max(t for t in scheduled if t < f.t)
            assert lanes[f.t] == prev  # the earlier observation, never relabelled as new
            assert f.valid_until - prev <= 0.25 + 1e-6
    carried = next(f for f in o.frames if f.t not in scheduled)
    source = next(f for f in p.frames if f.t == lanes[carried.t])
    assert carried.lane.left == source.ego_lane.left and carried.lane.state is source.ego_lane.state


def test_a_lane_is_not_carried_beyond_its_age_limit():
    times = [round(0.1 * i, 3) for i in range(60)] + [round(6.2 + 0.1 * i, 3) for i in range(60)]
    res, pk = make(times, info=S.video(duration=15.0))
    # 5.9 is scheduled, 6.2 is not: drawn from 6.2 the 5.9 lane would be older than 0.25 s.
    p = alternate(res.perception, scheduled=lambda i, f: f.t != 6.2 and i % 2 == 0 or f.t == 5.9)
    o = overlay(p, pk, ReviewConfig(overlay_max_age_s=0.25))
    assert lanes_by_t(o)[6.2] is None
    # With a tighter limit nothing is carried at all: each carried lane would be drawn 0.2 s after it.
    o = overlay(alternate(res.perception), pk, ReviewConfig(overlay_max_age_s=0.15))
    assert all(f.lane is None or f.lane.observed_t == f.t for f in o.frames)


@pytest.mark.parametrize("bad", ["failed", "unavailable", "missing"])
def test_failed_unavailable_or_missing_observations_are_not_carried(bad):
    res, pk = make()
    target = 5.0

    def change(f):
        if f.t != target:
            return f
        if bad == "failed":
            return f.model_copy(update={"status": {**f.status, Role.EGO_LANE: OutputStatus.FAILED}, "ego_lane": None})
        if bad == "unavailable":
            return f.model_copy(update={"ego_lane": EgoLaneObservation(state=EgoLaneState.UNAVAILABLE)})
        return f.model_copy(update={"ego_lane": None})

    o = overlay(alternate(res.perception, change=change), pk, ReviewConfig(overlay_max_age_s=0.25))
    lanes = {f.t: f.lane for f in o.frames}
    assert lanes[5.1] is None  # the frame after shows no lane: nothing valid to carry
    assert lanes[5.3].observed_t == 5.2  # a later valid observation is carried as usual
    if bad == "unavailable":
        assert lanes[5.0].state is EgoLaneState.UNAVAILABLE  # the observation itself is still shown


def test_records_with_lane_estimation_on_every_frame_are_unchanged():
    res, pk = make()
    o = overlay(res.perception, pk)
    assert all(f.lane is not None and f.lane.observed_t == f.t for f in o.frames)
