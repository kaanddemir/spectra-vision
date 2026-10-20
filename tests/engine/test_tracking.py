import numpy as np
import pytest

from spectra.engine.contracts import Box, Detection, ObjectCategory
from spectra.engine.video import Frame

pytest.importorskip("trackers")
from spectra.engine.tracking import BoTSortTracker  # noqa: E402


def car(x, y=100.0, w=60.0, h=40.0, score=0.9):
    return Detection(box=Box(x1=x, y1=y, x2=x + w, y2=y + h), score=score, category=ObjectCategory.CAR)


def frame(t, scale=1.0, size=(360, 640)):
    h, w = round(size[0] * scale), round(size[1] * scale)
    return Frame(decode_index=None, t=t, pts=None, image=np.zeros((h, w, 3), np.uint8), scale=scale)


def run(tracker, steps):
    """steps: (t, detections) -> list of track ids per step, aligned with the detections."""
    tracker.load()
    out = []
    for t, dets in steps:
        tracked = tracker.update(frame(t), dets)
        by_box = {o.box: o.track_id for o in tracked}
        out.append([by_box.get(d.box) for d in dets])
    return out


def plain(**kw):
    return BoTSortTracker(camera_motion=False, **kw)


def test_moving_vehicle_keeps_one_identity_at_irregular_intervals():
    times = [0.0, 0.1, 0.15, 0.4, 0.45, 0.9, 1.0, 1.3]
    ids = run(plain(), [(t, [car(100 + 80 * t)]) for t in times])
    confirmed = [i[0] for i in ids if i[0] is not None]
    assert len(set(confirmed)) == 1 and len(confirmed) >= len(times) - 1


@pytest.mark.parametrize("step", [0.1, 0.5])
def test_short_gap_is_bridged_and_long_gap_is_not_whatever_the_sampling_rate(step):
    def scenario(gap):
        seen = [round(k * step, 3) for k in range(int(1.0 / step) + 1)]
        back = [round(seen[-1] + gap + k * step, 3) for k in range(3)]
        tracker = plain(reference_rate=1 / step)
        steps = [(t, [car(100)]) for t in seen]
        steps += [(t, []) for t in np.arange(seen[-1] + step, back[0] - 1e-9, step)]
        steps += [(t, [car(100)]) for t in back]
        ids = run(tracker, steps)
        return ids[0][0], ids[-1][0]

    first, after_short = scenario(1.5)
    assert first is not None and after_short == first  # retained for up to 2 s of source time
    first, after_long = scenario(3.0)
    assert after_long is not None and after_long != first


def test_two_vehicles_side_by_side_keep_separate_identities():
    steps = [(0.1 * k, [car(100 + 2 * k), car(300 - 2 * k)]) for k in range(15)]
    ids = run(plain(), steps)
    left, right = {i[0] for i in ids[2:]}, {i[1] for i in ids[2:]}
    assert len(left) == len(right) == 1 and left != right


def test_boxes_are_reported_in_source_pixels_when_frames_are_downscaled():
    tracker = plain()
    tracker.load()
    d = car(400, 200, 120, 80)
    out = []
    for k in range(4):
        out = tracker.update(frame(0.1 * k, scale=0.5), [d])
    assert out and out[0].box == d.box and out[0].score == d.score


def test_no_detections_and_reset():
    tracker = plain()
    assert run(tracker, [(0.0, []), (0.1, [])]) == [[], []]
    ids = run(tracker, [(0.2 + 0.1 * k, [car(100)]) for k in range(3)])
    tracker.reset()
    again = run(tracker, [(0.0 + 0.1 * k, [car(100)]) for k in range(3)])
    assert ids[-1] == again[-1]  # identities restart after reset


def test_camera_motion_compensation_runs_on_textured_frames():
    rng = np.random.default_rng(0)
    texture = rng.integers(0, 255, (400, 700, 3), dtype=np.uint8)
    tracker = BoTSortTracker()
    tracker.load()
    ids = []
    for k in range(10):  # the whole scene pans 6 px per frame, and so does the vehicle
        img = np.ascontiguousarray(texture[20:380, 6 * k:6 * k + 640])
        f = Frame(decode_index=k, t=0.1 * k, pts=None, image=img)
        out = tracker.update(f, [car(200 - 6 * k)])
        ids.append(out[0].track_id if out else None)
    assert len({i for i in ids if i is not None}) == 1


def test_identity_reflects_configuration():
    assert plain().info != plain(lost_seconds=1.0).info
    assert BoTSortTracker().info.version.startswith("trackers-")


def _panning(scale=1.0, frames=6, step=6):
    rng = np.random.default_rng(0)
    texture = rng.integers(0, 255, (400, 700, 3), dtype=np.uint8)
    out = []
    for k in range(frames):
        img = np.ascontiguousarray(texture[20:380, step * k:step * k + 640])
        if scale != 1.0:
            import cv2
            img = cv2.resize(img, (round(640 * scale), round(360 * scale)), interpolation=cv2.INTER_AREA)
        out.append(Frame(decode_index=k, t=0.1 * k, pts=None, image=img, scale=scale))
    return out


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_camera_motion_is_reported_in_source_pixels_from_the_previous_update(scale):
    tracker = BoTSortTracker()
    tracker.load()
    seen = []
    for f in _panning(scale):
        tracker.update(f, [car(100, 300)])
        seen.append(tracker.camera_motion())
    assert seen[0] is None  # nothing to compare the first frame with
    for k, m in enumerate(seen[1:], 1):
        assert m.since_t == pytest.approx(0.1 * (k - 1))
        assert m.scale == pytest.approx(1.0, abs=0.01) and abs(m.rotation) < 0.01
        assert m.tx == pytest.approx(-6.0, abs=0.6) and abs(m.ty) < 0.6  # the scene moves 6 source px left


def test_motion_after_an_empty_stretch_is_measured_from_the_last_estimated_frame():
    tracker = BoTSortTracker(lost_seconds=0.1)
    tracker.load()
    frames = _panning(frames=8, step=2)
    got = []
    for k, f in enumerate(frames):  # frames 2-5: nothing to track, so the library skips its estimator
        tracker.update(f, [car(100, 300)] if k < 2 or k > 5 else [])
        got.append(tracker.camera_motion())
    assert got[1].since_t == pytest.approx(0.0)
    assert got[6].since_t == pytest.approx(0.1 * last_estimated(got, tracker))
    assert got[6].tx == pytest.approx(-2.0 * (6 - last_estimated(got, tracker)), abs=0.6)


def last_estimated(got, tracker):
    """Index of the last frame before frame 6 on which the estimator ran (it still runs while lost tracks
    remain, so this is not always frame 1)."""
    return max(k for k in range(6) if k <= 1 or got[k] is not None)


def test_no_camera_motion_without_texture_when_disabled_or_after_reset():
    tracker = BoTSortTracker()
    tracker.load()
    for k in range(3):  # featureless frames: the estimator fails and returns identity
        tracker.update(frame(0.1 * k), [])
        assert tracker.camera_motion() is None
    tracker.reset()
    assert tracker.camera_motion() is None
    off = plain()
    off.load()
    for f in _panning():
        off.update(f, [])
        assert off.camera_motion() is None


def test_reporting_camera_motion_leaves_tracking_unchanged():
    def ids(report):
        tracker = BoTSortTracker()
        tracker.load()
        if not report:  # the library's own estimator, without the adapter's subclass or wrapper
            from trackers.utils.cmc import CMC
            tracker._tracker.cmc = CMC(tracker._tracker.cmc.cfg)
        out = []
        for k, f in enumerate(_panning(frames=10)):
            got = tracker.update(f, [car(200 - 6 * k), car(420 - 6 * k, 200)])
            out.append([(o.track_id, o.box) for o in got])
        return out
    assert ids(True) == ids(False)


def _sequences():
    """Frame sequences exercising every branch of the sparse-flow estimator: panning, zooming, independent
    noise images (correspondences that disagree), featureless frames and a frame-size change."""
    import cv2
    rng = np.random.default_rng(4)
    texture = rng.integers(0, 255, (400, 700, 3), dtype=np.uint8)
    pan = [np.ascontiguousarray(texture[20:380, 5 * k:5 * k + 640]) for k in range(6)]
    zoom = [cv2.resize(texture[20 + 4 * k:380 - 4 * k, 30 + 7 * k:670 - 7 * k], (640, 360)) for k in range(6)]
    noise = [rng.integers(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(4)]
    flat = [np.zeros((360, 640, 3), np.uint8)] * 3
    resized = [pan[0], cv2.resize(pan[1], (320, 180)), cv2.resize(pan[2], (320, 180))]
    return [pan, zoom, noise, flat + pan[:3], resized]


def test_quality_estimator_returns_exactly_the_library_matrices():
    from trackers.utils.cmc import CMC, CMCConfig

    from spectra.engine.tracking import _quality_cmc_class
    for downscale in (1, 2):
        for seq in _sequences():
            ours, theirs = _quality_cmc_class()(CMCConfig(downscale=downscale)), CMC(CMCConfig(downscale=downscale))
            for img in seq:
                a, b = ours.estimate(img), theirs.estimate(img)
                assert a.dtype == b.dtype and np.array_equal(a, b)
                assert ours.frames_failed == theirs.frames_failed


def test_camera_motion_carries_fit_quality():
    tracker = BoTSortTracker()
    tracker.load()
    seen = []
    for f in _panning(0.5):
        tracker.update(f, [car(100, 300)])
        seen.append(tracker.camera_motion())
    for m in seen[1:]:
        assert 0 < m.inliers <= m.matches and m.inliers > 0.8 * m.matches
        assert m.residual_px < 1.0  # source pixels: the pan is exact
    from trackers.utils.cmc import CMCConfig

    from spectra.engine.tracking import _quality_cmc_class
    cmc = _quality_cmc_class()(CMCConfig())
    noise = _sequences()[2]
    cmc.estimate(noise[0])
    cmc.estimate(noise[1])
    matches, inliers, residual = cmc.quality  # unrelated images: few correspondences agree
    assert inliers < 0.5 * matches and residual > 1.0
    cmc.estimate(np.zeros((360, 640, 3), np.uint8))
    assert cmc.quality is None  # no fit, no quality


def test_camera_motion_quality_is_pinned_to_the_copied_library_version(monkeypatch):
    import spectra.engine.tracking as tracking
    monkeypatch.setattr(tracking, "_version", lambda package: "9.9.9" if package == "trackers" else "1")
    with pytest.raises(RuntimeError, match="implemented for trackers 2.6.1"):
        BoTSortTracker().load()
    plain().load()  # without camera motion nothing depends on the copy
