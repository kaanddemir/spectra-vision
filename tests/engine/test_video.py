import hashlib

import numpy as np
import pytest

from spectra.engine.contracts import TimestampSource
from spectra.engine.video import CorruptVideo, MissingTimestamps, UnsupportedVideo, VideoSource, file_sha256
from tests.engine.media import H, W, cfr_times, make_video, read_index

VFR = [0.0, 0.1, 0.2, 0.5, 0.55, 1.6, 1.7, 1.8, 1.9, 2.0]


@pytest.fixture
def cfr(tmp_path):
    return make_video(tmp_path / "cfr.mp4", cfr_times(60, 30))


@pytest.fixture
def vfr(tmp_path):
    return make_video(tmp_path / "vfr.mkv", VFR, fmt="matroska")


def test_metadata_of_a_constant_rate_file(cfr):
    info = VideoSource(cfr).info
    assert (info.width, info.height, info.rotation) == (W, H, 0)
    assert info.timestamp_source is TimestampSource.CONTAINER
    assert info.duration_s == pytest.approx(2.0, abs=0.05)
    assert len(info.sha256) == 64 and info.size_bytes == cfr.stat().st_size


def test_every_frame_in_presentation_order_with_true_timestamps(cfr):
    src = VideoSource(cfr)
    frames = list(src.frames())
    assert [f.t for f in frames] == pytest.approx(cfr_times(60, 30), abs=1e-3)
    assert [read_index(f.image) for f in frames] == list(range(60))
    assert [f.decode_index for f in frames] == list(range(60))
    report = src.stats.report(len(frames))
    assert report.variable_frame_rate is False and report.frames_analysed == 60


@pytest.mark.parametrize("codec,options", [("mpeg4", None), ("libx264", {"bf": "2"})])
def test_variable_frame_rate_timestamps_survive_reordering(tmp_path, codec, options):
    path = make_video(tmp_path / f"{codec}.mp4", VFR, codec=codec, options=options)
    src = VideoSource(path)
    frames = list(src.frames())
    assert [f.t for f in frames] == pytest.approx(VFR, abs=1e-3)
    assert [read_index(f.image) for f in frames] == list(range(len(VFR)))
    report = src.stats.report(len(frames))
    assert report.variable_frame_rate is True and report.max_interval_s == pytest.approx(1.05, abs=1e-3)


def test_sampling_follows_a_timestamp_grid_without_bursts(vfr):
    src = VideoSource(vfr)
    got = [f.t for f in src.frames(interval_s=0.25)]
    # grid 0, .25, .5, .75 ... : first frame at or after each; the 0.55-1.6 gap yields one frame
    assert got == pytest.approx([0.0, 0.5, 1.6, 1.8, 2.0], abs=1e-3)
    assert src.stats.report(len(got)).frames_analysed == 5


def test_sampling_is_deterministic(vfr):
    a = [(f.t, read_index(f.image)) for f in VideoSource(vfr).frames(interval_s=0.3)]
    b = [(f.t, read_index(f.image)) for f in VideoSource(vfr).frames(interval_s=0.3)]
    assert a == b


def test_span_selection_uses_timestamps(vfr):
    got = [f.t for f in VideoSource(vfr).frames(start_s=0.52, end_s=1.85)]
    assert got == pytest.approx([0.55, 1.6, 1.7, 1.8], abs=1e-3)



def test_frame_times_begin_at_a_keyframe_before_the_start_and_skip_pixels(tmp_path):
    times = cfr_times(60, 30)
    path = make_video(tmp_path / "gop.mp4", times, gop=12)
    got = list(VideoSource(path).frame_times(1.0, 1.5))
    assert got[0][2] and got[0][0] <= 1.0 < got[0][0] + 12 / 30  # the keyframe decoding starts from
    assert [round(t, 3) for t, _, _ in got] == [t for t in times if got[0][0] - 1e-3 <= t <= 1.5]
    assert all(pts is not None for _, pts, _ in got)
    whole = list(VideoSource(path).frame_times())
    assert whole[0][2] and len(whole) == len(times)

@pytest.mark.parametrize("t,expected", [(0.0, 0), (0.15, 1), (0.54, 3), (1.59, 4), (1.6, 5), (2.0, 9)])
def test_frame_at_returns_the_frame_displayed_at_that_time(vfr, t, expected):
    f = VideoSource(vfr).frame_at(t)
    assert read_index(f.image) == expected and f.t == pytest.approx(VFR[expected], abs=1e-3)


def test_frame_at_matches_linear_decoding_for_every_frame(tmp_path):
    times = cfr_times(90, 30)
    path = make_video(tmp_path / "long.mp4", times, gop=12)
    src = VideoSource(path)
    linear = {round(f.t, 3): f.image for f in src.frames()}
    for t in times[::7]:
        f = src.frame_at(t)
        assert f.t == pytest.approx(t, abs=1e-3)
        assert np.array_equal(f.image, linear[round(t, 3)])


def test_frame_at_rejects_times_outside_the_video(cfr):
    src = VideoSource(cfr)
    with pytest.raises(ValueError):
        src.frame_at(-0.1)
    with pytest.raises(ValueError):
        src.frame_at(30.0)


def test_rotation_metadata_is_applied(tmp_path):
    path = make_video(tmp_path / "rot.mp4", cfr_times(3, 10), rotation=90)
    src = VideoSource(path)
    assert (src.info.rotation, src.info.width, src.info.height) == (90, H, W)
    img = next(src.frames()).image
    assert img.shape == (W, H, 3)
    # counter-clockwise: the top-left marker moves to the bottom-left
    assert img[-4, 4, 2] > 200 and img[4, 4, 2] < 50


def test_downscaling_reports_scale_and_keeps_aspect(cfr):
    f = next(VideoSource(cfr).frames(max_width=64))
    assert f.image.shape == (32, 64, 3) and f.scale == pytest.approx(0.5)


def test_streams_without_timestamps_need_an_explicit_rate(tmp_path):
    path = make_video(tmp_path / "raw.h264", cfr_times(10, 20), codec="libx264", fmt="h264")
    with pytest.raises(MissingTimestamps):
        VideoSource(path)
    src = VideoSource(path, assumed_fps=20)
    assert src.info.timestamp_source is TimestampSource.ASSUMED_RATE
    assert [f.t for f in src.frames()] == pytest.approx([i / 20 for i in range(10)])
    f = src.frame_at(0.31)
    assert read_index(f.image) == 6 and f.decode_index == 6


def test_missing_file_and_non_video_inputs(tmp_path):
    with pytest.raises(UnsupportedVideo):
        VideoSource(tmp_path / "absent.mp4")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(np.random.default_rng(0).integers(0, 256, 4096, dtype=np.uint8).tobytes())
    with pytest.raises(CorruptVideo):
        VideoSource(junk)
    text = tmp_path / "notes.mov"
    text.write_text("not a video")
    with pytest.raises(CorruptVideo):
        VideoSource(text)


def test_audio_only_file_is_unsupported(tmp_path):
    import av
    path = tmp_path / "audio.wav"
    with av.open(str(path), "w") as c:
        s = c.add_stream("pcm_s16le", rate=8000)
        frame = av.AudioFrame.from_ndarray(np.zeros((1, 800), np.int16), format="s16", layout="mono")
        frame.rate = 8000
        for pk in s.encode(frame):
            c.mux(pk)
    with pytest.raises(UnsupportedVideo, match="no video stream"):
        VideoSource(path)


def test_truncated_mp4_is_reported_as_corrupt(tmp_path, cfr):
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(cfr.read_bytes()[: cfr.stat().st_size // 2])  # index (moov) is at the end
    with pytest.raises(CorruptVideo):
        VideoSource(cut)


def test_truncated_stream_yields_the_intact_frames_and_says_it_ended_early(tmp_path):
    full = make_video(tmp_path / "full.mkv", cfr_times(150, 30), fmt="matroska")
    cut = tmp_path / "cut.mkv"
    cut.write_bytes(full.read_bytes()[: int(full.stat().st_size * 0.6)])
    src = VideoSource(cut)
    frames = list(src.frames())
    assert 30 < len(frames) < 150
    assert [read_index(f.image) for f in frames] == list(range(len(frames)))
    assert "container declares" in src.stats.report(len(frames)).ended_early


def test_damaged_middle_is_reported_and_never_reorders_frames(tmp_path):
    full = make_video(tmp_path / "full.mkv", cfr_times(150, 30), fmt="matroska")
    data = bytearray(full.read_bytes())
    mid = len(data) // 2
    data[mid:mid + 400] = bytes(400)
    damaged = tmp_path / "damaged.mkv"
    damaged.write_bytes(bytes(data))
    src = VideoSource(damaged)
    frames = list(src.frames())
    times = [f.t for f in frames]
    assert times == sorted(set(times)) and times[0] == 0.0
    assert [read_index(f.image) for f in frames] == [round(t * 30) for t in times]  # no frame mislabelled
    report = src.stats.report(len(frames))
    assert len(frames) < 150
    # Matroska resynchronises past the damage without an error; the loss shows as a gap in time.
    assert report.ended_early or report.decode_errors or report.max_interval_s > 1.0


def test_complete_file_is_not_flagged(tmp_path):
    src = VideoSource(make_video(tmp_path / "ok.mkv", cfr_times(150, 30), fmt="matroska"))
    frames = list(src.frames())
    assert len(frames) == 150 and src.stats.report(150).ended_early is None
    list(src.frames(end_s=1.0))
    assert src.stats.ended_early is None


class _OvershootingSeek:
    """Container proxy whose seek lands later than requested, as keyframe seeking can on real files."""

    def __init__(self, container, overshoot):
        self._c, self._overshoot = container, overshoot

    def seek(self, offset, **kw):
        self._c.seek(offset + self._overshoot, **kw)

    def __getattr__(self, name):
        return getattr(self._c, name)


def test_inaccurate_seek_is_detected_and_retried(tmp_path, monkeypatch):
    import spectra.engine.video as video
    path = make_video(tmp_path / "gop.mp4", cfr_times(150, 30), gop=15)
    src = VideoSource(path)
    linear = {round(f.t, 3): f.image for f in src.frames()}
    real_open, calls = video._open, []

    def open_overshooting(p):
        container, stream = real_open(p)
        calls.append(1)
        return _OvershootingSeek(container, int(3 / stream.time_base)), stream  # 3 s too late

    monkeypatch.setattr(video, "_open", open_overshooting)
    f = src.frame_at(2.0)
    assert f.t == pytest.approx(2.0, abs=1e-3) and np.array_equal(f.image, linear[2.0])
    assert len(calls) > 1  # first attempt overshot and was abandoned


def test_iteration_releases_the_container_when_abandoned(cfr):
    import gc
    src = VideoSource(cfr)
    it = src.frames()
    next(it)
    it.close()
    gc.collect()
    assert list(f.t for f in src.frames(end_s=0.1)) == pytest.approx([0.0, 1 / 30, 2 / 30, 0.1], abs=1e-3)


def test_file_hash_is_streamed_sha256(tmp_path):
    path = tmp_path / "data"
    data = b"abc" * 800000  # larger than one read block
    path.write_bytes(data)
    assert file_sha256(path) == hashlib.sha256(data).hexdigest()
