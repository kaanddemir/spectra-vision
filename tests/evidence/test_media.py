import pytest

from spectra.engine.contracts import TimestampSource
from spectra.engine.runner import CancelToken
from spectra.engine.video import VideoSource, file_sha256
from spectra.evidence.contracts import ReviewInterval
from spectra.evidence.media import (ClipCancelled, ClipUnsupported, SourceStatus, SourceUnavailable, check_source,
                                    extract_clip, load_clip, resolve)
from tests.engine.media import make_video, read_index

# 25 fps with B-frames, starting at 0.5 s of container time, with a 1.5 s hole after 2.0 s
TIMES = [round(0.5 + i / 25, 3) for i in range(40)] + [round(3.6 + i / 25, 3) for i in range(40)]


def iv(a, b):
    return ReviewInterval(start_s=a, end_s=b, evidence_start_s=a, evidence_end_s=b)


@pytest.fixture
def clip(tmp_path):
    return make_video(tmp_path / "src.mp4", TIMES, gop=12, options={"bf": "2"})


def source_times(path, a=0.0, b=None):
    return [t for t, _, _ in VideoSource(path).frame_times(a, b)]


def test_source_identity_is_checked(clip, tmp_path):
    info = VideoSource(clip).info
    assert check_source(clip, info).status is SourceStatus.AVAILABLE
    assert check_source(tmp_path / "nope.mp4", info).status is SourceStatus.MISSING
    other = make_video(tmp_path / "other.mp4", TIMES[:-1])
    assert check_source(other, info).status is SourceStatus.CHANGED
    with pytest.raises(SourceUnavailable) as e:
        resolve(tmp_path / "nope.mp4", info, iv(0.5, 1.0))
    assert e.value.check.status is SourceStatus.MISSING


def test_resolve_reports_frames_keyframe_and_gaps_in_presentation_time(clip):
    info = VideoSource(clip).info
    assert info.start_s == 0.5  # times below are relative to the stream start, never frame / fps
    r = resolve(clip, info, iv(1.21, 3.5), gap_s=0.5)
    t = source_times(clip)
    inside = [x for x in t if 1.21 <= x <= 3.5]
    assert (r.first_frame_s, r.last_frame_s, r.frames) == (inside[0], inside[-1], len(inside))
    assert r.displayed_at_start_s == max(x for x in t if x <= 1.21)
    assert r.keyframe_s is not None and r.keyframe_s <= r.displayed_at_start_s
    assert [(g.start_s, g.end_s) for g in r.gaps] == [(1.56, 3.1)]
    assert r.stream_start_s == 0.5 and r.timestamp_source is TimestampSource.CONTAINER


def test_clip_is_a_stream_copy_with_exact_source_times(clip, tmp_path):
    info = VideoSource(clip).info
    before = file_sha256(clip)
    meta = extract_clip(clip, info, iv(1.0, 1.4), tmp_path / "clips", "e1")
    assert file_sha256(clip) == before  # the source is untouched
    assert meta.origin_s <= 1.0 and meta.last_s >= max(x for x in source_times(clip) if x <= 1.4)
    got = source_times(tmp_path / "clips" / meta.file)
    expected = source_times(clip, meta.origin_s, meta.last_s)
    assert [round(meta.origin_s + x - got[0], 6) for x in got] == [round(x, 6) for x in expected]
    # same pictures: the frame shown at a source time is the clip's frame at origin-relative time
    src = VideoSource(clip).frame_at(1.2)
    out = VideoSource(tmp_path / "clips" / meta.file).frame_at(round(src.t - meta.origin_s, 6))
    assert read_index(out.image) == read_index(src.image)
    assert sorted(p.name for p in (tmp_path / "clips").iterdir()) == ["e1.json", meta.file]


def test_clip_starting_after_a_gap_and_at_a_non_keyframe(clip, tmp_path):
    info = VideoSource(clip).info
    meta = extract_clip(clip, info, iv(3.3, 3.5), tmp_path, "gap")
    assert meta.origin_s <= 3.3 and meta.frames == len(source_times(tmp_path / meta.file))


def test_stored_clip_is_reused_only_while_it_verifies(clip, tmp_path):
    info = VideoSource(clip).info
    meta = extract_clip(clip, info, iv(1.0, 1.4), tmp_path, "e1")
    assert load_clip(tmp_path, "e1", info.sha256, iv(1.0, 1.4)) == meta
    assert load_clip(tmp_path, "e1", info.sha256, iv(1.0, 1.5)) is None
    (tmp_path / meta.file).write_bytes(b"x")
    assert load_clip(tmp_path, "e1", info.sha256, iv(1.0, 1.4)) is None


def test_cancelled_extraction_leaves_no_files(clip, tmp_path):
    info = VideoSource(clip).info
    token = CancelToken()
    token.cancel()
    with pytest.raises(ClipCancelled):
        extract_clip(clip, info, iv(1.0, 1.4), tmp_path / "out", "e1", cancel=token)
    assert list((tmp_path / "out").iterdir()) == []


def test_sources_without_timestamps_and_long_intervals_are_refused(clip, tmp_path):
    info = VideoSource(clip).info
    assumed = info.model_copy(update={"timestamp_source": TimestampSource.ASSUMED_RATE, "assumed_fps": 25.0})
    with pytest.raises(ClipUnsupported):
        extract_clip(clip, assumed, iv(1.0, 1.4), tmp_path, "e1")
    with pytest.raises(ClipUnsupported):
        extract_clip(clip, info, iv(0.0, 5.0), tmp_path, "e1", max_duration_s=2.0)
    with pytest.raises(SourceUnavailable):
        extract_clip(tmp_path / "missing.mp4", info, iv(1.0, 1.4), tmp_path, "e1")
    assert not any(tmp_path.glob("*.partial*"))


def test_webm_vp9_source_is_copied_into_webm(tmp_path):
    src = make_video(tmp_path / "v.webm", [round(i / 10, 3) for i in range(30)], codec="libvpx-vp9", gop=10)
    info = VideoSource(src).info
    meta = extract_clip(src, info, iv(1.25, 2.0), tmp_path / "c", "v")
    assert meta.format == "webm" and meta.file.endswith(".webm") and meta.origin_s == 1.0
