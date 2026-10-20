import pytest

from spectra.api.playback import assess
from spectra.engine.contracts import TimestampSource, VideoInfo
from tests.api.conftest import MATROSKA, MP4


def info(container, codec, timestamps=TimestampSource.CONTAINER):
    return VideoInfo(file_name="v", size_bytes=1, sha256="0" * 64, container=container, codec=codec, width=64,
                     height=48, coded_width=64, coded_height=48, rotation=0, timestamp_source=timestamps,
                     assumed_fps=25.0 if timestamps is TimestampSource.ASSUMED_RATE else None)


@pytest.mark.parametrize("container, codec, audio, suffix, expected, mime", [
    (MP4, "h264", ["aac"], ".mp4", "likely", "video/mp4"),
    (MP4, "h264", [], ".MOV", "uncertain", "video/quicktime"),
    (MP4, "h264", ["pcm_s16le"], ".mp4", "uncertain", "video/mp4"),
    (MP4, "hevc", ["aac"], ".mp4", "uncertain", "video/mp4"),
    (MP4, "av1", [], ".mp4", "uncertain", "video/mp4"),
    (MP4, "mpeg4", [], ".mp4", "unsupported", None),
    (MATROSKA, "vp9", ["opus"], ".webm", "likely", "video/webm"),
    (MATROSKA, "vp8", ["vorbis"], ".webm", "likely", "video/webm"),
    (MATROSKA, "av1", [], ".webm", "uncertain", "video/webm"),
    (MATROSKA, "vp9", ["aac"], ".mkv", "uncertain", "video/webm"),
    (MATROSKA, "h264", [], ".mkv", "uncertain", "video/webm"),
    (MATROSKA, "mpeg4", [], ".mkv", "unsupported", None),
    ("ogg", "theora", ["vorbis"], ".ogg", "uncertain", "video/ogg"),
    ("avi", "h264", [], ".avi", "unsupported", None),
    ("mpegts", "h264", [], ".ts", "unsupported", None),
    ("hevc", "hevc", [], ".hevc", "unsupported", None),
])
def test_assessment_uses_probed_container_and_codecs(container, codec, audio, suffix, expected, mime):
    result = assess(info(container, codec), audio, suffix)
    assert (result.assessment, result.mime_type) == (expected, mime)
    assert bool(result.reasons) == (expected != "likely")


def test_extension_never_overrides_the_probed_format():
    assert assess(info("avi", "mpeg4"), [], ".mp4").assessment == "unsupported"
    assert assess(info(MATROSKA, "vp9"), [], ".mp4").mime_type == "video/webm"


def test_streams_without_timestamps_are_unsupported():
    result = assess(info("hevc", "hevc", TimestampSource.ASSUMED_RATE), [], ".hevc")
    assert result.assessment == "unsupported"
    assert "timestamps" in result.reasons[0]
