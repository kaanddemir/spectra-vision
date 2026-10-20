"""Source-video access for review: identity checks, interval references and optional stream-copied clips.

The default is a reference: the source file, identified by its hash, and a span of presentation time to
play. ``resolve`` checks that span against the frames the file actually decodes to (first and last frame
times, the keyframe a player must start decoding from, gaps, decode errors) without converting pixels.

A clip file is made only when a separate file is needed. It is a stream copy: packets from the keyframe
at or before the interval start to the interval end, written to a new file without decoding or
re-encoding, so the pictures are the source's own and no encoder is involved. The clip therefore starts
at that keyframe, ``origin_s``, which is up to one keyframe interval before the review interval; clip time
``c`` is source time ``origin_s + c``. Every clip is decoded once after writing and its frame times are
compared with the source's; a clip that does not match is deleted and an error raised. Audio is not
copied. Whether a browser can play the codec is not decided here: a stream copy keeps the source codec.

Sources without container timestamps (times assume a frame rate) cannot be cut with correct times and are
refused. The source file is only ever opened for reading.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator

from spectra.engine.contracts import TimestampSource, VideoInfo, canonical_json
from spectra.engine.video import EPS, SEEK_BACKOFF_S, VideoError, VideoSource, file_sha256
from spectra.evidence.contracts import EVIDENCE_VERSION, Gap, ReviewInterval, _Strict

TIME_TOLERANCE_S = 1e-4  # timestamps are compared to well below one frame


class SourceStatus(str, Enum):
    AVAILABLE = "available"
    MISSING = "missing"  # no file at the path
    CHANGED = "changed"  # a file is there, but not the analysed one (size or content hash differs)
    UNREADABLE = "unreadable"  # the analysed file, but it cannot be decoded here


class SourceCheck(_Strict):
    status: SourceStatus
    message: Optional[str] = None


def check_source(path: Path | str, video: VideoInfo, verify_hash: bool = True) -> SourceCheck:
    """Whether ``path`` is the analysed video. Hashing reads the whole file once; skip it only when the
    caller has just verified the same file."""
    path = Path(path)
    if not path.is_file():
        return SourceCheck(status=SourceStatus.MISSING, message=f"video file not found: {path.name}")
    if path.stat().st_size != video.size_bytes or (verify_hash and file_sha256(path) != video.sha256):
        return SourceCheck(status=SourceStatus.CHANGED, message=f"{path.name} is not the analysed video")
    return SourceCheck(status=SourceStatus.AVAILABLE)


class ResolvedInterval(_Strict):
    """A review interval checked against the decoded source. Times are source presentation times."""
    sha256: str
    start_s: float = Field(ge=0, description="requested")
    end_s: float = Field(ge=0)
    first_frame_s: Optional[float] = Field(default=None, ge=0, description="first frame shown at or after start")
    displayed_at_start_s: Optional[float] = Field(default=None, ge=0,
                                                  description="frame on screen at start: the latest at or before it")
    last_frame_s: Optional[float] = Field(default=None, ge=0, description="last frame at or before end")
    keyframe_s: Optional[float] = Field(default=None, ge=0, description="where decoding must begin to show start")
    frames: int = Field(default=0, ge=0, description="frames in [start, end]")
    max_frame_interval_s: Optional[float] = Field(default=None, ge=0)
    gaps: tuple[Gap, ...] = Field(default=(), description="stretches without decoded frames longer than gap_s")
    decode_errors: int = Field(default=0, ge=0)
    dropped_timestamps: int = Field(default=0, ge=0, description="frames dropped for missing or repeated times")
    ended_early: Optional[str] = None
    stream_start_s: float = Field(description="container time of t = 0: add it for container-level seeking")
    timestamp_source: TimestampSource


def resolve(path: Path | str, video: VideoInfo, interval: ReviewInterval, gap_s: float = 1.0,
            verify_hash: bool = True) -> ResolvedInterval:
    """Decode the interval's timestamps (not its pixels) and report what a player will show."""
    check = check_source(path, video, verify_hash)
    if check.status is not SourceStatus.AVAILABLE:
        raise SourceUnavailable(check)
    try:
        source = VideoSource(path, assumed_fps=video.assumed_fps)
        frames = list(source.frame_times(interval.start_s, interval.end_s))
    except VideoError as exc:
        raise SourceUnavailable(SourceCheck(status=SourceStatus.UNREADABLE, message=str(exc))) from exc
    stats = source.stats
    before = [f for f in frames if f[0] <= interval.start_s + EPS]
    inside = [f for f in frames if f[0] >= interval.start_s - EPS]
    keys = [t for t, _, key in before if key]
    times = [t for t, _, _ in inside]
    steps = [b - a for a, b in zip(times, times[1:])]
    gaps = tuple(Gap(start_s=a, end_s=b) for a, b in zip(times, times[1:]) if b - a > gap_s + EPS)
    return ResolvedInterval(
        sha256=video.sha256, start_s=interval.start_s, end_s=interval.end_s,
        first_frame_s=times[0] if times else None, displayed_at_start_s=before[-1][0] if before else None,
        last_frame_s=times[-1] if times else None, keyframe_s=keys[-1] if keys else (frames[0][0] if frames else None),
        frames=len(times), max_frame_interval_s=round(max(steps), 6) if steps else None, gaps=gaps,
        decode_errors=stats.decode_errors, dropped_timestamps=stats.missing_timestamps + stats.non_monotonic,
        ended_early=stats.ended_early, stream_start_s=video.start_s, timestamp_source=video.timestamp_source)


class SourceUnavailable(Exception):
    def __init__(self, check: SourceCheck) -> None:
        super().__init__(check.message or check.status.value)
        self.check = check


class ClipUnsupported(Exception):
    """A clip cannot be cut from this source with correct times; play the source interval instead."""


class ClipCancelled(Exception):
    pass


# --- stream-copied clips --------------------------------------------------------------------------


_FORMATS = {"h264": ("mp4", ".mp4"), "hevc": ("mp4", ".mp4"), "mpeg4": ("mp4", ".mp4"), "av1": ("mp4", ".mp4"),
            "vp8": ("webm", ".webm"), "vp9": ("webm", ".webm")}
_DEFAULT_FORMAT = ("matroska", ".mkv")  # holds nearly any codec


class ClipFile(_Strict):
    """A stream-copied clip of the source. ``origin_s`` is the source time of clip time zero."""
    evidence_version: str = EVIDENCE_VERSION
    file: str = Field(description="file name, next to this record")
    sha256: str
    size_bytes: int = Field(ge=0)
    format: str
    codec: str = Field(description="the source codec, copied; browser support is not implied")
    source_sha256: str
    start_s: float = Field(ge=0, description="requested review interval")
    end_s: float = Field(ge=0)
    origin_s: float = Field(ge=0, description="source time of the clip's first frame, a keyframe at or before "
                                              "start_s")
    first_s: float = Field(ge=0, description="source time of the clip's first frame")
    last_s: float = Field(ge=0, description="source time of the clip's last frame; may pass end_s by the frames "
                                            "a decoder needs to show end_s")
    frames: int = Field(ge=1)

    @field_validator("file")
    @classmethod
    def _file_name(cls, value: str) -> str:
        if not value or value in (".", "..") or any(c in value for c in ("/", "\\", "\x00")):
            raise ValueError("clip file must be a single file name")
        return value


def clip_format(codec: str) -> tuple[str, str]:
    return _FORMATS.get(codec, _DEFAULT_FORMAT)


def extract_clip(path: Path | str, video: VideoInfo, interval: ReviewInterval, out_dir: Path | str, name: str,
                 *, cancel=None, max_duration_s: float = 120.0, disk_reserve_mb: float = 512.0,
                 verify_hash: bool = True) -> ClipFile:
    """Write ``<out_dir>/<name><ext>`` and its record ``<name>.json``, atomically. Raises ``ClipUnsupported``
    for sources without container timestamps or over-long intervals, ``SourceUnavailable`` when the
    source is missing or changed, and ``ClipCancelled`` when ``cancel`` is set; no partial file remains."""
    if video.timestamp_source is not TimestampSource.CONTAINER:
        raise ClipUnsupported("the source has no frame timestamps; a clip could not keep correct times")
    if interval.end_s - interval.start_s > max_duration_s:
        raise ClipUnsupported(f"interval longer than {max_duration_s:g} s")
    check = check_source(path, video, verify_hash)
    if check.status is not SourceStatus.AVAILABLE:
        raise SourceUnavailable(check)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt, ext = clip_format(video.codec)
    span = interval.end_s - interval.start_s + SEEK_BACKOFF_S[1]
    estimate = video.size_bytes * span / video.duration_s if video.duration_s else video.size_bytes
    if shutil.disk_usage(out_dir).free < estimate + disk_reserve_mb * 2 ** 20:
        raise ClipUnsupported("not enough free disk space for the clip")
    final, record = out_dir / f"{name}{ext}", out_dir / f"{name}.json"
    tmp = out_dir / f".{name}.{uuid.uuid4().hex}.partial{ext}"
    try:
        origin = _copy(Path(path), video, interval, tmp, fmt, cancel)
        clip = _verify(Path(path), video, interval, tmp, origin)
        meta = ClipFile(file=final.name, sha256=file_sha256(tmp), size_bytes=tmp.stat().st_size, format=fmt,
                        codec=video.codec, source_sha256=video.sha256, start_s=interval.start_s,
                        end_s=interval.end_s, **clip)
        os.replace(tmp, final)
        _atomic_text(record, canonical_json(meta))
        return meta
    finally:
        tmp.unlink(missing_ok=True)


def _copy(path: Path, video: VideoInfo, interval: ReviewInterval, tmp: Path, fmt: str, cancel) -> float:
    """Copy packets from the latest keyframe at or before the start; return its source time. Seeks are
    verified like decoding seeks: one landing after the start is retried further back, then from the start."""
    import av
    for backoff in (*SEEK_BACKOFF_S, None):
        with av.open(str(path)) as inp:
            ist = inp.streams.video[0]
            tb = ist.time_base
            if backoff is not None:
                target = max(0.0, interval.start_s - backoff) + video.start_s
                inp.seek(int(target / tb), stream=ist, backward=True, any_frame=False)
            out = av.open(str(tmp), "w", format=fmt)
            try:
                ost = out.add_stream_from_template(ist, opaque=True)
                origin = None
                for n, pk in enumerate(inp.demux(ist)):
                    if cancel is not None and n % 64 == 0 and cancel.cancelled:
                        raise ClipCancelled("clip extraction cancelled")
                    if pk.dts is None or pk.pts is None:
                        continue
                    if origin is None:
                        if not pk.is_keyframe:
                            continue
                        origin = pk.pts
                        if float(origin * tb) - video.start_s > interval.start_s + EPS:
                            break  # the seek overshot: retry further back
                    if float(pk.dts * tb) - video.start_s > interval.end_s + EPS:
                        break
                    pk.pts, pk.dts, pk.stream = pk.pts - origin, pk.dts - origin, ost
                    out.mux(pk)
            finally:
                out.close()
        if origin is not None and float(origin * tb) - video.start_s <= interval.start_s + EPS:
            return float(origin * tb) - video.start_s
    raise ClipUnsupported(f"no keyframe found at or before {interval.start_s:.3f} s")


def _verify(path: Path, video: VideoInfo, interval: ReviewInterval, clip: Path, origin: float) -> dict:
    """Decode the clip's timestamps and compare them with the source's over the same span."""
    got = [t for t, _, _ in VideoSource(clip).frame_times()]
    if not got:
        raise VideoError("the written clip has no decodable frames")
    shifted = [origin + t - got[0] for t in got]
    source = [t for t, _, _ in VideoSource(path).frame_times(origin, max(interval.end_s, shifted[-1]))]
    want = source[:len(shifted)]
    if len(want) != len(shifted) or any(abs(a - b) > TIME_TOLERANCE_S for a, b in zip(want, shifted)):
        raise VideoError("the written clip's frame times do not match the source")
    if want[0] > interval.start_s + EPS or len(want) < sum(t <= interval.end_s + EPS for t in source):
        raise VideoError("the written clip does not cover the interval")
    return dict(origin_s=round(origin, 6), first_s=round(want[0], 6), last_s=round(want[-1], 6), frames=len(got))


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_clip(out_dir: Path | str, name: str, source_sha256: str, interval: ReviewInterval) -> Optional[ClipFile]:
    """A previously written clip, when its record, file hash, source and interval all still match."""
    out_dir = Path(out_dir)
    try:
        meta = ClipFile.model_validate(json.loads((out_dir / f"{name}.json").read_text()))
    except (OSError, ValueError):
        return None
    if (meta.source_sha256, meta.start_s, meta.end_s) != (source_sha256, interval.start_s, interval.end_s):
        return None
    f = out_dir / meta.file
    if f.resolve().parent != out_dir.resolve():
        return None
    if not f.is_file() or f.stat().st_size != meta.size_bytes or file_sha256(f) != meta.sha256:
        return None
    return meta
