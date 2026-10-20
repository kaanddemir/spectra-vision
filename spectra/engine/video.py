"""PyAV video source: metadata, timestamp-driven sampling and verified frame retrieval.

Every iteration opens its own container and closes it when the iterator finishes or is closed, so
callers never share seek state and native resources are released deterministically. Frames are
converted to arrays one at a time; nothing accumulates decoded images.

Seeking lands on a keyframe that may be seconds away from the request (on variable-frame-rate
sources it can be far off), so positioning never trusts it: after a seek the first decoded frame must
not be later than the target, otherwise the seek is retried further back and finally replaced by
decoding from the start. Frames are then selected by their decoded timestamps.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from spectra.engine.contracts import TimestampSource, TimingReport, VideoInfo

EPS = 1e-6
SEEK_BACKOFF_S = (0.0, 2.0, 10.0, 60.0)  # extra pre-roll for each successive seek attempt
VFR_TOLERANCE = 0.1  # 5th-95th percentile interval spread, relative to the median, that counts as variable
GAP_WARNING_S = 1.0  # a longer stretch without frames is reported: evidence there is missing, whatever the cause


class VideoError(Exception):
    """A video that cannot be analysed; the message is suitable for a user."""


class UnsupportedVideo(VideoError):
    pass


class CorruptVideo(VideoError):
    pass


class MissingTimestamps(UnsupportedVideo):
    pass


@dataclass(frozen=True)
class Frame:
    decode_index: Optional[int]  # position among decoded frames; None when decoding began at a seek
    t: float
    pts: Optional[int]
    image: np.ndarray = field(repr=False)  # BGR uint8, display orientation
    scale: float = 1.0  # image width / display width


@dataclass
class DecodeStats:
    """Counters for one iteration. ``frames_decoded`` includes pre-roll decoded before the start time."""
    frames_decoded: int = 0
    missing_timestamps: int = 0
    non_monotonic: int = 0
    decode_errors: int = 0
    ended_early: Optional[str] = None
    times: list[float] = field(default_factory=list, repr=False)  # every in-span timestamp, not only sampled

    def report(self, frames_analysed: int) -> TimingReport:
        gaps = np.diff(self.times) if len(self.times) > 1 else np.array([])
        median = float(np.median(gaps)) if len(gaps) else None
        variable = None
        if median:
            variable = bool(np.percentile(gaps, 95) - np.percentile(gaps, 5) > VFR_TOLERANCE * median)
        return TimingReport(
            frames_decoded=self.frames_decoded, frames_analysed=frames_analysed,
            first_t=self.times[0] if self.times else None, last_t=self.times[-1] if self.times else None,
            median_interval_s=median, max_interval_s=float(gaps.max()) if len(gaps) else None,
            max_interval_start=self.times[int(gaps.argmax())] if len(gaps) else None,
            variable_frame_rate=variable, missing_timestamps=self.missing_timestamps,
            non_monotonic=self.non_monotonic, decode_errors=self.decode_errors, ended_early=self.ended_early)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def decoder_versions() -> dict[str, str]:
    import av
    return {"pyav": av.__version__, "ffmpeg": str(av.ffmpeg_version_info)}


def _open(path: Path):
    import av
    try:
        container = av.open(str(path), mode="r")
    except FileNotFoundError as exc:
        raise UnsupportedVideo(f"video file not found: {path.name}") from exc
    except av.error.FFmpegError as exc:
        raise CorruptVideo(f"cannot read {path.name} as a video container: {exc}") from exc
    if not container.streams.video:
        container.close()
        raise UnsupportedVideo(f"{path.name} contains no video stream")
    stream = container.streams.video[0]
    if stream.codec_context is None:
        container.close()
        raise UnsupportedVideo(f"{path.name}: no decoder available for this video codec")
    stream.thread_type = "AUTO"
    # Without this libavcodec only applies crops that keep buffers aligned, leaving padding in the
    # picture (a Theora sample decoded 858 px wide with a 4 px left offset instead of 854).
    stream.codec_context.options = {"flags": "unaligned"}
    return container, stream


class VideoSource:
    """A local video file, opened read-only and never modified."""

    def __init__(self, path: Path | str, assumed_fps: Optional[float] = None, max_consecutive_errors: int = 25):
        self.path = Path(path)
        self.assumed_fps = assumed_fps
        self.max_consecutive_errors = max_consecutive_errors
        self.stats = DecodeStats()
        self.info = self._probe()

    def _probe(self) -> VideoInfo:
        if not self.path.is_file():
            raise UnsupportedVideo(f"video file not found: {self.path.name}")
        container, stream = _open(self.path)
        try:
            frames = self._decode(container, stream, DecodeStats())
            first = next(frames, None)
            frames.close()
            if first is None:
                raise CorruptVideo(f"{self.path.name}: no decodable video frames")
            if first.pts is None:
                if self.assumed_fps is None:
                    raise MissingTimestamps(
                        f"{self.path.name} has no frame timestamps; analysis needs the camera frame rate")
                source = TimestampSource.ASSUMED_RATE
            else:
                source = TimestampSource.CONTAINER
            rotation = int(round(first.rotation / 90.0)) % 4 * 90
            cw, ch = first.width, first.height
            tb = stream.time_base
            start = float(stream.start_time * tb) if stream.start_time is not None and tb else 0.0
            if container.duration is not None:
                duration = container.duration / 1_000_000
            elif stream.duration is not None and tb:
                duration = float(stream.duration * tb)
            else:
                duration = None
            rate = stream.average_rate or stream.guessed_rate
            return VideoInfo(
                file_name=self.path.name, size_bytes=self.path.stat().st_size, sha256=file_sha256(self.path),
                container=container.format.name, codec=stream.codec_context.name,
                width=ch if rotation in (90, 270) else cw, height=cw if rotation in (90, 270) else ch,
                coded_width=cw, coded_height=ch, rotation=rotation,
                time_base=f"{tb.numerator}/{tb.denominator}" if tb else None, start_s=start,
                duration_s=duration if duration is not None and duration >= 0 else None,
                nominal_fps=float(rate) if rate else None, timestamp_source=source,
                assumed_fps=self.assumed_fps if source is TimestampSource.ASSUMED_RATE else None)
        finally:
            container.close()

    def _decode(self, container, stream, stats: DecodeStats):
        """Raw decoded frames in presentation order, tolerating isolated corrupt packets."""
        import av
        consecutive = 0
        packets = container.demux(stream)
        while True:
            try:
                packet = next(packets)
            except StopIteration:
                return
            except av.error.FFmpegError as exc:
                if stats.frames_decoded == 0:
                    raise CorruptVideo(f"{self.path.name}: unreadable from the start ({exc})") from exc
                stats.ended_early = f"demuxing stopped: {exc}"
                return
            try:
                decoded = packet.decode()
            except av.error.FFmpegError as exc:
                stats.decode_errors += 1
                consecutive += 1
                if consecutive > self.max_consecutive_errors:
                    raise CorruptVideo(
                        f"{self.path.name}: {consecutive} consecutive undecodable packets ({exc})") from exc
                continue
            for frame in decoded:
                consecutive = 0
                stats.frames_decoded += 1
                yield frame

    def _timed(self, start_s: float, stats: DecodeStats) -> Iterator[tuple[Optional[int], float, object]]:
        """(decode_index, t, av_frame) with validated, strictly increasing times.

        Starts at a frame no later than ``start_s``: either the stream start or a verified seek point.
        """
        info = self.info
        assumed = info.timestamp_source is TimestampSource.ASSUMED_RATE
        attempts = [None] if assumed or start_s <= EPS else [*SEEK_BACKOFF_S, None]
        for backoff in attempts:
            stats.__dict__.update(DecodeStats().__dict__)  # an abandoned attempt yields nothing; forget it
            container, stream = _open(self.path)
            frames = None
            try:
                if backoff is not None:
                    target = max(0.0, start_s - backoff) + info.start_s
                    container.seek(int(target / stream.time_base), stream=stream, backward=True, any_frame=False)
                frames = self._decode(container, stream, stats)
                last_t, index, overshot = None, (0 if backoff is None else None), False
                for frame in frames:
                    if assumed:
                        t = index / info.assumed_fps
                    elif frame.pts is None:
                        stats.missing_timestamps += 1
                        continue
                    else:
                        t = float(frame.pts * stream.time_base) - info.start_s
                    if last_t is None and backoff is not None and t > start_s + EPS:
                        overshot = True
                        break
                    if last_t is not None and t <= last_t + EPS:
                        stats.non_monotonic += 1
                        continue
                    last_t = t
                    yield index, t, frame
                    if index is not None:
                        index += 1
                if not overshot:
                    return
            finally:
                if frames is not None:
                    frames.close()
                container.close()
        raise CorruptVideo(f"{self.path.name}: could not position decoding at {start_s:.3f} s")

    def _image(self, frame, max_width: Optional[int]) -> tuple[np.ndarray, float]:
        info = self.info
        scale = min(1.0, max_width / info.width) if max_width else 1.0
        if scale < 1.0:
            w, h = max(2, round(info.coded_width * scale)), max(2, round(info.coded_height * scale))
            image = frame.to_ndarray(format="bgr24", width=w, height=h)
        else:
            image = frame.to_ndarray(format="bgr24")
        if info.rotation:
            image = np.ascontiguousarray(np.rot90(image, k=info.rotation // 90))
        return image, image.shape[1] / info.width

    def frames(self, interval_s: Optional[float] = None, start_s: float = 0.0, end_s: Optional[float] = None,
               max_width: Optional[int] = None) -> Iterator[Frame]:
        """Frames on a timestamp grid: the first frame at or after each ``start_s + k * interval_s``.

        Without an interval every frame in ``[start_s, end_s]`` is returned. A gap longer than the
        interval yields one frame, never a burst. Statistics for the iteration are in ``self.stats``.
        """
        self.stats = stats = DecodeStats()
        next_t = start_s
        timed = self._timed(start_s, stats)
        try:
            for index, t, frame in timed:
                if t + EPS < start_s:
                    continue
                if end_s is not None and t > end_s + EPS:
                    return
                stats.times.append(t)
                if interval_s is not None:
                    if t + EPS < next_t:
                        continue
                    next_t = start_s + (np.floor((t - start_s) / interval_s + EPS) + 1) * interval_s
                image, scale = self._image(frame, max_width)
                yield Frame(decode_index=index, t=t, pts=frame.pts, image=image, scale=scale)
            self._check_complete(stats, end_s)
        finally:
            timed.close()

    def _check_complete(self, stats: DecodeStats, end_s: Optional[float]) -> None:
        """Damaged or truncated files often just stop; compare with the declared duration to notice."""
        declared = self.info.duration_s
        if stats.ended_early or declared is None or not stats.times:
            return
        wanted = min(declared, end_s) if end_s is not None else declared
        gaps = np.diff(stats.times)
        slack = max(1.0, 2 * float(gaps.max())) if len(gaps) else 1.0
        if wanted - stats.times[-1] > slack:
            stats.ended_early = (f"decoding ended at {stats.times[-1]:.3f} s, but the container declares "
                                 f"{declared:.3f} s")

    def frame_times(self, start_s: float = 0.0, end_s: Optional[float] = None
                    ) -> Iterator[tuple[float, Optional[int], bool]]:
        """(t, pts, keyframe) of every decoded frame up to ``end_s``, without converting any pixels. Starts
        where decoding has to start to show ``start_s``: the stream start or a verified seek keyframe at or
        before it, so the frames before ``start_s`` are the pre-roll. Statistics are in ``self.stats``."""
        self.stats = stats = DecodeStats()
        timed = self._timed(start_s, stats)
        try:
            for _, t, frame in timed:
                if end_s is not None and t > end_s + EPS:
                    return
                stats.times.append(t)
                yield t, frame.pts, bool(frame.key_frame)
            self._check_complete(stats, end_s)
        finally:
            timed.close()

    def frame_at(self, t: float, max_width: Optional[int] = None) -> Frame:
        """The frame displayed at time ``t``: the last frame whose timestamp is at or before ``t``.

        Pass a timestamp recorded in an analysis result to get exactly that frame back.
        """
        if t < 0 or (self.info.duration_s is not None and t > self.info.duration_s + EPS):
            raise ValueError(f"{t:.3f} s is outside the video")
        timed = self._timed(t, DecodeStats())
        try:
            found = None
            for item in timed:
                if item[1] > t + EPS:
                    break
                found = item
            if found is None:
                raise ValueError(f"no frame is displayed at {t:.3f} s")
            index, ft, frame = found
            image, scale = self._image(frame, max_width)
            return Frame(decode_index=index, t=ft, pts=frame.pts, image=image, scale=scale)
        finally:
            timed.close()
