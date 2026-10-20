"""Preliminary browser-playback assessment from probed container and codecs.

The server cannot know what a particular browser decodes, so it never claims playback. It
classifies sources as ``likely`` (common browser formats), ``uncertain`` (depends on the browser or
platform) or ``unsupported`` (no current mainstream browser plays it as a file). Only an actual load
in the user's browser, reported back by the application, confirms playback. The engine still decodes
every format it supports; this restriction belongs to the review workflow, which needs playback.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from spectra.engine.contracts import TimestampSource, VideoInfo

Assessment = Literal["likely", "uncertain", "unsupported"]

# PyAV/FFmpeg demuxer names. MP4 and QuickTime share one demuxer, as do Matroska and WebM.
MP4_FAMILY = "mov,mp4,m4a,3gp,3g2,mj2"
MATROSKA_FAMILY = "matroska,webm"
WEBM_VIDEO = frozenset({"vp8", "vp9", "av1"})
WEBM_AUDIO = frozenset({"opus", "vorbis"})
MP4_VIDEO_LIKELY = frozenset({"h264", "vp9", "av1"})
MP4_AUDIO = frozenset({"aac", "mp3", "opus", "flac", "alac"})
OGG_AUDIO = frozenset({"vorbis", "opus", "flac"})


class PlaybackAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    assessment: Assessment
    reasons: list[str]
    mime_type: Optional[str] = None


def _audio_reasons(audio: list[str], allowed: frozenset[str]) -> list[str]:
    odd = sorted(set(audio) - allowed)
    return [f"Audio codec {', '.join(odd)} may prevent playback in some browsers."] if odd else []


def assess(info: VideoInfo, audio_codecs: list[str], suffix: str = "") -> PlaybackAssessment:
    """Classify a probed source. ``suffix`` only chooses between equivalent MIME labels."""
    container, codec, suffix = info.container, info.codec, suffix.lower()
    if info.timestamp_source is TimestampSource.ASSUMED_RATE:
        return PlaybackAssessment(assessment="unsupported",
                                  reasons=["The stream has no timestamps, so browsers cannot play it."])
    if container == MP4_FAMILY:
        mime = "video/quicktime" if suffix == ".mov" else "video/mp4"
        if codec in MP4_VIDEO_LIKELY:
            reasons = _audio_reasons(audio_codecs, MP4_AUDIO)
            if codec == "av1":
                reasons.insert(0, "AV1 needs a browser and device with AV1 support (not Safari on M1 Macs).")
            if mime == "video/quicktime":
                reasons.append("QuickTime files play natively in Safari; other browsers usually accept "
                               "them as MP4.")
            return PlaybackAssessment(assessment="uncertain" if reasons else "likely", reasons=reasons,
                                      mime_type=mime)
        if codec == "hevc":
            return PlaybackAssessment(assessment="uncertain", mime_type=mime, reasons=[
                "HEVC (H.265) plays only where the browser and system provide a decoder; Safari also "
                "requires the 'hvc1' tag.", *_audio_reasons(audio_codecs, MP4_AUDIO)])
        return PlaybackAssessment(assessment="unsupported",
                                  reasons=[f"Browsers do not play {codec} video in MP4 or QuickTime files."])
    if container == MATROSKA_FAMILY:
        if codec in WEBM_VIDEO and set(audio_codecs) <= WEBM_AUDIO:
            reasons = (["AV1 needs a browser and device with AV1 support (not Safari on M1 Macs)."]
                       if codec == "av1" else [])
            return PlaybackAssessment(assessment="uncertain" if reasons else "likely", reasons=reasons,
                                      mime_type="video/webm")
        if codec in WEBM_VIDEO | {"h264", "hevc"}:
            return PlaybackAssessment(assessment="uncertain", mime_type="video/webm", reasons=[
                "Matroska files with these codecs play in some browsers only (not Safari)."])
        return PlaybackAssessment(assessment="unsupported",
                                  reasons=[f"Browsers do not play {codec} video in Matroska files."])
    if container == "ogg":
        if codec == "theora":
            return PlaybackAssessment(assessment="uncertain", mime_type="video/ogg", reasons=[
                "Theora is no longer supported by Chrome or Safari; Firefox may still play it.",
                *_audio_reasons(audio_codecs, OGG_AUDIO)])
        return PlaybackAssessment(assessment="unsupported",
                                  reasons=[f"Browsers do not play {codec} video in Ogg files."])
    return PlaybackAssessment(assessment="unsupported", reasons=[
        f"Browsers do not play the {container} container directly. Convert the video to MP4 (H.264) "
        "or WebM (VP9) with another tool to review it here."])
