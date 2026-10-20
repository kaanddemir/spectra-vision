"""API responses used as fixtures by the web application's tests, built from the API's own models.

Videos use the server's playback assessment and eligibility rules; the timeline, evidence packages and
overlay are built by the evidence layer from a synthetic analysis result. Regenerate after a deliberate contract change with
``python -m tests.api.frontend_fixtures``; ``test_frontend_fixtures`` fails while the file is out of date.
"""
from __future__ import annotations

import json
from typing import Optional

from spectra.api import API_VERSION
from spectra.api.evidence import layers
from spectra.api.playback import assess
from spectra.api.schemas import (Analysis, AnalysisProgress, AnalysisResultSummary, AnalysisReadiness, BrowserCheck,
                                 ClipResponse, DatabaseHealth, EvidenceStatus, Health, LibraryEntryInfo,
                                 LibraryFolderInfo, LibraryListing, MediaInfo, MemoryHealth, ModelFile,
                                 ModelRegistryHealth, OverlayLayers, OverlayResponse, OverlayTiming, PlaybackInfo,
                                 ProbeStatus, ReviewDecision, ReviewDecisions, SourceState, Video,
                                 EventEdit, EventEdits)
from spectra.api.settings import REPOSITORY_ROOT
from spectra.api.store import SCHEMA_VERSION
from spectra.api.videos import VideoRegistry
from spectra.engine.contracts import TimestampSource, VideoInfo
from spectra.evidence.build import build
from spectra.evidence.media import ClipFile
from spectra.evidence.overlays import overlay
from tests.evidence import synth

FIXTURES_PATH = REPOSITORY_ROOT / "frontend" / "src" / "test" / "fixtures" / "api.json"
FOLDER = LibraryFolderInfo(id="0123456789abcdef", name="Dashcam", path="/Users/reviewer/Movies/Dashcam",
                           source="startup", available=True, registered_videos=2)
SECOND_FOLDER = LibraryFolderInfo(id="fedcba9876543210", name="Archive", path="/Volumes/Archive/Footage",
                                  source="added", available=True, registered_videos=0)
SAFARI = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15"


def _id(n: int) -> str:
    return f"{n:032x}"


def _sha(n: int) -> str:
    return f"{n:064x}"


def _video(n: int, name: str, *, probe: str = "ready", container: str = "mov,mp4,m4a,3gp,3g2,mj2",
           codec: str = "h264", audio: tuple[str, ...] = ("aac",), source: str = "available",
           check: Optional[bool] = None, error: Optional[str] = None, message: Optional[str] = None,
           path: Optional[str] = None) -> Video:
    """A registered video as the server reports it, using the server's playback and eligibility rules."""
    media, playback = None, PlaybackInfo(assessment="unknown", reasons=[])
    if probe == "ready":
        info = VideoInfo(file_name=name, size_bytes=48_000_000 + n, sha256=_sha(n), container=container,
                         codec=codec, width=1920, height=1080, coded_width=1920, coded_height=1080, rotation=0,
                         duration_s=62.5, nominal_fps=29.97, timestamp_source=TimestampSource.CONTAINER)
        media = MediaInfo(sha256=info.sha256, container=container, codec=codec, audio_codecs=list(audio),
                          width=1920, height=1080, rotation=0, duration_s=62.5, nominal_fps=29.97, start_s=0.0,
                          timestamp_source=info.timestamp_source.value)
        assessment = assess(info, list(audio), "." + name.rsplit(".", 1)[-1])
        browser = None if check is None else BrowserCheck(
            playable=check, checked_at="2026-10-18T09:15:00+00:00", user_agent=SAFARI,
            detail="Decoded a 1920×1080 frame." if check else "This browser could not decode the video.")
        playback = PlaybackInfo(assessment=assessment.assessment, reasons=assessment.reasons,
                                mime_type=assessment.mime_type, browser_check=browser)
    probe_status = ProbeStatus(state=probe, error=error, message=message,
                               probed_at="2026-10-18T09:00:05+00:00" if probe != "pending" else None)
    source_state = SourceState(status=source, message={
        "missing": f"{name} was moved or deleted",
        "changed": f"{name} changed since it was read; register it again",
    }.get(source))
    servable = source == "available" and playback.assessment in ("likely", "uncertain")
    return Video(id=_id(n), file_name=name, folder_id=FOLDER.id, path=path or name, size_bytes=48_000_000 + n,
                 modified_at="2026-10-12T17:40:00+00:00", registered_at=f"2026-10-18T09:00:{n:02d}+00:00",
                 probe=probe_status, media=media, source=source_state, playback=playback,
                 analysis=VideoRegistry._eligibility(probe_status, source_state, playback),
                 media_url=f"/api/videos/{_id(n)}/media" if servable else None)


def _videos() -> dict[str, Video]:
    return {
        "ready": _video(1, "drive.mp4", check=True),
        "needs_check": _video(2, "city.mp4", path="2026-10 trips/city.mp4"),
        "probing": _video(3, "rain.mp4", probe="pending"),
        "unsupported": _video(4, "old.avi", container="avi", codec="mpeg4", audio=()),
        "missing": _video(5, "deleted.mp4", source="missing", check=True),
        "unreadable": _video(6, "broken.mp4", probe="failed", error="corrupt",
                             message="broken.mp4: no decodable video frame"),
        "not_playable": _video(7, "tunnel.mov", codec="hevc", check=False),
    }


def _analysis(n: int, video: Video, state: str, *, created: str, proposals: Optional[int] = None,
              evidence: str = "not_applicable", stop_reason: Optional[str] = None,
              message: Optional[str] = None) -> Analysis:
    result = None if proposals is None else AnalysisResultSummary(
        cache_hit=False, perception_key=_sha(100 + n), analysis_key=_sha(200 + n), proposals=proposals, notes=[])
    return Analysis(id=_id(100 + n), video_id=video.id, file_name=video.file_name, source_sha256=_sha(1),
                    state=state, revision=12, queue_position=None, cancel_requested=False,
                    progress=AnalysisProgress(stage="analysing frames", frames_analysed=625, media_time_s=62.4,
                                              fraction=1.0 if result else None),
                    result=result, evidence=EvidenceStatus(state=evidence, build_key=_sha(300 + n) if evidence ==
                                                           "ready" else None),
                    stop_reason=stop_reason, message=message, created_at=created,
                    started_at=created, finished_at=created if state != "running" else None)


def build_fixtures() -> dict:
    videos = _videos()
    ready = videos["ready"]
    result = synth.result([synth.approach(12.0, 14.5), synth.cut_in(30.0, 31.2, entry_from=28.0),
                           synth.contact(44.0, 44.8)],
                          analyzers=(synth.APPROACH, synth.CUT_IN, synth.IMPACT, synth.CONTACT),
                          times=[round(0.1 * i, 3) for i in range(625)], info=synth.video(62.5))
    timeline, packages = build(result)
    first = packages[0]
    drawn = overlay(result.perception, first)
    overlay_response = OverlayResponse(
        overlay=drawn, layers=OverlayLayers(**layers(result.perception)), gaps=list(first.gaps),
        timing=OverlayTiming(stream_start_s=first.video.start_s, max_age_s=timeline.config.overlay_max_age_s,
                             analysed_start_s=first.analysed_start_s, analysed_end_s=first.analysed_end_s,
                             timestamp_source=first.video.timestamp_source.value))
    clip = ClipResponse(
        clip=ClipFile(file=f"{first.proposal.event_id}.mp4", sha256=_sha(400), size_bytes=2_400_000, format="mp4",
                      codec="h264", source_sha256=first.video.sha256, start_s=first.review.start_s,
                      end_s=first.review.end_s, origin_s=8.0, first_s=8.0, last_s=16.5, frames=86),
        media_url=f"/api/analyses/{_id(101)}/packages/{first.proposal.event_id}/clip", mime_type="video/mp4")
    analyses = [
        _analysis(1, ready, "completed", created="2026-10-18T10:00:00+00:00", proposals=len(timeline.entries),
                  evidence="ready"),
        _analysis(2, ready, "completed", created="2026-10-17T10:00:00+00:00", proposals=1, evidence="ready"),
        _analysis(3, videos["missing"], "failed", created="2026-10-18T11:00:00+00:00", stop_reason="video",
                  message="the video changed after it was registered"),
    ]
    listing = LibraryListing(folder=FOLDER, path="", parent=None, truncated=False, entries=[
        LibraryEntryInfo(name="2026-10 trips", path="2026-10 trips", kind="directory",
                         modified_at="2026-10-12T17:40:00+00:00"),
        LibraryEntryInfo(name="drive.mp4", path="drive.mp4", kind="video", size_bytes=ready.size_bytes,
                         modified_at=ready.modified_at, video_id=ready.id),
        LibraryEntryInfo(name="highway.mp4", path="highway.mp4", kind="video", size_bytes=120_500_000,
                         modified_at="2026-10-11T08:12:00+00:00"),
        LibraryEntryInfo(name="night.webm", path="night.webm", kind="video", size_bytes=88_000_000,
                         modified_at="2026-10-10T22:30:00+00:00"),
    ])
    sublisting = LibraryListing(folder=FOLDER, path="2026-10 trips", parent="", truncated=False, entries=[
        LibraryEntryInfo(name="city.mp4", path="2026-10 trips/city.mp4", kind="video",
                         size_bytes=videos["needs_check"].size_bytes, modified_at="2026-10-12T17:40:00+00:00",
                         video_id=videos["needs_check"].id),
        LibraryEntryInfo(name="coast.mp4", path="2026-10 trips/coast.mp4", kind="video", size_bytes=64_000_000,
                         modified_at="2026-10-12T18:05:00+00:00"),
    ])
    models = ModelRegistryHealth(available=True, models=[
        ModelFile(role="detector", file="detector.onnx", sha256=_sha(901)),
        ModelFile(role="segmenter", file="segmenter.onnx", sha256=_sha(902))])
    health = Health(status="ok", version=API_VERSION, database=DatabaseHealth(available=True, schema_version=SCHEMA_VERSION),
                    models=models, memory=MemoryHealth(pressure_level=1, state="normal"),
                    analysis=AnalysisReadiness(available=True, reasons=[]), library_folders=1)
    degraded = health.model_copy(update={
        "status": "degraded", "memory": MemoryHealth(pressure_level=4, state="critical"),
        "analysis": AnalysisReadiness(available=False, reasons=["Memory pressure is critical."])})
    dump = lambda value: value.model_dump(mode="json")  # noqa: E731
    return {
        "description": "API responses for the web application's tests; generated by tests/api/frontend_fixtures.py.",
        "health": dump(health),
        "healthDegraded": dump(degraded),
        "folders": [dump(FOLDER)],
        "twoFolders": [dump(FOLDER), dump(SECOND_FOLDER)],
        "listing": dump(listing),
        "sublisting": dump(sublisting),
        "videos": {key: dump(value) for key, value in videos.items()},
        "analyses": [dump(a) for a in analyses],
        "timeline": dump(timeline),
        "packages": {pk.proposal.event_id: dump(pk) for pk in packages},
        "overlay": dump(overlay_response),
        "clip": dump(clip),
        "reviews": dump(ReviewDecisions(analysis_id=analyses[0].id, decisions=[
            ReviewDecision(event_id=timeline.entries[1].event_id, decision="accepted", revision=1,
                           updated_at="2026-10-18T10:30:00+00:00")])),
        "edits": dump(EventEdits(analysis_id=analyses[0].id, edits=[
            EventEdit(event_id=timeline.entries[0].event_id, t_start=11.24, t_end=15.0,
                      description="Truck ahead brakes hard; wet road.", revision=2,
                      updated_at="2026-10-18T10:40:00+00:00")])),
    }


def render() -> str:
    return json.dumps(build_fixtures(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    FIXTURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURES_PATH.write_text(render(), encoding="utf-8")
