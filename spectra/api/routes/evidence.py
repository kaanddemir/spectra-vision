"""Timelines, evidence packages, overlays and clips of finished analyses.

Responses reuse the evidence contracts unchanged. Handlers are synchronous so cache verification,
evidence rebuilds and stream copies run on the thread pool, never on the event loop.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query, Request
from fastapi.responses import FileResponse, Response

from spectra.api.evidence import EvidenceNotFound, EvidenceUnavailable, layers
from spectra.api.export import ExportNotFound, ExportUnavailable, file_name, render
from spectra.api.schemas import ClipResponse, OverlayLayers, OverlayResponse, OverlayTiming
from spectra.evidence.build import Records
from spectra.evidence.contracts import EvidencePackage, Timeline

router = APIRouter(prefix="/analyses/{analysis_id}", tags=["evidence"])
AnalysisId = Path(pattern=r"^[0-9a-f]{32}$")
EventId = Path(pattern=r"^[0-9a-f]{16}$")
CLIP_TYPES = {"mp4": "video/mp4", "webm": "video/webm", "matroska": "video/x-matroska"}


def _call(function, *args):
    try:
        return function(*args)
    except EvidenceNotFound:
        raise HTTPException(404, "not found") from None
    except EvidenceUnavailable as exc:
        raise HTTPException(exc.status_code, str(exc)) from None


@router.get("/timeline", response_model=Timeline, responses={409: {"description": "no evidence available"}})
def timeline(request: Request, analysis_id: str = AnalysisId) -> Timeline:
    """The review timeline: proposals in review order, segments, gaps, warnings and analyser coverage.

    Runs without a result return a timeline with their state and message and no entries. An empty
    timeline of a completed run means no proposals, never that nothing happened: see ``analyzers``.
    """
    return _call(request.app.state.evidence.timeline, analysis_id)


@router.get("/packages/{event_id}", response_model=EvidencePackage)
def package(request: Request, analysis_id: str = AnalysisId, event_id: str = EventId) -> EvidencePackage:
    """One proposal's evidence package: the unchanged proposal, review interval, tracks, lanes and gaps."""
    return _call(request.app.state.evidence.package, analysis_id, event_id)


@router.get("/packages/{event_id}/overlay", response_model=OverlayResponse)
def overlay(request: Request, analysis_id: str = AnalysisId, event_id: str = EventId,
            start_s: Optional[float] = Query(None, ge=0), end_s: Optional[float] = Query(None, ge=0)) -> OverlayResponse:
    """Recorded boxes and lane boundaries for the review interval, or a requested span of up to 120 s."""
    drawn, pk, perception = _call(request.app.state.evidence.overlay, analysis_id, event_id, start_s, end_s)
    config = request.app.state.evidence.config
    gaps = [g for g in pk.gaps if g.end_s > drawn.start_s and g.start_s < drawn.end_s] if start_s is None and \
        end_s is None else list(Records(drawn.frames).gaps(config.gap_s, drawn.start_s, drawn.end_s)) \
        if drawn.frames else []
    return OverlayResponse(
        overlay=drawn, layers=OverlayLayers(**layers(perception)), gaps=gaps,
        timing=OverlayTiming(stream_start_s=pk.video.start_s, max_age_s=config.overlay_max_age_s,
                             analysed_start_s=pk.analysed_start_s, analysed_end_s=pk.analysed_end_s,
                             timestamp_source=pk.video.timestamp_source.value))


@router.get("/export", response_class=Response,
            responses={200: {"content": {"application/json": {}}, "description": "the export document as a download"},
                       404: {"description": "analysis not found"},
                       409: {"description": "no results, or the stored evidence is not ready or fails verification"}})
def export(request: Request, analysis_id: str = AnalysisId) -> Response:
    """The analysis, its evidence and the reviewer's decisions, edits and notes as one JSON file. Read-only:
    nothing is written, rebuilt or analysed."""
    try:
        document, video_name = request.app.state.export.export(analysis_id)
    except ExportNotFound:
        raise HTTPException(404, "analysis not found") from None
    except ExportUnavailable as exc:
        raise HTTPException(409, str(exc)) from None
    return Response(render(document), media_type="application/json; charset=utf-8", headers={
        "content-disposition": f'attachment; filename="{file_name(video_name, analysis_id)}"',
        "cache-control": "no-store"})


def _clip_response(analysis_id: str, event_id: str, clip) -> ClipResponse:
    return ClipResponse(clip=clip, media_url=f"/api/analyses/{analysis_id}/packages/{event_id}/clip",
                        mime_type=CLIP_TYPES.get(clip.format, "application/octet-stream"))


@router.post("/packages/{event_id}/clip", response_model=ClipResponse,
             responses={409: {"description": "source unavailable or changed"},
                        422: {"description": "no clip can be cut from this source"}})
def create_clip(request: Request, analysis_id: str = AnalysisId, event_id: str = EventId) -> ClipResponse:
    """Stream-copy the review interval from the registered source (no re-encoding), or reuse a verified clip.

    The source is verified by its full SHA-256 first. A valid clip may still not play in a browser.
    """
    state = request.app.state
    try:
        analysis = state.analyses.get(analysis_id)
    except LookupError:
        raise HTTPException(404, "not found") from None
    _, path, _ = state.videos.source_for_analysis(analysis.video_id)
    clip, _ = _call(state.evidence.clip, analysis_id, event_id, path)
    return _clip_response(analysis_id, event_id, clip)


@router.get("/packages/{event_id}/clip", response_class=FileResponse, responses={404: {"description": "no clip"}})
def clip_media(request: Request, analysis_id: str = AnalysisId, event_id: str = EventId) -> FileResponse:
    """A clip created earlier, with byte-range support."""
    clip, path = _call(request.app.state.evidence.stored_clip, analysis_id, event_id)
    return FileResponse(path, media_type=CLIP_TYPES.get(clip.format, "application/octet-stream"),
                        content_disposition_type="inline", headers={"cache-control": "private, no-cache"})
