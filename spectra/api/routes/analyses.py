"""Analysis jobs: request, status polling, Server-Sent Events progress and cancellation."""
from __future__ import annotations

from typing import AsyncIterator, Optional

import anyio
from fastapi import APIRouter, Header, HTTPException, Path, Query, Request, Response
from fastapi.responses import StreamingResponse

from spectra.api.jobs import TERMINAL_STATES, AnalysisNotFound, AnalysisRefused
from spectra.api.schemas import Analysis, AnalysisRequest
from spectra.api.videos import VideoNotFound

router = APIRouter(prefix="/analyses", tags=["analyses"])
AnalysisId = Path(pattern=r"^[0-9a-f]{32}$")
POLL_S = 0.25
KEEPALIVE_S = 15.0


def _refused(exc: AnalysisRefused) -> HTTPException:
    return HTTPException(exc.status_code, {"message": str(exc), "reasons": exc.reasons})


@router.post("", response_model=Analysis, status_code=201,
             responses={200: {"model": Analysis, "description": "the video's queued or running analysis"},
                        409: {"description": "the video is not eligible"},
                        503: {"description": "models, memory or shutdown prevent new analyses"}})
def create(body: AnalysisRequest, request: Request, response: Response) -> Analysis:
    """Queue an analysis. Repeating the request while one is queued or running returns that one."""
    try:
        analysis, created = request.app.state.analyses.submit(body.video_id)
    except VideoNotFound:
        raise HTTPException(404, "video not registered") from None
    except AnalysisRefused as exc:
        raise _refused(exc) from None
    if not created:
        response.status_code = 200
    return analysis


@router.get("", response_model=list[Analysis])
def list_analyses(request: Request, video_id: Optional[str] = Query(None, pattern=r"^[0-9a-f]{32}$")) -> list[Analysis]:
    """Analyses, newest first, optionally for one video."""
    return request.app.state.analyses.list(video_id)


@router.get("/{analysis_id}", response_model=Analysis)
def get_analysis(request: Request, analysis_id: str = AnalysisId) -> Analysis:
    """Current status and progress; the polling fallback for the event stream."""
    try:
        return request.app.state.analyses.get(analysis_id)
    except AnalysisNotFound:
        raise HTTPException(404, "analysis not found") from None


@router.post("/{analysis_id}/cancel", response_model=Analysis, responses={409: {"description": "already finished"}})
def cancel(request: Request, analysis_id: str = AnalysisId) -> Analysis:
    """Cancel a queued analysis, or ask a running one to stop; poll or stream until it is cancelled."""
    try:
        return request.app.state.analyses.cancel(analysis_id)
    except AnalysisNotFound:
        raise HTTPException(404, "analysis not found") from None
    except AnalysisRefused as exc:
        raise _refused(exc) from None


@router.get("/{analysis_id}/events", response_class=StreamingResponse,
            responses={200: {"content": {"text/event-stream": {}},
                             "description": "`analysis` events carry the Analysis JSON; `end` follows a final state "
                                            "once evidence preparation has settled"}})
async def events(request: Request, analysis_id: str = AnalysisId,
                 last_event_id: Optional[str] = Header(None)) -> StreamingResponse:
    """Stream status changes. Each event id is the analysis revision; a reconnect with Last-Event-ID
    receives only newer states. The stream ends after a final state or when the server stops."""
    service = request.app.state.analyses
    try:
        await anyio.to_thread.run_sync(service.get, analysis_id)
    except AnalysisNotFound:
        raise HTTPException(404, "analysis not found") from None
    sent = int(last_event_id) if last_event_id and last_event_id.isdigit() else None

    async def stream() -> AsyncIterator[str]:
        nonlocal sent
        service.subscribers += 1
        try:
            yield "retry: 2000\n\n"
            quiet = 0.0
            while True:
                analysis = await anyio.to_thread.run_sync(service.get, analysis_id)
                if analysis.revision != sent:
                    sent = analysis.revision
                    quiet = 0.0
                    yield f"id: {sent}\nevent: analysis\ndata: {analysis.model_dump_json()}\n\n"
                if analysis.state in TERMINAL_STATES and analysis.evidence.state != "pending":
                    yield "event: end\ndata: {}\n\n"
                    return
                if service.closing or await request.is_disconnected():
                    return
                await anyio.sleep(POLL_S)
                quiet += POLL_S
                if quiet >= KEEPALIVE_S:
                    quiet = 0.0
                    yield ": keep-alive\n\n"
        finally:
            service.subscribers -= 1

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"cache-control": "no-cache", "x-accel-buffering": "no"})
