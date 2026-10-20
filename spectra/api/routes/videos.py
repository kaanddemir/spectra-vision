"""Registered videos: registration, metadata, browser playback checks and media serving."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Request, Response
from fastapi.responses import FileResponse

from spectra.api.library import LibraryPathError
from spectra.api.schemas import PlaybackReport, Video, VideoReference
from spectra.api.videos import VideoNotFound, VideoUnavailable

router = APIRouter(prefix="/videos", tags=["videos"])
VideoId = Path(pattern=r"^[0-9a-f]{32}$")


def _not_found() -> HTTPException:
    return HTTPException(404, "video not registered")


@router.post("", response_model=Video, status_code=201,
             responses={200: {"model": Video, "description": "already registered"}})
def register(reference: VideoReference, request: Request, response: Response) -> Video:
    """Register a library file. Probing continues in the background; poll the video until
    ``probe.state`` is no longer ``pending``."""
    try:
        video, created = request.app.state.videos.register(reference.folder_id, reference.path)
    except LibraryPathError as exc:
        raise HTTPException(exc.status_code, str(exc)) from None
    if not created:
        response.status_code = 200
    return video


@router.get("", response_model=list[Video])
def list_videos(request: Request) -> list[Video]:
    return request.app.state.videos.list()


@router.get("/{video_id}", response_model=Video)
def get_video(request: Request, video_id: str = VideoId) -> Video:
    try:
        return request.app.state.videos.get(video_id)
    except VideoNotFound:
        raise _not_found() from None


@router.delete("/{video_id}", status_code=204)
def remove_video(request: Request, video_id: str = VideoId) -> Response:
    """Forget a registration. The source file is never touched."""
    try:
        request.app.state.videos.remove(video_id)
    except VideoNotFound:
        raise _not_found() from None
    except VideoUnavailable as exc:
        raise HTTPException(409, str(exc)) from None
    return Response(status_code=204)


@router.post("/{video_id}/playback-check", response_model=Video)
def playback_check(report: PlaybackReport, request: Request, video_id: str = VideoId) -> Video:
    """Record whether this browser actually loaded and decoded the video. With ``sha256``, a result for
    contents other than the current file's is refused (409) instead of being attached to the new file."""
    try:
        return request.app.state.videos.record_browser_check(
            video_id, report.playable, report.detail, request.headers.get("user-agent"), report.sha256)
    except VideoNotFound:
        raise _not_found() from None
    except VideoUnavailable as exc:
        raise HTTPException(409, str(exc)) from None


@router.get("/{video_id}/media", response_class=FileResponse,
            responses={206: {"description": "partial content"}, 409: {"description": "unavailable"},
                       416: {"description": "range not satisfiable"}})
def media(request: Request, video_id: str = VideoId) -> FileResponse:
    """Stream the original file with byte-range support. Refused when it changed since probing."""
    try:
        path, info, mime = request.app.state.videos.media(video_id)
    except VideoNotFound:
        raise _not_found() from None
    except VideoUnavailable as exc:
        raise HTTPException(409, str(exc)) from None
    return FileResponse(path, stat_result=info, media_type=mime, content_disposition_type="inline",
                        headers={"cache-control": "private, no-cache"})


router.add_api_route("/{video_id}/media", media, methods=["HEAD"], include_in_schema=False)
