"""Library folders: listing, adding through the native folder dialog, removing and browsing.

Only paths relative to a folder are accepted from clients. Adding a folder takes no path: the folder is
chosen in the operating system's dialog on this computer.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath

from fastapi import APIRouter, HTTPException, Path, Query, Request, Response

from spectra.api.library import ConfiguredFolder, LibraryPathError, list_directory
from spectra.api.picker import PickerBusy, PickerUnavailable
from spectra.api.schemas import FolderSelection, LibraryEntryInfo, LibraryFolderInfo, LibraryListing

router = APIRouter(prefix="/library", tags=["library"])
FOLDER_ID = Path(pattern=r"^[0-9a-f]{16}$")


def _folder_info(state, item: ConfiguredFolder) -> LibraryFolderInfo:
    return LibraryFolderInfo(id=item.folder.id, name=item.folder.name, path=str(item.folder.path),
                             source=item.source, available=item.available,
                             registered_videos=state.library.registered_videos(item.folder))


@router.get("", response_model=list[LibraryFolderInfo])
def folders(request: Request) -> list[LibraryFolderInfo]:
    """Library folders given at startup or added in the application; videos are selected only inside these."""
    state = request.app.state
    return [_folder_info(state, item) for item in state.library.configured()]


@router.post("/folders", response_model=FolderSelection)
def add_folder(request: Request, response: Response) -> FolderSelection:
    """Open the native folder dialog on this computer and add the chosen folder. 201 when it is new,
    200 when it was already a library folder or the dialog was cancelled."""
    state = request.app.state
    try:
        chosen = state.pick_folder()
    except PickerBusy as exc:
        raise HTTPException(409, str(exc)) from None
    except PickerUnavailable as exc:
        raise HTTPException(501, str(exc)) from None
    if chosen is None:
        return FolderSelection(folder=None, created=False)
    try:
        item, created = state.library.add(chosen, datetime.now(timezone.utc).isoformat(timespec="seconds"))
    except LibraryPathError as exc:
        raise HTTPException(exc.status_code, str(exc)) from None
    if created:
        response.status_code = 201
    return FolderSelection(folder=_folder_info(state, item), created=created)


@router.delete("/folders/{folder_id}", status_code=204)
def remove_folder(request: Request, folder_id: str = FOLDER_ID) -> Response:
    """Stop offering a folder. Files, registrations and analyses are kept; its videos become unavailable
    for playback and analysis until the folder is added again."""
    try:
        request.app.state.library.remove(folder_id)
    except LibraryPathError as exc:
        raise HTTPException(exc.status_code, str(exc)) from None
    return Response(status_code=204)


@router.get("/{folder_id}/entries", response_model=LibraryListing)
def entries(request: Request, folder_id: str = FOLDER_ID,
            path: str = Query("", max_length=4096)) -> LibraryListing:
    """Subfolders and video files (by extension) directly inside a library directory."""
    state = request.app.state
    try:
        item = state.library.get(folder_id)
        if not item.available:
            raise HTTPException(404, "the folder was moved, deleted or cannot be read")
        current, items, truncated = list_directory(state.library, folder_id, path,
                                                   exclude=[state.settings.data_dir])
    except LibraryPathError as exc:
        raise HTTPException(exc.status_code, str(exc)) from None
    registered = state.videos.registered_paths()
    entries = [LibraryEntryInfo(
        name=item_.name, path=item_.path, kind=item_.kind, size_bytes=item_.size_bytes,
        modified_at=(datetime.fromtimestamp(item_.mtime_ns / 1e9, timezone.utc).isoformat(timespec="seconds")
                     if item_.mtime_ns is not None else None),
        video_id=registered.get(str(item_.resolved)) if item_.resolved else None) for item_ in items]
    parent = None if not current else ("" if PurePosixPath(current).parent == PurePosixPath(".")
                                       else str(PurePosixPath(current).parent))
    return LibraryListing(folder=_folder_info(state, item), path=current, parent=parent, entries=entries,
                          truncated=truncated)
