"""Video selection from library folders chosen by the person who starts the server.

Browsers never reveal a selected file's absolute path, and a page served on localhost must not be
able to name arbitrary files. The server is therefore started with one or more library folders;
clients address a video as (folder id, path relative to that folder). A reference resolves only to
a readable regular file inside the folder after symlinks are followed. Videos are read in place
and never copied, written, renamed or deleted.
"""
from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional

# Candidates offered for selection. The extension only filters the listing; the container and codec
# are identified by probing, never trusted from the name.
VIDEO_EXTENSIONS = frozenset({".mp4", ".m4v", ".mov", ".webm", ".mkv", ".ogv", ".ogg", ".avi", ".ts",
                              ".mts", ".m2ts", ".3gp", ".hevc", ".h264", ".h265", ".264", ".265"})
MAX_ENTRIES = 2000


class LibraryPathError(ValueError):
    """A reference that does not name a usable entry inside a library folder (HTTP 400)."""
    status_code = 400


class LibraryNotFound(LibraryPathError):
    status_code = 404


class UnsupportedFileType(LibraryPathError):
    status_code = 415


@dataclass(frozen=True)
class LibraryFolder:
    id: str
    path: Path

    @property
    def name(self) -> str:
        return self.path.name or str(self.path)


@dataclass(frozen=True)
class LibraryEntry:
    name: str
    path: str  # relative to the library folder, POSIX separators
    kind: str  # "directory" or "video"
    size_bytes: Optional[int] = None
    mtime_ns: Optional[int] = None
    resolved: Optional[Path] = None  # never sent to clients


def library_folders(paths: Iterable[Path]) -> tuple[LibraryFolder, ...]:
    """Identify each resolved folder by a stable digest of its path, independent of option order."""
    return tuple(LibraryFolder(hashlib.sha256(str(path).encode()).hexdigest()[:16], path) for path in paths)


def find_folder(folders: Iterable[LibraryFolder], folder_id: str) -> LibraryFolder:
    folder = next((f for f in folders if f.id == folder_id), None)
    if folder is None:
        raise LibraryNotFound("unknown library folder")
    return folder


def _resolve(folder: LibraryFolder, relative_path: str, exclude: Iterable[Path], *, allow_root: bool) -> Path:
    if "\x00" in relative_path:
        raise LibraryPathError("the path contains an invalid character")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise LibraryPathError("the path must stay inside the library folder")
    if not relative.parts and not allow_root:
        raise LibraryPathError("a file path inside the library folder is required")
    try:
        resolved = folder.path.joinpath(*relative.parts).resolve(strict=True)
    except (OSError, RuntimeError):
        raise LibraryNotFound("not found in the library folder") from None
    if not resolved.is_relative_to(folder.path):
        raise LibraryPathError("the path must stay inside the library folder")
    if any(resolved.is_relative_to(path) for path in exclude):
        raise LibraryPathError("application data cannot be used as a source video")
    return resolved


def _regular_readable(path: Path) -> os.stat_result:
    try:
        info = path.stat()
    except OSError:
        raise LibraryNotFound("not found in the library folder") from None
    if not stat.S_ISREG(info.st_mode):
        raise LibraryPathError("the path is not a regular file")
    if not os.access(path, os.R_OK):
        raise LibraryPathError("the file is not readable")
    return info


def resolve_video_reference(folders: Iterable[LibraryFolder], folder_id: str, relative_path: str,
                            *, exclude: Iterable[Path] = ()) -> Path:
    """Return the real path of a library video file, refusing anything outside its folder.

    ``exclude`` lists resolved directories (such as the data directory) that never hold sources.
    The check opens nothing, so a FIFO or device in a library folder cannot block the server.
    """
    resolved = _resolve(find_folder(folders, folder_id), relative_path, exclude, allow_root=False)
    _regular_readable(resolved)
    if resolved.suffix.lower() not in VIDEO_EXTENSIONS:
        raise UnsupportedFileType(f"{resolved.suffix or 'files without an extension'} is not a supported "
                                  f"video file type")
    return resolved


def list_directory(folders: Iterable[LibraryFolder], folder_id: str, relative_path: str = "",
                   *, exclude: Iterable[Path] = ()) -> tuple[str, list[LibraryEntry], bool]:
    """Subfolders and video candidates directly inside a library directory.

    Returns (normalised relative path, entries, truncated). Hidden entries, entries that resolve
    outside the folder or into excluded directories, and unreadable files are omitted.
    """
    exclude = tuple(exclude)
    folder = find_folder(folders, folder_id)
    directory = _resolve(folder, relative_path, exclude, allow_root=True)
    if not directory.is_dir():
        raise LibraryPathError("the path is not a folder")
    base = PurePosixPath(*directory.relative_to(folder.path).parts) if directory != folder.path else None
    entries: list[LibraryEntry] = []
    truncated = False
    try:
        children = sorted(os.scandir(directory), key=lambda e: e.name.lower())
    except OSError:
        raise LibraryPathError("the folder cannot be read") from None
    for child in children:
        if child.name.startswith("."):
            continue
        try:
            target = Path(child.path).resolve(strict=True)
            info = target.stat()
        except (OSError, RuntimeError):
            continue
        if not target.is_relative_to(folder.path) or any(target.is_relative_to(p) for p in exclude):
            continue
        relative = str(base / child.name) if base else child.name
        if stat.S_ISDIR(info.st_mode):
            entry = LibraryEntry(child.name, relative, "directory")
        elif (stat.S_ISREG(info.st_mode) and Path(child.name).suffix.lower() in VIDEO_EXTENSIONS
              and os.access(target, os.R_OK)):
            entry = LibraryEntry(child.name, relative, "video", info.st_size, info.st_mtime_ns, target)
        else:
            continue
        if len(entries) == MAX_ENTRIES:
            truncated = True
            break
        entries.append(entry)
    entries.sort(key=lambda e: (e.kind != "directory", e.name.lower()))
    return ("" if base is None else str(base)), entries, truncated


def locate(folders: Iterable[LibraryFolder], path: Path) -> Optional[tuple[LibraryFolder, str]]:
    """The configured folder containing a stored real path, with the path relative to it."""
    for folder in folders:
        if path.is_relative_to(folder.path):
            return folder, str(PurePosixPath(*path.relative_to(folder.path).parts))
    return None


class FolderRejected(LibraryPathError):
    """A chosen folder that cannot become a library folder (HTTP 400)."""


class FolderNotRemovable(LibraryPathError):
    """A folder given on the command line; it returns on every start (HTTP 409)."""
    status_code = 409


@dataclass(frozen=True)
class ConfiguredFolder:
    folder: LibraryFolder
    source: str  # "startup" (command line) or "added" (chosen in the application)
    available: bool  # the folder exists and is a readable directory now


def validate_folder(raw: Path, data_dir: Path) -> Path:
    """The resolved path of a directory that may hold source videos, or FolderRejected."""
    try:
        path = Path(raw).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise FolderRejected("the folder does not exist") from None
    if not path.is_dir():
        raise FolderRejected("the selection is not a folder")
    if path == Path(path.anchor):
        raise FolderRejected("a whole disk cannot be a library folder; choose a folder on it")
    if path.is_relative_to(data_dir):
        raise FolderRejected("Spectra's own data folder cannot be a library folder")
    if not os.access(path, os.R_OK | os.X_OK):
        raise FolderRejected("the folder cannot be read")
    return path


class LibraryFolders:
    """The allowlist of library folders: those given at startup plus those chosen in the application and
    stored in the database. Iterating yields the folders usable now, so a folder added or removed takes
    effect immediately. Removing a folder only removes it from the allowlist; no file is touched."""

    def __init__(self, database, startup: Iterable[Path], data_dir: Path) -> None:
        self.database = database
        self.data_dir = data_dir
        self.startup = library_folders(startup)

    def _added(self) -> list[LibraryFolder]:
        with self.database.connect() as conn:
            rows = conn.execute("SELECT id, path FROM library_folders ORDER BY rowid").fetchall()
        return [LibraryFolder(row["id"], Path(row["path"])) for row in rows]

    def configured(self) -> list[ConfiguredFolder]:
        startup_ids = {f.id for f in self.startup}
        items = [("startup", f) for f in self.startup]
        items += [("added", f) for f in self._added() if f.id not in startup_ids]
        return [ConfiguredFolder(f, source, _is_readable_dir(f.path)) for source, f in items]

    def __iter__(self):
        return iter([item.folder for item in self.configured() if item.available])

    def __len__(self) -> int:
        return len(self.configured())

    def get(self, folder_id: str) -> ConfiguredFolder:
        item = next((i for i in self.configured() if i.folder.id == folder_id), None)
        if item is None:
            raise LibraryNotFound("unknown library folder")
        return item

    def add(self, raw: Path, added_at: str) -> tuple[ConfiguredFolder, bool]:
        """Store a chosen folder; returns (folder, created). A folder already present is returned as is."""
        path = validate_folder(raw, self.data_dir)
        folder = library_folders([path])[0]
        with self.database.connect() as conn:
            created = conn.execute("INSERT INTO library_folders (id, path, added_at) VALUES (?, ?, ?) "
                                   "ON CONFLICT DO NOTHING", (folder.id, str(path), added_at)).rowcount == 1
        if any(f.id == folder.id for f in self.startup):
            created = False
        return self.get(folder.id), created

    def remove(self, folder_id: str) -> ConfiguredFolder:
        item = self.get(folder_id)
        if item.source == "startup":
            raise FolderNotRemovable("this folder was given when Spectra started; start it without that folder")
        with self.database.connect() as conn:
            conn.execute("DELETE FROM library_folders WHERE id = ?", (folder_id,))
        return item

    def registered_videos(self, folder: LibraryFolder) -> int:
        """Registered videos stored under the folder's path."""
        with self.database.connect() as conn:
            paths = [Path(row[0]) for row in conn.execute("SELECT path FROM videos")]
        return sum(path.is_relative_to(folder.path) for path in paths)


def _is_readable_dir(path: Path) -> bool:
    try:
        return path.resolve(strict=True) == path and path.is_dir() and os.access(path, os.R_OK | os.X_OK)
    except (OSError, RuntimeError):
        return False
