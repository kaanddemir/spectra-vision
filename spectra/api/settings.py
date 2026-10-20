"""Server configuration. The server listens on the IPv4 loopback interface only."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HOST = "127.0.0.1"
LOCAL_HOSTNAMES = ("127.0.0.1", "localhost")
DEFAULT_PORT = 8765
FRONTEND_BUILD = REPOSITORY_ROOT / "frontend" / "dist"


class SettingsError(ValueError):
    """Invalid configuration, reported before the server starts."""


def _loopback_origin(origin: str) -> str:
    parts = urlsplit(origin)
    try:
        port = parts.port
    except ValueError:
        port = None
    if (parts.scheme != "http" or parts.hostname not in LOCAL_HOSTNAMES or port is None
            or parts.username or parts.password or parts.path or parts.query or parts.fragment):
        raise SettingsError(f"allowed origin must look like http://localhost:PORT, got {origin!r}")
    return f"http://{parts.hostname}:{port}"


@dataclass(frozen=True)
class Settings:
    """Paths and limits for one server. Paths are resolved; library folders must exist."""

    data_dir: Path = REPOSITORY_ROOT / "var"
    models_dir: Path = REPOSITORY_ROOT / "models" / "engine"
    port: int = DEFAULT_PORT
    library_dirs: tuple[Path, ...] = ()
    extra_origins: tuple[str, ...] = ()
    frontend_dir: Optional[Path] = None  # a production build of the web application to serve, if any

    def __post_init__(self) -> None:
        if not isinstance(self.port, int) or isinstance(self.port, bool) or not 1 <= self.port <= 65535:
            raise SettingsError(f"port must be between 1 and 65535, got {self.port!r}")
        data_dir = Path(self.data_dir).expanduser().resolve()
        libraries: list[Path] = []
        for raw in self.library_dirs:
            try:
                path = Path(raw).expanduser().resolve(strict=True)
            except (OSError, RuntimeError):
                raise SettingsError(f"library folder does not exist: {raw}") from None
            if not path.is_dir():
                raise SettingsError(f"library folder is not a directory: {raw}")
            if path.is_relative_to(data_dir):
                raise SettingsError(f"library folder must not be inside the data directory: {raw}")
            if path not in libraries:
                libraries.append(path)
        origins = tuple(dict.fromkeys(_loopback_origin(origin) for origin in self.extra_origins))
        object.__setattr__(self, "data_dir", data_dir)
        object.__setattr__(self, "models_dir", Path(self.models_dir).expanduser().resolve())
        object.__setattr__(self, "library_dirs", tuple(libraries))
        object.__setattr__(self, "extra_origins", origins)
        if self.frontend_dir is not None:
            object.__setattr__(self, "frontend_dir", Path(self.frontend_dir).expanduser().resolve())

    @property
    def database_path(self) -> Path:
        return self.data_dir / "spectra.sqlite3"

    @property
    def cache_dir(self) -> Path:
        """Perception cache written by analysis workers."""
        return self.data_dir / "cache"

    @property
    def evidence_dir(self) -> Path:
        """Evidence store: timelines, packages and on-request clips."""
        return self.data_dir / "evidence"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def allowed_origins(self) -> frozenset[str]:
        own = {f"http://{name}:{self.port}" for name in LOCAL_HOSTNAMES}
        return frozenset(own | set(self.extra_origins))

    def prepare(self) -> None:
        """Create the private data directories. Source videos are never stored here."""
        for path in (self.data_dir, self.cache_dir, self.evidence_dir, self.tmp_dir):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
