"""Serving the built web application from the API server, on the same origin as the API.

Only files inside the build directory are served. Any other path outside ``/api`` returns the
application's ``index.html`` so client-side routes survive a reload; ``/api`` paths never fall back.
The page is limited by a Content-Security-Policy that allows only this origin.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self'; "
    "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; "
    "frame-ancestors 'none'"
)
PAGE_HEADERS = {"cache-control": "no-cache", "content-security-policy": CONTENT_SECURITY_POLICY}
HASHED_ASSET_HEADERS = {"cache-control": "public, max-age=31536000, immutable"}


def is_build(directory: Path) -> bool:
    return (directory / "index.html").is_file()


def mount_frontend(app: FastAPI, directory: Path) -> None:
    """Serve ``directory`` (a production build). Call after every API router has been included."""
    root = directory.resolve()
    index = root / "index.html"

    @app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        parts = PurePosixPath(path).parts
        if parts[:1] == ("api",):
            raise HTTPException(404, "Not Found")
        if parts and not any(part.startswith(".") for part in parts):
            candidate = (root / path).resolve()
            if candidate.is_relative_to(root) and candidate.is_file():
                hashed = parts[0] == "assets"
                return FileResponse(candidate, headers=HASHED_ASSET_HEADERS if hashed else {"cache-control": "no-cache"})
            if "." in parts[-1]:
                raise HTTPException(404, "Not Found")
        return FileResponse(index, media_type="text/html", headers=PAGE_HEADERS)
