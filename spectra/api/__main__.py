"""Start the local server: ``python -m spectra.api`` (``--library DIR`` also offers a folder from the start)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from spectra.api.frontend import is_build
from spectra.api.settings import DEFAULT_PORT, FRONTEND_BUILD, HOST, Settings, SettingsError


def parse_settings(argv: Optional[Sequence[str]] = None) -> Settings:
    parser = argparse.ArgumentParser(prog="python -m spectra.api",
                                     description=f"Serve Spectra on http://{HOST}:PORT (loopback only).")
    defaults = Settings.__dataclass_fields__
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--data-dir", type=Path, default=defaults["data_dir"].default,
                        help="application state, caches and evidence (default: var/ in the repository)")
    parser.add_argument("--models", type=Path, default=defaults["models_dir"].default,
                        help="runtime model registry folder (default: models/engine)")
    parser.add_argument("--library", type=Path, action="append", default=[], metavar="DIR",
                        help="folder offered from the start, in addition to folders added in the application; repeatable")
    parser.add_argument("--allow-origin", action="append", default=[], metavar="URL",
                        help="extra loopback page origin, such as a frontend development server")
    parser.add_argument("--log-level", default="info", choices=("debug", "info", "warning", "error"),
                        help="server log detail (default: info); warnings and errors always show")
    parser.add_argument("--frontend", type=Path, default=None, metavar="DIR",
                        help="web application build to serve at / (default: frontend/dist when it has been "
                             "built; otherwise only the API is served)")
    args = parser.parse_args(argv)
    if args.frontend is not None and not is_build(args.frontend.expanduser()):
        parser.error(f"no web application build (index.html) in {args.frontend}")
    frontend = args.frontend if args.frontend is not None else (FRONTEND_BUILD if is_build(FRONTEND_BUILD) else None)
    try:
        return Settings(data_dir=args.data_dir, models_dir=args.models, port=args.port,
                        library_dirs=tuple(args.library), extra_origins=tuple(args.allow_origin),
                        frontend_dir=frontend)
    except SettingsError as exc:
        parser.error(str(exc))


# Open responses (an analysis event stream follows a job until it ends) get this long to finish after a stop
# signal; then they are closed and the application's shutdown cancels the running analysis.
GRACEFUL_SHUTDOWN_S = 3


def server_options(port: int, log_level: str = "info") -> dict:
    """uvicorn options for the local server; the application has no WebSocket routes."""
    return dict(host=HOST, port=port, proxy_headers=False, server_header=False, log_level=log_level, ws="none",
                timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S)


def main(argv: Optional[Sequence[str]] = None) -> None:
    settings = parse_settings(argv)  # validates every option, --log-level included
    log_level = argparse.ArgumentParser(add_help=False)
    log_level.add_argument("--log-level", default="info")
    import uvicorn

    from spectra.api.app import create_app

    uvicorn.run(create_app(settings), **server_options(settings.port, log_level.parse_known_args(argv)[0].log_level))


if __name__ == "__main__":
    main()
