"""Entrypoint: `python run.py` starts Spectra. Ctrl+C stops it.

Spectra needs three model files that git does not carry and a set of heavy
dependencies, so most first runs fail on setup rather than on code. Uvicorn
cannot say anything useful about that: it reports the import error or the
taken port in its own words, several frames deep. Everything below runs
before the server does, so the first thing anyone sees is either the banner
or the one command that fixes what is missing.
"""

import importlib.util
import logging
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DEFAULT_PORT = 8000

# Import name -> what to say it is, in the order the pipeline needs them.
REQUIRED_PACKAGES = (
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("multipart", "python-multipart"),
    ("cv2", "OpenCV"),
    ("numpy", "NumPy"),
    ("PIL", "Pillow"),
    ("onnxruntime", "ONNX Runtime"),
    ("ultralytics", "Ultralytics"),
    ("torch", "PyTorch"),
)


def _paint(tty):
    """Return a colouring function that is a no-op unless stdout is a terminal."""
    codes = {"bold": "1", "dim": "2", "green": "32", "yellow": "33"}

    def paint(text, *styles):
        if not tty:
            return text
        prefix = "".join(f"\033[{codes[style]}m" for style in styles)
        return f"{prefix}{text}\033[0m"

    return paint


def _problem(paint, title, *fix):
    """Format a setup problem as the one thing to read and the command to run."""
    lines = ["", f"  {paint(title, 'yellow')}"]
    lines += [f"  {paint(line, 'dim')}" for line in fix]
    lines.append("")
    return "\n".join(lines)


def _missing_packages():
    """Which required distributions are not importable, by display name.

    ``find_spec`` locates a package without importing it. That matters here:
    importing Ultralytics alone costs several seconds, which is a long time to
    spend on a check that passes on every run after the first.
    """
    missing = []
    for module_name, label in REQUIRED_PACKAGES:
        try:
            found = importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(label)
    return missing


def _venv_python():
    """The project venv's interpreter, if that venv exists."""
    for relative in ("bin/python", "Scripts/python.exe"):
        candidate = ROOT / ".venv" / relative
        if candidate.exists():
            return candidate
    return None


def _running_in_venv():
    """Whether this interpreter is the project venv's.

    Compares prefixes rather than executables: a venv's `python` is usually a
    symlink to the interpreter it was built from, so resolving the two paths
    makes them equal even when the imports differ.
    """
    return Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()


def _reexec_into_venv():
    """Hand the process to the venv interpreter, or return if there is none.

    `python run.py` without activating first is the ordinary way to start
    this, and it fails on every import while the packages sit one directory
    away. Switching interpreters is what the activation would have done, so
    it happens here instead of being explained in an error.
    """
    venv_python = _venv_python()
    if venv_python is None or _running_in_venv():
        return
    os.environ["SPECTRA_REEXEC"] = "1"
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


def _python_hint():
    """How to get an interpreter that has the dependencies.

    Only reached when the venv is missing or when it is the venv itself that
    is short a package — anything else has already been switched over.
    """
    if _venv_python() is None:
        return ("python -m venv .venv", "source .venv/bin/activate")
    return ()


def _model_status():
    """Each required model file, whether it is present, and how to get it.

    The paths come from the vision modules rather than from a second list
    here, so `SPECTRA_DEPTH_MODEL` and `SPECTRA_LANENET_MODEL` point this
    check at the same file the pipeline will load.
    """
    from spectra.vision.detection import YOLO_MODEL_PATH
    from spectra.vision.lanenet import LANENET_MODEL_PATH
    from spectra.vision.models import DEPTH_MODEL_PATH

    return (
        ("depth", DEPTH_MODEL_PATH, "scripts/download_depth_model.py"),
        ("lane", LANENET_MODEL_PATH, "scripts/download_lanenet_model.py"),
        ("detector", YOLO_MODEL_PATH, "scripts/download_yolo_model.py"),
    )


def _setup_problem(paint):
    """Return the first missing piece of local setup, or None if all is present."""
    missing = _missing_packages()
    if missing:
        return _problem(
            paint,
            f"Missing dependencies: {', '.join(missing)}.",
            *_python_hint(),
            "pip install -r requirements.txt",
        )

    absent = [(name, script) for name, path, script in _model_status() if not path.is_file()]
    if absent:
        # All three are hard requirements — analysis refuses to start without
        # any one of them — so every missing script is listed at once rather
        # than one per attempted run.
        return _problem(
            paint,
            "Model files are missing: " + ", ".join(name for name, _ in absent) + ".",
            "They are large runtime artifacts and are not in git. Run:",
            *(f"    python {script}" for _, script in absent),
        )
    return None


def _onnx_runtime_line():
    """Which execution provider the ONNX models will actually get."""
    from spectra.vision.models import build_coreml_providers

    providers = build_coreml_providers()
    names = [p[0] if isinstance(p, tuple) else p for p in providers]
    if "CoreMLExecutionProvider" in names:
        return "CoreML, CPU fallback"
    return "CPU"


def _banner(port, reload_enabled, paint):
    rows = [
        ("Open", paint(f"http://localhost:{port}", "bold")),
        ("Models", "depth, lane, detector ready"),
        ("ONNX", _onnx_runtime_line()),
        ("Stop", "Ctrl+C"),
    ]
    lines = [
        "",
        f"  {paint('Spectra', 'bold', 'green')}",
        f"  {paint('─' * 44, 'dim')}",
    ]
    lines += [f"  {paint(label.ljust(10), 'dim')}{value}" for label, value in rows]
    if reload_enabled:
        # Worth saying on screen: a reload drops the loaded models, so the next
        # analysis pays the full load cost again.
        lines += [
            "",
            f"  {paint('! Auto-reload on: the models are reloaded on every file change.', 'yellow')}",
        ]
    lines.append("")
    return "\n".join(lines)


def _port_is_free(port):
    """Whether the app can still take `port`, checked before Uvicorn does.

    Uvicorn handles this failure itself and exits with a bare "Address already
    in use", which does not say what is holding the port or what to do about
    it. Asking first is what keeps the message ours.
    """
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _port_in_use(port, paint):
    return "\n".join((
        "",
        f"  {paint('Port ' + str(port) + ' is already in use.', 'yellow')}",
        f"  {paint('Another copy is probably still running. Stop it there with', 'dim')}",
        f"  {paint('Ctrl+C, or start this one elsewhere:', 'dim')}",
        "",
        f"      PORT={port + 1} python run.py",
        "",
    ))


def _quiet_framework_output(log_level):
    """Silence Uvicorn's startup notes and its per-request access log.

    Uvicorn logs one line per request, and a single analysis produces a burst
    of preview traffic that buries anything the app itself has to say.
    LOG_LEVEL=DEBUG is the way back to all of it.
    """
    if log_level == "DEBUG":
        return
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def main():
    paint = _paint(sys.stdout.isatty())
    _reexec_into_venv()

    if os.environ.get("SPECTRA_REEXEC") == "1":
        # Said once, because which interpreter is serving is worth knowing when
        # the one that was typed is not the one that ended up running.
        print(f"\n  {paint('Switched to .venv, where the dependencies are.', 'dim')}", flush=True)

    problem = _setup_problem(paint)
    if problem:
        print(problem, flush=True)
        return 1

    port = int(os.environ.get("PORT", DEFAULT_PORT))
    if not _port_is_free(port):
        print(_port_in_use(port, paint), flush=True)
        return 1

    # Off by default: a reload discards the loaded depth, lane and detector
    # models, so it is a real cost here rather than the free convenience it is
    # in an app that holds nothing.
    reload_enabled = os.environ.get("RELOAD", "").strip().lower() in {"1", "true", "yes", "on"}
    log_level = os.environ.get("LOG_LEVEL", "").strip().upper()

    import uvicorn

    _quiet_framework_output(log_level)

    # flush: a piped stdout is block-buffered, and a banner that only appears
    # once the server is stopped is worse than none.
    print(_banner(port, reload_enabled, paint), flush=True)

    try:
        uvicorn.run(
            "spectra.app:app",
            host="127.0.0.1",
            port=port,
            reload=reload_enabled,
            log_level=log_level.lower() if log_level else "warning",
        )
    except KeyboardInterrupt:
        pass

    # Uploads are written to a temporary directory that is removed as soon as
    # the analysis returns, so nothing of the video outlives this process.
    print(f"\n  {paint('Stopped. The uploaded video is gone with it.', 'dim')}\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
