"""Start the local Spectra application with one command.

    python3 run.py          # serve Spectra on :8765 and open it in the browser
    python3 run.py --dev    # API plus the Vite development server on :5173 (needs Node.js)

``--library DIR`` (repeatable) also offers a folder from the start.

Any Python may start it: the launcher replaces itself with the repository's ``.venv`` interpreter.
Node.js is needed to build the web application when it is missing or older than its sources, and for
``--dev``. ``--no-browser`` skips opening the browser.
Other options (``--port``, ``--data-dir``, ``--models``, repeated ``--library``, ``--log-level``) are passed
to ``python -m spectra.api``; ``--verbose`` shows the servers' routine logs. Warnings and errors always
show. Ctrl+C stops everything this script started.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
DEFAULT_PORT = 8765
DEV_PORT = 5173
DEV_ORIGIN = f"http://127.0.0.1:{DEV_PORT}"
REEXEC_ENV = "SPECTRA_RUN_REEXEC"
SETUP = ("Spectra's environment is missing ({venv}). Create it from the repository root:\n"
         "  python3.12 -m venv .venv\n"
         "  .venv/bin/python -m pip install -e .")
# The API cancels a running analysis and waits up to 120 s for its worker; the development server stops at once.
API_GRACE_S = 130
WEB_GRACE_S = 5
# Routine Vite lines that repeat the banner. Anything else (warnings, errors) passes through.
VITE_NOISE = re.compile(r"^(> |VITE v[\d.]+\s+ready|➜\s+(Local|Network|press)|$)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def ensure_venv(root: Path = ROOT, prefix: str = sys.prefix, argv: list[str] = sys.argv,
                environ=os.environ, execv=os.execv) -> None:
    """Continue in the repository's virtual environment, replacing this process (no child process,
    so arguments, signals and the exit code are the launcher's own). Never installs anything."""
    venv = root / ".venv"
    python = venv / "bin" / "python"
    if Path(prefix).resolve() == venv.resolve():
        return
    if not python.is_file():
        sys.exit(SETUP.format(venv=venv))
    if environ.get(REEXEC_ENV):  # the environment's interpreter did not report the environment as its prefix
        sys.exit(f"could not start Spectra with {python}; recreate the environment.\n" + SETUP.format(venv=venv))
    environ[REEXEC_ENV] = "1"
    execv(str(python), [str(python), str(root / "run.py"), *argv[1:]])


# Presentation ------------------------------------------------------------------------------------

def make_paint(tty: bool, environ=os.environ):
    """A colouring function that is a no-op off a terminal, with NO_COLOR set, or on a dumb terminal."""
    enabled = tty and "NO_COLOR" not in environ and environ.get("TERM") != "dumb"
    codes = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33"}

    def paint(text: str, *styles: str) -> str:
        if not enabled or not styles:
            return text
        return "".join(f"\033[{codes[s]}m" for s in styles) + f"{text}\033[0m"

    return paint


def problem(paint, title: str, *details: str) -> str:
    """One thing to read and, dimmed, what to do about it."""
    return "\n".join(["", f"  {paint(title, 'yellow')}", *(f"  {paint(d, 'dim')}" for d in details), ""])


def banner(paint, *, url: str, dev: bool, api_url: str) -> str:
    rows = [("Open", paint(url, "bold"))]
    if dev:
        rows.append(("API", api_url))
    rows.append(("Stop", "Ctrl+C"))
    title = paint("Spectra", "bold", "red") + (paint("  development", "dim") if dev else "")
    lines = ["", f"  {title}", f"  {paint('─' * 44, 'dim')}"]
    lines += [f"  {paint(label.ljust(10), 'dim')}{value}" for label, value in rows]
    return "\n".join([*lines, ""])


def option_values(args: list[str], name: str) -> list[str]:
    """Every value given for `name`, as `--name value` or `--name=value`."""
    values = []
    for i, arg in enumerate(args):
        if arg == name and i + 1 < len(args):
            values.append(args[i + 1])
        elif arg.startswith(name + "="):
            values.append(arg.split("=", 1)[1])
    return values


def forward(stream, noise: re.Pattern | None) -> None:
    """Copy a child's output to ours, minus routine lines."""
    for line in iter(stream.readline, ""):
        if noise is None or not noise.match(ANSI.sub("", line).strip()):
            sys.stdout.write(line)
            sys.stdout.flush()


# Processes ---------------------------------------------------------------------------------------

def port_is_free(port: int) -> bool:
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def accepts(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def start(command: list[str], cwd: Path, noise: re.Pattern | None = None) -> subprocess.Popen:
    """A child in its own process group, so Ctrl+C reaches only this launcher, which stops the children in
    order. stdout is filtered when `noise` is given; stderr (errors) always goes straight to the terminal."""
    if noise is None:
        return subprocess.Popen(command, cwd=cwd, start_new_session=True)
    process = subprocess.Popen(command, cwd=cwd, start_new_session=True, stdout=subprocess.PIPE, text=True,
                               encoding="utf-8", errors="replace", bufsize=1)
    threading.Thread(target=forward, args=(process.stdout, noise), daemon=True).start()
    return process


def stop(process: subprocess.Popen, sig: int, grace_s: float) -> None:
    """Signal the child's process group, wait, then kill what remains."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    except ProcessLookupError:
        pass


def settings_error(api_args: list[str]) -> str | None:
    """The API's own reason for refusing these options, as one sentence, or None when they are valid.
    Checked before anything starts so a typo never produces a banner, a usage dump and a stopped server."""
    from spectra.api.__main__ import parse_settings

    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            parse_settings(api_args)
    except SystemExit:
        message = stderr.getvalue().strip().splitlines()[-1] if stderr.getvalue().strip() else "invalid options"
        message = message.split(" error: ", 1)[-1]
        return message[:1].upper() + message[1:] + ("" if message.endswith(".") else ".")
    return None


def interrupt(*_) -> None:
    """SIGTERM takes the same path as Ctrl+C, so `kill` also stops both servers."""
    raise KeyboardInterrupt


def silence(*_) -> None:
    """The terminal is gone: discard the launcher's output from now on."""
    sys.stdout = sys.stderr = open(os.devnull, "w")


def hang_up(*_) -> None:
    """Closing the terminal (SIGHUP) also stops both servers. They run in their own process group and never
    see it, so without this they would outlive the launcher."""
    silence()
    raise KeyboardInterrupt


def quiet_run(paint, title: str, command: list[str]) -> bool:
    """Run a setup step with one status line; its output is shown only if it fails."""
    print(f"  {paint('…', 'dim')} {title}", flush=True)
    result = subprocess.run(command, cwd=FRONTEND, capture_output=True, text=True)
    if result.returncode == 0:
        return True
    print((result.stdout + result.stderr).rstrip()[-4000:], file=sys.stderr)
    print(problem(paint, f"{title.rstrip('…')} failed.", "Its output is above."), flush=True)
    return False


def build_is_current(frontend: Path = FRONTEND) -> bool:
    """Whether frontend/dist exists and is newer than every source file of the web application."""
    index = frontend / "dist" / "index.html"
    if not index.is_file():
        return False
    built = index.stat().st_mtime
    sources = [*frontend.joinpath("src").rglob("*"), *frontend.joinpath("public").rglob("*"),
               frontend / "index.html", frontend / "package-lock.json"]
    return all(p.stat().st_mtime <= built for p in sources if p.is_file())


def web_build_step(frontend: Path = FRONTEND, npm: bool = True) -> str:
    """What normal mode does about the web application: "serve" the current build, "build" it first, serve a
    "stale" build because npm is missing, or stop because it is "missing" and cannot be built."""
    if build_is_current(frontend):
        return "serve"
    if npm:
        return "build"
    return "stale" if (frontend / "dist" / "index.html").is_file() else "missing"


# Main --------------------------------------------------------------------------------------------

def main() -> int:
    ensure_venv()
    parser = argparse.ArgumentParser(description="Start the local Spectra application.")
    parser.add_argument("--dev", action="store_true", help=f"also run the Vite development server on :{DEV_PORT}")
    parser.add_argument("--verbose", action="store_true", help="show the servers' routine logs")
    parser.add_argument("--no-browser", action="store_true", help="do not open Spectra in the browser")
    args, api_args = parser.parse_known_args()
    paint = make_paint(sys.stdout.isatty())

    error = settings_error(api_args)
    if error:
        print(problem(paint, "Spectra could not start.", error), flush=True)
        return 2
    ports = option_values(api_args, "--port")
    port = int(ports[-1]) if ports else DEFAULT_PORT  # validated above
    npm = shutil.which("npm") is not None
    step = "dev" if args.dev else web_build_step(npm=npm)
    if step == "dev" and not npm:
        print(problem(paint, "npm was not found.", "Development mode needs Node.js 22.12 or newer."), flush=True)
        return 1
    if step == "missing":
        print(problem(paint, "The web application has not been built (frontend/dist).",
                      "Install Node.js 22.12 or newer; Spectra then builds it on start."), flush=True)
        return 1
    for name, p in (("API", port), *((("web", DEV_PORT),) if args.dev else ())):
        if not port_is_free(p):
            hint = f"python3 run.py --port {port + 1} …" if name == "API" else "Stop the other development server."
            print(problem(paint, f"Port {p} ({name}) is already in use.",
                          "Another copy is probably still running. Stop it with Ctrl+C, or:", hint), flush=True)
            return 1
    if step == "stale":
        print(problem(paint, "The web application build is older than its sources.",
                      "Serving it anyway; install Node.js 22.12 or newer to rebuild it."), flush=True)
    if step in ("dev", "build") and not (FRONTEND / "node_modules").is_dir() \
            and not quiet_run(paint, "Installing web dependencies…", ["npm", "ci"]):
        return 1
    if step == "build" and not quiet_run(paint, "Building the web application…", ["npm", "run", "build"]):
        return 1

    api = [sys.executable, "-m", "spectra.api", *api_args]
    if not option_values(api_args, "--log-level"):
        api += ["--log-level", "info" if args.verbose else "warning"]
    if args.dev:
        api += ["--allow-origin", DEV_ORIGIN]
    servers: list[tuple[str, subprocess.Popen, int, int, float]] = []  # name, process, port, stop signal, grace
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, hang_up)
    try:
        servers.append(("API", start(api, ROOT), port, signal.SIGINT, API_GRACE_S))
        if args.dev:
            web = start(["npm", "run", "dev", "--", "--clearScreen", "false"], FRONTEND,
                        None if args.verbose else VITE_NOISE)
            servers.append(("Web", web, DEV_PORT, signal.SIGTERM, WEB_GRACE_S))
    except OSError as exc:
        for _, process, _, sig, grace in servers:
            stop(process, sig, grace)
        print(problem(paint, f"Could not start a server: {exc}"), flush=True)
        return 1

    url = DEV_ORIGIN if args.dev else f"http://127.0.0.1:{port}"
    print(banner(paint, url=url, dev=args.dev, api_url=f"http://127.0.0.1:{port}/api"), flush=True)

    code, died = 0, None
    waiting = {name: p for name, _, p, _, _ in servers}
    try:
        while True:
            for name, process, *_ in servers:
                if process.poll() is not None:
                    code, died = process.returncode or 1, name
                    raise SystemExit
            for name, p in list(waiting.items()):
                if accepts(p):
                    del waiting[name]
                    print(f"  {paint('✓', 'green')} {name} ready", flush=True)
                    if not waiting and not args.no_browser:
                        webbrowser.open(url)
            time.sleep(0.2)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl+C must not cut the orderly stop short
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, silence)  # closing the terminal now must not cut the stop short either
        if died is None:
            print(f"\n  {paint('Stopping…', 'dim')}", flush=True)
        for _, process, _, sig, grace in reversed(servers):
            stop(process, sig, grace)

    if died:
        others = " The other server was stopped too." if len(servers) > 1 else ""
        print(problem(paint, f"The {died} server stopped.{others}",
                      "Its error is above." if died == "API" else "The development server's output is above."),
              flush=True)
    else:
        print(f"  {paint('Stopped.', 'dim')}\n", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
