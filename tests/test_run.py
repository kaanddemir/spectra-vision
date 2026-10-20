"""The root launcher: its virtual-environment hand-over, terminal presentation and server lifecycle."""
import importlib.util
import os
import pty
import re
import select
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("spectra_run", ROOT / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def fake_root(tmp_path, with_venv=True):
    if with_venv:
        python = tmp_path / ".venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("")
    return tmp_path


def test_inside_the_environment_nothing_happens(tmp_path):
    root = fake_root(tmp_path)
    calls = []
    run.ensure_venv(root, str(root / ".venv"), ["run.py"], {}, lambda *a: calls.append(a))
    assert calls == []


def test_another_interpreter_is_replaced_with_all_arguments(tmp_path):
    root = fake_root(tmp_path)
    calls, environ = [], {}
    argv = ["run.py", "--library", "/videos one", "--dev", "--port", "9001"]
    run.ensure_venv(root, "/usr", argv, environ, lambda *a: calls.append(a))
    python = str(root / ".venv" / "bin" / "python")
    assert calls == [(python, [python, str(root / "run.py"), *argv[1:]])]
    assert environ[run.REEXEC_ENV] == "1"


def test_a_missing_environment_explains_the_setup_and_installs_nothing(tmp_path):
    root = fake_root(tmp_path, with_venv=False)
    calls = []
    with pytest.raises(SystemExit) as stop:
        run.ensure_venv(root, "/usr", ["run.py"], {}, lambda *a: calls.append(a))
    assert "python3.12 -m venv .venv" in str(stop.value.code)
    assert calls == [] and not (root / ".venv").exists()


def test_a_second_replacement_is_refused_instead_of_looping(tmp_path):
    root = fake_root(tmp_path)
    calls = []
    with pytest.raises(SystemExit) as stop:
        run.ensure_venv(root, "/usr", ["run.py"], {run.REEXEC_ENV: "1"}, lambda *a: calls.append(a))
    assert "recreate the environment" in str(stop.value.code)
    assert calls == []


def test_system_python_starts_the_launcher_in_the_environment():
    base = getattr(sys, "_base_executable", None)
    if not base or sys.prefix == sys.base_prefix:
        pytest.skip("not running inside a virtual environment")
    result = subprocess.run([base, str(ROOT / "run.py"), "--help"], capture_output=True, text=True, timeout=60,
                            env={"PATH": "/usr/bin:/bin"})
    assert result.returncode == 0, result.stderr
    assert "--dev" in result.stdout
    # The replacement process's exit code is the launcher's own: an argument error stays 2.
    failed = subprocess.run([base, str(ROOT / "run.py"), "--dev=1"], capture_output=True, text=True, timeout=60,
                            env={"PATH": "/usr/bin:/bin"})
    assert failed.returncode == 2 and "--dev" in failed.stderr


# Presentation ------------------------------------------------------------------------------------

ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


@pytest.mark.parametrize("tty, environ, coloured", [(True, {}, True), (False, {}, False),
                                                     (True, {"NO_COLOR": ""}, False), (True, {"TERM": "dumb"}, False)])
def test_colour_only_on_a_capable_terminal(tty, environ, coloured):
    text = run.make_paint(tty, environ)("Spectra", "bold", "red")
    assert ESCAPE.sub("", text) == "Spectra"
    assert (text != "Spectra") is coloured


def test_banner_lists_only_what_is_needed():
    plain = run.make_paint(False, {})
    normal = run.banner(plain, url="http://127.0.0.1:8765", dev=False, api_url="http://127.0.0.1:8765/api")
    assert [line.split()[0] for line in normal.strip().splitlines()] == ["Spectra", "─" * 44, "Open", "Stop"]
    assert "Videos" not in normal and "Data" not in normal
    dev = run.banner(plain, url=run.DEV_ORIGIN, dev=True, api_url="http://127.0.0.1:8765/api")
    assert "development" in dev and "API       http://127.0.0.1:8765/api" in dev


def test_options_are_read_in_both_spellings():
    args = ["--library", "/a", "--port=9001", "--library=/b c", "--data-dir", "x"]
    assert run.option_values(args, "--library") == ["/a", "/b c"]
    assert run.option_values(args, "--port") == ["9001"]
    assert run.option_values(args, "--models") == []


def test_routine_development_server_lines_are_hidden_and_problems_shown():
    routine = ["> spectra-frontend@0.0.0 dev", "  VITE v8.3.1  ready in 312 ms", "  ➜  Local:   http://127.0.0.1:5173/",
               "  ➜  press h + enter to show help", ""]
    shown = ["[vite] Internal server error: Failed to resolve import", "warning: something changed"]
    assert all(run.VITE_NOISE.match(run.ANSI.sub("", line).strip()) for line in routine)
    assert not any(run.VITE_NOISE.match(line) for line in shown)


@pytest.mark.parametrize("args, message", [(["--library", "/nowhere/videos"], "Library folder does not exist"),
                                           (["--port", "70000"], "Port must be between 1 and 65535")])
def test_invalid_options_are_refused_before_anything_starts(args, message):
    assert run.settings_error(args).startswith(message)
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), *args], capture_output=True, text=True, timeout=60)
    assert result.returncode == 2
    assert "Spectra could not start." in result.stdout and message in result.stdout
    assert "usage:" not in result.stdout + result.stderr and "Open" not in result.stdout


def web_tree(tmp_path, built: bool, newer_sources: bool = False):
    frontend = tmp_path / "frontend"
    (frontend / "src").mkdir(parents=True)
    source = frontend / "src" / "main.tsx"
    source.write_text("source")
    if built:
        (frontend / "dist").mkdir()
        (frontend / "dist" / "index.html").write_text("built")
        built_at = (frontend / "dist" / "index.html").stat().st_mtime
        os.utime(source, (built_at + (10 if newer_sources else -10),) * 2)
    return frontend


@pytest.mark.parametrize("built, newer, npm, step", [
    (True, False, False, "serve"),   # a current build needs no Node.js
    (True, False, True, "serve"),
    (True, True, True, "build"),     # sources changed
    (True, True, False, "stale"),    # ... and no npm to rebuild: serve the existing build
    (False, False, True, "build"),
    (False, False, False, "missing"),
])
def test_the_web_application_needs_npm_only_to_build_it(tmp_path, built, newer, npm, step):
    assert run.web_build_step(web_tree(tmp_path, built, newer), npm=npm) == step


def test_a_busy_port_is_reported_with_the_way_out():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        result = subprocess.run([sys.executable, str(ROOT / "run.py"), "--port", str(port)], capture_output=True,
                                text=True, timeout=60)
    assert result.returncode == 1
    assert f"Port {port} (API) is already in use." in result.stdout and f"--port {port + 1}" in result.stdout


# Lifecycle ---------------------------------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def in_terminal(args, ready: list[str], environ=None, timeout_s=90, close_terminal=False) -> tuple[str, int]:
    """Run the launcher in a pseudo-terminal until every `ready` text appeared, then send SIGTERM, or with
    `close_terminal` close the terminal as a closed window does (the launcher then receives SIGHUP)."""
    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover - the child process
        os.chdir(ROOT)
        os.execve(sys.executable, [sys.executable, "run.py", "--no-browser", *args], {**os.environ, **(environ or {})})
    out, sent, deadline = b"", False, time.monotonic() + timeout_s
    while True:
        text = out.decode(errors="replace")
        if not sent and (all(r in text for r in ready) or time.monotonic() > deadline):
            sent = True
            if close_terminal:
                os.close(fd)
                break
            os.kill(pid, signal.SIGTERM)
        if select.select([fd], [], [], 0.2)[0]:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                chunk = b""
            if not chunk:
                break
            out += chunk
    _, status = os.waitpid(pid, 0)
    return out.decode(errors="replace").replace("\r", ""), os.waitstatus_to_exitcode(status)


def launcher_children(data_dir) -> list[str]:
    ps = subprocess.run(["ps", "-A", "-o", "command="], capture_output=True, text=True).stdout
    return [line for line in ps.splitlines() if str(data_dir) in line]


def test_normal_mode_starts_quietly_and_stops_cleanly(tmp_path):
    library, data = tmp_path / "videos", tmp_path / "state"
    library.mkdir()
    port = free_port()
    out, code = in_terminal(["--library", str(library), "--data-dir", str(data), "--port", str(port)], ["API ready"])
    plain = ESCAPE.sub("", out)
    assert code == 0, out
    assert "\x1b[1m\x1b[31mSpectra" in out  # brand title in a colour terminal
    assert f"Open      http://127.0.0.1:{port}" in plain and "✓ API ready" in plain
    assert plain.index("Stopping…") < plain.index("Stopped.")
    assert "Uvicorn running" not in plain and "Started server process" not in plain and "Web ready" not in plain
    assert launcher_children(data) == []
    out, code = in_terminal(["--library", str(library), "--data-dir", str(data), "--port", str(port)], ["API ready"],
                            environ={"NO_COLOR": "1"})
    assert code == 0 and not ESCAPE.search(out)


@pytest.mark.slow
def test_development_mode_runs_both_servers_and_stops_both(tmp_path):
    if shutil.which("npm") is None or not (ROOT / "frontend" / "node_modules").is_dir():
        pytest.skip("the web application's dependencies are not installed")
    if not run.port_is_free(run.DEV_PORT):
        pytest.skip(f"port {run.DEV_PORT} is in use")
    library, data = tmp_path / "videos", tmp_path / "state"
    library.mkdir()
    out, code = in_terminal(["--dev", "--library", str(library), "--data-dir", str(data), "--port", str(free_port())],
                            ["API ready", "Web ready"])
    plain = ESCAPE.sub("", out)
    assert code == 0, out
    assert "development" in plain and f"Open      {run.DEV_ORIGIN}" in plain
    assert "✓ Web ready" in plain and "Stopped." in plain
    assert "VITE v" not in plain and "➜" not in plain  # routine development-server lines are hidden
    assert launcher_children(data) == [] and run.port_is_free(run.DEV_PORT)


def test_normal_mode_starts_without_node(tmp_path):
    if not run.build_is_current():
        pytest.skip("the web application build is missing or older than its sources")
    data, port = tmp_path / "state", free_port()
    out, code = in_terminal(["--data-dir", str(data), "--port", str(port)], ["API ready"],
                            environ={"PATH": "/usr/bin:/bin"})
    plain = ESCAPE.sub("", out)
    assert code == 0, out
    assert "✓ API ready" in plain and "npm was not found" not in plain and "Building" not in plain
    assert launcher_children(data) == []


def test_development_mode_without_npm_is_refused(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), "--dev", "--port", str(free_port())],
                            capture_output=True, text=True, timeout=60, env={**os.environ, "PATH": "/usr/bin:/bin"})
    assert result.returncode == 1 and "Development mode needs Node.js" in result.stdout


def test_closing_the_terminal_stops_the_servers(tmp_path):
    data, port = tmp_path / "state", free_port()
    out, code = in_terminal(["--data-dir", str(data), "--port", str(port)], ["API ready"], close_terminal=True)
    assert "API ready" in ESCAPE.sub("", out)
    assert code == 0
    assert launcher_children(data) == []
    assert run.port_is_free(port)
