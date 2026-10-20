"""Media probing in a short-lived helper process.

Demuxers and decoders run on untrusted files. A crash, hang or memory spike in native code must
not affect the server, and decoder memory must not raise the server's lifetime peak RSS. Each probe
therefore runs ``python -m spectra.api.probe PATH`` in its own process session, prints one JSON
report and exits. The parent enforces a size-scaled timeout and kills the whole process group on
timeout or shutdown. The source is opened read-only.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from spectra.engine.contracts import VideoInfo

ProbeErrorKind = Literal["not_found", "unsupported", "corrupt", "missing_timestamps", "changed", "failed",
                         "timeout", "cancelled"]
BASE_TIMEOUT_S = 60.0
HASH_RATE_BYTES_S = 20e6  # conservative full-file hashing rate for the timeout
STDOUT_LIMIT = 1 << 20
HELPER_COMMAND = (sys.executable, "-B", "-m", "spectra.api.probe")


class FileStat(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    size_bytes: int
    mtime_ns: int
    inode: int


class ProbeReport(BaseModel):
    """Exactly one of ``info`` (with the stat it was read under) or ``error`` is present."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    info: Optional[VideoInfo] = None
    audio_codecs: list[str] = []
    stat: Optional[FileStat] = None
    error: Optional[ProbeErrorKind] = None
    message: Optional[str] = None


def file_stat(path: Path) -> FileStat:
    info = path.stat()
    return FileStat(size_bytes=info.st_size, mtime_ns=info.st_mtime_ns, inode=info.st_ino)


def timeout_for(size_bytes: int) -> float:
    return BASE_TIMEOUT_S + size_bytes / HASH_RATE_BYTES_S


def _audio_codecs(path: Path) -> list[str]:
    import av
    with av.open(str(path), mode="r") as container:
        return [s.codec_context.name for s in container.streams.audio if s.codec_context is not None]


def probe_in_process(path: Path) -> ProbeReport:
    """The helper's work: identity, metadata and audio codecs. Called only inside the helper."""
    from spectra.engine.video import CorruptVideo, MissingTimestamps, UnsupportedVideo, VideoError, VideoSource
    try:
        before = file_stat(path)
    except OSError:
        return ProbeReport(error="not_found", message=f"video file not found: {path.name}")
    try:
        info = VideoSource(path).info  # hashes the whole file and decodes the first frame
        audio = _audio_codecs(path)
    except MissingTimestamps as exc:
        return ProbeReport(error="missing_timestamps", message=str(exc))
    except CorruptVideo as exc:
        return ProbeReport(error="corrupt", message=str(exc))
    except UnsupportedVideo as exc:
        kind = "not_found" if "not found" in str(exc) else "unsupported"
        return ProbeReport(error=kind, message=str(exc))
    except VideoError as exc:
        return ProbeReport(error="unsupported", message=str(exc))
    except Exception as exc:  # native decoder errors surface with many types
        return ProbeReport(error="corrupt", message=f"{path.name} could not be read: {exc}")
    try:
        after = file_stat(path)
    except OSError:
        return ProbeReport(error="not_found", message=f"{path.name} disappeared while it was read")
    if after != before or info.size_bytes != after.size_bytes:
        return ProbeReport(error="changed", message=f"{path.name} changed while it was read")
    return ProbeReport(info=info, audio_codecs=audio, stat=after)


class ProbeRunner:
    """Runs one helper process at a time and can kill it from another thread."""

    def __init__(self, command: Sequence[str] = HELPER_COMMAND) -> None:
        self.command = tuple(command)
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._closed = False

    def run(self, path: Path, timeout_s: float) -> ProbeReport:
        with self._lock:
            if self._closed:
                return ProbeReport(error="cancelled", message="the server is shutting down")
            process = subprocess.Popen(
                [*self.command, str(path)], stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            self._process = process
        try:
            out, _ = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._kill(process)
            process.communicate()
            return ProbeReport(error="timeout", message=f"reading {path.name} took longer than {timeout_s:.0f} s")
        finally:
            with self._lock:
                self._process = None
        if self._closed:
            return ProbeReport(error="cancelled", message="the server is shutting down")
        if process.returncode != 0 or len(out) > STDOUT_LIMIT:
            return ProbeReport(error="failed",
                               message=f"the media reader stopped unexpectedly (exit status {process.returncode})")
        try:
            return ProbeReport.model_validate_json(out)
        except ValueError:
            return ProbeReport(error="failed", message="the media reader returned an invalid report")

    def close(self) -> None:
        """Refuse new probes and kill a running helper."""
        with self._lock:
            self._closed = True
            if self._process is not None:
                self._kill(self._process)

    @staticmethod
    def _kill(process: subprocess.Popen) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m spectra.api.probe PATH", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(probe_in_process(Path(argv[0])).model_dump(mode="json")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
