"""Native folder selection on the computer running the server.

A browser never reveals the absolute path of a chosen folder, and an HTTP request must never be able to
name one. Adding a library folder therefore opens the operating system's own folder dialog on this
computer; only the person using it can choose, and the request carries no path. One dialog is open at a time.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

FolderPicker = Callable[[], Optional[Path]]
PICK_TIMEOUT_S = 600
PROMPT = "Choose a video folder"


class PickerUnavailable(RuntimeError):
    """No native folder dialog on this system (HTTP 501)."""


class PickerBusy(RuntimeError):
    """A folder dialog is already open (HTTP 409)."""


_lock = threading.Lock()


def _command() -> list[str]:
    if sys.platform == "darwin" and shutil.which("osascript"):
        script = f'POSIX path of (choose folder with prompt "{PROMPT}")'
        return ["osascript", "-e", script]
    if shutil.which("zenity"):
        return ["zenity", "--file-selection", "--directory", f"--title={PROMPT}"]
    raise PickerUnavailable("this system has no supported folder dialog; start Spectra with --library DIR")


def choose_folder(run=subprocess.run) -> Optional[Path]:
    """Show the dialog and return the chosen folder, or None when it is cancelled."""
    command = _command()
    if not _lock.acquire(blocking=False):
        raise PickerBusy("a folder dialog is already open")
    try:
        result = run(command, capture_output=True, text=True, timeout=PICK_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None
    finally:
        _lock.release()
    chosen = result.stdout.strip()
    if result.returncode != 0:
        if "-128" in result.stderr or not result.stderr.strip():  # cancelled by the person
            return None
        raise PickerUnavailable("the folder dialog could not be shown")
    return Path(chosen) if chosen else None
