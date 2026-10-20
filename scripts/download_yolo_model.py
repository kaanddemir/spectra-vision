"""Download the YOLOv8n detection weights.

Run from the project root:
    .venv/bin/python scripts/download_yolo_model.py

The app expects:
    models/yolov8n.pt

Ultralytics can fetch this file itself on first use, but it writes it to its
own working directory rather than to ``models/``, which is where every other
Spectra model lives and where ``spectra/vision/detection.py`` looks. Fetching
it here keeps setup to one command per model.

Optional override:
    YOLOV8N_URL=<direct .pt URL>
"""

from __future__ import annotations

import os
import ssl
import sys
import urllib.request
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "yolov8n.pt"

_DEFAULT_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
)
MODEL_URL = os.environ.get("YOLOV8N_URL", _DEFAULT_URL).strip() or _DEFAULT_URL

# The published weights are ~6.2 MB. A file far below that is an error page or
# a truncated transfer, both of which fail later as an unreadable checkpoint.
MIN_EXPECTED_BYTES = 4 * 1024 * 1024


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _print_progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    percent = min(100.0, downloaded / total_size * 100.0)
    mb_done = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    sys.stdout.write(f"\rDownloading: {percent:5.1f}%  ({mb_done:6.1f} / {mb_total:6.1f} MB)")
    sys.stdout.flush()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {url}")
    print(f"Saving to {destination}")
    with urllib.request.urlopen(url, context=_ssl_context()) as response:
        total_size = int(response.headers.get("Content-Length") or 0)
        block_size = 1024 * 256
        with open(destination, "wb") as out:
            block_num = 0
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                out.write(chunk)
                block_num += 1
                _print_progress(block_num, block_size, total_size)
    print()


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"YOLO weights already present at {MODEL_PATH} ({size_mb:.1f} MB).")
        return 0

    # Downloaded beside the target and moved into place only once it is whole,
    # so an interrupted run cannot leave a half file that looks installed.
    temp_path = MODEL_PATH.with_suffix(".pt.part")
    try:
        _download(MODEL_URL, temp_path)
        size = temp_path.stat().st_size
        if size < MIN_EXPECTED_BYTES:
            raise RuntimeError(
                f"downloaded file is only {size / (1024 * 1024):.1f} MB, "
                "which is too small to be the YOLOv8n weights"
            )
        temp_path.replace(MODEL_PATH)
    except Exception as exc:
        print(f"\nYOLO weights download failed: {exc}", file=sys.stderr)
        temp_path.unlink(missing_ok=True)
        return 1

    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Done. {size_mb:.1f} MB at {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
