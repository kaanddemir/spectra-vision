"""Download the NeuFlow v2 ONNX optical flow model.

Run from the project root:
    .venv/bin/python scripts/download_flow_model.py

If the default URL has moved, set NEUFLOW_MODEL_URL to override it.
"""

from __future__ import annotations

import os
import ssl
import sys
import urllib.request
from pathlib import Path


_DEFAULT_URL = "https://github.com/ibaiGorordo/ONNX-NeuFlowV2-Optical-Flow/releases/download/0.1.0/neuflow_mixed.onnx"
MODEL_URL = os.environ.get("NEUFLOW_MODEL_URL", _DEFAULT_URL)
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "neuflow_v2.onnx"


def _print_progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    percent = min(100.0, downloaded / total_size * 100.0)
    mb_done = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    sys.stdout.write(f"\rDownloading: {percent:5.1f}%  ({mb_done:6.1f} / {mb_total:6.1f} MB)")
    sys.stdout.flush()


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"Model already present at {MODEL_PATH} ({size_mb:.1f} MB).")
        return 0

    print(f"Fetching {MODEL_URL}")
    print(f"Saving to {MODEL_PATH}")

    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()

    try:
        with urllib.request.urlopen(MODEL_URL, context=context) as response:
            total_size = int(response.headers.get("Content-Length") or 0)
            block_size = 1024 * 256
            with open(MODEL_PATH, "wb") as out:
                block_num = 0
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    block_num += 1
                    _print_progress(block_num, block_size, total_size)
        print()
    except Exception as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        print(
            "\nIf the default URL is unavailable, set NEUFLOW_MODEL_URL to a working\n"
            "ONNX export of NeuFlow v2 or drop the model file manually at models/neuflow_v2.onnx.",
            file=sys.stderr,
        )
        return 1

    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Done. {size_mb:.1f} MB at {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
