"""Download and export Depth Anything V2 Metric VKITTI Small to ONNX.

Run from the project root:
    .venv/bin/python scripts/download_depth_model.py

The runtime app only needs the exported ONNX file. This script uses the
official Depth Anything V2 metric-depth implementation at build time to load
the VKITTI Small checkpoint and export a static 518x518 ONNX graph.
"""

from __future__ import annotations

import shutil
import ssl
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


OFFICIAL_REPO_ZIP_URL = "https://github.com/DepthAnything/Depth-Anything-V2/archive/refs/heads/main.zip"
CHECKPOINT_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/"
    "resolve/main/depth_anything_v2_metric_vkitti_vits.pth?download=true"
)
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
CHECKPOINT_PATH = MODEL_DIR / "depth_anything_v2_metric_vkitti_vits.pth"
ONNX_PATH = MODEL_DIR / "depth_anything_v2_metric_vkitti_vits.onnx"
INPUT_SIZE = 518
MAX_DEPTH_M = 80.0


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


def _prepare_official_source(temp_dir: Path) -> Path:
    archive_path = temp_dir / "depth_anything_v2_main.zip"
    _download(OFFICIAL_REPO_ZIP_URL, archive_path)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(temp_dir)
    candidates = sorted(temp_dir.glob("Depth-Anything-V2-main/metric_depth"))
    if not candidates:
        raise RuntimeError("Official source archive did not contain metric_depth")
    return candidates[0]


def _ensure_checkpoint() -> None:
    if CHECKPOINT_PATH.exists():
        size_mb = CHECKPOINT_PATH.stat().st_size / (1024 * 1024)
        print(f"Checkpoint already present at {CHECKPOINT_PATH} ({size_mb:.1f} MB).")
        return
    _download(CHECKPOINT_URL, CHECKPOINT_PATH)


def _export_onnx(metric_source: Path) -> None:
    import torch

    sys.path.insert(0, str(metric_source))
    try:
        from depth_anything_v2.dpt import DepthAnythingV2

        model = DepthAnythingV2(
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384],
            max_depth=MAX_DEPTH_M,
        )
        state = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
        ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = ONNX_PATH.with_suffix(".tmp.onnx")
        print(f"Exporting ONNX to {ONNX_PATH}")
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy,
                tmp_path,
                input_names=["image"],
                output_names=["depth_m"],
                opset_version=17,
                do_constant_folding=True,
            )

        import onnx

        onnx_model = onnx.load(str(tmp_path), load_external_data=True)
        onnx.checker.check_model(onnx_model)
        onnx.save_model(onnx_model, str(ONNX_PATH), save_as_external_data=False)
        tmp_path.unlink(missing_ok=True)
        tmp_data_path = tmp_path.with_suffix(tmp_path.suffix + ".data")
        tmp_data_path.unlink(missing_ok=True)
        size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
        print(f"Done. {size_mb:.1f} MB at {ONNX_PATH}")
    finally:
        try:
            sys.path.remove(str(metric_source))
        except ValueError:
            pass


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if ONNX_PATH.exists():
        size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
        print(f"ONNX model already present at {ONNX_PATH} ({size_mb:.1f} MB).")
        return 0

    try:
        _ensure_checkpoint()
        with tempfile.TemporaryDirectory() as tmp:
            metric_source = _prepare_official_source(Path(tmp))
            _export_onnx(metric_source)
    except Exception as exc:
        print(f"\nMetric depth model export failed: {exc}", file=sys.stderr)
        tmp_path = ONNX_PATH.with_suffix(".tmp.onnx")
        if tmp_path.exists():
            tmp_path.unlink()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
