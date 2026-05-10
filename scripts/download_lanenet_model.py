"""Download/export the UFLDv2 CULane ResNet-18 lane model.

Run from the project root:
    .venv/bin/python scripts/download_lanenet_model.py

The app expects:
    models/ufld_v2_culane_r18.onnx

On Apple Silicon this script keeps runtime inference in ONNX Runtime, where
Spectra prefers CoreMLExecutionProvider and falls back to CPU. The default
path downloads the official UFLDv2 CULane ResNet-18 PyTorch checkpoint and
exports it to ONNX locally, avoiding the much larger prepackaged model archive.

Optional overrides:
    UFLDV2_ONNX_URL=<direct .onnx URL>
    UFLDV2_CHECKPOINT_URL=<gdown-supported checkpoint URL or file id>
    UFLDV2_SOURCE_ZIP_URL=<official source zip URL>
"""

from __future__ import annotations

import os
import shutil
import ssl
import sys
import tempfile
import types
import urllib.request
import zipfile
from importlib import import_module
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "ufld_v2_culane_r18.onnx"
CACHE_DIR = MODEL_DIR / ".ufldv2_downloads"

_DEFAULT_SOURCE_ZIP_URL = "https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/archive/refs/heads/master.zip"
_DEFAULT_CHECKPOINT_ID = "1oEjJraFr-3lxhX_OXduAGFWalWa6Xh3W"

SOURCE_ZIP_URL = os.environ.get("UFLDV2_SOURCE_ZIP_URL", _DEFAULT_SOURCE_ZIP_URL)
CHECKPOINT_URL = os.environ.get("UFLDV2_CHECKPOINT_URL", _DEFAULT_CHECKPOINT_ID)
DIRECT_ONNX_URL = os.environ.get("UFLDV2_ONNX_URL", "").strip()


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
    sys.stdout.write(f"\rDownloading: {percent:5.1f}%  ({mb_done:7.1f} / {mb_total:7.1f} MB)")
    sys.stdout.flush()


def _download_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    context = _ssl_context()
    with urllib.request.urlopen(url, context=context) as response:
        total_size = int(response.headers.get("Content-Length") or 0)
        block_size = 1024 * 512
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


def _download_checkpoint(destination: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required for the official Google Drive checkpoint. "
            "Install it with: .venv/bin/python -m pip install gdown"
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = CHECKPOINT_URL
    if url.startswith(("http://", "https://")):
        print(f"Fetching UFLDv2 checkpoint from {url}")
        result = gdown.download(url=url, output=str(destination), quiet=False)
    else:
        print(f"Fetching UFLDv2 checkpoint id {url}")
        result = gdown.download(id=url, output=str(destination), quiet=False)
    if result is None or not destination.is_file():
        raise RuntimeError("checkpoint download failed")


def _extract_source(zip_path: Path, destination: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)
    candidates = sorted(p for p in destination.iterdir() if p.is_dir() and p.name.startswith("Ultra-Fast-Lane-Detection"))
    if not candidates:
        raise RuntimeError("UFLDv2 source zip did not contain the expected directory")
    return candidates[0]


def _install_minimal_utils_stub() -> None:
    """Avoid importing UFLDv2's training-only utils.common dependencies."""

    import torch

    def real_init_weights(module: object) -> None:
        if isinstance(module, list):
            for child in module:
                real_init_weights(child)
            return
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(module.weight, 1)
            torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.Module):
            for child in module.children():
                real_init_weights(child)

    def initialize_weights(*models: object) -> None:
        for model in models:
            real_init_weights(model)

    utils_pkg = types.ModuleType("utils")
    common_mod = types.ModuleType("utils.common")
    common_mod.initialize_weights = initialize_weights  # type: ignore[attr-defined]
    sys.modules["utils"] = utils_pkg
    sys.modules["utils.common"] = common_mod


def _export_onnx(source_root: Path, checkpoint_path: Path, output_path: Path) -> None:
    import torch

    sys.path.insert(0, str(source_root))
    _install_minimal_utils_stub()

    parsingNet = import_module("model.model_culane").parsingNet

    net = parsingNet(
        pretrained=False,
        backbone="18",
        num_grid_row=200,
        num_cls_row=72,
        num_grid_col=100,
        num_cls_col=81,
        num_lane_on_row=4,
        num_lane_on_col=4,
        use_aux=False,
        input_height=320,
        input_width=1600,
        fc_norm=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    compatible_state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_name(f"{output_path.name}.export.onnx")
    tmp_external_data = tmp_output.with_name(f"{tmp_output.name}.data")
    if tmp_output.exists():
        tmp_output.unlink()
    if tmp_external_data.exists():
        tmp_external_data.unlink()
    dummy = torch.ones((1, 3, 320, 1600), dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            net,
            dummy,
            str(tmp_output),
            opset_version=18,
            input_names=["input"],
            output_names=["loc_row", "loc_col", "exist_row", "exist_col"],
            external_data=False,
        )
    tmp_output.replace(output_path)
    if tmp_external_data.exists():
        tmp_external_data.unlink()


def _validate_model(path: Path) -> None:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required to validate the exported model") from exc

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    output_names = [output.name for output in session.get_outputs()]
    print(f"Validated ONNX input={input_shape}, outputs={output_names}")


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        if MODEL_PATH.stat().st_size > 1024 * 1024:
            print(f"Model already present at {MODEL_PATH} ({size_mb:.1f} MB).")
            return 0
        print(f"Existing model at {MODEL_PATH} is only {size_mb:.1f} MB; regenerating it.")
        MODEL_PATH.unlink()

    try:
        if DIRECT_ONNX_URL:
            print(f"Fetching {DIRECT_ONNX_URL}")
            print(f"Saving to {MODEL_PATH}")
            _download_url(DIRECT_ONNX_URL, MODEL_PATH)
        else:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="spectra-ufldv2-") as tmp:
                tmp_dir = Path(tmp)
                source_zip = CACHE_DIR / "ufldv2-master.zip"
                checkpoint_path = CACHE_DIR / "culane_res18.pth"

                if not source_zip.exists():
                    print(f"Fetching UFLDv2 source from {SOURCE_ZIP_URL}")
                    _download_url(SOURCE_ZIP_URL, source_zip)
                else:
                    print(f"Using cached UFLDv2 source at {source_zip}")
                source_root = _extract_source(source_zip, tmp_dir / "source")

                if not checkpoint_path.exists():
                    _download_checkpoint(checkpoint_path)
                else:
                    print(f"Using cached UFLDv2 checkpoint at {checkpoint_path}")
                print(f"Exporting ONNX to {MODEL_PATH}")
                _export_onnx(source_root, checkpoint_path, MODEL_PATH)

        _validate_model(MODEL_PATH)
    except Exception as exc:
        print(f"\nUFLDv2 model setup failed: {exc}", file=sys.stderr)
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        print(
            "\nYou can also set UFLDV2_ONNX_URL to a direct UFLDv2 CULane "
            "ResNet-18 ONNX file and rerun this script.",
            file=sys.stderr,
        )
        return 1
    finally:
        for leftover in (
            MODEL_PATH.with_name(f"{MODEL_PATH.name}.export.onnx"),
            MODEL_PATH.with_name(f"{MODEL_PATH.name}.export.onnx.data"),
            MODEL_PATH.with_name(f"{MODEL_PATH.name}.tmp.data"),
        ):
            if leftover.exists():
                leftover.unlink()

    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Done. {size_mb:.1f} MB at {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
