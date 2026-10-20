"""Export the local comma10k-segnet checkpoint to ONNX for the analysis engine, and check it.

    python -m evaluation.export_segnet_onnx OUT.onnx

Needs ``segmentation-models-pytorch`` 0.5.0 and ``timm`` 1.0.30 in the exporting environment only; the
engine runs the result on ONNX Runtime. Uses only ``evaluation/weights/comma10k-segnet``; nothing is
downloaded. The graph takes one RGB 0-255 float image at 384 x 512 (NCHW) and returns the five class
logits. After export the ONNX output is compared with PyTorch on a random input and the sha256 printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation import segnet


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("--opset", type=int, default=17)
    a = p.parse_args()

    import onnxruntime as ort
    import torch

    model = segnet.load()
    w, h = segnet.INPUT_SIZE
    x = torch.rand(1, 3, h, w, generator=torch.Generator().manual_seed(0)) * 255
    with torch.inference_mode():
        ref = model(x).numpy()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (x,), str(a.out), input_names=["image"], output_names=["logits"],
                      opset_version=a.opset, do_constant_folding=True, dynamo=False)
    got = ort.InferenceSession(str(a.out), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
    weights = segnet.WEIGHTS / "model.safetensors"
    print(json.dumps({
        "out": str(a.out), "bytes": a.out.stat().st_size,
        "sha256": hashlib.sha256(a.out.read_bytes()).hexdigest(),
        "source_checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "max_logit_diff": float(np.abs(ref - got).max()),
        "argmax_agreement": float((ref.argmax(1) == got.argmax(1)).mean()),
    }, indent=2))


if __name__ == "__main__":
    main()
