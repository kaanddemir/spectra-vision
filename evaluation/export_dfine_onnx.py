"""Export a local D-FINE checkpoint (model + post-processor) to ONNX for the analysis engine, and check it.

    evaluation/.venv/bin/python -m evaluation.export_dfine_onnx OUT.onnx [--size n] [--input-px 640]

Uses only the checkpoint already in ``evaluation/.venv/weights``; nothing is downloaded. Batch size is
fixed at one. After export the ONNX graph is compared with the PyTorch model on a random input and the
sha256 of the file is printed for the engine configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("--size", default="n")
    p.add_argument("--input-px", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    a = p.parse_args()

    import onnxruntime as ort
    import torch

    from evaluation.detectors.dfine import DFine

    det = DFine(size=a.size, input_px=a.input_px, device="cpu", threads=4)
    det.load()

    class Exported(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model, self.post = det._model, det._post

        def forward(self, images, orig_target_sizes):
            return self.post(self.model(images), orig_target_sizes)

    model = Exported().eval()
    x = torch.rand(1, 3, a.input_px, a.input_px, generator=torch.Generator().manual_seed(0))
    size = torch.tensor([[1280, 720]])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        ref = [t.numpy() for t in model(x, size)]
    torch.onnx.export(model, (x, size), str(a.out), input_names=["images", "orig_target_sizes"],
                      output_names=["labels", "boxes", "scores"], opset_version=a.opset, do_constant_folding=True,
                      dynamo=False)
    sess = ort.InferenceSession(str(a.out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"images": x.numpy(), "orig_target_sizes": size.numpy()})
    top = np.argsort(-ref[2][0])[:50]
    report = {
        "out": str(a.out), "bytes": a.out.stat().st_size,
        "sha256": hashlib.sha256(a.out.read_bytes()).hexdigest(),
        "source_checkpoint_sha256": hashlib.sha256(det.weights.read_bytes()).hexdigest(),
        "labels_equal_top50": bool(np.array_equal(ref[0][0][top], got[0][0][top])),
        "max_box_diff_px_top50": float(np.abs(ref[1][0][top] - got[1][0][top]).max()),
        "max_score_diff_top50": float(np.abs(ref[2][0][top] - got[2][0][top]).max()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
