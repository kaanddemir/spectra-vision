"""D-FINE adapter (Apache-2.0 code, COCO-only checkpoint) running on PyTorch CPU or MPS."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

VENV = Path(__file__).resolve().parents[1] / ".venv"


class DFine:
    def __init__(self, size: str = "n", input_px: int = 640, device: str = "cpu", threads: int = 4,
                 score_threshold: float = 0.3, repo: Path = VENV / "src" / "D-FINE",
                 weights: Path | None = None) -> None:
        self.name = f"dfine_{size}_{input_px}_{device}"
        self.size, self.input_px, self.device, self.threads = size, input_px, device, threads
        self.score_threshold = score_threshold
        self.repo = Path(repo)
        self.weights = Path(weights) if weights else VENV / "weights" / f"dfine_{size}_coco.pth"

    def load(self) -> None:
        import torch
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        if "calflops" not in sys.modules:  # imported only for FLOP profiling; avoids its dependency tree
            stub = types.ModuleType("calflops")
            stub.calculate_flops = lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("calflops stub"))
            sys.modules["calflops"] = stub
        from src.core import YAMLConfig  # type: ignore[import-not-found]

        torch.set_num_threads(self.threads)
        cfg = YAMLConfig(str(self.repo / f"configs/dfine/dfine_hgnetv2_{self.size}_coco.yml"),
                         HGNetv2={"pretrained": False},  # full checkpoint supplies backbone weights; no download
                         eval_spatial_size=[self.input_px, self.input_px],
                         remap_mscoco_category=False)  # contiguous zero-based COCO ids
        state = torch.load(self.weights, map_location="cpu", weights_only=False)
        state = state["ema"]["module"] if "ema" in state else state["model"]
        if self.input_px != 640:  # anchors are derived from the input size, not learned; rebuilt by the model
            state = {k: v for k, v in state.items() if k not in ("decoder.anchors", "decoder.valid_mask")}
            missing, unexpected = cfg.model.load_state_dict(state, strict=False)
            if unexpected or set(missing) - {"decoder.anchors", "decoder.valid_mask"}:
                raise RuntimeError(f"unexpected checkpoint mismatch: {missing} {unexpected}")
        else:
            cfg.model.load_state_dict(state)
        self._model = cfg.model.deploy().eval().to(self.device)
        self._post = cfg.postprocessor.deploy().eval().to(self.device)
        self._torch = torch

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2
        torch = self._torch
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (self.input_px, self.input_px), interpolation=cv2.INTER_LINEAR),
                           cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            labels, boxes, scores = self._post(self._model(x), torch.tensor([[w, h]], device=self.device))
        labels, boxes, scores = labels[0].cpu().numpy(), boxes[0].cpu().numpy(), scores[0].cpu().numpy()
        keep = scores >= self.score_threshold
        return np.column_stack([boxes[keep], scores[keep], labels[keep]]).astype(np.float64)
