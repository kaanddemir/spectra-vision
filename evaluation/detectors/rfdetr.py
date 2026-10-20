"""RF-DETR adapter (Apache-2.0 code and Nano/Small weights) on PyTorch CPU or MPS."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parents[1] / ".venv" / "weights"

# RF-DETR reports original COCO category ids (1..90 with gaps); the adapter contract is the
# contiguous zero-based 80-class index used by the other detectors (car = 2, truck = 7).
COCO_CATEGORY_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
                     31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                     56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
                     85, 86, 87, 88, 89, 90)
_TO_CONTIGUOUS = np.full(91, -1.0)
_TO_CONTIGUOUS[list(COCO_CATEGORY_IDS)] = np.arange(80)


def to_contiguous(category_ids: np.ndarray) -> np.ndarray:
    out = _TO_CONTIGUOUS[np.clip(category_ids.astype(int), 0, 90)]
    if np.any(out < 0):
        raise ValueError(f"unknown COCO category ids: {sorted(set(category_ids[out < 0].astype(int)))}")
    return out


class RfDetr:
    def __init__(self, size: str = "nano", device: str = "cpu", threads: int = 4, score_threshold: float = 0.3,
                 input_px: int | None = None) -> None:
        self.size, self.device, self.threads, self.score_threshold = size, device, threads, score_threshold
        self.input_px = input_px
        self.weights = WEIGHTS / f"rf-detr-{self.size}.pth"
        self.name = f"rfdetr_{size}_{input_px or 'default'}_{device}"

    def load(self) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"  # weights come only from the verified local checkpoint
        import torch
        import rfdetr
        torch.set_num_threads(self.threads)
        cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall}[self.size]
        kw = {"pretrain_weights": str(self.weights), "device": self.device}
        if self.input_px:
            kw["resolution"] = self.input_px
        self._model = cls(**kw)

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        d = self._model.predict(np.ascontiguousarray(frame_bgr[:, :, ::-1]), threshold=self.score_threshold)
        return np.column_stack([d.xyxy, d.confidence, to_contiguous(np.asarray(d.class_id))]).astype(np.float64).reshape(-1, 6)
