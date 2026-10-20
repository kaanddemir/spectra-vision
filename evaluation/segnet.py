"""PyTorch reference for the comma10k-segnet road segmentation that the runtime uses as an ONNX export.

The model (commaai/comma10k-segnet, MIT) is a segmentation_models_pytorch U-Net loaded from local files in
``evaluation/weights/comma10k-segnet``. Class indices follow comma10k: 0 road, 1 lane markings,
2 undrivable, 3 movable, 4 my car.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROAD, MARKING, UNDRIVABLE, MOVABLE, MY_CAR = range(5)
WEIGHTS = Path(__file__).parent / "weights" / "comma10k-segnet"
INPUT_SIZE = (512, 384)


def load(weights: Path = WEIGHTS, device: str = "cpu"):
    import segmentation_models_pytorch as smp
    from safetensors.torch import load_file
    cfg = json.loads((weights / "config.json").read_text())
    cfg.pop("_model_class")
    model = smp.Unet(**cfg)
    model.load_state_dict(load_file(weights / "model.safetensors"))
    return model.eval().to(device)


def segment(model, frame, device: str = "cpu") -> np.ndarray:
    """Class map (uint8, INPUT_SIZE) for one BGR frame. Input is RGB 0-255, as in the published transform."""
    import cv2
    import torch
    rgb = cv2.cvtColor(cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(device)
    with torch.inference_mode():
        return model(x)[0].argmax(0).byte().cpu().numpy()

