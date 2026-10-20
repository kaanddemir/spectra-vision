"""Local model locations; no model is loaded while importing this module."""
from pathlib import Path
import pytest
MODEL = Path(__file__).resolve().parents[2] / "models/engine/dfine_n_coco_640.onnx"
MODEL_SHA = "9533e039b26796c7eec06a8394b212c5561979cb147fcad0b3e0805b5db5ed9a"
needs_model = pytest.mark.skipif(not MODEL.is_file(), reason="local D-FINE ONNX export not present")
