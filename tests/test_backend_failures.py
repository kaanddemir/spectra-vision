"""Hard-failure contracts for required vision backends."""

import numpy as np
import pytest

from spectra.analysis import video
from spectra.vision import lanenet, models
from spectra.vision.detection import ObjectDetector


def _reset_depth_singleton(monkeypatch):
    monkeypatch.setattr(models, "_depth_model_singleton", None)
    monkeypatch.setattr(models, "_depth_load_failed", False)
    monkeypatch.setattr(models, "_depth_load_error", None)


def _reset_lanenet_singleton(monkeypatch):
    monkeypatch.setattr(lanenet, "_lanenet_singleton", None)
    monkeypatch.setattr(lanenet, "_lanenet_load_failed", False)
    monkeypatch.setattr(lanenet, "_lanenet_load_error", None)


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = _FakeTensor(xyxy)
        self.cls = _FakeTensor(cls)
        self.conf = _FakeTensor(conf)

    def __len__(self):
        return len(self.conf.value)


class _FakeYoloResult:
    names = {2: "car", 7: "truck"}

    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYoloModel:
    names = {2: "car", 7: "truck"}

    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, **kwargs):
        return [_FakeYoloResult(self.boxes)]


def test_get_depth_model_raises_and_caches_load_failure(monkeypatch):
    _reset_depth_singleton(monkeypatch)

    class BrokenDepthModel:
        def __init__(self, model_path):
            raise ValueError("bad depth graph")

    monkeypatch.setattr(models, "is_depth_available", lambda: True)
    monkeypatch.setattr(models, "DepthAnythingONNX", BrokenDepthModel)

    with pytest.raises(RuntimeError, match="Depth Anything metric ONNX model failed to load"):
        models.get_depth_model()
    with pytest.raises(RuntimeError, match="bad depth graph"):
        models.get_depth_model()


def test_get_lanenet_model_raises_and_caches_load_failure(monkeypatch):
    _reset_lanenet_singleton(monkeypatch)

    class BrokenLaneModel:
        def __init__(self, model_path):
            raise ValueError("bad lane graph")

    monkeypatch.setattr(lanenet, "is_lanenet_available", lambda: True)
    monkeypatch.setattr(lanenet, "UFLDv2ONNX", BrokenLaneModel)

    with pytest.raises(RuntimeError, match="UFLDv2 ONNX model failed to load"):
        lanenet.get_lanenet_model()
    with pytest.raises(RuntimeError, match="bad lane graph"):
        lanenet.get_lanenet_model()


def test_ensure_required_models_surfaces_depth_load_failure(monkeypatch):
    monkeypatch.setattr(video, "is_depth_available", lambda: True)
    monkeypatch.setattr(video, "is_yolo_available", lambda: True)
    monkeypatch.setattr(video, "is_lanenet_available", lambda: True)
    monkeypatch.setattr(video, "get_depth_model", lambda: (_ for _ in ()).throw(RuntimeError("depth load failed")))
    monkeypatch.setattr(video, "get_lanenet_model", lambda: object())
    monkeypatch.setattr(video, "get_detector", lambda: object())

    with pytest.raises(RuntimeError, match="depth load failed"):
        video._ensure_required_models()


def test_ensure_required_models_surfaces_lanenet_load_failure(monkeypatch):
    monkeypatch.setattr(video, "is_depth_available", lambda: True)
    monkeypatch.setattr(video, "is_yolo_available", lambda: True)
    monkeypatch.setattr(video, "is_lanenet_available", lambda: True)
    monkeypatch.setattr(video, "get_depth_model", lambda: object())
    monkeypatch.setattr(video, "get_lanenet_model", lambda: (_ for _ in ()).throw(RuntimeError("lane load failed")))
    monkeypatch.setattr(video, "get_detector", lambda: object())

    with pytest.raises(RuntimeError, match="lane load failed"):
        video._ensure_required_models()


def test_yolo_inference_failure_raises_runtime_error():
    class FailingModel:
        def predict(self, **kwargs):
            raise ValueError("predict crashed")

    detector = ObjectDetector()
    detector._model = FailingModel()
    detector._device = "cpu"

    with pytest.raises(RuntimeError, match="YOLOv8 inference failed"):
        detector.detect(np.zeros((32, 48, 3), dtype=np.uint8))


def test_yolo_empty_results_remain_empty_detections():
    class EmptyModel:
        def predict(self, **kwargs):
            return []

    detector = ObjectDetector()
    detector._model = EmptyModel()
    detector._device = "cpu"

    assert detector.detect(np.zeros((32, 48, 3), dtype=np.uint8)) == []


def test_vehicle_detection_below_class_confidence_gate_is_filtered():
    detector = ObjectDetector()
    detector._model = _FakeYoloModel(
        _FakeBoxes(
            xyxy=[[4, 5, 24, 25]],
            cls=[2],
            conf=[0.35],
        )
    )
    detector._device = "cpu"

    detections = detector.detect(np.zeros((32, 48, 3), dtype=np.uint8))

    assert detections == []


def test_vehicle_detection_above_class_confidence_gate_is_kept():
    detector = ObjectDetector()
    detector._model = _FakeYoloModel(
        _FakeBoxes(
            xyxy=[[4, 5, 24, 25]],
            cls=[7],
            conf=[0.55],
        )
    )
    detector._device = "cpu"

    detections = detector.detect(np.zeros((32, 48, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].class_name == "truck"
    assert detections[0].confidence == pytest.approx(0.55)


def test_yolo_unloaded_detector_raises_runtime_error():
    detector = ObjectDetector()

    with pytest.raises(RuntimeError, match="YOLOv8 detector is not loaded"):
        detector.detect(np.zeros((32, 48, 3), dtype=np.uint8))
