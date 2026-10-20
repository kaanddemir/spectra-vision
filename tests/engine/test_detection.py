
import numpy as np
import pytest

from spectra.engine.adapters import ComponentFailure
from spectra.engine.contracts import Box, Detection, ObjectCategory as C
from spectra.engine.detection import DFineOnnxDetector, suppress_duplicates
from spectra.engine.video import file_sha256
from spectra.engine.video import Frame

from tests.engine.models import MODEL, MODEL_SHA, needs_model


def det(x1, y1, x2, y2, score, category=C.CAR):
    return Detection(box=Box(x1=x1, y1=y1, x2=x2, y2=y2), score=score, category=category)


def detector(**kw):
    return DFineOnnxDetector(MODEL, MODEL_SHA, **kw)


def test_one_vehicle_reported_under_two_classes_is_kept_once():
    out = suppress_duplicates([det(100, 100, 200, 180, 0.8, C.CAR), det(102, 98, 201, 182, 0.6, C.TRUCK)], 0.7)
    assert out == [det(100, 100, 200, 180, 0.8, C.CAR)]


def test_distinct_overlapping_vehicles_are_both_kept():
    # a car partly hidden behind another: IoU about 0.43
    a, b = det(100, 100, 200, 180, 0.9), det(140, 105, 240, 185, 0.7)
    assert suppress_duplicates([a, b], 0.7) == [a, b]


def test_people_are_never_suppressed_by_vehicles():
    rider, bike = det(100, 60, 140, 180, 0.8, C.PERSON), det(100, 62, 140, 182, 0.7, C.MOTORCYCLE)
    assert set(suppress_duplicates([rider, bike], 0.7)) == {rider, bike}


def _parse(**arrays):
    base = dict(labels=np.array([2, 7, 0, 9, 2]), scores=np.array([0.9, 0.5, 0.4, 0.95, 0.05]),
                boxes=np.array([[10, 10, 50, 40], [300, 20, 360, 70], [-5, 100, 20, 160],
                                [0, 0, 5, 5], [1, 1, 2, 2]], float))
    base.update(arrays)
    return detector(duplicate_iou=None)._parse(base["labels"], base["boxes"], base["scores"], 320, 160)


def test_parsing_maps_classes_thresholds_and_clips():
    out = _parse()
    assert [d.category for d in out] == [C.CAR, C.TRUCK, C.PERSON]  # traffic light (9) and the 0.05 box dropped
    person = out[2]
    assert person.box.x1 == 0 and person.box.y2 == 160  # clipped to the image
    truck = out[1]
    assert truck.box.x2 == 320 and truck.score == pytest.approx(0.5)


def test_empty_output_is_an_empty_list():
    assert _parse(labels=np.zeros(0, int), scores=np.zeros(0), boxes=np.zeros((0, 4))) == []


@pytest.mark.parametrize("bad", [dict(boxes=np.full((5, 4), np.nan)), dict(scores=np.array([np.inf] * 5)),
                                 dict(boxes=np.zeros((4, 4)))])
def test_malformed_output_is_a_frame_failure(bad):
    with pytest.raises(ComponentFailure):
        _parse(**bad)


def test_boxes_entirely_outside_the_image_are_dropped():
    out = _parse(labels=np.array([2]), scores=np.array([0.9]), boxes=np.array([[400, 10, 450, 40]], float))
    assert out == []


def test_identity_includes_checkpoint_and_output_affecting_parameters():
    a, b = detector(), detector(input_size=512)
    assert a.info.checkpoint_sha256 == MODEL_SHA and a.info != b.info
    assert detector(score_threshold=0.3).info != a.info and detector(duplicate_iou=None).info != a.info


def test_missing_or_altered_checkpoint_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        DFineOnnxDetector(tmp_path / "absent.onnx", MODEL_SHA).load()
    fake = tmp_path / "model.onnx"
    fake.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="sha256"):
        DFineOnnxDetector(fake, MODEL_SHA).load()


def test_use_before_load_is_an_error():
    with pytest.raises(RuntimeError):
        detector().detect(Frame(decode_index=0, t=0.0, pts=0, image=np.zeros((8, 8, 3), np.uint8)))


@pytest.mark.model
@needs_model
def test_real_model_loads_runs_and_releases():
    assert file_sha256(MODEL) == MODEL_SHA
    d = detector()
    d.load()
    blank = Frame(decode_index=0, t=0.0, pts=0, image=np.full((360, 640, 3), 127, np.uint8))
    out = d.detect(blank)
    assert isinstance(out, list) and all(0.1 <= o.score <= 1 for o in out)
    assert all(0 <= o.box.x1 < o.box.x2 <= 640 and 0 <= o.box.y1 < o.box.y2 <= 360 for o in out)
    assert d.detect(blank) == out  # deterministic on CPU
    d.close()
    with pytest.raises(RuntimeError):
        d.detect(blank)
