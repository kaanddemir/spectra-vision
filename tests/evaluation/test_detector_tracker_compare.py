from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.detector_compare import agreement, describe
from evaluation.detectors.rfdetr_onnx import decode
from evaluation.tracker_compare import McByteTracker, indicators
from spectra.engine.contracts import Box, Detection, ObjectCategory, TrackedObject

CAR = ObjectCategory.CAR


def det(x1, y1, x2, y2, score=0.9, cat=CAR):
    return Detection(box=Box(x1=x1, y1=y1, x2=x2, y2=y2), score=score, category=cat)


def frame(t, *dets):
    return SimpleNamespace(t=t, detections=dets)


def test_rfdetr_decode_uses_independent_sigmoids_and_sparse_coco_ids():
    # At most Q query/class pairs are kept (Q = 3 queries here), as in rfdetr's own decoder.
    boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]], np.float32)
    logits = np.full((3, 91), -9.0, np.float32)
    logits[0, 3] = 4.0   # COCO id 3 (car) -> contiguous 2
    logits[0, 8] = 2.0   # same query also scores as truck (id 8 -> 7): both pairs kept
    logits[1, 12] = 3.0  # id 12 is unused in COCO -> -1, dropped later by the class filter
    xyxy, scores, labels = decode(boxes, logits, 200, 100, threshold=0.5)
    assert labels.tolist() == [2, -1, 7]
    assert np.all(np.diff(scores) <= 0)
    assert xyxy[0] == pytest.approx([80, 30, 120, 70])


def test_rfdetr_decode_threshold_and_clipping():
    boxes = np.array([[0.95, 0.5, 0.2, 0.2]], np.float32)
    logits = np.full((1, 91), -9.0, np.float32)
    logits[0, 3] = 0.1
    assert decode(boxes, logits, 100, 100, threshold=0.6)[0].shape == (0, 4)
    xyxy, _, _ = decode(boxes, logits, 100, 100, threshold=0.5)
    assert xyxy[0, 2] == 100


def test_agreement_matches_one_to_one_by_height_and_ignores_unpaired_frames():
    a = [frame(0.0, det(0, 0, 10, 20), det(100, 0, 150, 50)), frame(0.1, det(0, 0, 10, 100))]
    b = [frame(0.0, det(1, 0, 11, 20), det(1, 0, 11, 20)), frame(0.5, det(0, 0, 10, 100))]
    r = agreement(a, b)
    assert r["paired_frames"] == 1
    assert r["both"] == {"0-30": 1, "30-80": 0, "80-inf": 0}
    assert r["only_a"] == {"0-30": 0, "30-80": 1, "80-inf": 0}
    assert r["only_b"] == {"0-30": 1, "30-80": 0, "80-inf": 0}


def test_describe_counts_above_threshold_only():
    r = describe([frame(0.0, det(0, 0, 10, 20), det(0, 0, 10, 90, score=0.3),
                        det(0, 0, 5, 40, cat=ObjectCategory.PERSON))])
    assert r["by_category"] == {"car": 1, "person": 1}
    assert r["vehicles_by_height_px"] == {"0-30": 1, "30-80": 0, "80-inf": 0}


def trk(tid, y2, x1=0.0):
    return TrackedObject(track_id=tid, box=Box(x1=x1, y1=0, x2=x1 + 10, y2=y2), score=0.9, category=CAR)


def test_indicators_count_fragments_and_size_jumps():
    recs = [(0.0, [trk(1, 20)]), (0.1, [trk(1, 20)]), (0.2, [trk(1, 60)]),   # height x3 in 0.1 s: a jump
            (1.0, [trk(2, 60)]), (1.1, [trk(2, 60)])]                       # restarts where 1 ended: a fragment
    r = indicators(recs)
    assert r["tracks"] == 2
    assert r["jumps_per_min"] == pytest.approx(1 / (1.1 / 60), rel=0.01)
    assert r["probable_fragments_per_min"] == pytest.approx(1 / (1.1 / 60), rel=0.01)


def test_mcbyte_adapter_builds_mcbyte_with_motion_compensation_and_restores_the_library():
    import trackers
    original = trackers.BoTSORTTracker
    t = McByteTracker()
    t.load()
    assert type(t._tracker).__name__ == "McByteTracker" and t._tracker.cmc is not None
    assert trackers.BoTSORTTracker is original and t.info.name == "mcbyte"
