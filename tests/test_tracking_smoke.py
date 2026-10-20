import numpy as np

from spectra.analysis.tracking import IoUTracker
from spectra.vision.detection import Detection


def _det(bbox, cls="car", conf=0.9):
    return Detection(bbox=bbox, class_name=cls, confidence=conf)


def _frame(color, region=(100, 60, 160, 120), *, size=(200, 300)):
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    x1, y1, x2, y2 = region
    img[y1:y2, x1:x2] = color
    return img


def _confirm_and_lose(tracker, frame):
    out = []
    for i in range(4):
        out = tracker.update([_det((100, 60, 160, 120))], frame_index=i, timestamp_sec=i * 0.1, frame_bgr=frame)
    track_id = out[0].track_id
    for i in range(4, 13):
        tracker.update([], frame_index=i, timestamp_sec=i * 0.1, frame_bgr=frame)
    return track_id


def test_tracker_confirms_overlapping_detections_with_stable_display_id():
    tracker = IoUTracker(iou_threshold=0.2)

    assert tracker.update([_det((0, 0, 50, 50), conf=0.6)], frame_index=0, timestamp_sec=0.0) == []
    assert tracker.update([_det((5, 5, 55, 55), conf=0.6)], frame_index=1, timestamp_sec=0.1) == []
    tracks = tracker.update([_det((10, 10, 60, 60), conf=0.6)], frame_index=2, timestamp_sec=0.2)

    assert tracks[0].track_id == 1
    assert tracks[0].confirmed
    assert tracks[0].display_id == 1


def test_tracker_coasts_through_short_detection_miss_and_reconnects():
    tracker = IoUTracker(iou_threshold=0.5, max_misses=5)
    tracker.update([_det((0, 0, 100, 100))], frame_index=0, timestamp_sec=0.0, frame_shape=(200, 300, 3))
    tracker.update([_det((10, 0, 110, 100))], frame_index=1, timestamp_sec=0.1, frame_shape=(200, 300, 3))

    coasting = tracker.update([], frame_index=2, timestamp_sec=0.2, frame_shape=(200, 300, 3))
    assert coasting[0].track_id == 1
    assert coasting[0].misses == 1

    tracks = tracker.update([_det((75, 0, 175, 100))], frame_index=3, timestamp_sec=0.3, frame_shape=(200, 300, 3))

    assert tracks[0].track_id == 1
    assert tracks[0].misses == 0


def test_hot_track_coasts_longer_than_a_normal_track():
    # A normal track goes silent after coast_limit (2) missed detections, while a
    # "hot" threat id keeps emitting up to hot_coast_limit (6) so a near lead
    # vehicle the detector briefly drops stays in the active set.
    tracker = IoUTracker(iou_threshold=0.5, coast_limit=2, hot_coast_limit=6, max_misses=8)
    for i in range(3):  # confirm the track
        tracker.update([_det((0, 0, 100, 100))], frame_index=i, timestamp_sec=i * 0.1, frame_shape=(200, 300, 3))
    track_id = 1

    # 4 missed detections: beyond coast_limit but within hot_coast_limit.
    for i in range(3, 7):
        normal = tracker.update([], frame_index=i, timestamp_sec=i * 0.1, frame_shape=(200, 300, 3))
    assert normal == []  # not hot → silent after 2 misses

    # Same miss streak, but the id is flagged hot → still emitted.
    hot = tracker.propagate(hot_ids={track_id})
    assert [t.track_id for t in hot] == [track_id]
    assert hot[0].misses == 4


def test_reid_reconnects_fast_growing_box_to_original_id():
    # A close vehicle that doubles in size across a short gap must keep its id
    # (re-id scale tolerance), preserving its depth-Kalman history.
    tracker = IoUTracker()
    red = _frame((0, 0, 200))
    track_id = _confirm_and_lose(tracker, red)

    # Re-detect a much larger box (≈4× area) at a moved-down position.
    big = _frame((0, 0, 200), region=(96, 56, 200, 124))
    out = tracker.update([_det((96, 56, 200, 124))], frame_index=13, timestamp_sec=1.3, frame_bgr=big)

    assert out[0].track_id == track_id


def test_tracker_does_not_link_across_classes():
    tracker = IoUTracker(iou_threshold=0.2)
    tracker.update([_det((0, 0, 50, 50), cls="car", conf=0.6)], frame_index=0, timestamp_sec=0.0)

    assert tracker.update([_det((0, 0, 50, 50), cls="person", conf=0.6)], frame_index=1, timestamp_sec=0.1) == []


def test_display_ids_are_scoped_per_class():
    tracks = IoUTracker().update(
        [_det((0, 0, 80, 80), cls="car"), _det((120, 0, 200, 80), cls="truck")],
        frame_index=0,
        timestamp_sec=0.0,
        frame_shape=(200, 320, 3),
    )

    by_class = {track.class_name: track for track in tracks}
    assert by_class["car"].display_id == 1
    assert by_class["truck"].display_id == 1


def test_lost_pool_reidentifies_matching_object_without_minting_new_id():
    tracker = IoUTracker()
    red = _frame((0, 0, 200))
    track_id = _confirm_and_lose(tracker, red)

    out = tracker.update([_det((138, 60, 198, 120))], frame_index=13, timestamp_sec=1.3, frame_bgr=red)

    assert out[0].track_id == track_id
    assert list(tracker._tracks) == [track_id]
