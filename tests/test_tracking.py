"""Unit tests for the IoU tracker: linking, confirmation, display IDs, and the
appearance-gated lost-track re-identification that keeps a physical object on a
single ID across detection gaps (the ID-duplication regression)."""

import numpy as np

from spectra.analysis.tracking import (
    IoUTracker,
    _appearance_descriptor,
    _appearance_similarity,
    _blend_appearance,
)
from spectra.vision.detection import Detection


def _det(bbox, cls="car", conf=0.9):
    return Detection(bbox=bbox, class_name=cls, confidence=conf)


def _frame(color, region, *, size=(200, 300)):
    """A blank frame with one solid-colour rectangle (BGR)."""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    x1, y1, x2, y2 = region
    img[y1:y2, x1:x2] = color
    return img


class TestIoUTracker:
    def test_pending_tracks_are_hidden_on_first_detection(self):
        tracker = IoUTracker()
        dets = [
            Detection(bbox=(0, 0, 20, 20), class_name="car", confidence=0.6),
        ]
        tracks = tracker.update(dets, frame_index=0, timestamp_sec=0.0)

        assert tracks == []

    def test_links_and_confirms_overlapping_detections_across_frames(self):
        tracker = IoUTracker(iou_threshold=0.2)
        first = [Detection(bbox=(0, 0, 50, 50), class_name="car", confidence=0.6)]
        tracks_t0 = tracker.update(first, frame_index=0, timestamp_sec=0.0)
        second = [Detection(bbox=(5, 5, 55, 55), class_name="car", confidence=0.6)]
        tracks_t1 = tracker.update(second, frame_index=1, timestamp_sec=0.1)
        third = [Detection(bbox=(10, 10, 60, 60), class_name="car", confidence=0.6)]
        tracks_t2 = tracker.update(third, frame_index=2, timestamp_sec=0.2)

        assert tracks_t0 == []
        assert tracks_t1 == []
        assert tracks_t2[0].track_id == 1
        assert tracks_t2[0].confirmed
        assert tracks_t2[0].display_id == 1
        assert len(tracks_t2[0].history) == 2

    def test_fast_confirms_large_high_confidence_detection(self):
        tracker = IoUTracker()
        tracks = tracker.update(
            [Detection(bbox=(0, 0, 50, 50), class_name="car", confidence=0.9)],
            frame_index=0,
            timestamp_sec=0.0,
            frame_shape=(200, 300, 3),
        )

        assert len(tracks) == 1
        assert tracks[0].confirmed
        assert tracks[0].display_id == 1

    def test_links_low_iou_detection_by_center_and_scale(self):
        tracker = IoUTracker(iou_threshold=0.5)
        first = [Detection(bbox=(0, 0, 100, 100), class_name="car", confidence=0.9)]
        tracks_t0 = tracker.update(first, frame_index=0, timestamp_sec=0.0, frame_shape=(200, 300, 3))
        second = [Detection(bbox=(35, 10, 135, 110), class_name="car", confidence=0.8)]
        tracks_t1 = tracker.update(second, frame_index=1, timestamp_sec=0.1, frame_shape=(200, 300, 3))

        assert tracks_t0[0].track_id == 1
        assert tracks_t1[0].track_id == 1
        assert tracks_t1[0].display_id == 1
        assert len(tracks_t1[0].history) == 1

    def test_reconnects_after_short_detection_miss(self):
        tracker = IoUTracker(iou_threshold=0.5, max_misses=5)
        tracker.update(
            [Detection(bbox=(0, 0, 100, 100), class_name="car", confidence=0.9)],
            frame_index=0,
            timestamp_sec=0.0,
            frame_shape=(200, 300, 3),
        )
        tracker.update(
            [Detection(bbox=(10, 0, 110, 100), class_name="car", confidence=0.9)],
            frame_index=1,
            timestamp_sec=0.1,
            frame_shape=(200, 300, 3),
        )
        # A confirmed track coasts through a short detection miss: it keeps
        # emitting (with its last bbox) so a live threat does not vanish from
        # the active set on a single missed detection frame.
        coasting = tracker.update([], frame_index=2, timestamp_sec=0.2, frame_shape=(200, 300, 3))
        assert len(coasting) == 1
        assert coasting[0].track_id == 1
        assert coasting[0].misses == 1

        tracks = tracker.update(
            [Detection(bbox=(75, 0, 175, 100), class_name="car", confidence=0.9)],
            frame_index=3,
            timestamp_sec=0.3,
            frame_shape=(200, 300, 3),
        )

        assert len(tracks) == 1
        assert tracks[0].track_id == 1
        assert tracks[0].display_id == 1
        assert tracks[0].misses == 0

    def test_center_scale_gate_does_not_merge_adjacent_tracks(self):
        tracker = IoUTracker(iou_threshold=0.5)
        tracks_t0 = tracker.update(
            [
                Detection(bbox=(0, 0, 100, 100), class_name="car", confidence=0.9),
                Detection(bbox=(160, 0, 260, 100), class_name="car", confidence=0.9),
            ],
            frame_index=0,
            timestamp_sec=0.0,
            frame_shape=(200, 320, 3),
        )

        tracks_t1 = tracker.update(
            [
                Detection(bbox=(35, 0, 135, 100), class_name="car", confidence=0.9),
                Detection(bbox=(195, 0, 295, 100), class_name="car", confidence=0.9),
            ],
            frame_index=1,
            timestamp_sec=0.1,
            frame_shape=(200, 320, 3),
        )

        assert sorted(track.track_id for track in tracks_t0) == [1, 2]
        assert sorted(track.track_id for track in tracks_t1) == [1, 2]
        assert sorted(track.display_id for track in tracks_t1) == [1, 2]

    def test_display_ids_are_scoped_per_class(self):
        tracker = IoUTracker()
        tracks = tracker.update(
            [
                Detection(bbox=(0, 0, 80, 80), class_name="car", confidence=0.9),
                Detection(bbox=(120, 0, 200, 80), class_name="truck", confidence=0.9),
            ],
            frame_index=0,
            timestamp_sec=0.0,
            frame_shape=(200, 320, 3),
        )

        by_class = {track.class_name: track for track in tracks}
        assert by_class["car"].track_id == 1
        assert by_class["truck"].track_id == 2
        assert by_class["car"].display_id == 1
        assert by_class["truck"].display_id == 1

    def test_pending_false_positive_does_not_consume_display_id(self):
        tracker = IoUTracker()
        tracker.update(
            [Detection(bbox=(0, 0, 20, 20), class_name="car", confidence=0.6)],
            frame_index=0,
            timestamp_sec=0.0,
        )

        tracks = tracker.update(
            [Detection(bbox=(40, 40, 100, 100), class_name="car", confidence=0.9)],
            frame_index=1,
            timestamp_sec=0.1,
            frame_shape=(200, 300, 3),
        )

        assert len(tracks) == 1
        assert tracks[0].track_id == 2
        assert tracks[0].display_id == 1

    def test_pending_false_positive_does_not_propagate(self):
        tracker = IoUTracker()
        tracker.update(
            [Detection(bbox=(0, 0, 20, 20), class_name="car", confidence=0.6)],
            frame_index=0,
            timestamp_sec=0.0,
        )

        assert tracker.propagate() == []

    def test_does_not_link_across_classes(self):
        tracker = IoUTracker(iou_threshold=0.2)
        tracker.update(
            [Detection(bbox=(0, 0, 50, 50), class_name="car", confidence=0.6)],
            frame_index=0,
            timestamp_sec=0.0,
        )
        tracks = tracker.update(
            [Detection(bbox=(0, 0, 50, 50), class_name="person", confidence=0.6)],
            frame_index=1,
            timestamp_sec=0.1,
        )

        assert tracks == []


class TestAppearanceDescriptor:
    def test_descriptor_normalized_for_valid_bbox(self):
        frame = _frame((0, 0, 200), (100, 60, 160, 120))
        desc = _appearance_descriptor(frame, (100, 60, 160, 120))
        assert desc is not None
        assert desc.dtype == np.float32
        # L1-normalized histogram sums to ~1.0.
        assert float(desc.sum()) == np.float32(1.0) or abs(float(desc.sum()) - 1.0) < 1e-3

    def test_descriptor_none_for_missing_frame_or_degenerate_bbox(self):
        frame = _frame((0, 0, 200), (100, 60, 160, 120))
        assert _appearance_descriptor(None, (0, 0, 10, 10)) is None
        assert _appearance_descriptor(frame, (50, 50, 51, 51)) is None  # < 2px

    def test_similarity_high_for_identical_low_for_distinct(self):
        red = _frame((0, 0, 200), (100, 60, 160, 120))
        blue = _frame((200, 0, 0), (100, 60, 160, 120))
        a = _appearance_descriptor(red, (100, 60, 160, 120))
        b = _appearance_descriptor(red, (100, 60, 160, 120))
        c = _appearance_descriptor(blue, (100, 60, 160, 120))

        assert _appearance_similarity(a, b) > 0.95
        assert _appearance_similarity(a, c) < 0.30  # below the re-id gate
        assert _appearance_similarity(a, None) is None

    def test_blend_updates_descriptor_in_place_field(self):
        from spectra.analysis.tracking import Track

        red = _appearance_descriptor(_frame((0, 0, 200), (100, 60, 160, 120)), (100, 60, 160, 120))
        track = Track(
            track_id=1, class_name="car", confidence=0.9,
            bbox=(100, 60, 160, 120), frame_index=0, timestamp_sec=0.0,
        )
        assert track.appearance is None
        _blend_appearance(track, red)
        assert track.appearance is not None
        # Blending toward a new (blue) descriptor shifts the signature.
        blue = _appearance_descriptor(_frame((200, 0, 0), (100, 60, 160, 120)), (100, 60, 160, 120))
        before = track.appearance.copy()
        _blend_appearance(track, blue)
        assert not np.array_equal(track.appearance, before)


class TestLostPoolReidentification:
    """An object that leaves the frame (corridor drop / occlusion) longer than
    ``max_misses`` is demoted to the lost pool, then re-identified onto its
    original ID when it reappears within ``max_lost_sec`` *and* its colour
    matches — the core fix for ID duplication."""

    def _confirm_and_lose(self, tracker, frame):
        # Confirm a track over four detection frames.
        for i in range(4):
            out = tracker.update(
                [_det((100, 60, 160, 120))],
                frame_index=i, timestamp_sec=i * 0.1, frame_bgr=frame,
            )
        track_id = out[0].track_id
        # Nine empty detection frames exceed max_misses (8) -> demote to lost.
        for i in range(4, 13):
            tracker.update([], frame_index=i, timestamp_sec=i * 0.1, frame_bgr=frame)
        return track_id

    def test_lost_pool_revival_keeps_id(self):
        tracker = IoUTracker()
        red = _frame((0, 0, 200), (100, 60, 160, 120))
        track_id = self._confirm_and_lose(tracker, red)

        assert not tracker._tracks
        assert track_id in tracker._lost_tracks

        # Reappear within the 2.5s window with a matching-colour crop.
        out = tracker.update(
            [_det((138, 60, 198, 120))],
            frame_index=13, timestamp_sec=1.3, frame_bgr=red,
        )

        assert track_id in tracker._tracks
        assert tracker._tracks[track_id].misses == 0
        # No fresh ID was minted for the same physical object.
        assert [t for t in tracker._tracks if t != track_id] == []

    def test_appearance_gate_blocks_different_object(self):
        tracker = IoUTracker()
        red = _frame((0, 0, 200), (100, 60, 160, 120))
        track_id = self._confirm_and_lose(tracker, red)

        # A differently-coloured object at the same plausible position must NOT
        # hijack the stale ID: the appearance gate rejects re-id, a new ID mints.
        blue = _frame((200, 0, 0), (130, 60, 190, 120))
        tracker.update(
            [_det((138, 60, 198, 120))],
            frame_index=13, timestamp_sec=1.3, frame_bgr=blue,
        )

        revived = track_id in tracker._tracks and tracker._tracks[track_id].misses == 0
        minted = [t for t in tracker._tracks if t != track_id]
        assert not revived
        assert minted  # a brand-new id was created instead

    def test_lost_pool_expires_after_window(self):
        tracker = IoUTracker()
        red = _frame((0, 0, 200), (100, 60, 160, 120))
        track_id = self._confirm_and_lose(tracker, red)

        # Reappear AFTER max_lost_sec (2.5s) -> the lost track is purged, so even
        # a colour match cannot revive it; a new ID is assigned.
        tracker.update(
            [_det((138, 60, 198, 120))],
            frame_index=120, timestamp_sec=4.0, frame_bgr=red,
        )

        assert track_id not in tracker._tracks
        assert track_id not in tracker._lost_tracks
