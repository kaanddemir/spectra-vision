import numpy as np
import pytest

from evaluation.perception_cache import PerceptionCache
from evaluation.tracking import TrackerConfig, agnostic_nms, lead_vehicle_switches, track_quality


def test_agnostic_nms_merges_same_vehicle_reported_as_car_and_truck():
    det = np.array([[0, 10, 10, 110, 90, 0.9, 2, -1],    # car
                    [0, 12, 11, 111, 92, 0.6, 7, -1],    # same object as truck
                    [0, 300, 10, 400, 90, 0.8, 2, -1],   # different car
                    [0, 12, 11, 111, 92, 0.6, 0, -1]])   # person overlapping: not a vehicle, kept
    kept = agnostic_nms(det, 0.7)
    assert len(kept) == 3 and not np.any((kept[:, 6] == 7))


def _cache(rows, n=30, dt=0.1):
    return PerceptionCache(media_sha256="x", frame_size=(1280, 720), timestamps=np.arange(n) * dt,
                           detections=np.array(rows, dtype=float))


def test_track_quality_flags_split_tracks_as_probable_fragments():
    rows = [(f, 600, 300, 700, 380, 0.9, 2, 1) for f in range(0, 10)]
    rows += [(f, 602, 301, 702, 381, 0.9, 2, 2) for f in range(12, 30)]   # same car, new id after a gap
    q = track_quality(_cache(rows))
    assert q["tracks"] == 2 and q["probable_fragments_per_min"] > 0


def test_lead_vehicle_switches_counts_identity_changes():
    rows = [(f, 600, 300, 700, 380, 0.9, 2, 1 if f < 15 else 2) for f in range(30)]
    r = lead_vehicle_switches(_cache(rows), 0, 3)
    assert r["switches"] == 1 and r["ids"] == [1, 2]


def test_tracker_lost_buffer_is_expressed_in_seconds():
    pytest.importorskip("supervision")
    from evaluation.tracking import retrack
    rows = [(f, 600, 300, 700, 380, 0.9, 2, -1) for f in range(30) if not 10 <= f < 25]  # 1.5 s gap
    cache = _cache(rows)
    short, _ = retrack(cache, TrackerConfig("short", lost_seconds=0.6), fps=10)
    long, _ = retrack(cache, TrackerConfig("long", lost_seconds=2.0), fps=10)
    ids = lambda c: {int(x) for x in c.detections[:, 7] if x >= 0}
    assert len(ids(long)) == 1 and len(ids(short)) == 2
