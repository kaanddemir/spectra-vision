"""Streamed frame records, lazy replay, and hood withholding inside a run."""
import hashlib
import json
import pickle

import numpy as np
import pytest

from spectra.engine.adapters import Segmentation
from spectra.engine.cache import FRAMES, PerceptionCache, StoredFrames, perception_key
from spectra.engine.contracts import Box, Detection, FrameRecord, ObjectCategory, OutputStatus, Role
from spectra.engine.runner import AnalysisRun, RunState
from tests.engine.fakes import FakeDetector, FakeSegmenter, FakeTracker
from tests.engine.media import make_video
from tests.engine.cache_fixtures import result, store
from tests.engine.media import VFR
from tests.engine.fakes import calm


def test_replay_reads_frames_lazily_and_equals_the_written_result(tmp_path):
    cache = PerceptionCache(tmp_path)
    r = result()
    key = store(cache, r)
    found = cache.lookup(key).result
    assert isinstance(found.frames, StoredFrames)
    assert found == r and list(found.frames) == list(r.frames)
    assert found.frames[-1] == r.frames[-1] and found.frames[1:] == r.frames[1:]
    with pytest.raises(IndexError):
        found.frames[3]
    with pytest.raises(TypeError, match="not picklable"):
        pickle.dumps(found.frames)
    assert json.loads(found.model_dump_json())["frames"] == json.loads(r.model_dump_json())["frames"]


def test_entry_layout_keeps_frames_out_of_the_header(tmp_path):
    cache = PerceptionCache(tmp_path)
    key = store(cache, result())
    entry = cache.entry(key)
    assert "frames" not in json.loads((entry / "perception.json").read_text())
    assert len((entry / FRAMES).read_text().splitlines()) == 3


def _rehash(entry, name):
    manifest = json.loads((entry / "manifest.json").read_text())
    manifest["files"][name] = hashlib.sha256((entry / name).read_bytes()).hexdigest()
    (entry / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("edit,reason", [
    (lambda lines: lines[:1] + lines[2:], "frame indexes"),
    (lambda lines: lines + [lines[-1]], "frame indexes"),
    (lambda lines: lines[:-1] + ["{not json"], "frames fail validation"),
])
def test_inconsistent_frame_files_are_rejected_even_with_matching_hashes(tmp_path, edit, reason):
    cache = PerceptionCache(tmp_path)
    key = store(cache, result())
    entry = cache.entry(key)
    lines = (entry / FRAMES).read_text().splitlines()
    (entry / FRAMES).write_text("\n".join(edit(lines)) + "\n")
    _rehash(entry, FRAMES)
    found = cache.lookup(key)
    assert found.result is None and "frames fail validation" in found.reason and (
        reason in found.reason or reason == "frames fail validation")


def test_entries_in_the_earlier_single_file_layout_are_rejected_with_a_reason(tmp_path):
    cache = PerceptionCache(tmp_path)
    key = store(cache, result())
    entry = cache.entry(key)
    (entry / FRAMES).unlink()
    manifest = json.loads((entry / "manifest.json").read_text())
    del manifest["files"][FRAMES]
    (entry / "manifest.json").write_text(json.dumps(manifest))
    assert "frames.jsonl" in cache.lookup(key).reason


def test_appended_frames_must_arrive_in_order(tmp_path):
    cache = PerceptionCache(tmp_path)
    r = result()
    writer = cache.begin(perception_key(r.video, r.config, r.provenance))
    writer.append_frame(r.frames[0])
    with pytest.raises(ValueError, match="in order"):
        writer.append_frame(r.frames[2])
    with pytest.raises(ValueError, match="in order"):
        writer.append_frame(r.frames[0])
    writer.abort()
    assert not writer.staging.exists()


def test_a_streamed_run_returns_lazily_read_frames(tmp_path):
    clip = make_video(tmp_path / "vfr.mp4", VFR)
    outcome = AnalysisRun(clip, detector=FakeDetector(), tracker=FakeTracker(),
                          cache=PerceptionCache(tmp_path / "cache"), guard=calm()).run()
    assert outcome.state is RunState.COMPLETED
    assert isinstance(outcome.result.perception.frames, StoredFrames)
    assert len(outcome.result.perception.frames) == len(VFR)


# --- hood withholding -----------------------------------------------------------------------------


class HoodSegmenter(FakeSegmenter):
    """Bottom quarter is the camera's own vehicle."""
    CLASSES = ("road", "lane_marking", "ego_vehicle")

    def segment(self, frame):
        self._call(frame)
        mask = np.zeros((24, 32), np.uint8)
        mask[18:] = 2
        return Segmentation(mask=mask, classes=self.CLASSES)


class HoodAndCarDetector(FakeDetector):
    def detect(self, frame):
        self._call(frame)
        h, w = frame.image.shape[:2]
        return [Detection(box=Box(x1=0, y1=h * 0.78, x2=w, y2=h), score=0.8, category=ObjectCategory.CAR),
                Detection(box=Box(x1=w * 0.4, y1=h * 0.4, x2=w * 0.6, y2=h * 0.7), score=0.9,
                          category=ObjectCategory.CAR)]


class RecordingTracker(FakeTracker):
    def update(self, frame, detections):
        self.seen = getattr(self, "seen", []) + [len(detections)]
        return super().update(frame, detections)


def test_hood_detections_stay_in_the_record_but_never_reach_the_tracker(tmp_path):
    clip = make_video(tmp_path / "vfr.mp4", VFR)
    tracker = RecordingTracker()
    outcome = AnalysisRun(clip, detector=HoodAndCarDetector(), tracker=tracker, segmenter=HoodSegmenter(),
                          cache=PerceptionCache(tmp_path / "cache"), guard=calm()).run()
    assert outcome.state is RunState.COMPLETED, outcome.error
    frames = list(outcome.result.perception.frames)
    for f in frames:
        hood, car = f.detections
        assert hood.ego_vehicle and hood.ego_overlap > 0.9
        assert not car.ego_vehicle and car.ego_overlap == 0
        assert len(f.tracks) == 1
    assert tracker.seen == [1] * len(VFR)


def test_without_a_segmenter_detections_carry_no_ego_judgement(tmp_path):
    clip = make_video(tmp_path / "vfr.mp4", VFR)
    tracker = RecordingTracker()
    outcome = AnalysisRun(clip, detector=HoodAndCarDetector(), tracker=tracker,
                          cache=PerceptionCache(tmp_path / "cache"), guard=calm()).run()
    f = outcome.result.perception.frames[0]
    assert all(d.ego_overlap is None and not d.ego_vehicle for d in f.detections)
    assert tracker.seen == [2] * len(VFR)  # unchanged behaviour: everything is tracked


def test_a_stale_mask_is_not_used_for_hood_judgement(tmp_path):
    clip = make_video(tmp_path / "vfr.mp4", VFR)
    run = AnalysisRun(clip, detector=HoodAndCarDetector(), tracker=RecordingTracker(),
                      segmenter=HoodSegmenter(interval_s=10.0), cache=PerceptionCache(tmp_path / "cache"),
                      guard=calm(), ego_mask_max_age_s=0.3)
    frames = list(run.run().result.perception.frames)
    assert frames[0].status[Role.SEGMENTER] is OutputStatus.OK
    late = [f for f in frames if f.t > 0.31]
    assert late and all(d.ego_overlap is None for f in late for d in f.detections)


def test_frame_records_reject_relations_for_unknown_tracks():
    from spectra.engine.contracts import LaneRelation, TrackLaneRelation
    with pytest.raises(ValueError, match="lane relation"):
        FrameRecord(index=0, t=0.0, status={}, lane_relations=(
            TrackLaneRelation(track_id=3, relation=LaneRelation.EGO_LANE, lane_t=0.0),))
