"""Engine result -> proposal -> evidence package -> timeline entry -> video reference, on a synthetic video."""
from spectra.engine.cache import PerceptionCache
from spectra.engine.contracts import canonical_json
from spectra.engine.runner import AnalysisRun
from spectra.engine.video import file_sha256
from spectra.evidence.build import from_outcome
from spectra.evidence.media import resolve
from spectra.evidence.overlays import overlay
from spectra.evidence.store import EvidenceStore
from tests.engine.fakes import FakeAnalyzer, FakeDetector, FakeEgoLane, FakeSegmenter, FakeTracker
from tests.engine.media import make_video
from tests.engine.fakes import calm

VFR = [round(0.3 + i * 0.04, 3) for i in range(30)] + [round(2.0 + i * 0.1, 3) for i in range(20)]


def run(video, cache):
    return AnalysisRun(video, cache=cache, guard=calm(), detector=FakeDetector(), tracker=FakeTracker(),
                       segmenter=FakeSegmenter(), ego_lane=FakeEgoLane(), analyzers=[FakeAnalyzer()]).run()


def test_end_to_end_from_engine_result_to_video_reference(tmp_path):
    video = make_video(tmp_path / "vfr.mp4", VFR, gop=8)
    source_hash = file_sha256(video)
    outcome = run(video, PerceptionCache(tmp_path / "cache"))
    def proposals():
        return canonical_json([e.model_dump(mode="json") for e in outcome.result.events])
    events = proposals()

    timeline, packages = from_outcome(outcome)
    store = EvidenceStore(tmp_path / "evidence")
    store.write(timeline, packages)
    assert proposals() == events  # evidence preparation changes nothing

    replay = run(video, PerceptionCache(tmp_path / "cache"))
    assert replay.cache_hit and from_outcome(replay)[0] == timeline  # same records, same timeline

    [entry] = timeline.entries
    pk = store.package(timeline.build_key, entry.event_id)
    assert pk.proposal == outcome.result.events[0] and entry.package.endswith(f"{entry.event_id}.json")

    o = overlay(replay.result.perception, pk)  # read from the cache entry, no model involved
    assert o.frames and all(f.tracks is not None for f in o.frames)

    r = resolve(video, pk.video, pk.review)
    assert r.first_frame_s >= pk.review.start_s and r.last_frame_s <= pk.review.end_s
    clip = store.clip(pk, video)
    assert clip.origin_s <= pk.review.start_s and store.clip(pk, video) == clip
    assert store.clip_path(pk, clip).is_file() and file_sha256(video) == source_hash
