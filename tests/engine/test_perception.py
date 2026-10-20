"""End-to-end perception with the real local detector and tracker, in process and in a worker."""
import pytest

from spectra.engine.cache import STAGING, PerceptionCache
from spectra.engine.contracts import OutputStatus, Role, RunConfig
from spectra.engine.runner import AnalysisRun, CancelToken, RunState
from spectra.engine.worker import DetectorSpec, TrackerSpec, WorkerJob, run_in_worker
from tests.engine.media import cfr_times, make_video
from tests.engine.models import MODEL, MODEL_SHA, needs_model
from tests.engine.fakes import calm

pytest.importorskip("trackers")
from spectra.engine.detection import DFineOnnxDetector  # noqa: E402
from spectra.engine.tracking import BoTSortTracker  # noqa: E402

pytestmark = [needs_model, pytest.mark.model, pytest.mark.slow]


class Counting(DFineOnnxDetector):
    loads = 0

    def load(self):
        Counting.loads += 1
        super().load()


@pytest.fixture
def clip(tmp_path):
    return make_video(tmp_path / "clip.mp4", cfr_times(20, 10))


def job(clip, tmp_path, **kw):
    base = dict(video=str(clip), cache_root=str(tmp_path / "cache"), config={"interval_s": 0.2},
                detector=DetectorSpec(str(MODEL), MODEL_SHA), tracker=TrackerSpec(reference_rate=5.0))
    return WorkerJob(**{**base, **kw})


def test_in_process_run_records_detector_and_tracker_and_replays_without_models(clip, tmp_path):
    cache = PerceptionCache(tmp_path / "cache")

    def run():
        return AnalysisRun(clip, RunConfig(interval_s=0.2), detector=Counting(MODEL, MODEL_SHA),
                           tracker=BoTSortTracker(reference_rate=5.0), cache=cache, guard=calm()).run()

    Counting.loads = 0
    first = run()
    assert first.state is RunState.COMPLETED and len(first.result.perception.frames) == 10
    assert all(f.status[Role.DETECTOR] is OutputStatus.OK and f.status[Role.TRACKER] is OutputStatus.OK
               for f in first.result.perception.frames)
    names = [c.name for c in first.result.perception.provenance.components]
    assert names == ["dfine-n-coco", "botsort"]
    second = run()
    assert second.cache_hit and second.result == first.result and Counting.loads == 1


def test_worker_completes_and_matches_the_in_process_result(clip, tmp_path):
    seen = []
    outcome = run_in_worker(job(clip, tmp_path), on_progress=seen.append, timeout_s=300)
    assert outcome.state is RunState.COMPLETED, outcome.error
    assert seen and seen[-1].state is RunState.COMPLETED
    local = AnalysisRun(clip, RunConfig(interval_s=0.2), detector=DFineOnnxDetector(MODEL, MODEL_SHA),
                        tracker=BoTSortTracker(reference_rate=5.0), guard=calm()).run()
    assert local.result.perception.frames == outcome.result.perception.frames
    assert run_in_worker(job(clip, tmp_path), timeout_s=300).cache_hit


def test_worker_cancellation(clip, tmp_path):
    token = CancelToken()
    outcome = run_in_worker(job(clip, tmp_path, config={}), cancel=token,
                            on_progress=lambda p: token.cancel() if p.frames_analysed >= 1 else None, timeout_s=300)
    assert outcome.state is RunState.CANCELLED and outcome.result is None
    root = tmp_path / "cache"
    assert not root.exists() or not [p for p in root.iterdir() if p.name != STAGING]


def test_worker_reports_a_checkpoint_mismatch_as_a_failed_run(clip, tmp_path):
    outcome = run_in_worker(job(clip, tmp_path, detector=DetectorSpec(str(MODEL), "0" * 64)), timeout_s=300)
    assert outcome.state is RunState.FAILED and "sha256" in outcome.error


def test_worker_reports_a_missing_video(tmp_path):
    outcome = run_in_worker(WorkerJob(video=str(tmp_path / "absent.mp4"), cache_root=str(tmp_path)))
    assert outcome.state is RunState.FAILED and "not found" in outcome.error
