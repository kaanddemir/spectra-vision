import pytest

from spectra.engine.cache import PerceptionCache
from spectra.engine.runner import AnalysisRun, RunState, StopReason
from spectra.engine.worker import DetectorSpec, WorkerJob, run_in_worker
from tests.engine.fakes import FakeAnalyzer, FakeDetector, calm
from tests.engine.media import make_video


def test_cache_only_hit_preserves_keys_and_never_loads_detector(tmp_path):
    video = make_video(tmp_path / 'clip.mp4', [0.0, 0.1, 0.2])
    cache = PerceptionCache(tmp_path / 'cache')
    first = AnalysisRun(video, detector=FakeDetector(), analyzers=[FakeAnalyzer()], cache=cache, guard=calm()).run()
    detector = FakeDetector()
    def forbidden(): raise AssertionError('perception model must not load')
    detector.load = forbidden
    replay = AnalysisRun(video, detector=detector, analyzers=[FakeAnalyzer()], cache=cache,
                         cache_only=True, guard=calm()).run()
    assert replay.state is RunState.COMPLETED and replay.cache_hit
    assert replay.result == first.result


@pytest.mark.parametrize('corrupt', [False, True])
def test_cache_only_miss_never_loads_or_writes(tmp_path, corrupt):
    video = make_video(tmp_path / 'clip.mp4', [0.0, 0.1, 0.2])
    cache = PerceptionCache(tmp_path / 'cache')
    if corrupt:
        original = AnalysisRun(video, detector=FakeDetector(), cache=cache, guard=calm()).run()
        (cache.entry(original.result.perception_key) / 'manifest.json').write_text('{}')
    before = {str(p): p.read_bytes() for p in cache.root.rglob('*') if p.is_file()}
    detector = FakeDetector()
    result = AnalysisRun(video, detector=detector, cache=cache, cache_only=True, guard=calm()).run()
    assert result.state is RunState.FAILED and result.stop is StopReason.ERROR
    assert 'cache-only' in result.error and detector.loaded == 0
    assert {str(p): p.read_bytes() for p in cache.root.rglob('*') if p.is_file()} == before


def test_worker_propagates_cache_only_before_model_loading(tmp_path):
    video = make_video(tmp_path / 'clip.mp4', [0.0, 0.1, 0.2])
    job = WorkerJob(str(video), str(tmp_path / 'cache'), detector=DetectorSpec('absent.onnx', 'a'*64), cache_only=True)
    outcome = run_in_worker(job, timeout_s=20)
    assert outcome.state is RunState.FAILED and outcome.stop is StopReason.ERROR
    assert 'cache-only' in outcome.error
    assert not (tmp_path / 'cache').exists()
