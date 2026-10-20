from dataclasses import replace
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from scripts import check_real_clips as check
from spectra.engine.cache import PerceptionCache
from spectra.engine.resources import ResourceExhausted
from spectra.engine.runner import AnalysisRun, RunState, StopReason, RunOutcome
from spectra.engine.worker import DetectorSpec, SegmenterSpec
from tests.engine.fakes import FakeAnalyzer, FakeDetector, FakeTracker, calm
from tests.engine.media import make_video, cfr_times


def test_guard_refusal_precedes_any_input_or_output_access(tmp_path, monkeypatch):
    def refuse(**kwargs): raise ResourceExhausted('pressure 2')
    monkeypatch.setattr(check, 'ResourceGuard', lambda **kw: SimpleNamespace(check=refuse))
    out = tmp_path / 'out'
    assert check.main(['--out', str(out)]) == 3
    assert not out.exists()


def test_importing_script_does_not_start_work():
    loaded = runpy.run_path(str(check.REPO / 'scripts/check_real_clips.py'), run_name='import_only')
    assert callable(loaded['main'])


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    root = tmp_path / 'source'; root.mkdir()
    video = make_video(root / 'tiny.mp4', cfr_times(20, 10), gop=8)
    cache = PerceptionCache(tmp_path / 'synthetic-cache')
    outcome = AnalysisRun(video, detector=FakeDetector(), tracker=FakeTracker(), analyzers=[FakeAnalyzer()],
                          cache=cache, guard=calm()).run()
    assert outcome.state is RunState.COMPLETED
    m = tmp_path / 'manifest.json'
    m.write_text(json.dumps(dict(clips=[dict(id='unseen-source', filename=video.name, bytes=video.stat().st_size,
                                           sha256=hashlib.sha256(video.read_bytes()).hexdigest(), role='test',
                                           licence='synthetic', assumed_fps=17)])))
    def guard(**kw):
        assert kw['policy'] == 'strict'
        return SimpleNamespace(check=lambda **kwargs: None)
    monkeypatch.setattr(check, 'ResourceGuard', guard)
    monkeypatch.setattr(check, 'memory_pressure_level', lambda: 1)
    monkeypatch.setattr(check, 'default_specs', lambda _: (DetectorSpec('unused', 'a'*64), SegmenterSpec('unused', 'b'*64)))
    jobs = []
    def worker(job, **kw):
        jobs.append(job)
        assert job.cache_only and job.policy == 'strict'
        assert job.config['assumed_fps'] == 17
        return replace(outcome, cache_hit=True)
    monkeypatch.setattr(check, 'run_in_worker', worker)
    return root, m, video, jobs


def test_synthetic_pipeline_writes_verified_records(tmp_path, synthetic):
    root, manifest, video, jobs = synthetic
    before = video.read_bytes()
    output = tmp_path / 'result'
    assert check.main(['--manifest', str(manifest), '--root', str(root), '--clips', 'unseen',
                       '--cache-only', '--out', str(output)]) == 0
    record = json.loads((output / '000.json').read_text())
    assert record['status'] == 'passed' and record['cache_hit']
    assert record['packages'] == record['overlays_checked'] == 1
    assert record['media'][0]['clip']['frames'] > 0
    assert len(jobs) == 1 and video.read_bytes() == before


def test_failed_worker_stops_and_returns_nonzero(tmp_path, synthetic, monkeypatch):
    root, manifest, _, jobs = synthetic
    monkeypatch.setattr(check, 'run_in_worker', lambda *a, **k: RunOutcome(RunState.FAILED, error='cache miss', stop=StopReason.ERROR))
    output = tmp_path / 'result'
    assert check.main(['--manifest', str(manifest), '--root', str(root), '--cache-only', '--out', str(output)]) == 1
    assert 'cache miss' in json.loads((output / 'failure.json').read_text())['error']


def test_existing_output_is_not_modified(tmp_path, synthetic):
    root, manifest, _, jobs = synthetic
    output = tmp_path / 'existing'; output.mkdir()
    sentinel = output / 'keep'; sentinel.write_text('unchanged')
    assert check.main(['--manifest', str(manifest), '--root', str(root), '--cache-only', '--out', str(output)]) == 1
    assert list(output.iterdir()) == [sentinel] and sentinel.read_text() == 'unchanged'
    assert not jobs


def test_no_proposals_still_checks_media_without_inventing_evidence(tmp_path, synthetic, monkeypatch):
    root, manifest, video, _ = synthetic
    outcome = AnalysisRun(video, cache=PerceptionCache(tmp_path / 'no-events-cache'), guard=calm()).run()
    monkeypatch.setattr(check, 'run_in_worker', lambda *a, **kw: replace(outcome, cache_hit=True))
    output = tmp_path / 'no-events-result'
    assert check.main(['--manifest', str(manifest), '--root', str(root), '--cache-only', '--out', str(output)]) == 0
    record = json.loads((output / '000.json').read_text())
    assert record['no_proposals'] and record['overlays_checked'] == 0
    assert len(record['media']) == 1 and record['media'][0]['clip']['frames'] > 0


def test_clip_selection_rejects_ambiguity_and_duplicates():
    from evaluation.datasets import Clip, ClipMetadata
    def c(name):
        return Clip(Path('unused'), ClipMetadata(id=name, role='test', licence='test', bytes=1))
    available = [c('one-a'), c('one-b')]
    with pytest.raises(ValueError, match='matches 2'):
        check._select(available, ['one'])
    with pytest.raises(ValueError, match='duplicate'):
        check._select(available, ['one-a', 'one-a'])


def test_worker_resource_refusal_is_recorded_and_not_retried(tmp_path, synthetic, monkeypatch):
    root, manifest, _, _ = synthetic
    calls = []
    def worker(*a, **kw):
        calls.append(1)
        return RunOutcome(RunState.FAILED, error='strict guard refused', stop=StopReason.RESOURCES)
    monkeypatch.setattr(check, 'run_in_worker', worker)
    output = tmp_path / 'resource-stop'
    assert check.main(['--manifest', str(manifest), '--root', str(root), '--cache-only', '--out', str(output)]) == 3
    assert len(calls) == 1
    assert json.loads((output / 'failure.json').read_text())['clip_id'] == 'unseen-source'
