"""Fixed pre-refactor identities and additive worker configuration contracts."""
import json
from pathlib import Path

import pytest

from spectra.engine.cache import perception_key
from spectra.engine.contracts import ComponentInfo, Provenance, RunConfig, VideoInfo, analysis_key
from spectra.engine.worker import WorkerJob

FIXTURES = Path(__file__).parent / "fixtures"


def test_fixed_fake_component_keys():
    from tests.engine.fakes import FakeDetector, FakeTracker, FakeSegmenter, FakeEgoLane
    f = json.loads((FIXTURES / "keys_before_refactor.json").read_text())
    provenance = Provenance(**f["provenance"])
    assert tuple(c().info for c in (FakeDetector, FakeTracker, FakeSegmenter, FakeEgoLane)) == provenance.components
    key = perception_key(VideoInfo(**f["video"]), RunConfig(**f["config"]), provenance)
    assert key == f["perception_key"]
    assert analysis_key(key, [ComponentInfo(**a) for a in f["analyzers"]]) == f["analysis_key"]


@pytest.mark.parametrize("lanes", [False, True])
def test_original_worker_analyzers_and_keys_are_byte_identical(lanes):
    from spectra.engine.worker import build_analyzers
    f = json.loads((FIXTURES / "analyzers_before_refactor.json").read_text())[str(lanes).lower()]
    key = json.loads((FIXTURES / "keys_before_refactor.json").read_text())["perception_key"]
    analyzers = build_analyzers(WorkerJob("unused", "unused", events=True, lanes=lanes))
    assert [a.info.model_dump(mode="json") for a in analyzers] == f["analyzers"]
    assert analysis_key(key, [a.info for a in analyzers]) == f["analysis_key"]


def test_explicit_analyzers_and_nested_approach_configuration():
    from spectra.engine.worker import AnalyzerSpec, build_analyzers
    specs = (AnalyzerSpec("apparent-approach", {"tau_s": 4.0}),
             AnalyzerSpec("camera-vehicle-impact", {"approach": {"tau_s": 4.0}}))
    a, impact = build_analyzers(WorkerJob("unused", "unused", analyzers=specs))
    assert a.info.params["tau_s"] == 4.0
    assert impact.info.params["approach"]["params"] == a.info.params
    assert build_analyzers(WorkerJob("unused", "unused", events=True, analyzers=())) == []
    assert build_analyzers(WorkerJob("unused", "unused")) == []


@pytest.mark.parametrize("name,params", [("unknown", {}), ("apparent-approach", {"bad_key": 1})])
def test_invalid_analyzer_specs_are_rejected(name, params):
    from spectra.engine.worker import AnalyzerSpec, build_analyzers
    with pytest.raises((ValueError, TypeError)):
        build_analyzers(WorkerJob("unused", "unused", analyzers=(AnalyzerSpec(name, params),)))


def test_worker_uses_explicit_analyzer_specs_without_loading_models(tmp_path):
    from spectra.engine.runner import RunState
    from spectra.engine.worker import AnalyzerSpec, run_in_worker
    from tests.engine.media import make_video
    video = make_video(tmp_path / "tiny.mp4", [0.0, 0.1, 0.2])
    job = WorkerJob(str(video), str(tmp_path / "cache"),
                    analyzers=(AnalyzerSpec("apparent-approach", {"tau_s": 4.0}),))
    outcome = run_in_worker(job, timeout_s=20)
    assert outcome.state is RunState.COMPLETED, outcome.error
    assert len(outcome.result.reports) == 1
    assert outcome.result.reports[0].analyzer.params["tau_s"] == 4.0
    # No tracking input is available; this must not manufacture proposals.
    assert outcome.result.reports[0].status.value == "unavailable"
    assert outcome.result.events == ()
