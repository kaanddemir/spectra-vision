import json

import pytest

from spectra.engine.__main__ import main
from tests.engine.media import make_video


def test_probe_and_model_free_analysis(tmp_path, capsys):
    video = make_video(tmp_path / "tiny.mp4", [0.0, 0.1, 0.2])
    assert main(["probe", str(video)]) == 0
    assert json.loads(capsys.readouterr().out)["width"] > 0
    assert main(["analyze", str(video), "--cache", str(tmp_path / "cache")]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "completed"


def test_tracking_needs_a_detector(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["analyze", "unused", "--cache", str(tmp_path), "--track"])
    assert exc.value.code == 2


def test_registry_shortcut_and_explicit_model_flags(tmp_path, monkeypatch, capsys):
    from dataclasses import asdict
    import spectra.engine.__main__ as cli
    from spectra.engine.runner import RunOutcome, RunState, StopReason
    from spectra.engine.worker import DetectorSpec, SegmenterSpec
    import spectra.engine.models as models
    detector = DetectorSpec('registry-detector.onnx', 'a' * 64, input_size=320)
    segmenter = SegmenterSpec('registry-segmenter.onnx', 'b' * 64)
    monkeypatch.setattr(models, 'default_specs', lambda directory: (detector, segmenter))
    jobs = []
    def run(job, **kwargs):
        jobs.append(job)
        return RunOutcome(RunState.FAILED, error="synthetic dispatch sentinel", stop=StopReason.ERROR)
    monkeypatch.setattr(cli, 'run_in_worker', run)
    args = ['analyze', 'unused', '--cache', str(tmp_path), '--models', 'registry', '--track', '--lanes', '--events']
    assert main(args) == 1
    assert asdict(jobs[-1].detector) == asdict(detector)
    assert asdict(jobs[-1].segmenter) == asdict(segmenter)
    assert main(args + ['--detector', 'explicit.onnx', '--detector-sha256', 'c' * 64]) == 1
    assert jobs[-1].detector.model_path == 'explicit.onnx'
    assert jobs[-1].detector.sha256 == 'c' * 64
    assert main(args + ['--segmenter', 'explicit-seg.onnx', '--segmenter-sha256', 'd' * 64]) == 1
    assert jobs[-1].segmenter.model_path == 'explicit-seg.onnx'
    with pytest.raises(SystemExit) as exc:
        main(args + ['--detector', 'explicit.onnx'])
    assert exc.value.code == 2


def test_registry_problem_is_a_cli_error(tmp_path, monkeypatch):
    import spectra.engine.models as models
    def invalid(directory):
        raise ValueError('sha256 mismatch')
    monkeypatch.setattr(models, 'default_specs', invalid)
    with pytest.raises(SystemExit) as exc:
        main(['analyze', 'unused', '--cache', str(tmp_path), '--models', 'bad'])
    assert exc.value.code == 2
