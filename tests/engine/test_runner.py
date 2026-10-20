import pytest

from spectra.engine.adapters import AnalyzerUnavailable, ComponentFailure
from spectra.engine.cache import STAGING, PerceptionCache
from spectra.engine.contracts import AnalyzerStatus, OutputStatus, ReasonCode, Role, RunConfig
from spectra.engine.resources import ResourceGuard
from spectra.engine.runner import AnalysisRun, CancelToken, RunOutcome, RunState, StopReason
from tests.engine.fakes import FakeAnalyzer, FakeDetector, FakeEgoLane, FakeSegmenter, FakeTracker
from tests.engine.media import H, W, cfr_times, make_video

from tests.engine.media import VFR
from tests.engine.fakes import calm




@pytest.fixture
def clip(tmp_path):
    return make_video(tmp_path / "vfr.mp4", VFR)


@pytest.fixture
def cache(tmp_path):
    return PerceptionCache(tmp_path / "cache")


def full_run(clip, cache, config=RunConfig(), **kw):
    parts = dict(detector=FakeDetector(), tracker=FakeTracker(), segmenter=FakeSegmenter(), ego_lane=FakeEgoLane(),
                 analyzers=[FakeAnalyzer()])
    parts.update(kw)
    run = AnalysisRun(clip, config, cache=cache, guard=parts.pop("guard", calm()), **parts)
    return run, parts


def test_complete_run_produces_validated_cached_results(clip, cache):
    run, parts = full_run(clip, cache)
    outcome = run.run()
    assert outcome.state is RunState.COMPLETED and outcome.error is None and not outcome.cache_hit
    p = outcome.result.perception
    assert [f.t for f in p.frames] == pytest.approx(VFR, abs=1e-3)
    assert all(set(f.status.values()) == {OutputStatus.OK} for f in p.frames)
    assert p.frames[3].detections[0].box.x1 == pytest.approx(3 * W / 20)
    assert p.frames[3].tracks[0].track_id == 0
    assert p.frames[2].segmentation.path == "masks/000002.png"
    assert p.frames[1].ego_lane.offset is not None
    assert p.timing.variable_frame_rate is True
    assert [w.code.value for w in p.warnings] == ["frame_gap"] and "no frames for 1.05 s" in p.warnings[0].message
    r = outcome.result
    assert [e.analyzer for e in r.events] == ["fake-event_analyzer"]
    assert [(x.status.value, x.considered, x.proposals, x.rejected) for x in r.reports] == [
        ("ok", 2, 1, {ReasonCode.OTHER_LANE: 1})]
    assert all(c.loaded == 1 and c.closed == 1 for c in parts.values() if hasattr(c, "loaded"))
    assert parts["analyzers"][0].closed == 1
    assert cache.lookup(outcome.result.perception_key).result == p
    assert not any((cache.root / STAGING).iterdir())


def test_replay_from_cache_loads_no_perception_model(clip, cache):
    first = full_run(clip, cache)[0].run()
    run, parts = full_run(clip, cache)
    outcome = run.run()
    assert outcome.cache_hit and outcome.result == first.result
    assert all(parts[k].loaded == 0 for k in ("detector", "tracker", "segmenter", "ego_lane"))
    assert parts["analyzers"][0].loaded == 1  # event analysis re-runs on cached perception


def test_changed_component_configuration_is_a_miss(clip, cache):
    full_run(clip, cache)[0].run()
    run, _ = full_run(clip, cache, detector=FakeDetector(threshold=0.4))
    assert not run.run().cache_hit


def test_corrupt_cache_entry_is_recomputed_with_a_note(clip, cache):
    first = full_run(clip, cache)[0].run()
    entry = cache.entry(first.result.perception_key)
    (entry / "perception.json").write_text("{}")
    outcome = full_run(clip, cache)[0].run()
    assert not outcome.cache_hit and outcome.result == first.result
    assert any("cache entry rejected" in n for n in outcome.notes)


def test_runs_are_deterministic(clip, tmp_path):
    a, b = PerceptionCache(tmp_path / "a"), PerceptionCache(tmp_path / "b")
    ka = full_run(clip, a)[0].run().result.perception_key
    kb = full_run(clip, b)[0].run().result.perception_key
    assert ka == kb
    assert (a.entry(ka) / "perception.json").read_bytes() == (b.entry(kb) / "perception.json").read_bytes()


def test_components_run_on_their_own_schedules(clip, cache):
    run, parts = full_run(clip, cache, segmenter=FakeSegmenter(interval_s=0.5))
    p = run.run().result.perception
    seg = [f.t for f in p.frames if f.status[Role.SEGMENTER] is OutputStatus.OK]
    assert seg == pytest.approx([0.0, 0.5, 1.6, 2.0], abs=1e-3)
    lane = [f.status[Role.EGO_LANE] for f in p.frames]
    assert lane == [f.status[Role.SEGMENTER] for f in p.frames]
    assert all(f.status[Role.DETECTOR] is OutputStatus.OK for f in p.frames)


def test_sampling_interval_and_downscaling(clip, cache):
    run, parts = full_run(clip, cache, RunConfig(interval_s=0.25, max_width=64))
    p = run.run().result.perception
    assert [f.t for f in p.frames] == pytest.approx([0.0, 0.5, 1.6, 1.8, 2.0], abs=1e-3)
    box = p.frames[0].detections[0].box
    assert (box.x2 - box.x1) == pytest.approx(W / 4) and box.y2 == pytest.approx(H * 0.75)  # source pixels


def test_progress_is_monotonic_and_ends_in_the_terminal_state(clip, cache):
    seen = []
    run, _ = full_run(clip, cache)
    run.on_progress = seen.append
    run.run()
    assert seen[0].state is RunState.RUNNING and seen[-1].state is RunState.COMPLETED
    fractions = [p.fraction for p in seen]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    counts = [p.frames_analysed for p in seen if p.stage == "analysing frames"]
    assert counts == list(range(1, len(VFR) + 1))


def test_cancellation_stops_cleanly_without_a_result_or_cache_entry(clip, cache):
    token = CancelToken()
    run, parts = full_run(clip, cache)
    run.cancel = token
    run.on_progress = lambda p: token.cancel() if p.frames_analysed == 3 else None
    outcome = run.run()
    assert outcome.state is RunState.CANCELLED and outcome.result is None
    assert len(parts["detector"].calls) == 3
    assert all(parts[k].closed == 1 for k in ("detector", "tracker", "segmenter", "ego_lane"))
    assert parts["analyzers"][0].loaded == 0
    assert not [p for p in cache.root.iterdir() if p.name != STAGING]
    assert not any((cache.root / STAGING).iterdir())


def test_cancellation_before_start_loads_nothing(clip, cache):
    token = CancelToken()
    token.cancel()
    run, parts = full_run(clip, cache)
    run.cancel = token
    assert run.run().state is RunState.CANCELLED
    assert parts["detector"].loaded == 0


def test_recoverable_frame_failure_is_recorded_and_the_run_continues(clip, cache):
    run, _ = full_run(clip, cache, detector=FakeDetector(fail_on={2}))
    outcome = run.run()
    assert outcome.state is RunState.COMPLETED
    f = outcome.result.perception.frames[2]
    assert f.status[Role.DETECTOR] is OutputStatus.FAILED and "frame 2" in f.errors[Role.DETECTOR]
    assert f.detections is None and f.status[Role.TRACKER] is OutputStatus.INPUT_UNAVAILABLE
    assert outcome.result.perception.frames[3].status[Role.DETECTOR] is OutputStatus.OK


def test_persistent_failure_fails_the_run(clip, cache):
    run, parts = full_run(clip, cache, detector=FakeDetector(fail_on=set(range(10))))
    run.max_consecutive_failures = 3
    outcome = run.run()
    assert outcome.state is RunState.FAILED and "4 consecutive frames" in outcome.error
    assert outcome.result is None and parts["detector"].closed == 1
    assert not any((cache.root / STAGING).iterdir())


def test_unexpected_adapter_error_fails_the_run(clip, cache):
    det = FakeDetector()
    det.detect = lambda frame: 1 / 0
    outcome = full_run(clip, cache, detector=det)[0].run()
    assert outcome.state is RunState.FAILED and outcome.error.startswith("ZeroDivisionError")
    assert outcome.stop is StopReason.ERROR and det.closed == 1


def test_invalid_adapter_output_is_a_frame_failure(clip, cache):
    det = FakeDetector()
    det.detect = lambda frame: [("not", "a", "detection")]
    outcome = full_run(clip, cache, detector=det)[0].run()
    frames = outcome.result.perception.frames
    assert all(f.status[Role.DETECTOR] is OutputStatus.FAILED for f in frames[:5])
    assert "invalid output" in frames[0].errors[Role.DETECTOR]


def test_load_failure_fails_and_still_closes(clip, cache):
    seg = FakeSegmenter()
    seg.load = lambda: (_ for _ in ()).throw(RuntimeError("weights missing"))
    run, parts = full_run(clip, cache, segmenter=seg)
    outcome = run.run()
    assert outcome.state is RunState.FAILED and "weights missing" in outcome.error
    assert seg.closed == 1 and parts["detector"].closed == 1 and parts["ego_lane"].loaded == 0


def test_close_errors_are_reported_without_changing_the_outcome(clip, cache):
    det = FakeDetector()
    det.close = lambda: 1 / 0
    outcome = full_run(clip, cache, detector=det)[0].run()
    assert outcome.state is RunState.COMPLETED and any("closing fake-detector failed" in n for n in outcome.notes)


def test_analyser_failure_is_reported_and_other_output_kept(clip, cache):
    bad = FakeAnalyzer(name="bad")
    bad.analyze = lambda frames, w, h: (_ for _ in ()).throw(ComponentFailure("not enough tracks"))
    missing = FakeAnalyzer(name="missing")
    missing.analyze = lambda frames, w, h: (_ for _ in ()).throw(AnalyzerUnavailable("no lane relations"))
    outcome = full_run(clip, cache, analyzers=[bad, missing, FakeAnalyzer()])[0].run()
    r = outcome.result
    assert outcome.state is RunState.COMPLETED and len(r.events) == 1
    assert [(x.analyzer.name, x.status, x.reason) for x in r.reports] == [
        ("bad", AnalyzerStatus.FAILED, "not enough tracks"), ("missing", AnalyzerStatus.UNAVAILABLE, "no lane relations"),
        ("fake-event_analyzer", AnalyzerStatus.OK, None)]
    assert not any("analyser" in n for n in outcome.notes)  # in the result, not in run notes


def test_analyser_returning_an_invalid_output_fails_the_run(clip, cache):
    bad = FakeAnalyzer(name="bad")
    bad.analyze = lambda frames, w, h: []
    outcome = full_run(clip, cache, analyzers=[bad])[0].run()
    assert outcome.state is RunState.FAILED and "invalid output" in outcome.error


def test_event_ids_are_stable_across_replays_and_change_with_the_analysis(clip, cache):
    first = full_run(clip, cache)[0].run().result
    again = full_run(clip, cache)[0].run().result
    other = full_run(clip, cache, analyzers=[FakeAnalyzer(name="other")])[0].run().result
    assert [e.event_id for e in first.events] == [e.event_id for e in again.events]
    assert first.analysis_key == again.analysis_key != other.analysis_key


def test_unsupported_video_fails_before_loading_anything(tmp_path, cache):
    bad = tmp_path / "bad.mp4"
    bad.write_text("nope")
    run, parts = full_run(bad, cache)
    outcome = run.run()
    assert outcome.state is RunState.FAILED and "bad.mp4" in outcome.error
    assert outcome.stop is StopReason.VIDEO and parts["detector"].loaded == 0


def test_empty_span_fails(clip, cache):
    outcome = full_run(clip, cache, RunConfig(start_s=5.0))[0].run()
    assert outcome.state is RunState.FAILED and "no frames" in outcome.error


def test_strict_policy_refuses_to_start_above_normal_pressure(clip, cache):
    guard = ResourceGuard(policy="strict", pressure=lambda: 2, rss=lambda: 0.0)
    run, parts = full_run(clip, cache, guard=guard)
    outcome = run.run()
    assert outcome.state is RunState.FAILED and "memory pressure level 2" in outcome.error
    assert outcome.stop is StopReason.RESOURCES and parts["detector"].loaded == 0


def test_standard_policy_continues_at_warning_level_and_says_so(clip, cache):
    outcome = full_run(clip, cache, guard=ResourceGuard(pressure=lambda: 2, rss=lambda: 0.0))[0].run()
    assert outcome.state is RunState.COMPLETED
    assert any("pressure reached level 2" in n for n in outcome.notes)


def test_critical_pressure_refuses_to_start_under_any_policy(clip, cache):
    run, parts = full_run(clip, cache, guard=ResourceGuard(pressure=lambda: 4, rss=lambda: 0.0))
    assert run.run().state is RunState.FAILED and parts["detector"].loaded == 0


def test_critical_pressure_during_a_run_stops_it(clip, cache):
    levels = iter([1, 1, 1, 4])
    guard = ResourceGuard(pressure=lambda: next(levels, 4), rss=lambda: 0.0, check_every_s=0)
    run, parts = full_run(clip, cache, guard=guard)
    outcome = run.run()
    assert outcome.state is RunState.FAILED and "level 4" in outcome.error
    assert parts["detector"].closed == 1 and not any((cache.root / STAGING).iterdir())


def test_process_memory_budget_stops_a_run(clip, cache):
    rss = iter([100.0, 100.0, 5000.0])
    guard = ResourceGuard(pressure=lambda: 1, rss=lambda: next(rss, 5000.0), rss_budget_mb=1000, check_every_s=0)
    outcome = full_run(clip, cache, guard=guard)[0].run()
    assert outcome.state is RunState.FAILED and "exceeds budget" in outcome.error


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        ResourceGuard(policy="lenient")


def test_a_run_starts_once(clip, cache):
    run, _ = full_run(clip, cache)
    run.run()
    with pytest.raises(RuntimeError):
        run.run()


def test_component_wiring_is_validated(clip, cache):
    with pytest.raises(ValueError, match="needs a detector"):
        AnalysisRun(clip, tracker=FakeTracker())
    with pytest.raises(ValueError, match="pass a cache"):
        AnalysisRun(clip, segmenter=FakeSegmenter())
    with pytest.raises(ValueError, match="expected detector"):
        AnalysisRun(clip, detector=FakeTracker())


def test_decode_only_run_needs_no_components(clip, cache):
    outcome = AnalysisRun(clip, RunConfig(interval_s=0.5), cache=cache, guard=calm()).run()
    assert outcome.state is RunState.COMPLETED
    assert [f.status for f in outcome.result.perception.frames] == [{}] * 4


def test_only_completed_outcomes_carry_results():
    with pytest.raises(ValueError):
        RunOutcome(RunState.CANCELLED, result=object())
    with pytest.raises(ValueError):
        RunOutcome(RunState.RUNNING)
    with pytest.raises(ValueError):
        RunOutcome(RunState.FAILED, error="no reason")
    with pytest.raises(ValueError):
        RunOutcome(RunState.CANCELLED, stop=StopReason.RESOURCES)


def test_long_input_keeps_no_frames_in_memory(tmp_path, cache):
    import tracemalloc
    path = make_video(tmp_path / "long.mp4", cfr_times(600, 30), gop=30)
    tracemalloc.start()
    outcome = AnalysisRun(path, cache=cache, detector=FakeDetector(), guard=calm()).run()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert outcome.state is RunState.COMPLETED and len(outcome.result.perception.frames) == 600
    assert peak < 6 * 2**20  # retaining the 600 decoded frames would take 14 MB on its own


class MotionTracker(FakeTracker):
    """Reports a fixed image motion from the second update on."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._last = None

    def update(self, frame, detections):
        self._motion = None
        if self._last is not None:
            from spectra.engine.contracts import CameraMotion
            self._motion = CameraMotion(since_t=self._last, scale=1.02, rotation=0.0, tx=-4.0, ty=0.0, matches=50,
                                        inliers=40, residual_px=0.5)
        self._last = frame.t
        return super().update(frame, detections)

    def camera_motion(self):
        return self._motion


def test_camera_motion_from_the_tracker_is_stored_with_each_frame(clip, cache):
    outcome = full_run(clip, cache, tracker=MotionTracker())[0].run()
    frames = list(outcome.result.perception.frames)
    assert frames[0].camera_motion is None
    assert all(f.camera_motion.since_t == p.t for p, f in zip(frames, frames[1:]))
    plain = full_run(clip, PerceptionCache(cache.root / "other"))[0].run()
    assert all(f.camera_motion is None for f in plain.result.perception.frames)  # tracker without motion


def test_changing_event_analysers_reuses_cached_perception(clip, cache):
    from spectra.engine.events import ApproachAnalyzer
    first = full_run(clip, cache, analyzers=[ApproachAnalyzer()])[0].run()
    run, parts = full_run(clip, cache, analyzers=[ApproachAnalyzer(tau_s=2.0)])
    second = run.run()
    assert first.state is second.state is RunState.COMPLETED and second.cache_hit
    assert parts["detector"].loaded == 0 and parts["segmenter"].loaded == 0
    assert first.result.perception_key == second.result.perception_key
    assert first.result.analysis_key != second.result.analysis_key  # review decisions bind to the analysers
