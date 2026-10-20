import json

import pytest
from pydantic import ValidationError

from spectra.engine.contracts import (SCHEMA_VERSION, AnalysisResult, AnalyzerReport, AnalyzerStatus, Basis, Box,
                                      CameraMotion, ComponentInfo, Detection, EgoLaneObservation, EventDraft,
                                      EventProposal, EventSubtype, EventType, EvidenceItem, EvidenceState,
                                      FrameRecord, LaneBoundary, LaneChange, LaneReason, ObjectCategory,
                                      OutputStatus, PerceptionResult, Provenance, ReasonCode, ReviewDecision, Role,
                                      RunConfig, RunWarning, SegmentationRef, TimestampSource, TimingReport,
                                      WarningCode, analysis_key, canonical_json, event_id)

from tests.engine.fakes import SHA, video
DET = ComponentInfo(role=Role.DETECTOR, name="det", version="1")
ANALYZER = ComponentInfo(role=Role.EVENT_ANALYZER, name="approach", version="1")




def frame(i, t, detections=()):
    return FrameRecord(index=i, decode_index=i, t=t, pts=int(t * 1000), status={Role.DETECTOR: OutputStatus.OK},
                       detections=tuple(detections))


def perception(frames):
    return PerceptionResult(video=video(), config=RunConfig(interval_s=0.2),
                            provenance=Provenance(engine_version="0", decoder={"pyav": "x"}, components=(DET,)),
                            timing=TimingReport(frames_decoded=len(frames), frames_analysed=len(frames)),
                            frames=tuple(frames))


def car(x1=10.0):
    return Detection(box=Box(x1=x1, y1=20, x2=x1 + 50, y2=60), score=0.8, category=ObjectCategory.CAR)


def draft(**kw):
    base = dict(type=EventType.CUT_IN, state=EvidenceState.SUPPORTED, t_start=0.0, t_peak=0.2, t_end=0.4,
                track_ids=(3,), analyzer="approach")
    return EventDraft(**{**base, **kw})


def report(proposals=1, **kw):
    return AnalyzerReport(**{**dict(analyzer=ANALYZER, status=AnalyzerStatus.OK, considered=proposals,
                                    proposals=proposals), **kw})


def result(p, drafts=(), reports=None):
    reports = (report(len(drafts)),) if reports is None else reports
    key = analysis_key(SHA, [r.analyzer for r in reports])
    return AnalysisResult(analysis_key=key, perception_key=SHA, perception=p, reports=reports,
                          events=tuple(d.bind(key) for d in drafts))


def test_result_round_trip_is_lossless_and_deterministic():
    p = perception([frame(0, 0.0, [car()]), frame(1, 0.2, []), frame(2, 0.4, [car(30)])])
    ev = draft(evidence=(EvidenceItem(basis=Basis.ESTIMATED, name="apparent_tau_s", t=0.2, track_id=3, value=1.5,
                                      unit="s"),
                         EvidenceItem(basis=Basis.OBSERVED, name="lane_relation", t=0.0, t_end=0.2,
                                      value="left_adjacent")))
    r = result(p, [ev])
    text = canonical_json(r)
    back = AnalysisResult.model_validate_json(text)
    assert back == r
    assert canonical_json(back) == text
    assert json.loads(text)["schema_version"] == SCHEMA_VERSION == "1.0"
    assert back.events[0].evidence[1].value == "left_adjacent" and back.events[0].evidence[0].value == 1.5


def test_unknown_fields_and_wrong_schema_version_are_rejected():
    data = json.loads(canonical_json(perception([frame(0, 0.0)])))
    with pytest.raises(ValidationError):
        PerceptionResult.model_validate({**data, "extra": 1})
    with pytest.raises(ValidationError, match="unsupported schema_version"):
        PerceptionResult.model_validate({**data, "schema_version": "0.9"})


def test_frame_times_must_increase_strictly():
    with pytest.raises(ValidationError, match="strictly increasing"):
        perception([frame(0, 0.2), frame(1, 0.2)])
    with pytest.raises(ValidationError, match="0..n-1"):
        perception([frame(1, 0.0)])


def test_non_finite_values_are_rejected():
    with pytest.raises(ValidationError):
        Box(x1=0, y1=0, x2=float("nan"), y2=1)
    with pytest.raises(ValidationError):
        LaneBoundary(near=(0.1, float("inf")), far=(0.2, 0.5))


def test_box_coordinates_and_normalisation():
    with pytest.raises(ValidationError, match="x2 > x1"):
        Box(x1=5, y1=0, x2=5, y2=1)
    assert Box(x1=0, y1=0, x2=1280, y2=720).normalized(1280, 720) == (0, 0, 1, 1)


def test_display_size_must_match_rotation():
    assert video(rotation=90, width=720, height=1280).rotation == 90
    with pytest.raises(ValidationError, match="after rotation"):
        video(rotation=90)


def test_assumed_rate_requires_an_explicit_rate():
    with pytest.raises(ValidationError):
        video(timestamp_source=TimestampSource.ASSUMED_RATE)
    with pytest.raises(ValidationError):
        video(assumed_fps=20.0)
    assert video(timestamp_source=TimestampSource.ASSUMED_RATE, assumed_fps=20.0).assumed_fps == 20.0


def test_missing_output_is_distinct_from_an_empty_detection():
    empty = frame(0, 0.0, [])
    assert empty.detections == () and empty.status[Role.DETECTOR] is OutputStatus.OK
    skipped = FrameRecord(index=0, t=0.0, status={Role.DETECTOR: OutputStatus.NOT_SCHEDULED})
    assert skipped.detections is None
    with pytest.raises(ValidationError, match="exactly when"):
        FrameRecord(index=0, t=0.0, status={Role.DETECTOR: OutputStatus.FAILED}, detections=())
    with pytest.raises(ValidationError, match="exactly when"):
        FrameRecord(index=0, t=0.0, status={Role.DETECTOR: OutputStatus.OK})
    with pytest.raises(ValidationError, match="without failed status"):
        FrameRecord(index=0, t=0.0, status={Role.DETECTOR: OutputStatus.NOT_SCHEDULED},
                    errors={Role.DETECTOR: "x"})


def test_frame_status_must_cover_exactly_the_configured_roles():
    stray = FrameRecord(index=0, t=0.0, status={Role.DETECTOR: OutputStatus.NOT_SCHEDULED,
                                                Role.SEGMENTER: OutputStatus.NOT_SCHEDULED})
    with pytest.raises(ValidationError, match="configured components"):
        perception([stray])


def test_ego_lane_offset_requires_both_boundaries():
    b = LaneBoundary(near=(0.3, 0.9), far=(0.45, 0.6))
    assert EgoLaneObservation(left=b).offset is None
    with pytest.raises(ValidationError, match="both boundaries"):
        EgoLaneObservation(left=b, offset=0.1)


def test_proposal_shape_rules():
    with pytest.raises(ValidationError, match="t_start <= t_peak <= t_end"):
        draft(t_peak=0.5)
    with pytest.raises(ValidationError):
        draft(rank=1.5)
    with pytest.raises(ValidationError, match="subtype"):
        draft(type=EventType.IMPACT)
    with pytest.raises(ValidationError, match="subtype"):
        draft(subtype=EventSubtype.CAMERA_VEHICLE)
    assert draft(type=EventType.IMPACT, subtype=EventSubtype.CAMERA_VEHICLE).subtype is EventSubtype.CAMERA_VEHICLE
    with pytest.raises(ValidationError, match="sorted and unique"):
        draft(track_ids=(4, 3))


def test_uncertain_proposals_must_give_reasons_and_only_they():
    with pytest.raises(ValidationError, match="reasons"):
        draft(state=EvidenceState.UNCERTAIN)
    with pytest.raises(ValidationError, match="reasons"):
        draft(reasons=(ReasonCode.IDENTITY_JUMP,))
    d = draft(state=EvidenceState.UNCERTAIN, reasons=(ReasonCode.BOUNDARIES_REIDENTIFIED,))
    assert d.reasons == (ReasonCode.BOUNDARIES_REIDENTIFIED,)


def test_event_ids_are_bound_to_the_analysis_and_stable():
    d = draft()
    a, b = d.bind("1" * 64), d.bind("2" * 64)
    assert a.event_id == d.bind("1" * 64).event_id != b.event_id and len(a.event_id) == 16
    assert draft(t_start=0.001).bind("1" * 64).event_id == a.event_id  # rounded to 0.01 s
    assert draft(track_ids=(4,)).bind("1" * 64).event_id != a.event_id
    assert event_id("1" * 64, a) == a.event_id


def test_events_must_lie_in_span_and_match_their_reports():
    p = perception([frame(0, 0.0), frame(1, 0.4)])
    with pytest.raises(ValidationError, match="outside the analysed span"):
        result(p, [draft(t_end=0.6)])
    with pytest.raises(ValidationError, match="without a completed report"):
        result(p, [draft()], reports=())
    with pytest.raises(ValidationError, match="counts 2 proposals"):
        result(p, [draft()], reports=(report(2),))
    r = result(p, [draft()])
    with pytest.raises(ValidationError, match="analysis_key"):
        AnalysisResult(analysis_key="0" * 64, perception_key=SHA, perception=p, reports=r.reports, events=r.events)
    forged = EventProposal(**{**dict(r.events[0]), "event_id": "0" * 16})
    with pytest.raises(ValidationError, match="not bound"):
        AnalysisResult(analysis_key=r.analysis_key, perception_key=SHA, perception=p, reports=r.reports,
                       events=(forged,))


def test_analyser_reports_reconcile_and_failures_carry_a_reason():
    ok = report(2, considered=5, rejected={ReasonCode.OTHER_LANE: 2, ReasonCode.MERGED: 1})
    assert ok.considered == 5
    with pytest.raises(ValidationError, match="considered"):
        report(2, considered=4, rejected={ReasonCode.OTHER_LANE: 1})
    with pytest.raises(ValidationError, match="reason is required"):
        report(0, status=AnalyzerStatus.FAILED)
    with pytest.raises(ValidationError, match="no counts"):
        report(1, status=AnalyzerStatus.UNAVAILABLE, reason="no lane relations")
    failed = report(0, status=AnalyzerStatus.FAILED, reason="boom")
    assert result(perception([frame(0, 0.0)]), reports=(failed,)).reports[0].status is AnalyzerStatus.FAILED
    with pytest.raises(ValidationError, match="event analysers"):
        report(0, analyzer=DET)


def test_evidence_shapes():
    assert EvidenceItem(basis=Basis.OBSERVED, name="x", t=1.0, t_end=2.0).t_end == 2.0
    for bad in (dict(t_end=0.5), dict(value="ego_lane", unit="s"), dict(basis="measurement"), dict(name="")):
        with pytest.raises(ValidationError):
            EvidenceItem(**{**dict(basis=Basis.OBSERVED, name="x", t=1.0), **bad})


def test_review_decisions_are_separate_from_results_and_carry_the_proposal_interval():
    r = result(perception([frame(0, 0.0), frame(1, 0.4)]), [draft()])
    data = json.loads(canonical_json(r))
    data["events"][0]["verdict"] = "confirmed"
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(data)
    d = ReviewDecision.of(r.events[0], r.analysis_key, verdict="confirmed", reviewer="kd",
                          decided_at="2026-01-01T00:00:00Z")
    assert (d.verdict.value, d.type, d.t_start, d.t_end, d.event_id) == (
        "confirmed", EventType.CUT_IN, 0.0, 0.4, r.events[0].event_id)
    with pytest.raises(ValidationError):
        ReviewDecision.of(r.events[0], "not-a-hash", verdict="confirmed", reviewer="kd", decided_at="x")
    with pytest.raises(ValidationError, match="corrected"):
        ReviewDecision.of(r.events[0], r.analysis_key, verdict="unsure", reviewer="kd", decided_at="x",
                          corrected_t_start=2.0, corrected_t_end=1.0)


def test_lane_observations_use_reason_codes_and_lane_changes_can_be_cancelled():
    obs = EgoLaneObservation(reasons=(LaneReason.SHARP_BEND, LaneReason.ON_BOUNDARY))
    assert json.loads(obs.model_dump_json())["reasons"] == ["sharp_bend", "on_boundary"]
    with pytest.raises(ValidationError):
        EgoLaneObservation(reasons=("bend too sharp",))
    assert LaneChange(direction="left", phase="cancelled", t_start=1.0).phase == "cancelled"


def test_run_warnings_are_coded():
    w = RunWarning(code=WarningCode.FRAME_GAP, message="no frames for 2 s", t=3.0)
    assert w.code is WarningCode.FRAME_GAP
    with pytest.raises(ValidationError):
        RunWarning(code="something", message="x")


def test_segmentation_paths_stay_inside_the_entry():
    ok = SegmentationRef(path="masks/000001.png", sha256=SHA, width=4, height=3, classes=("road",))
    assert ok.path.startswith("masks/")
    for bad in ("/etc/passwd", "../x.png", "masks/../../x.png"):
        with pytest.raises(ValidationError):
            SegmentationRef(path=bad, sha256=SHA, width=4, height=3, classes=("road",))


def test_component_identity_and_provenance_rules():
    with pytest.raises(ValidationError):
        ComponentInfo(role=Role.DETECTOR, name="d", version="1", params={"bad": float("nan")})
    with pytest.raises(ValidationError):
        ComponentInfo(role=Role.DETECTOR, name="d", version="1", checkpoint_sha256="abc")
    with pytest.raises(ValidationError, match="per perception role"):
        Provenance(engine_version="0", decoder={}, components=(DET, DET))
    with pytest.raises(ValidationError, match="per perception role"):
        Provenance(engine_version="0", decoder={}, components=(ANALYZER,))


def test_run_config_span():
    with pytest.raises(ValidationError):
        RunConfig(start_s=5, end_s=5)
    with pytest.raises(ValidationError):
        RunConfig(interval_s=0)


def motion(**kw):
    return CameraMotion(**{**dict(since_t=0.9, scale=1.01, rotation=0.0, tx=-3.0, ty=1.0, matches=200, inliers=150,
                                  residual_px=0.4), **kw})


def test_camera_motion_needs_a_tracker_that_ran_and_an_earlier_reference():
    motion_ = motion()
    ok = {Role.DETECTOR: OutputStatus.OK, Role.TRACKER: OutputStatus.OK}
    FrameRecord(index=1, t=1.0, status=ok, detections=(), tracks=(), camera_motion=motion_)
    with pytest.raises(ValueError, match="tracker that ran"):
        FrameRecord(index=1, t=1.0, status={Role.DETECTOR: OutputStatus.OK}, detections=(), camera_motion=motion_)
    with pytest.raises(ValueError, match="earlier frame"):
        FrameRecord(index=1, t=0.9, status=ok, detections=(), tracks=(), camera_motion=motion_)
    with pytest.raises(ValueError):
        motion(scale=0.0)
    with pytest.raises(ValueError, match="inliers cannot exceed"):
        motion(inliers=201)
    with pytest.raises(ValueError):
        CameraMotion(since_t=0.0, scale=1.0, rotation=0.0, tx=0.0, ty=0.0)  # quality is required
