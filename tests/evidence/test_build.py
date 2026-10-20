from spectra.engine.contracts import (AnalyzerStatus, EgoLaneState, EventType, LaneRelation, ReasonCode, RunWarning,
                                      UnresolvedPattern, WarningCode, canonical_json)
from spectra.engine.runner import RunOutcome, RunState, StopReason
from spectra.evidence.build import build, build_key, from_outcome, review_interval
from spectra.evidence.contracts import (AnalysisState, Coverage, Note, RelationKind, ReviewConfig)
from tests.evidence import synth as S
from tests.evidence.synth import APPROACH, CONTACT, CUT_IN, IMPACT


def by_type(timeline):
    return {(e.type.value, e.subtype and e.subtype.value): e for e in timeline.entries}


def test_every_proposal_gets_one_package_with_the_proposal_unchanged():
    res = S.result([S.approach(5.0, 7.0), S.cut_in(9.0, 9.5, entry_from=8.0)])
    timeline, packages = build(res)
    assert sorted(p.proposal.event_id for p in packages) == sorted(e.event_id for e in res.events)
    for p in packages:
        original = next(e for e in res.events if e.event_id == p.proposal.event_id)
        assert canonical_json(p.proposal) == canonical_json(original)
        assert p.analysis_key == res.analysis_key and p.perception_key == res.perception_key
        assert p.analyzer.name == original.analyzer and p.video == res.perception.video
    assert [e.package for e in timeline.entries] == [f"packages/{p.proposal.event_id}.json" for p in packages]


def test_evidence_basis_and_units_are_preserved():
    res = S.result([S.approach(5.0, 7.0)])
    _, [pk] = build(res)
    assert [(e.basis, e.name, e.unit, e.value) for e in pk.proposal.evidence] == [
        (e.basis, e.name, e.unit, e.value) for e in res.events[0].evidence]


def test_building_is_byte_for_byte_repeatable():
    res = S.result([S.approach(5.0, 7.0), S.impact(5.0, 7.0, 7.2), S.cut_in(9.0, 9.5)])
    a, b = build(res), build(res)
    assert canonical_json(a[0]) == canonical_json(b[0])
    assert [canonical_json(p) for p in a[1]] == [canonical_json(p) for p in b[1]]


def test_entries_are_chronological_with_a_fixed_tie_break():
    res = S.result([S.cut_in(4.0, 4.5), S.approach(4.0, 6.0, track=3, peak=4.5),
                    S.approach(2.0, 3.0, track=4)])
    timeline, packages = build(res)
    assert [(e.t_start, e.type) for e in timeline.entries] == [
        (2.0, EventType.RAPID_APPROACH), (4.0, EventType.RAPID_APPROACH), (4.0, EventType.CUT_IN)]
    assert [p.proposal.event_id for p in packages] == [e.event_id for e in timeline.entries]


def test_titles_have_stable_keys():
    timeline, _ = build(S.result([S.approach(5.0, 7.0), S.impact(5.0, 7.0, 7.2)]))
    t = by_type(timeline)
    assert t[("rapid_approach", None)].title_key == "event.rapid_approach"
    assert t[("impact", "camera_vehicle")].title_key == "event.impact.camera_vehicle"
    assert "Possible" in t[("impact", "camera_vehicle")].title


# --- review intervals ---------------------------------------------------------------------------------


def test_interval_adds_context_around_proposal_and_evidence():
    res = S.result([S.cut_in(9.0, 9.5, entry_from=7.5)])
    _, [pk] = build(res, ReviewConfig(pre_s=2.0, post_s=1.0))
    assert (pk.review.evidence_start_s, pk.review.evidence_end_s) == (7.5, 9.5)
    assert (pk.review.start_s, pk.review.end_s) == (5.5, 10.5)
    assert not {Note.CLIPPED_AT_VIDEO_START, Note.CLIPPED_AT_VIDEO_END} & set(pk.notes)


def test_interval_is_clipped_to_the_video_and_says_so():
    res = S.result([S.approach(0.5, 1.0, track=1), S.approach(19.5, 20.0, track=2)])
    _, [early, late] = build(res, ReviewConfig(pre_s=3.0, post_s=2.0))
    assert early.review.start_s == 0.0 and Note.CLIPPED_AT_VIDEO_START in early.notes
    assert late.review.end_s == 20.0 and Note.CLIPPED_AT_VIDEO_END in late.notes


def test_undeclared_duration_clips_at_the_last_analysed_frame():
    res = S.result([S.approach(19.0, 19.5)], info=S.video(duration=None))
    _, [pk] = build(res, ReviewConfig(post_s=2.0))
    assert pk.review.end_s == 20.0 and Note.CLIPPED_AT_VIDEO_END in pk.notes


def test_context_is_capped_but_the_evidence_never_is():
    res = S.result([S.approach(5.0, 12.0)])
    _, [pk] = build(res, ReviewConfig(pre_s=4.0, post_s=4.0, max_context_s=2.0))
    assert (pk.review.start_s, pk.review.end_s) == (4.0, 13.0)
    assert Note.CONTEXT_CAPPED in pk.notes


def test_context_beyond_the_analysed_span_is_marked():
    res = S.result([S.approach(10.5, 11.0)], times=[round(10 + 0.1 * i, 3) for i in range(21)],
                   info=S.video(duration=30.0))
    _, [pk] = build(res, ReviewConfig(pre_s=2.0, post_s=2.0))
    assert (pk.review.start_s, pk.review.end_s) == (8.5, 13.0)
    assert Note.OUTSIDE_ANALYSIS in pk.notes and (pk.analysed_start_s, pk.analysed_end_s) == (10.0, 12.0)


def test_variable_rate_and_offset_times_are_used_as_recorded():
    times = [0.0, 0.07, 0.21, 0.24, 0.5, 0.52, 0.9, 1.33, 1.4, 1.41, 2.0, 2.3]
    res = S.result([S.approach(0.21, 1.33, peak=0.9)], times=times, info=S.video(duration=2.3, start_s=0.733))
    _, [pk] = build(res, ReviewConfig(pre_s=0.1, post_s=0.07))
    assert (pk.review.start_s, pk.review.end_s) == (0.11, 1.4)
    assert pk.tracks[0].first_s == 0.21 and pk.tracks[0].last_s == 1.4


def test_frame_gaps_in_the_interval_are_reported():
    times = [round(0.1 * i, 3) for i in range(60)] + [round(9 + 0.1 * i, 3) for i in range(60)]
    res = S.result([S.approach(4.0, 9.5)], times=times, info=S.video(duration=15.0))
    timeline, [pk] = build(res)
    assert [(g.start_s, g.end_s) for g in pk.gaps] == [(5.9, 9.0)] and Note.FRAME_GAP in pk.notes
    assert [(g.start_s, g.end_s) for g in timeline.gaps] == [(5.9, 9.0)]


# --- overlaps and relations ---------------------------------------------------------------------------


def test_overlapping_proposals_share_a_segment_and_stay_separate():
    res = S.result([S.approach(5.0, 7.0, track=1), S.cut_in(6.0, 6.5, track=2), S.approach(15.0, 16.0, track=3)])
    timeline, _ = build(res)
    assert len(timeline.entries) == 3 and len(timeline.segments) == 2
    first, second = timeline.segments
    assert len(first.event_ids) == 2 and second.event_ids == (timeline.entries[2].event_id,)
    assert [e.segment for e in timeline.entries] == [0, 0, 1]
    assert first.start_s == min(e.review.start_s for e in timeline.entries[:2])


def test_impact_is_linked_to_the_approach_it_was_built_on():
    res = S.result([S.approach(5.0, 7.0), S.impact(5.0, 7.0, 7.3), S.approach(5.5, 6.0, track=9)])
    timeline, _ = build(res)
    t = {e.event_id: e for e in timeline.entries}
    imp = next(e for e in timeline.entries if e.type is EventType.IMPACT)
    app = next(e for e in timeline.entries if e.type is EventType.RAPID_APPROACH and e.track_ids == (1,))
    other = next(e for e in timeline.entries if e.track_ids == (9,))
    assert [(r.kind, r.event_id) for r in imp.related] == [(RelationKind.BASED_ON, app.event_id)]
    assert [(r.kind, r.event_id) for r in app.related] == [(RelationKind.BASIS_OF, imp.event_id)]
    assert other.related == () and len(t) == 3  # both stay independent entries


def test_no_based_on_link_when_the_impact_used_a_different_approach_configuration():
    changed = IMPACT.model_copy(update={"params": {"approach": {**IMPACT.params["approach"], "params": {"tau_s": 3}}}})
    res = S.result([S.approach(5.0, 7.0), S.impact(5.0, 7.0, 7.3, analyzer=changed)],
                   analyzers=(APPROACH, CUT_IN, changed))
    timeline, _ = build(res)
    kinds = {r.kind for e in timeline.entries for r in e.related}
    assert kinds == {RelationKind.SHARES_TRACK}


def test_proposals_about_a_common_track_are_linked_when_they_overlap():
    res = S.result([S.approach(5.0, 7.0, track=2), S.cut_in(6.0, 6.5, track=2), S.cut_in(12.0, 12.5, track=2)])
    timeline, _ = build(res)
    first, second, third = timeline.entries
    assert [r.kind for r in first.related] == [RelationKind.SHARES_TRACK] and third.related == ()


# --- incomplete and missing data ----------------------------------------------------------------------


def test_unknown_lane_relations_stay_unknown():
    res = S.result([S.approach(5.0, 7.0)], relation=lambda t, i: LaneRelation.LEFT_ADJACENT if t < 5 else None)
    _, [pk] = build(res)
    runs = [(r.relation, r.start_s, r.end_s) for r in pk.lanes.relations]
    assert runs[0][0] is LaneRelation.LEFT_ADJACENT and runs[-1][0] is None
    assert Note.LANE_UNOBSERVED in pk.notes


def test_without_a_lane_estimator_lanes_are_unavailable_not_negative():
    res = S.result([S.approach(5.0, 7.0)], lanes=False)
    _, [pk] = build(res)
    assert not pk.lanes.available and pk.lanes.relations == ()
    assert Note.EGO_LANE_UNAVAILABLE in pk.notes and Note.LANE_UNOBSERVED not in pk.notes


def test_ego_lane_states_are_summarised_as_runs():
    res = S.result([S.approach(5.0, 7.0)],
                   lane_state=lambda t: EgoLaneState.AMBIGUOUS if 4.5 <= t < 8 else EgoLaneState.ESTIMATED)
    _, [pk] = build(res)
    assert [r.state for r in pk.lanes.ego_lane] == [EgoLaneState.ESTIMATED, EgoLaneState.AMBIGUOUS,
                                                    EgoLaneState.ESTIMATED]
    assert Note.EGO_LANE_UNAVAILABLE in pk.notes


def test_a_track_ending_early_and_predicted_boxes_are_noted():
    res = S.result([S.approach(5.0, 7.0)],
                   tracks=lambda t: [(1, 6.5 < t <= 7.0)] if t <= 7.0 else [(5, False)])
    _, [pk] = build(res)
    tr = pk.tracks[0]
    assert tr.last_s == 6.5 and tr.ends_early and tr.predicted == 5
    assert {Note.TRACK_ENDS_EARLY, Note.TRACK_PREDICTED} <= set(pk.notes)


def test_missing_camera_motion_is_noted():
    res = S.result([S.impact(5.0, 7.0, 7.2)], motion=lambda t: not 6.0 <= t <= 6.3)
    _, [pk] = build(res)
    assert Note.MOTION_MISSING in pk.notes


def test_analyser_outcomes_are_distinguished():
    un = UnresolvedPattern(reason=ReasonCode.MOTION_ESTIMATE_MISSING, t_start=11.0, t_end=12.0, track_ids=(4,))
    res = S.result([S.approach(10.0, 11.0)],
                   status={CUT_IN.name: (AnalyzerStatus.UNAVAILABLE, "no lane relations")},
                   unresolved={IMPACT.name: [un]})
    timeline, [pk] = build(res)
    cov = {c.name: c for c in timeline.analyzers}
    assert cov[APPROACH.name].coverage is Coverage.PROPOSED
    assert cov[CUT_IN.name].coverage is Coverage.UNAVAILABLE and cov[CUT_IN.name].reason == "no lane relations"
    assert cov[IMPACT.name].coverage is Coverage.NO_PROPOSAL and cov[IMPACT.name].unresolved == (un,)
    assert pk.unresolved == (un,) and Note.UNRESOLVED_NEARBY in pk.notes
    failed = S.result(status={IMPACT.name: (AnalyzerStatus.FAILED, "boom")})
    assert {c.name: c.coverage for c in build(failed)[0].analyzers}[IMPACT.name] is Coverage.FAILED


def test_experimental_proposals_are_labelled():
    res = S.result([S.contact(5.0, 5.5)], analyzers=(APPROACH, CONTACT))
    timeline, [pk] = build(res)
    entry = timeline.entries[0]
    assert entry.experimental and Note.EXPERIMENTAL in entry.notes and entry.reasons == (ReasonCode.NOT_VALIDATED,)
    assert {c.name: c.experimental for c in timeline.analyzers} == {APPROACH.name: False, CONTACT.name: True}


def test_run_warnings_in_the_interval_are_attached():
    w = [RunWarning(code=WarningCode.FRAME_GAP, message="gap", t=6.0),
         RunWarning(code=WarningCode.DECODE_ERRORS, message="errors"),
         RunWarning(code=WarningCode.FRAME_GAP, message="later", t=18.0)]
    _, [pk] = build(S.result([S.approach(5.0, 7.0)], warnings=w))
    assert [x.message for x in pk.warnings] == ["gap", "errors"] and Note.RUN_WARNING in pk.notes


def test_decoding_that_ended_early_makes_the_timeline_incomplete():
    timeline, _ = build(S.result(ended_early="demuxing stopped"))
    assert timeline.state is AnalysisState.INCOMPLETE and "demuxing stopped" in timeline.message


def test_runs_without_a_result_say_why_and_claim_nothing():
    cases = [(RunOutcome(RunState.CANCELLED), AnalysisState.CANCELLED),
             (RunOutcome(RunState.FAILED, error="memory", stop=StopReason.RESOURCES), AnalysisState.RESOURCE_STOPPED),
             (RunOutcome(RunState.FAILED, error="bad file", stop=StopReason.VIDEO), AnalysisState.FAILED)]
    for outcome, state in cases:
        timeline, packages = from_outcome(outcome)
        assert timeline.state is state and timeline.entries == () and timeline.analyzers == () and packages == []
        assert timeline.analysis_key is None and timeline.message


def test_build_key_depends_on_analysis_and_configuration():
    assert build_key(S.KEY, ReviewConfig()) == build_key(S.KEY, ReviewConfig())
    assert build_key(S.KEY, ReviewConfig()) != build_key(S.KEY, ReviewConfig(pre_s=4.0))
    assert build_key(S.KEY, ReviewConfig()) != build_key("c" * 64, ReviewConfig())


def test_review_interval_function_is_type_independent():
    res = S.result([S.approach(5.0, 7.0), S.cut_in(5.0, 7.0, track=1)])
    a, b = (review_interval(e, 20.0, (0.0, 20.0), ReviewConfig())[0] for e in res.events)
    assert (a.start_s, a.end_s) == (b.start_s, b.end_s) == (2.0, 9.0)
