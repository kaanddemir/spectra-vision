import pytest

from evaluation.annotations import (ClipAnnotation, Conditions, Event, EventType, Licence, Light, Provenance,
                                    Split)
from evaluation.candidates import Candidate, load_candidates, save_candidates
from evaluation.metrics import evaluate, temporal_iou

A = EventType.APPROACH
LIC = Licence(dataset_id="d", terms="t")
PROV = Provenance(annotator="a", method="manual", created="2026-10-16")


def clip(cid, events=(), negative=(), duration=3600.0, **kw):
    return ClipAnnotation(clip_id=cid, source="s", licence=LIC, split=Split.DEV, duration_s=duration,
                          events=tuple(events), negative_for=tuple(negative), provenance=PROV, **kw)


def cand(cid, peak, start=None, end=None, etype=A):
    return Candidate(cid, etype, start if start is not None else peak - 0.5, peak,
                     end if end is not None else peak + 0.5, 1.0, "x", "1")


def ev(eid, peak=None, start=None, end=None, **kw):
    return Event(event_id=eid, type=A, t_peak=peak, t_start=start, t_end=end, **kw)


def test_temporal_iou():
    assert temporal_iou((0, 2), (1, 3)) == pytest.approx(1 / 3)
    assert temporal_iou((0, 1), (2, 3)) == 0


def test_peak_tolerance_boundary():
    clips = [clip("c", [ev("e", peak=10.0)])]
    r, _, _ = evaluate(clips, [cand("c", 11.0)])
    assert (r[A].tp, r[A].fp) == (1, 0)
    r, _, _ = evaluate(clips, [cand("c", 11.2, 10.7, 11.7)])
    assert (r[A].tp, r[A].fp, r[A].fn) == (0, 1, 1)


def test_interval_overlap_matches_when_peak_is_far():
    clips = [clip("c", [ev("e", peak=10, start=8, end=14)])]
    r, _, _ = evaluate(clips, [cand("c", 12.5, 9, 13)])
    assert r[A].tp == 1


def test_one_to_one_assignment_prefers_global_optimum():
    clips = [clip("c", [ev("e1", peak=10.0), ev("e2", peak=11.5)])]
    r, m, _ = evaluate(clips, [cand("c", 10.8), cand("c", 12.3)])
    assert (r[A].tp, r[A].fp, r[A].fn) == (2, 0, 0)
    assert {x.event.event_id for x in m} == {"e1", "e2"}


def test_duplicate_candidates_count_as_false_positives():
    r, _, _ = evaluate([clip("c", [ev("e", peak=10)])], [cand("c", 10), cand("c", 10.1)])
    assert (r[A].tp, r[A].fp) == (1, 1)


def test_ambiguous_labels_neither_reward_nor_penalise():
    clips = [clip("c", [ev("e", peak=10, ambiguous=True)])]
    r, _, _ = evaluate(clips, [cand("c", 10)])
    assert (r[A].tp, r[A].fp, r[A].fn, r[A].ambiguous_hits) == (0, 0, 0, 1)
    r, _, _ = evaluate(clips, [])
    assert r[A].fn == 0


def test_false_per_hour_counts_only_clips_evaluated_for_the_type():
    clips = [clip("neg", negative=[A], duration=1800), clip("unlabelled", duration=1800)]
    r, _, unscored = evaluate(clips, [cand("neg", 5), cand("unlabelled", 5), cand("unknown", 1)])
    assert r[A].fp == 1 and r[A].hours == pytest.approx(0.5) and r[A].false_per_hour == pytest.approx(2.0)
    assert unscored == 2


def test_timing_errors_use_only_known_bounds():
    r, _, _ = evaluate([clip("c", [ev("e", peak=10, start=9)])], [cand("c", 10.4, 8.5, 11)])
    d = r[A].as_dict()["timing"]
    assert d["peak"]["mean_signed_s"] == pytest.approx(0.4)
    assert d["start"]["mean_signed_s"] == pytest.approx(-0.5)
    assert d["end"] is None


def test_recall_breakdown_falls_back_to_clip_conditions():
    clips = [clip("c", [ev("e1", peak=10), ev("e2", peak=50, conditions=Conditions(light=Light.DAY))],
                  conditions=Conditions(light=Light.NIGHT))]
    r, _, _ = evaluate(clips, [cand("c", 10)])
    bc = r[A].as_dict()["recall_by_condition"]
    assert bc["light=night"]["recall"] == 1.0 and bc["light=day"]["recall"] == 0.0


def test_empty_inputs_report_undefined_not_zero():
    r, _, _ = evaluate([], [])
    d = r[A].as_dict()
    assert d["precision"] is None and d["recall"] is None and d["false_candidates_per_hour"] is None


def test_candidate_round_trip_and_ordering_check(tmp_path):
    c = Candidate("c", A, 1, 2, 3, 0.5, "x", "1", (4,), {"k": 1})
    save_candidates([c], tmp_path / "c.jsonl")
    assert load_candidates(tmp_path / "c.jsonl") == [c]
    with pytest.raises(ValueError):
        Candidate("c", A, 3, 2, 1, 0.5, "x", "1")
