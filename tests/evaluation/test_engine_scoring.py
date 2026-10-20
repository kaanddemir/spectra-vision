import json

import pytest

from evaluation.annotations import (ClipAnnotation, Event, EventType, Licence, Provenance, SourceLabel, Split,
                                    dump_annotations)
from evaluation.engine_scoring import from_engine, score, window

C = EventType.CUT_IN


def clip(events=(), labels=(), allowed=None, method="manual"):
    return ClipAnnotation(clip_id="c", source="s", licence=Licence(dataset_id="d", terms="t", evaluation_allowed=allowed),
                          split=Split.DEV, duration_s=400.0, events=tuple(events), source_labels=tuple(labels),
                          provenance=Provenance(annotator="a", method=method, created="2026-10-17"))


def ev(eid, start, peak, end):
    return Event(event_id=eid, type=C, t_start=start, t_peak=peak, t_end=end)


def engine_cand(cid, peak, state="supported", etype="cut_in"):
    return {"event_id": cid, "type": etype, "subtype": "camera_vehicle" if etype == "impact" else None,
            "t_start": peak - 1.0, "t_peak": peak, "t_end": peak + 0.5, "rank": 0.5, "track_ids": [3],
            "analyzer": "cut_in", "state": state,
            "reasons": [] if state == "supported" else ["boundaries_reidentified"]}


def test_window_keeps_inside_events_shifts_times_and_marks_crossing_events_ambiguous():
    w = window(clip([ev("in", 110, 112, 114), ev("cross", 95, 101, 103), ev("out", 10, 11, 12)]), 100, 200)
    assert w.clip_id == "c@100-200" and w.duration_s == 100
    inside, crossing = w.events
    assert (inside.t_start, inside.t_peak, inside.t_end, inside.ambiguous) == (10, 12, 14, False)
    assert (crossing.t_start, crossing.ambiguous) == (0, True) and "crosses" in crossing.note


def test_window_filters_source_labels_and_keeps_untimed_ones():
    labels = [SourceLabel(label_id=i, dataset_id="d", record="r", label="x", t_start=t, t_end=t and t + 2)
              for i, t in (("a", 150.0), ("b", 10.0), ("c", None))]
    w = window(clip(labels=labels, method="imported:x"), 100, 200)
    assert [(s.label_id, s.t_start) for s in w.source_labels] == [("a", 50.0), ("c", None)]


def test_window_outside_the_clip_is_rejected():
    with pytest.raises(ValueError, match="not within the clip"):
        window(clip(), 300, 500)


def test_engine_candidates_are_mapped_shifted_and_filtered():
    raw = [engine_cand("1", 150.0), engine_cand("2", 160.0, "uncertain"), engine_cand("3", 250.0),
           engine_cand("4", 120.0, etype="rapid_approach"), engine_cand("5", 130.0, etype="impact")]
    cands = from_engine("c@100-200", raw, 100, 200)
    assert [(c.type, c.t_peak, c.observations["state"]) for c in cands] == [
        (C, 50.0, "supported"), (C, 60.0, "uncertain"), (EventType.APPROACH, 20.0, "supported"),
        (EventType.COLLISION, 30.0, "supported")]
    assert cands[1].observations["reasons"] == ["boundaries_reidentified"]
    assert [c.observations["event_id"] for c in from_engine("c", raw, 100, 200, supported_only=True)] == ["1", "4", "5"]
    with pytest.raises(ValueError, match="unknown engine event type"):
        from_engine("c", [engine_cand("x", 150, etype="near_miss")], 100, 200)


def write(tmp_path, clips, cands, state="completed"):
    ann = tmp_path / "a.jsonl"
    dump_annotations(clips, ann)
    out = tmp_path / "engine.json"
    out.write_text(json.dumps({"state": state, "error": None, "perception_key": "k", "events": cands}))
    return ann, out


def test_score_reports_all_and_supported_candidates_and_never_overwrites(tmp_path):
    ann, out = write(tmp_path, [clip([ev("e", 140, 150.5, 152)])], [engine_cand("1", 150.0), engine_cand("2", 180.0, "uncertain")])
    r = score(ann, "c", out, 100, 200, "base", tmp_path / "results")
    all_, sup = (r["algorithms"][k]["by_type"]["cut_in"] for k in ("engine_all", "engine_supported"))
    assert (all_["tp"], all_["fp"], all_["fn"]) == (1, 1, 0)
    assert (sup["tp"], sup["fp"], sup["fn"]) == (1, 0, 0)
    assert all_["timing"]["peak"]["median_abs_s"] == pytest.approx(0.5)
    assert (tmp_path / "results" / "results_log.jsonl").exists()
    with pytest.raises(FileExistsError):
        score(ann, "c", out, 100, 200, "base", tmp_path / "results")


def test_source_derived_labels_are_not_scored_and_licence_and_failed_runs_are_refused(tmp_path):
    ann, out = write(tmp_path, [clip([ev("e", 140, 150, 152)], method="imported:x")], [engine_cand("1", 150.0)])
    r = score(ann, "c", out, 100, 200, "src", tmp_path / "results")
    assert r["algorithms"]["engine_all"]["by_type"]["cut_in"]["tp"] == 0
    assert r["algorithms"]["engine_all"]["unscored_candidates"] == 1
    ann, out = write(tmp_path, [clip(allowed=False)], [])
    with pytest.raises(ValueError, match="does not allow evaluation"):
        score(ann, "c", out, 100, 200, "lic", tmp_path / "results")
    ann, out = write(tmp_path, [clip()], [], state="failed")
    with pytest.raises(ValueError, match="did not complete"):
        score(ann, "c", out, 100, 200, "fail", tmp_path / "results")
