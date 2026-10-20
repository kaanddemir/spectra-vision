import json

import pytest
from pydantic import ValidationError

from evaluation.annotations import (ClipAnnotation, Event, EventType, Licence, Origin, Provenance, SourceLabel,
                                    Split, Unmeasurable, load_annotations)
from evaluation.candidates import Candidate
from evaluation.metrics import evaluate

A, C = EventType.APPROACH, EventType.CUT_IN
LIC = Licence(dataset_id="d", terms="t")


def prov(method):
    return Provenance(annotator="a", method=method, created="2026-10-17")


def clip(method="manual", **kw):
    base = dict(clip_id="c", source="s", licence=LIC, split=Split.DEV, duration_s=60.0, provenance=prov(method))
    base.update(kw)
    return ClipAnnotation(**base)


def ev(eid="e", etype=A, peak=10.0, **kw):
    return Event(event_id=eid, type=etype, t_peak=peak, **kw)


def cand(peak, etype=A):
    return Candidate("c", etype, peak - 0.5, peak, peak + 0.5, 1.0, "x", "1")


@pytest.mark.parametrize("method,origin", [("manual", Origin.HUMAN_CONFIRMED), ("synthetic", Origin.SYNTHETIC),
                                           ("imported:dota", Origin.SOURCE_DERIVED),
                                           ("converted:ccd", Origin.SOURCE_DERIVED),
                                           ("draft:model-assisted-frame-review", Origin.DRAFT)])
def test_origin_follows_provenance_method(method, origin):
    c = clip(method, events=(ev(),))
    assert c.origin_of(c.events[0]) is origin


def test_schema_1_0_lines_load_and_draft_events_are_not_ground_truth(tmp_path):
    line = clip("draft:x", events=(ev(),)).model_dump(mode="json", exclude_defaults=True)
    line["schema_version"] = "1.0"
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps(line) + "\n")
    (c,) = load_annotations(path)
    assert c.schema_version == "1.1"
    assert not c.is_evaluated_for(A)


def test_imported_and_draft_events_are_not_scored_by_default():
    for method in ("imported:x", "draft:x"):
        r, _, unscored = evaluate([clip(method, events=(ev(),))], [cand(10.0)])
        assert (r[A].tp, r[A].fp, r[A].fn, unscored) == (0, 0, 0, 1)


def test_imported_events_score_only_when_explicitly_requested():
    r, _, _ = evaluate([clip("imported:x", events=(ev(),))], [cand(10.0)],
                       origins=frozenset({Origin.SOURCE_DERIVED}))
    assert r[A].tp == 1


def test_individually_confirmed_event_needs_a_reviewer_and_is_then_scored():
    with pytest.raises(ValidationError, match="reviewed_by"):
        ev(origin=Origin.HUMAN_CONFIRMED)
    c = clip("imported:x", events=(ev("a", origin=Origin.HUMAN_CONFIRMED, reviewed_by="r"), ev("b", peak=30.0)))
    r, _, _ = evaluate([c], [cand(10.0), cand(30.0)])
    assert (r[A].tp, r[A].fp, r[A].fn) == (1, 1, 0)


def test_negatives_cannot_be_imported_or_drafted():
    for method in ("imported:x", "draft:x"):
        with pytest.raises(ValidationError, match="negative_for"):
            clip(method, negative_for=(A,))
    clip("synthetic", negative_for=(A,))


def test_unmeasurable_type_is_not_scored_and_cannot_also_be_negative():
    c = clip(events=(ev(),), unmeasurable=(Unmeasurable(type=A, reason="lead vehicle out of frame"),))
    r, _, unscored = evaluate([c], [cand(10.0)])
    assert (r[A].tp, r[A].fn, unscored) == (0, 0, 1)
    with pytest.raises(ValidationError, match="both negative and unmeasurable"):
        clip(negative_for=(A,), unmeasurable=(Unmeasurable(type=A, reason="r"),))


def label(lid="s1", **kw):
    return SourceLabel(label_id=lid, dataset_id="d", record="f.json", label="lateral", **kw)


def test_source_labels_keep_raw_fields_and_refs_must_resolve():
    c = clip("imported:x", source_labels=(label(fields={"accident_id": "3", "ego_involve": True}),),
             events=(ev(source_refs=("s1",)),))
    again = ClipAnnotation.model_validate_json(c.model_dump_json())
    assert again.source_labels[0].fields == {"accident_id": "3", "ego_involve": True}
    with pytest.raises(ValidationError, match="unknown source labels"):
        clip("imported:x", events=(ev(source_refs=("missing",)),))


@pytest.mark.parametrize("kw,msg", [(dict(t_start=5.0, t_end=4.0), "t_end before t_start"),
                                    (dict(t_start=1.0, t_end=61.0), "exceeds clip duration")])
def test_invalid_source_label_times_are_rejected(kw, msg):
    with pytest.raises(ValidationError, match=msg):
        clip("imported:x", source_labels=(label(**kw),))


def test_duplicate_source_label_ids_are_rejected():
    with pytest.raises(ValidationError, match="duplicate label_id"):
        clip("imported:x", source_labels=(label(), label()))
