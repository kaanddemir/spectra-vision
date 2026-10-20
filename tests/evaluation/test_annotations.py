import pytest
from pydantic import ValidationError

from evaluation.annotations import (ClipAnnotation, Event, EventType, Licence, Provenance, Split,
                                    dump_annotations, load_annotations)

LIC = Licence(dataset_id="d", terms="t")
PROV = Provenance(annotator="a", method="manual", created="2026-10-16")


def clip(**kw):
    base = dict(clip_id="c1", source="s", licence=LIC, split=Split.DEV, duration_s=10.0, provenance=PROV)
    base.update(kw)
    return ClipAnnotation(**base)


def test_partial_event_timing_is_allowed_and_not_filled_in():
    e = Event(event_id="e", type=EventType.COLLISION, t_peak=3.0)
    assert e.t_start is None and e.t_end is None and e.anchor() == 3.0


def test_event_without_any_time_is_rejected():
    with pytest.raises(ValidationError):
        Event(event_id="e", type=EventType.APPROACH)


def test_out_of_order_timestamps_are_rejected():
    with pytest.raises(ValidationError):
        Event(event_id="e", type=EventType.APPROACH, t_start=5, t_peak=4)


def test_anchor_uses_midpoint_without_peak():
    assert Event(event_id="e", type=EventType.CUT_IN, t_start=2, t_end=4).anchor() == 3


def test_event_beyond_duration_is_rejected():
    with pytest.raises(ValidationError):
        clip(events=(Event(event_id="e", type=EventType.APPROACH, t_peak=11),))


def test_negative_label_conflicting_with_event_is_rejected():
    with pytest.raises(ValidationError):
        clip(events=(Event(event_id="e", type=EventType.APPROACH, t_peak=1),), negative_for=(EventType.APPROACH,))


def test_duplicate_event_ids_are_rejected():
    e = Event(event_id="e", type=EventType.APPROACH, t_peak=1)
    with pytest.raises(ValidationError):
        clip(events=(e, e))


def test_decreasing_frame_timestamps_are_rejected():
    with pytest.raises(ValidationError):
        clip(frame_timestamps_s=(0.0, 0.2, 0.1))


def test_unknown_fields_and_schema_versions_are_rejected():
    with pytest.raises(ValidationError):
        clip(colour="red")
    with pytest.raises(ValidationError):
        clip(schema_version="0.9")


def test_evaluated_for_requires_event_or_explicit_negative():
    c = clip(events=(Event(event_id="e", type=EventType.APPROACH, t_peak=1),), negative_for=(EventType.CUT_IN,))
    assert c.is_evaluated_for(EventType.APPROACH)
    assert c.is_evaluated_for(EventType.CUT_IN)
    assert not c.is_evaluated_for(EventType.COLLISION)


def test_jsonl_round_trip_and_line_numbers_in_errors(tmp_path):
    path = tmp_path / "a.jsonl"
    dump_annotations([clip(), clip(clip_id="c2")], path)
    assert [c.clip_id for c in load_annotations(path)] == ["c1", "c2"]
    path.write_text(path.read_text() + '{"clip_id": "bad"}\n')
    with pytest.raises(ValueError, match=":3:"):
        load_annotations(path)


def test_duplicate_clip_ids_in_file_are_rejected(tmp_path):
    path = tmp_path / "a.jsonl"
    dump_annotations([clip(), clip()], path)
    with pytest.raises(ValueError, match="duplicate clip_id"):
        load_annotations(path)
