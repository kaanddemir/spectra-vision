import copy
import json

import pytest

from evaluation.annotations import EventType, Origin
from evaluation.importers import dota, prevention, registry_licence
from evaluation.metrics import evaluate

# Shape copied from DoTA annotations/0RJPQ_97dcs_000199.json, shortened to 6 frames.
DOTA = {
    "video_name": "vid_000010", "channel": "AnAn", "num_frames": 6, "ignore": False, "ego_involve": True,
    "night": True, "anomaly_start": 2, "anomaly_end": 4, "video_start": 10, "video_end": 15,
    "accident_id": "3", "accident_name": "lateral",
    "labels": [{"frame_id": i, "image_path": f"frames/vid_000010/images/{i:06d}.jpg",
                "accident_id": 3 if 2 <= i < 4 else 0, "accident_name": "lateral" if 2 <= i < 4 else "normal",
                "objects": [{"obj_track_id": 7, "bbox": [10.0, 20.0, 50.0, 60.0], "category": "car",
                             "category ID": 3, "trunc": False}] if 2 <= i < 4 else []}
               for i in range(6)],
}


def test_dota_preserves_source_labels_converts_times_and_creates_no_events():
    c = dota.convert(DOTA, "f.json")
    assert c.events == () and c.origin is Origin.SOURCE_DERIVED
    (s,) = c.source_labels
    assert (s.label, s.t_start, s.t_end, c.duration_s) == ("lateral", 0.2, 0.4, 0.6)
    assert s.fields["accident_id"] == "3" and s.fields["ego_involve"] is True and "labels" not in s.fields
    assert s.participant_ids == ("7",) and [b.t for b in s.boxes] == [0.2, 0.3]
    assert s.review_for == (EventType.COLLISION, EventType.CUT_IN)
    assert c.conditions.light.value == "night"


def test_dota_review_hints_depend_on_ego_involvement_and_category():
    other = dict(DOTA, ego_involve=False)
    assert dota.convert(other, "f").source_labels[0].review_for == (EventType.COLLISION,)
    assert dota.convert(dict(DOTA, accident_name="leave_to_left"), "f").source_labels[0].review_for == ()


def test_dota_clip_is_never_scored_and_inherits_the_registry_licence():
    c = dota.convert(DOTA, "f")
    assert c.licence == registry_licence("dota")
    assert not any(c.is_evaluated_for(t) for t in EventType)
    r, _, _ = evaluate([c], [])
    assert all(r[t].fn == 0 for t in EventType)


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d.pop("ego_involve"), "unexpected keys"),
    (lambda d: d["labels"].pop(1), "frame_id must run"),
    (lambda d: d.update(anomaly_end=9), "outside"),
    (lambda d: d["labels"][2]["objects"][0].pop("trunc"), "unexpected object keys"),
])
def test_dota_invalid_metadata_is_rejected(mutate, msg):
    d = copy.deepcopy(DOTA)
    mutate(d)
    with pytest.raises(ValueError, match=msg):
        dota.convert(d, "f.json")


def test_dota_import_dir_filters_by_name(tmp_path):
    (tmp_path / "vid_000010.json").write_text(json.dumps(DOTA))
    (tmp_path / "vid_000099.json").write_text(json.dumps(dict(DOTA, video_name="vid_000099")))
    assert [c.clip_id for c in dota.import_dir(tmp_path, names={"vid_000099"})] == ["dota-vid_000099"]


# Lines 1 and 4 are verbatim from RECORD4/DRIVE3 and RECORD2/DRIVE1; the others cover code 1 (absent from
# every released drive), an undefined code, an unknown start and inconsistent frames.
LC = "\n".join(["-00001 000882 000003 000962 000991 001007 000001",
                "12 5 1 100 110 125 0",
                "-1 7 9 300 301 302 0",
                "000062 -00001 000003 -00001 001010 001010 000000",
                "-1 8 4 50 40 60 0", ""])
TIMES = [i * 0.091 for i in range(1100)]


def log(n=4, step=91074):
    return "".join(f"{i:011d} {i:011d} {45506924038 + i * step} {45506924038 + i * step + 500}\n" for i in range(n))


def test_camera_log_gives_frame_times_in_seconds():
    assert prevention.parse_camera_log(log()) == pytest.approx([0.0, 0.091074, 0.182148, 0.273222])


@pytest.mark.parametrize("text,msg", [(log(1), "at least two"), (log().replace("00000000001 00000000001", "00000000001 00000000005"), "out of sequence"),
                                      (log(step=0), "must increase"), ("1 2 3\n", "expected 4 fields")])
def test_invalid_camera_logs_are_rejected(text, msg):
    with pytest.raises(ValueError, match=msg):
        prevention.parse_camera_log(text)


def test_prevention_keeps_every_line_with_raw_fields_and_log_times():
    c = prevention.convert(LC, "RECORD4/DRIVE3", TIMES)
    assert c.clip_id == "prevention-record4-drive3" and c.origin is Origin.SOURCE_DERIVED
    assert c.fps_nominal == pytest.approx(1 / 0.091) and c.duration_s == pytest.approx(1100 * 0.091)
    left, cut_in, unknown, end_only, bad = c.source_labels
    assert left.label == "left_lane_change" and left.participant_ids == ("d:882",)
    assert left.fields == {"id_l": -1, "id_d": 882, "type": 3, "f0": 962, "fe": 991, "ff": 1007, "blk": 1}
    assert (left.t_start, left.t_end) == pytest.approx((962 * 0.091, 1007 * 0.091))
    assert left.review_for == (EventType.CUT_IN,)
    assert unknown.label == "type_9" and "not defined" in unknown.note
    assert end_only.t_start is None and end_only.t_end == pytest.approx(1010 * 0.091)
    assert end_only.participant_ids == ("l:62",) and "start frame unknown" in end_only.note
    assert bad.t_start is None and bad.t_end is None and bad.review_for == () and "out of order" in bad.note


def test_prevention_only_cut_in_code_becomes_a_source_derived_event():
    c = prevention.convert(LC, "R/D", TIMES)
    (e,) = c.events
    assert (e.type, e.t_start, e.t_peak, e.t_end) == pytest.approx((EventType.CUT_IN, 9.1, 10.01, 11.375))
    assert e.participant_ids == ("l:12", "d:5") and e.source_refs == (c.source_labels[1].label_id,)
    assert not c.is_evaluated_for(EventType.CUT_IN)


def test_prevention_frames_beyond_the_camera_log_are_not_converted():
    (s,) = prevention.convert("-1 5 3 10 20 30 1\n", "R/D", TIMES[:25]).source_labels
    assert s.t_start is None and "beyond the camera log" in s.note


@pytest.mark.parametrize("text,msg", [("1 2 3 4 5 6", "expected 7 fields"), ("1 2 x 4 5 6 7", "integers")])
def test_prevention_malformed_lines_are_rejected_with_line_numbers(text, msg):
    with pytest.raises(ValueError, match=f":2: .*{msg}"):
        prevention.convert("1 2 3 4 5 6 7\n" + text, "R/D", TIMES)


def test_unregistered_dataset_is_refused():
    with pytest.raises(ValueError, match="not in the registry"):
        registry_licence("unknown-dataset")
