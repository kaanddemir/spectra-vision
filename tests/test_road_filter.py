from spectra.vision.detection import Detection
from spectra.vision.road import (
    LaneFrame,
    detection_corridor_score,
    filter_relevant_detections,
)


def _lane(confidence: float = 0.25, detected: bool = False) -> LaneFrame:
    # A centered corridor spanning x≈[60,240] at the bottom of a 300x200 frame.
    return LaneFrame(
        vanishing_point=(150.0, 80.0),
        left_line=(120, 100, 60, 199),
        right_line=(180, 100, 240, 199),
        left_x_at_bottom=60.0,
        right_x_at_bottom=240.0,
        lane_width_at_bottom=180.0,
        lane_center_x_at_bottom=150.0,
        confidence=confidence,
        detected=detected,
        width=300,
        height=200,
    )


def test_oncoming_far_left_vehicle_is_dropped_even_when_confident():
    # A confident truck firmly on the far-left (opposite lane), not touching the
    # ego corridor, must be rejected — even with untrusted lane geometry, where
    # the confidence escapes would otherwise admit it.
    lane = _lane(confidence=0.25, detected=False)  # default/untrusted geometry
    oncoming = Detection(bbox=(0, 150, 40, 199), class_name="truck", confidence=0.9)
    assert detection_corridor_score(oncoming, lane) == 0.0
    assert filter_relevant_detections([oncoming], lane) == []


def test_in_corridor_lead_is_kept():
    # A large lead straddling the corridor bottom-center is kept.
    lane = _lane(confidence=0.25, detected=False)
    lead = Detection(bbox=(90, 150, 210, 199), class_name="car", confidence=0.9)
    assert detection_corridor_score(lead, lane) > 0.0
    assert filter_relevant_detections([lead], lane) == [lead]


def test_left_vehicle_touching_corridor_is_kept_as_cut_in():
    # A left-adjacent vehicle that overlaps the ego corridor is a cut-in
    # candidate and must survive the oncoming gate (overlap_px > 0).
    lane = _lane(confidence=0.9, detected=True)
    cut_in = Detection(bbox=(30, 150, 110, 199), class_name="car", confidence=0.9)
    assert detection_corridor_score(cut_in, lane) > 0.0
