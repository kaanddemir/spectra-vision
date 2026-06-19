"""Traffic-light colour classification + frame-level aggregation."""

import numpy as np

from spectra.vision.detection import Detection
from spectra.vision.traffic_light import classify_light_state, frame_light_state

_BBOX = (40, 40, 70, 90)


def _img_with(color):
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    img[40:90, 40:70] = color  # fill the bbox region
    return img


def test_red_green_yellow_classified():
    assert classify_light_state(_img_with((0, 0, 255)), _BBOX) == "red"
    assert classify_light_state(_img_with((0, 255, 0)), _BBOX) == "green"
    assert classify_light_state(_img_with((0, 255, 255)), _BBOX) == "yellow"


def test_dark_or_empty_is_unknown():
    assert classify_light_state(np.zeros((120, 120, 3), np.uint8), _BBOX) == "unknown"


def test_frame_state_none_without_lights():
    assert frame_light_state(np.zeros((120, 120, 3), np.uint8), []) == "none"


def test_frame_state_picks_largest_light():
    img = _img_with((0, 0, 255))  # red in the larger bbox
    small = Detection(bbox=(5, 5, 12, 18), class_name="traffic_light", confidence=0.5)
    big = Detection(bbox=_BBOX, class_name="traffic_light", confidence=0.5)
    assert frame_light_state(img, [small, big]) == "red"
