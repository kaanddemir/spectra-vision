import numpy as np

from spectra.vision.brake_lights import brake_score
from spectra.vision.detection import Detection
from spectra.vision.traffic_light import classify_light_state, frame_light_state


_VEHICLE_BBOX = (50, 50, 150, 150)
_LIGHT_BBOX = (40, 40, 70, 90)


def _blank(color=(0, 0, 0)):
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:] = color
    return img


def _bright_red(img, x1, y1, x2, y2):
    img[y1:y2, x1:x2] = (0, 0, 255)


def _light_img(color):
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    img[40:90, 40:70] = color
    return img


def test_symmetric_brake_light_pair_scores_high():
    img = _blank((40, 40, 40))
    _bright_red(img, 55, 120, 78, 145)
    _bright_red(img, 122, 120, 145, 145)

    assert brake_score(img, _VEHICLE_BBOX) > 0.5


def test_brake_light_false_positives_are_suppressed():
    single_lamp = _blank((40, 40, 40))
    _bright_red(single_lamp, 55, 120, 78, 145)

    assert brake_score(single_lamp, _VEHICLE_BBOX) < 0.3
    assert brake_score(_blank((0, 0, 255)), _VEHICLE_BBOX) < 0.3
    assert brake_score(None, _VEHICLE_BBOX) == 0.0


def test_traffic_light_primary_colours_classify():
    for color, expected in (((0, 0, 255), "red"), ((0, 255, 0), "green"), ((0, 255, 255), "yellow")):
        state, confidence = classify_light_state(_light_img(color), _LIGHT_BBOX)
        assert state == expected
        assert confidence > 0.0


def test_traffic_light_dark_frame_is_unknown():
    assert classify_light_state(np.zeros((120, 120, 3), np.uint8), _LIGHT_BBOX) == ("unknown", 0.0)


def test_frame_light_state_picks_largest_light_detection():
    img = _light_img((0, 0, 255))
    small = Detection(bbox=(5, 5, 12, 18), class_name="traffic_light", confidence=0.5)
    big = Detection(bbox=_LIGHT_BBOX, class_name="traffic_light", confidence=0.5)

    assert frame_light_state(img, [small, big])[0] == "red"
