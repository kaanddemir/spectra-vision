"""Brake-light cue: bright symmetric red pair scores high; paint/neutral don't."""

import numpy as np

from spectra.vision.brake_lights import brake_score

_BBOX = (50, 50, 150, 150)  # 100x100; lamp band is the lower ~55%


def _blank(color=(0, 0, 0)):
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:] = color
    return img


def _bright_red(img, x1, y1, x2, y2):
    img[y1:y2, x1:x2] = (0, 0, 255)  # BGR red, full saturation/value


def test_lit_pair_scores_high():
    img = _blank((40, 40, 40))
    # Two bright-red lamps in the lower-left and lower-right of the bbox band.
    _bright_red(img, 55, 120, 78, 145)
    _bright_red(img, 122, 120, 145, 145)
    assert brake_score(img, _BBOX) > 0.5


def test_neutral_vehicle_scores_zero():
    assert brake_score(_blank((90, 90, 90)), _BBOX) == 0.0


def test_single_lamp_is_not_a_pair():
    img = _blank((40, 40, 40))
    _bright_red(img, 55, 120, 78, 145)  # only left side lit
    # Symmetric-minimum logic means one side alone should not score high.
    assert brake_score(img, _BBOX) < 0.3


def test_full_red_bodywork_is_suppressed():
    # An entirely red vehicle must not read as braking (locality suppression).
    img = _blank((0, 0, 255))
    assert brake_score(img, _BBOX) < 0.3


def test_none_frame_and_tiny_bbox():
    assert brake_score(None, _BBOX) == 0.0
    assert brake_score(_blank((0, 0, 255)), (10, 10, 18, 18)) == 0.0
