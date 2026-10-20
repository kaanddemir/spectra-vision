import pytest

from tests.conftest import requests_real_video


@pytest.mark.parametrize("expression,expected", [
    ("", False), ("not real_video", False), ("not slow and not model", False),
    ("real_video", True), ("real_video and not model", True),
    ("not (real_video or model)", False), ("not (not real_video)", True),
])
def test_real_video_requires_positive_marker_selection(expression, expected):
    assert requests_real_video(expression) is expected
