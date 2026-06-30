from pathlib import Path


STATIC = Path(__file__).resolve().parents[1] / "spectra" / "web" / "static"


def _read(relative_path):
    return (STATIC / relative_path).read_text(encoding="utf-8")


def test_index_exposes_analysis_controls_and_static_assets():
    index = _read("index.html")

    for required in (
        'name="max_processed_frames"',
        'name="max_saved_events"',
        'name="resize_max_side"',
        'name="depth_every"',
        'name="detect_every"',
        'name="lane_every"',
        'name="flow_every"',
        'src="/static/js/main.js"',
        'href="/static/css/main.css"',
    ):
        assert required in index


def test_how_it_works_is_routed_as_standalone_page():
    index = _read("index.html")
    page = _read("how-it-works.html")

    assert 'href="/how-it-works"' in index
    assert 'src="/static/js/how-it-works.js"' in page
    assert 'href="/"' in page
