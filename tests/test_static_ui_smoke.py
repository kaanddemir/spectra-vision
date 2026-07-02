from pathlib import Path
import re


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
        'name="start_frame"',
        'name="end_frame"',
        'data-window-panel="frames"',
        'data-window-panel="time"',
        'data-frame-preset="balanced"',
        'data-sampling-preset="balanced"',
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


def test_main_css_imports_existing_stylesheets():
    main_css = _read("css/main.css")
    imports = re.findall(r'@import\s+"\.\/([^"]+)"', main_css)

    assert imports
    for relative_path in imports:
        assert (STATIC / "css" / relative_path).exists()


def test_controls_literal_id_references_exist_in_index():
    index = _read("index.html")
    controls = _read("js/controls.js")

    index_ids = set(re.findall(r'id="([^"]+)"', index))
    js_ids = set(re.findall(r'byId\("([^"]+)"\)', controls))
    js_ids.update(re.findall(r'querySelector\("#([^"]+)"\)', controls))

    assert js_ids
    assert js_ids - index_ids == set()


def test_how_it_works_interaction_contract_is_present():
    page = _read("how-it-works.html")
    script = _read("js/how-it-works.js")

    for required in (
        "data-doc-menu-toggle",
        "data-doc-info-toggle",
        "data-doc-open",
        "data-doc-close",
        'id="doc-topic-menu-list"',
        'id="doc-color-info-panel"',
        'class="doc-modal"',
    ):
        assert required in page

    for required in (
        "[data-doc-menu-toggle]",
        "[data-doc-info-toggle]",
        "[data-doc-open]",
        "[data-doc-close]",
        "doc-modal-open",
    ):
        assert required in script
