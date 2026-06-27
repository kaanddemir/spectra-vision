"""Static contract tests for the browser UI.

These tests intentionally inspect HTML/CSS/JS as text because the project does
not use a frontend test runner. They protect the settings UI and recent
conservative cleanup from accidental regressions.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "spectra" / "web" / "static"


def read_static(relative_path: str) -> str:
    return (STATIC / relative_path).read_text(encoding="utf-8")


def test_analysis_settings_exposes_supported_api_controls():
    index = read_static("index.html")

    for hidden in (
        'name="max_processed_frames" value="180"',
        'name="max_saved_events" value="20"',
        'name="resize_max_side" value="512"',
        'name="depth_every" value="10"',
        'name="adaptive_depth" value="1"',
        'name="detect_every" value="3"',
        'name="lane_every" value="3"',
        'name="flow_every" value="1"',
        'name="start_sec" value="0"',
        'name="end_sec" value="0"',
    ):
        assert hidden in index

    for visible_control in (
        "Analysis Scope",
        'data-window-mode="frames"',
        'data-window-mode="time"',
        "Advanced Sampling",
        'id="reset-advanced-sampling"',
        'id="max-saved-events-input"',
        'data-param="resize_max_side"',
        'data-value="128">128</button>',
        'data-param="depth_every"',
        'data-value="15">15x</button>',
        'data-param="adaptive_depth"',
        'data-value="0">Off</button>',
        'data-value="1">On</button>',
        'data-param="detect_every"',
        'data-param="lane_every"',
        'data-param="flow_every"',
        "Depth Anything V2 Metric",
        "Estimated metric distance",
    ):
        assert visible_control in index


def test_video_view_toggle_cleanup_does_not_regress():
    index = read_static("index.html")
    controls = read_static("js/controls.js")
    preview_css = read_static("css/preview.css")
    combined = "\n".join((index, controls, preview_css))

    for removed in (
        'data-view="video"',
        'viewMode === "video"',
        "side-bar-sep",
        "refreshEmptyStates(true)",
        "refreshEmptyStates(false)",
    ):
        assert removed not in combined


def test_active_ui_contracts_are_kept():
    index = read_static("index.html")
    controls = read_static("js/controls.js")
    drawers_css = read_static("css/drawers.css")

    object_tracking_heading = index.index("<h5>Object Tracking</h5>")
    assert '<symbol id="icon-car"' in index
    assert index.rfind('<use href="#icon-car"></use>', 0, object_tracking_heading) != -1

    assert 'id="reset-advanced-sampling"' in index
    assert "resetAdvancedSampling" in controls
    assert 'setSegmentedValue("adaptive_depth", 1)' in controls
    assert ".drawer-icon-btn" in drawers_css


def test_timeline_uses_backend_risk_score():
    controls = read_static("js/controls.js")
    score_fn_start = controls.index("const eventSeverityScore = (ev) => {")
    score_fn_end = controls.index("  const eventStateClass", score_fn_start)
    score_fn = controls[score_fn_start:score_fn_end]

    assert "riskScore" in score_fn
    assert "primaryRiskScore" in score_fn
    assert "etaSeconds" not in score_fn
    assert "riskFactors" not in score_fn


def test_banner_metrics_are_unambiguous():
    index = read_static("index.html")
    controls = read_static("js/controls.js")
    risk_css = read_static("css/risk.css")

    grid_start = index.index('<div class="modern-metrics-grid">')
    grid_end = index.index("</div>\n            </div>", grid_start)
    metrics_grid = index[grid_start:grid_end]

    for label in ("Primary Object", "Lane", "Distance", "Motion", "Approach Speed", "Confidence"):
        assert label in metrics_grid
    assert '<span class="metric-lbl">Closing</span>' not in metrics_grid

    for element_id in (
        "risk-object",
        "risk-lane",
        "risk-distance",
        "risk-motion",
        "risk-approach-speed",
        "risk-confidence",
    ):
        assert f'id="{element_id}"' in metrics_grid

    motion_start = controls.index("const motionLabel = (value) => {")
    motion_end = controls.index("  const approachSpeedLabel", motion_start)
    motion_fn = controls[motion_start:motion_end]
    assert "Closing" in motion_fn
    assert "Estimating" not in motion_fn
    assert "MISSING" in motion_fn
    assert "m/s" not in motion_fn

    speed_start = controls.index("const approachSpeedLabel = (value) => {")
    speed_end = controls.index("\n\n  function focusSummaryFrame", speed_start)
    speed_fn = controls[speed_start:speed_end]
    assert "m/s" in speed_fn
    assert "MISSING" in speed_fn
    assert "repeat(auto-fit, minmax(150px, 1fr))" in risk_css


def test_risk_panel_has_objects_tab_and_unified_object_inspector():
    index = read_static("index.html")
    controls = read_static("js/controls.js")
    risk_css = read_static("css/risk.css")

    for button_id in ("toggle-mode-live", "toggle-mode-summary", "toggle-mode-objects"):
        assert f'id="{button_id}"' in index
    assert "Objects" in index
    assert "Active Objects" not in index
    assert "object-inspector-panel" in index
    assert "objects-menu" in index
    assert "No objects in this frame" not in index
    assert "risk-factors-title" in index
    inspector_start = index.index('id="object-inspector-panel"')
    inspector_end = index.index("<!-- SUB PANEL 3: Chart -->", inspector_start)
    inspector_markup = index[inspector_start:inspector_end]
    assert "risk-sub-panel-head" not in inspector_markup

    assert 'state.uiMode === "objects"' in controls
    assert 'setUiMode("objects"' in controls
    assert 'byId("toggle-mode-objects")' in controls
    assert 'byId("objects-menu")' in controls
    assert "highestRiskObject(objects)" in controls
    assert 'list.hidden = state.uiMode !== "objects" || !objects.length' in controls
    assert "No objects in this frame" not in controls
    assert ".objects-menu" in risk_css
    assert ".object-selector-list" not in risk_css
