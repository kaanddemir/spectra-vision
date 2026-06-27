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


def test_banner_metrics_have_no_duplicated_data():
    """Each datum appears exactly once: measurements, score contributors,
    multipliers, and advanced diagnostics never repeat the same value."""
    index = read_static("index.html")
    controls = read_static("js/controls.js")

    # The old ambiguous 6-box grid is gone for good.
    assert "modern-metrics-grid" not in index
    for dead_id in ("risk-object", "risk-motion", "risk-approach-speed", "risk-confidence"):
        assert f'id="{dead_id}"' not in index

    # Measurements strip: physical readings only, each once. Confidence is NOT here.
    facts_start = index.index('<span class="section-cap">Measurements</span>')
    facts_end = index.index("<!-- SUB PANEL 2", facts_start)
    facts = index[facts_start:facts_end]
    for label in ("Distance", "Closing speed", "Lane"):
        assert label in facts
    assert "Confidence" not in facts
    for element_id in ("risk-distance", "risk-approach", "risk-lane"):
        assert f'id="{element_id}"' in facts

    # Risk Score contributors mirror score_raw: four weighted bars; crossing is
    # a multiplier (not a bar) and confidence is a multiplier (not a fact box).
    for bar_id in ("signal-eta", "signal-near", "signal-closing", "signal-brake"):
        assert f'id="{bar_id}"' in index
    assert 'id="signal-crossing"' not in index
    for weight in ("40%", "30%", "25%", "5%"):
        assert f'<span class="signal-weight">{weight}</span>' in index
    assert 'id="mult-relevance"' in index
    assert 'id="mult-confidence"' in index

    # Advanced section holds only non-duplicated diagnostics (no distance /
    # closing / crossing / confidence repeats).
    for removed_ev in ("ev-depth-distance", "ev-depth-closing", "ev-lane-position", "ev-conf-detection"):
        assert f'id="{removed_ev}"' not in index
    for kept_ev in ("ev-flow-expansion", "ev-flow-radial", "ev-depth-status"):
        assert f'id="{kept_ev}"' in index

    # ETA pressure is derived client-side from the collision ETA seconds.
    assert "const etaPressure" in controls
    assert 'setSignalBar("eta", etaPressure(' in controls


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
