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


def test_video_risk_analysis_player_controls_are_present():
    index = read_static("index.html")
    controls = read_static("js/controls.js")
    timeline_css = read_static("css/timeline.css")
    preview_css = read_static("css/preview.css")

    for element_id in (
        "playback-mode-normal",
        "playback-mode-risk",
        "playback-mode-pause-risk",
        "player-settings-button",
        "player-settings-menu",
        "export-snapshot",
        "export-evidence-json",
        "export-evidence-image",
        "player-risk-segments",
    ):
        assert f'id="{element_id}"' in index

    for contract in (
        'playbackMode: "normal"',
        "function riskSegmentsFromTimelineRows",
        "function exportSnapshot",
        "function exportEvidenceJson",
        "function exportEvidenceImage",
        "canvas.toBlob",
        "selectedEvidenceSource",
    ):
        assert contract in controls

    assert ".timeline-risk-segment.seg-danger" in timeline_css
    assert ".heat-cell.heat-danger" in timeline_css
    assert ".player-settings" in preview_css
    assert ".settings-menu" in preview_css
    assert ".mode-picker-menu" in preview_css
    assert ".player-risk-segments" in preview_css
    assert ".side-bar-btn:disabled" in preview_css


def test_live_risk_overlay_and_player_polish_present():
    index = read_static("index.html")
    controls = read_static("js/controls.js")
    preview_css = read_static("css/preview.css")

    # New DOM: canvas overlay layer + player-bar controls.
    for element_id in (
        "visual-overlay",
        "frame-step-back",
        "frame-step-forward",
        "player-settings-button",
        "overlay-toggle",
        "replay-event-btn",
        "playback-mode-slow-risk",
        "volume-btn",
        "volume-slider",
        "time-readout",
        "event-prev",
        "event-next",
        "seek-tooltip",
        "shortcuts-modal",
        "open-shortcuts",
        "timeline-heat",
    ):
        assert f'id="{element_id}"' in index

    # The canvas overlay is driven from the per-frame bbox/lane metadata that
    # the backend now serializes, synced to playback.
    for contract in (
        "function drawOverlay",
        "function containRect",
        "function resizeOverlayCanvas",
        "function stepFrame",
        "laneGeometry",
        "obj.bbox",
        "ResizeObserver",
        '"slow-risk"',
        "function toggleReplayLoop",
        "requestAnimationFrame(overlayLoop)",
        "function toggleMute",
        "function renderTimeReadout",
        'timeDisplayMode',
        "function jumpToEvent",
        "function showSeekTooltipAt",
        "function openShortcuts",
        "function renderTimelineHeat",
        "function seekFromRail",
        "formatAxisTime",
    ):
        assert contract in controls

    assert ".preview-canvas" in preview_css


def test_banner_metrics_have_no_duplicated_data():
    """Each datum appears exactly once: measurements, score contributors,
    multipliers, and advanced diagnostics never repeat the same value."""
    index = read_static("index.html")
    controls = read_static("js/controls.js")

    # The old ambiguous 6-box grid is gone for good.
    assert "modern-metrics-grid" not in index
    for dead_id in ("risk-object", "risk-motion", "risk-approach-speed", "risk-confidence"):
        assert f'id="{dead_id}"' not in index

    # Collision ETA builder: its two longitudinal inputs, each once. Lane is NOT
    # a measurement box anymore (it lives in the hero subtitle + lane relevance),
    # and the old Measurements strip is gone.
    assert "Measurements" not in index
    assert 'id="risk-lane"' not in index
    eta_start = index.index('What builds the Collision ETA')
    eta_end = index.index("<!-- SUB PANEL 2", eta_start)
    eta = index[eta_start:eta_end]
    for label in ("Distance", "Closing speed"):
        assert label in eta
    for element_id in ("risk-distance", "risk-approach"):
        assert f'id="{element_id}"' in eta

    # Risk Score contributors mirror score_raw: four weighted bars (additive),
    # then lane relevance + confidence as two multiplier bars (no × prefix, but
    # grouped under the dashed divider). crossing keeps no "signal-crossing" id.
    for bar_id in ("signal-eta", "signal-near", "signal-closing", "signal-brake"):
        assert f'id="{bar_id}"' in index
    assert 'id="signal-crossing"' not in index
    for weight in ("40%", "30%", "25%", "5%"):
        assert f'<span class="signal-weight">{weight}</span>' in index
    assert 'id="signal-relevance"' in index
    assert 'id="signal-confidence"' in index
    assert 'id="mult-relevance"' not in index

    # Detail Mode is the full diagnostic report: raw pipeline inputs plus fusion outputs,
    # intentionally repeating values shown elsewhere so the panel is self-contained.
    for kept_ev in (
        "ev-detector-class", "ev-detector-conf", "ev-tracking-conf",
        "ev-depth-distance", "ev-depth-closing", "ev-depth-conf",
        "ev-flow-expansion", "ev-flow-radial",
        "ev-lane-bucket", "ev-lane-pos", "ev-lane-crossing",
        "ev-fusion-eta", "ev-fusion-score", "ev-fusion-state",
        "ev-fusion-approach", "ev-fusion-brake",
        "ev-fusion-confidence",
    ):
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
    assert 'objectsMenuCollapsed: false' in controls
    assert 'state.objectsMenuCollapsed = true' in controls
    assert 'list.hidden = state.uiMode !== "objects" || state.objectsMenuCollapsed || !objects.length' in controls
    assert "No objects in this frame" not in controls
    assert ".objects-menu" in risk_css
    assert ".object-selector-list" not in risk_css
