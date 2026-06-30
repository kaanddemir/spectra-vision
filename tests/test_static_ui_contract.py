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

    assert '<symbol id="icon-help"' in index
    assert '<use href="#icon-help"></use>' in index

    assert 'id="reset-advanced-sampling"' in index
    assert "resetAdvancedSampling" in controls
    assert 'setSegmentedValue("adaptive_depth", 1)' in controls
    assert ".drawer-icon-btn" in drawers_css


def test_how_it_works_is_a_standalone_routed_page():
    index = read_static("index.html")
    controls = read_static("js/controls.js")
    page = read_static("how-it-works.html")
    page_css = read_static("css/how-it-works.css")
    page_js = read_static("js/how-it-works.js")

    # The in-app help modal moved out to /how-it-works: the header button is now
    # a plain link, and the modal markup + its JS are gone from the app shell.
    assert 'href="/how-it-works"' in index
    assert 'id="help-modal"' not in index
    assert "data-help-page" not in index
    for removed_js in ("const HELP_PAGES = {", "function showHelpPage", "function openHelpModal"):
        assert removed_js not in controls

    # The landing carries the pipeline map (only) plus its own popup driver.
    for flow_term in (
        "flow-board--vertical",
        "pipeline-group",
        "pipeline-steps",
        "flow-node--link",
        "Frame Ingest",
        "Preprocess",
        "Lane Pipeline",
        "UFLDv2 Lane Detection",
        "Lane Frame",
        "Vanishing Point",
        "Lane Kalman Smoothing / Coasting",
        "Lane Confidence Smoothing",
        "Motion + Depth",
        "Motion + Depth Models",
        "OpenCV DIS Optical Flow",
        "Depth Anything V2 Metric VKITTI",
        "Optical Flow",
        "Quick Motion Risk",
        "Depth Estimation",
        "Near Map",
        "Adaptive Depth Refresh",
        "Detection + Tracking",
        "YOLO Detection",
        "Traffic Light Split",
        "Traffic Light Classification",
        "Lane-Relevance Filter",
        "Tracking",
        "IoU Tracker Update",
        "Track Propagation",
        "Per-Object Risk Evaluation",
        "Depth / Kalman Distance",
        "BBox Expansion",
        "Flow TTC",
        "Lane Crossing Risk",
        "Brake Light Cue",
        "Collision ETA",
        "ETA Pressure",
        "Proximity",
        "Approach",
        "Proximity Score",
        "Approach Score",
        "Confidence Gate",
        "Lane Relevance",
        "Collision Object Set",
        "Road-Class Filter",
        "Lane Line Detection",
        "Lane Corridor Build",
        "Lane Position",
        "Corridor Overlap",
        "Cut-In Motion",
        "TTC Fusion",
        "Confidence Smoothing",
        "Risk Score",
        "Decision + Delivery",
        "Primary Object Selection",
        "State Stabilization",
        "Frame Row",
        "Saved Event Pipeline",
        "Render Overlay Images",
        "Peak Event",
        "Performance Summary",
        "Per-Frame Schedule",
        "Saved Events + Diagnostics",
    ):
        assert flow_term in page
    assert 'src="/static/js/how-it-works.js"' in page
    assert 'href="/"' in page  # back-to-app link

    # Each page in the menu mirrors a top-level pipeline group and has a
    # matching full-screen popup.
    for topic in (
        "input",
        "lane-pipeline",
        "motion-depth",
        "detection-pipeline",
        "per-object-risk",
        "decision-delivery",
        "performance-summary",
        "per-frame",
    ):
        assert f'data-doc-open="{topic}"' in page
        assert f'id="doc-modal-{topic}"' in page
    assert 'data-doc-menu-toggle' in page
    assert ">Pages<" in page
    assert 'id="doc-topic-menu-list"' in page
    assert 'data-doc-info-toggle' in page
    assert 'id="doc-color-info-panel"' in page
    assert ">Color key<" in page
    # The "Card style" legend was removed from the info panel; the per-popup
    # style keys still document Dashed/Brighter inside the modals.
    assert ">Card style<" not in page
    assert ">Dashed<" in page and ">Advisory/context only<" in page
    assert ">Brighter<" in page and ">Used downstream<" in page
    assert "Internal lane step" in page
    assert "doc-modal-info__btn" in page and "Lane card style info" in page
    assert "Motion card style info" in page
    assert "Detection card style info" in page
    assert 'data-info-module="detection"' in page
    assert "Risk card style info" in page
    assert 'data-info-module="risk"' in page
    assert "Internal risk step" in page
    assert "Decision card style info" in page
    assert 'data-info-module="decision"' in page
    assert "Internal decision step" in page
    assert 'data-doc-open="tracking"' not in page
    assert 'id="doc-modal-tracking"' not in page
    # The map's Detection + Tracking group is reduced to the YOLO model card
    # only (like the other model groups); the internal steps live in the popup.
    assert 'pipeline-group pipeline-group--single">\n          <h3>Detection + Tracking</h3>' in page
    assert 'data-doc-open="detection-pipeline" data-module="detection">\n              <h5>YOLO Detection</h5>' in page
    assert 'data-doc-open="detection-pipeline" data-module="detection">\n              <h5>Traffic Light Split</h5>' not in page
    assert 'data-doc-open="detection-pipeline" data-module="tracking">\n              <h5>IoU Tracker Update</h5>' not in page
    # Those steps remain documented inside the detail popup.
    assert 'flow-node--advisory" data-module="detection" data-route="Feeds Traffic Light Classification only; it does not become a tracked collision object."><h5>Traffic Light Split</h5>' in page
    assert 'data-route="Feeds stable track_id, object age, history and distance smoothing."><h5>IoU Tracker Update</h5>' in page

    # The map's Per-Object Risk group is reduced to its two headline outputs;
    # the cues, signal terms and gates live in the popup with risk colouring.
    assert 'data-doc-open="per-object-risk" data-module="risk">\n              <h5>Collision ETA</h5>' in page
    assert 'data-doc-open="per-object-risk" data-module="risk">\n              <h5>Risk Score</h5>' in page
    assert 'data-doc-open="per-object-risk" data-module="risk">\n              <h5>BBox Expansion</h5>' not in page
    assert 'data-doc-open="per-object-risk" data-module="risk">\n              <h5>TTC Fusion</h5>' not in page
    # Popup cards are coloured (data-module) and tag internal vs hub, with formulas.
    assert 'data-module="risk" data-route="Feeds TTC Fusion as the depth/kinematic cue and Collision ETA."><h5>Depth / Kalman Distance</h5>' in page
    assert 'flow-node--hub" data-module="risk" data-route="Feeds ETA Pressure, imminent-TTC escalation and the TTC shown in the UI."><h5>TTC Fusion</h5>' in page
    # Decision + Delivery is its own section, not duplicated inside the risk popup.
    assert 'data-module="decision" data-route="Feeds State From Score with' not in page
    # Formulas are split into separate labelled blocks and use the × sign.
    assert "doc-formula-set" in page
    for label in ("Time-to-collision", "Risk score", "State banding"):
        assert f'<div class="doc-formula__label">{label}</div>' in page
    assert "<b>eta_pressure</b>(ttc) = clamp((3 − ttc) / 3, 0, 1)" in page
    assert "<b>risk.score</b> = gate × relevance × signal" in page
    assert "·" not in page  # multiplication uses × everywhere now

    # The map's Decision + Delivery group is reduced to its two delivered
    # outputs; the steps and rules live in the popup with decision colouring.
    assert 'data-doc-open="decision-delivery" data-module="decision">\n              <h5>Frame Row</h5>' in page
    assert 'data-doc-open="decision-delivery" data-module="decision">\n              <h5>Peak Event</h5>' in page
    assert 'data-doc-open="decision-delivery" data-module="decision">\n              <h5>Primary Object Selection</h5>' not in page
    assert 'data-module="decision" data-route="Feeds the Frame Row and risk banner with the stabilized state."><h5>State Stabilization</h5>' in page
    assert 'flow-node--hub" data-module="decision" data-route="Feeds the summary view as the headline event of the analysis."><h5>Peak Event</h5>' in page
    for label in ("Primary object", "State stabilization", "Saved events"):
        assert f'<div class="doc-formula__label">{label}</div>' in page

    assert 'data-route="Feeds Depth / Kalman Distance, BBox Expansion, Flow TTC and Lane Crossing Risk."><h5>Tracked Object Stream</h5>' in page
    assert 'data-module="input" data-route="Feeds Preprocess with the raw source frame."><h5>Frame Ingest</h5>' in page
    assert "Input card style info" in page
    assert 'data-info-module="input"' in page
    assert "Internal input step" in page
    assert 'flow-node--hub" data-module="input" data-route="Feeds the Depth Anything V2 metric depth model."><h5>RGB View</h5>' in page
    assert 'data-module="lane" data-route="Feeds Lane Corridor Build."><h5>Lane Line Detection</h5>' in page
    assert 'flow-node--hub" data-module="lane" data-route=' in page
    for downstream_title in (
        "Lane Frame",
        "Vanishing Point",
        "Lane Confidence Smoothing",
        "Lane Position",
        "Cut-In Motion",
        "Lane Crossing Risk",
        "Lane Relevance",
    ):
        assert f"<h5>{downstream_title}</h5>" in page
    assert 'data-route="Feeds Flow TTC and far-lane reliability checks."' in page
    assert 'data-route="Feeds Risk Score as the multiplicative lane gate."' in page
    assert 'data-info-module="motion-depth"' in page
    assert 'data-route="Feeds Optical Flow and downstream Flow TTC."><h5>OpenCV DIS Optical Flow</h5>' in page
    assert 'data-route="Feeds Depth Estimation, metric distance and downstream Collision ETA."><h5>Depth Anything V2 Metric VKITTI</h5>' in page
    assert '<h5>OpenCV DIS Optical Flow</h5>\n              <p>Computes dense optical flow with ego-motion compensation.</p>' in page
    assert '<h5>Depth Anything V2 Metric VKITTI</h5>\n              <p>Produces metric depth for distance, nearness and collision ETA.</p>' in page
    assert 'data-module="motion-depth" data-route="Feeds Quick Motion Risk, Flow TTC, Approach Score and flow confidence."><h5>Optical Flow</h5>' in page
    assert 'data-route="Feeds Lane-Relevance Filter as the near-object map."><h5>Near Map</h5>' in page
    assert 'flow-node--hub" data-module="motion-depth" data-route="Feeds ETA Pressure, state escalation and the UI collision ETA display."><h5>Collision ETA</h5>' in page
    assert 'data-doc-output-toggle' not in page
    for module in (
        "input",
        "lane",
        "motion-depth",
        "detection",
        "risk",
        "decision",
        "performance",
    ):
        assert f'data-module="{module}"' in page
        assert f'data-legend-module="{module}"' in page
    # The tracking module folds into the detection colour now.
    assert 'data-legend-module="tracking"' not in page
    assert 'data-module="tracking"' not in page

    # Popups open/close via the standalone driver (no inline-section anchors).
    assert "data-doc-close" in page
    assert "openModal" in page_js and "closeModal" in page_js
    assert "setInfoOpen" in page_js and "doc-color-info-panel" in page_js
    assert "enhanceInfoTooltips" in page_js and "flow-info-icon" in page_js
    assert "flow-route-icon" in page_js and "dataset.route" in page_js
    assert "closeInfoTooltips" in page_js and "aria-expanded" in page_js

    # Output chip lists were removed from the landing and detail pages; this is
    # now a heading-based process map, not a variable/schema view.
    for removed_output_contract in (
        "flow-node__io",
        "flow-node__io-label",
        "flow-badge--out",
        ">Outputs<",
        "data-doc-modal-output-toggle",
        "doc-hide-outputs",
    ):
        assert removed_output_contract not in page
        assert removed_output_contract not in page_js

    # Page-specific styling for the topbar nav, grouped pipeline map, popups,
    # and title info tooltips.
    for style_contract in (
        ".doc-topbar",
        ".doc-icon-btn",
        ".doc-info-btn",
        ".doc-color-info",
        ".doc-color-row",
        ".doc-style-row",
        ".doc-style-swatch--advisory",
        ".doc-style-swatch--hub",
        ".doc-local-note",
        ".doc-local-note__item",
        ".doc-local-note .doc-style-swatch--hub",
        "64, 210, 148",
        ".doc-modal__actions",
        ".doc-modal-info__btn",
        ".doc-modal-info:hover .doc-local-note",
        '.doc-color-row[data-legend-module="detection"]',
        ".doc-menu-btn",
        ".doc-menu-btn.is-active",
        ".pipeline-group",
        ".pipeline-steps",
        ".pipeline-group--single .pipeline-steps",
        ".pipeline-group--hub",
        ".doc-modal__body > .pipeline-group",
        ".doc-section--map .flow-node",
        ".doc-section--map .flow-node[data-module]",
        ".doc-modal .flow-node[data-module]",
        '.doc-section--map .flow-node[data-module="input"]',
        '.doc-modal .flow-node[data-module="input"]',
        '.doc-section--map .flow-node[data-module="lane"]',
        '.doc-modal .flow-node[data-module="lane"]',
        '.doc-section--map .flow-node[data-module="motion-depth"]',
        '.doc-modal .flow-node[data-module="motion-depth"]',
        '.doc-modal-info[data-info-module="motion-depth"] .doc-local-note .doc-style-swatch--hub',
        '.doc-modal-info[data-info-module="detection"] .doc-local-note .doc-style-swatch--hub',
        '.doc-section--map .flow-node[data-module="detection"]',
        '.doc-modal .flow-node[data-module="detection"]',
        '.doc-section--map .flow-node[data-module="risk"]',
        '.doc-section--map .flow-node[data-module="decision"]',
        '.doc-section--map .flow-node[data-module="performance"]',
        ".doc-section--map .flow-node--hub[data-module]",
        ".doc-section--map .flow-node--advisory[data-module]",
        ".doc-modal .flow-node--advisory[data-module]",
        "border-style: dashed",
        "border-width: 1.5px",
        ".doc-section--map .flow-node:has(.flow-info-icon:hover)",
        ".doc-section--map .flow-node--link[data-module]:hover",
        "--module-rgb",
        "overflow: visible",
        ".doc-modal .flow-node",
        "min-height: 82px",
        ".flow-info-icon",
        ".flow-route-icon",
        ".flow-info-icon.is-open",
        ".flow-route-icon.is-open",
        ".flow-info-icon::after",
        ".flow-route-icon::after",
        "grid-template-rows: auto minmax(58px, auto) auto",
        "justify-items: center",
        "min-height: 148px",
        ".doc-modal",
        ".flow-node--link",
    ):
        assert style_contract in page_css

    # Shared flow primitives still live in the bundled stylesheet.
    responsive_css = read_static("css/responsive.css")
    for style_contract in (".flow-board", ".flow-node", ".flow-badge"):
        assert style_contract in responsive_css

    assert ".flow-io" not in responsive_css
    assert ".flow-quick-nav" not in responsive_css


def test_timeline_uses_backend_risk_score():
    controls = read_static("js/controls.js")
    score_fn_start = controls.index("const eventSeverityScore = (ev) => {")
    score_fn_end = controls.index("  const eventStateClass", score_fn_start)
    score_fn = controls[score_fn_start:score_fn_end]

    assert "riskScore" in score_fn
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
        # Detection + Tracking: detector outputs plus tracker-assigned id.
        "ev-detector-class", "ev-detector-conf", "ev-tracking-id",
        # Depth / Kinematics: proximity is a depth-derived measurement.
        "ev-depth-distance", "ev-depth-closing", "ev-depth-proximity", "ev-depth-conf",
        # Expansion: bbox growth plus radial optical-flow cue.
        "ev-expansion-rate", "ev-flow-radial", "ev-flow-conf",
        "ev-lane-bucket", "ev-lane-pos", "ev-lane-crossing", "ev-lane-conf",
        # Advisory: cues that do not gate collision logic.
        "ev-advisory-traffic", "ev-advisory-brake",
        # Fusion / Risk: fused outputs + the one genuinely-fused contributor.
        "ev-fusion-state", "ev-fusion-eta", "ev-fusion-eta-pressure",
        "ev-fusion-score", "ev-fusion-approach",
        "ev-fusion-agreement", "ev-fusion-confidence",
    ):
        assert f'id="{kept_ev}"' in index

    assert "Detection &amp; Tracking (IoU)" in index
    assert '<span class="evidence-group-lbl">Optical Flow (DIS)</span>' in index
    assert '<span class="evidence-group-lbl">Lane</span>' in index
    assert '<div class="evidence-row"><span>Expansion</span><span id="ev-expansion-rate">' in index
    assert "Expansion rate" not in index
    assert '"None"' not in controls
    assert '<span class="evidence-group-lbl">Detector (YOLO)</span>' not in index
    assert '<span class="evidence-group-lbl">Detection (YOLO) &amp; Tracking (IoU)</span>' not in index
    assert '<span class="evidence-group-lbl">Tracking (IoU)</span>' not in index
    assert '<span class="evidence-group-lbl">Lane (UFLDv2)</span>' not in index
    assert '<span class="evidence-group-lbl">Expansion (bbox)</span>' not in index
    assert '<span class="evidence-group-lbl">Flow (DIS)</span>' not in index

    detail_mode = index[index.index("<summary>Detail Mode</summary>"):index.index("<!-- SUB PANEL 3", index.index("<summary>Detail Mode</summary>"))]
    assert detail_mode.index("Detection &amp; Tracking (IoU)") < detail_mode.index("Optical Flow (DIS)")
    assert detail_mode.index("Optical Flow (DIS)") < detail_mode.index('<span class="evidence-group-lbl">Lane</span>')
    assert detail_mode.index('<span class="evidence-group-lbl">Lane</span>') < detail_mode.index("Depth / Kinematics")

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
