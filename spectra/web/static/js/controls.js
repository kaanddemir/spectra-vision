export function initializeSpectra() {
  const MISSING = "-";


  const form = document.querySelector("#analysis-form");
  const fileInput = document.querySelector("#source-file");
  const runButton = document.querySelector("#run-analysis");
  const uploadButton = document.querySelector("#top-upload");
  const removeBtn = document.querySelector("#remove-source");
  const selectedChip = document.querySelector("#selected-chip");
  const settingsDrawer = document.querySelector("#settings-drawer");

  const previewVideo = document.querySelector("#visual-original-video");
  const previewBlend = document.querySelector("#visual-blend");
  const previewCanvas = document.querySelector("#visual-overlay");
  const overlayCtx = previewCanvas ? previewCanvas.getContext("2d") : null;
  const previewFrame = document.querySelector("#frame-original");
  const playToggle = document.querySelector("#play-toggle");
  const seekBar = document.querySelector("#seek-bar");
  const timeCurrent = document.querySelector("#time-current");
  const timeTotal = document.querySelector("#time-total");

  // CSS-side risk colours (mirror of overlay.py COLORS, BGR→hex).
  const RISK_COLORS = { DANGER: "#ff3b3b", CAUTION: "#ffb020", SAFE: "#12d492" };

  const state = {
    previewUrl: "",
    lastResult: null,
    currentResult: null,
    sourceMeta: null,
    analyzing: false,

    progressTimer: null,
    progressStart: 0,
    previewOverlayTimer: null,
    timelineRows: [],
    events: [],
    syncFollowVideo: true,
    previewWs: null,
    previewSessionId: null,
    livePreviewActive: false,
    liveTimelineRows: [],
    liveEvents: [],
    uiMode: "live",
    analysisWindowMode: "frames",
    // Time domain that timeline/chart/player render against. When an analysis
    // covers only part of the video (time window, or a capped frame budget),
    // everything is scaled to [startSec, endSec] so the analysis fills the
    // timeline and the player is scoped to that segment. endSec=null → fall
    // back to the full media duration.
    analysisWindow: { startSec: 0, endSec: null },
    selectedSummaryEvent: null,
    currentTimelineRow: null,
    suppressVideoSyncCount: 0,
    selectedObjectId: null,
    objectsMenuCollapsed: false,
    playbackMode: "normal",
    pauseRiskLastKey: null,
    overlayEnabled: true,
    playbackRate: 1,
    overlayRafId: null,
    loopSegment: null,
    timeDisplayMode: "time",
    lastVolume: 1,
  };

  const byId = (id) => document.getElementById(id);
  const formField = (name) => form.elements.namedItem(name);

  // ─── helpers ──────────────────────────────────────────────
  const isReal = (value) => {
    if (value === undefined || value === null) return false;
    if (typeof value === "string") {
      const t = value.trim();
      return t !== "" && t !== "—" && !/^n\/a$/i.test(t);
    }
    return true;
  };
  const num = (value, defaultValue = null) => {
    if (value === null || value === undefined || value === "") return defaultValue;
    const n = Number(value);
    return Number.isFinite(n) ? n : defaultValue;
  };
  const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
  const titleCase = (v) => isReal(v)
    ? String(v).replace(/[_-]+/g, " ").replace(/\w\S*/g, (w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    : null;

  const formatSeconds = (totalSec) => {
    const n = num(totalSec, null);
    if (n === null) return MISSING;
    const m = Math.floor(Math.max(0, n) / 60);
    const s = Math.floor(Math.max(0, n) % 60);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  };
  // Axis tick formatter: when the spacing between ticks is under a second
  // (short analysis windows), add a decimal so labels don't repeat (00:09 00:09).
  const formatAxisTime = (totalSec, stepSec) => {
    const n = num(totalSec, null);
    if (n === null) return MISSING;
    if (num(stepSec, 1) >= 1) return formatSeconds(n);
    const m = Math.floor(Math.max(0, n) / 60);
    const s = Math.max(0, n) % 60;
    return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
  };
  const etaDisplay = (eta) => {
    const display = eta?.display;
    if (!display) return MISSING;
    return ["Estimating", "No closing", "Low confidence"].includes(display) ? MISSING : display;
  };
  const etaSeconds = (eta) => num(eta?.sec, null);
  // ETA pressure mirrors the backend: (3 - ttc)/3 clamped, 0 when not closing.
  const etaPressure = (eta) => {
    const sec = etaSeconds(eta);
    return sec === null ? 0 : clamp((3 - sec) / 3, 0, 1);
  };
  const laneWithPosition = (lane, lanePosition) => {
    const laneText = lane ? shortLane(lane) : MISSING;
    const pos = num(lanePosition, null);
    return pos === null ? laneText : `${laneText} (${pos.toFixed(1)})`;
  };

  const mediaSrc = (value) => {
    if (!isReal(value)) return "";
    if (/^(data:|blob:|https?:)/i.test(value)) return value;
    return `data:image/png;base64,${value}`;
  };

  const roundedTimelineTime = (value) => {
    const t = num(value, null);
    return t === null ? null : Math.round(t * 100) / 100;
  };
  const timelineKey = (row) => {
    const fi = num(row?.frameIndex, null);
    if (fi !== null) return `f${fi}`;
    const t = roundedTimelineTime(row?.timeSec);
    return t === null ? "" : t.toFixed(2);
  };

  // Lift the primary object's fields onto the row so chart, banner and panel
  // consumers can read the selected object's ETA/risk data directly.
  const getPrimaryObject = (row) => {
    if (!row || !Array.isArray(row.objects)) return null;
    const id = row.primaryObjectId;
    if (id === null || id === undefined) return null;
    return row.objects.find((o) => o && o.objectId === id) || null;
  };

  // Translate one v5 object-centric metric into the flat field names the panels,
  // chart, banner, overlay and exports consume. This is the single migration
  // seam: the backend speaks v5 (``risk``/``eta``/``motion``/``lane``/
  // ``tracking`` groups), everything downstream keeps reading the flat keys.
  const flattenObject = (obj) => {
    if (!obj) return obj;
    const risk = obj.risk || {};
    const eta = obj.eta || {};
    const motion = obj.motion || {};
    const laneInfo = obj.lane || {};
    const conf = obj.confidence || {};
    const overall = num(conf.overall, null);
    return {
      ...obj,
      // Identity + state
      objectId: obj.tracking?.trackId ?? null,
      displayId: obj.id ?? null,
      objectType: obj.class ?? null,
      riskState: obj.state ?? null,
      rawRiskState: obj.state ?? null,
      riskScore: num(risk.score, null),
      // Flat signal blocks the panels/overlay/exports read directly.
      riskFactors: risk.factors || {},
      collisionEta: { display: eta.display ?? null, sec: num(eta.collisionSec, null) },
      kinematics: {
        distanceM: num(motion.distanceM, null),
        closingMps: num(motion.closingSpeedMps, null),
      },
      evidence: {
        flow: {
          expansionScore: num(motion.expansionScore, null),
          radialScore: num(motion.radialScore, null),
        },
      },
      confidence: conf,
      overallConfidence: overall,
      ttcAgreement: num(eta.agreement, null),
      lane: laneInfo.bucket ?? null,
      lanePosition: num(laneInfo.position, null),
      bbox: obj.bbox ?? null,
      confidencePct: overall === null ? null : Math.round(overall * 1000) / 10,
    };
  };

  const flattenFrame = (frame) => {
    if (!frame) return frame;
    const objs = Array.isArray(frame.objects) ? frame.objects.map(flattenObject) : [];
    const primaryInfo = frame.primary || {};
    const primaryId = primaryInfo.trackId ?? null;
    const primary = objs.find((o) => o && o.objectId === primaryId) || null;
    const ts = num(frame.timestampSec, null);
    const riskFactors = primary?.riskFactors || {};
    const kinematics = primary?.kinematics || {};
    return {
      ...frame,
      objects: objs,
      // Normalized ego-lane corridor polygon for the canvas overlay.
      laneGeometry: frame.laneGeometry ?? null,
      // Frame-level traffic-light advisory: collapse {state, confidence} to the
      // state string the renderers compare against.
      trafficLight: frame.trafficLight?.state ?? null,
      timeSec: ts === null ? null : Math.round(ts * 100) / 100,
      riskState: frame.state ?? null,
      riskScore: num(primaryInfo.score, null),
      primaryObjectId: primaryId,
      objectId: primaryId,
      displayId: primary?.displayId ?? primaryId,
      objectType: primary?.objectType ?? null,
      lane: primary ? (primaryInfo.lane ?? null) : null,
      collisionEta: primary?.collisionEta ?? null,
      riskFactors: primary?.riskFactors ?? null,
      kinematics: primary?.kinematics ?? null,
      evidence: primary?.evidence ?? null,
      confidence: primary?.confidence ?? null,
      ttcAgreement: primary?.ttcAgreement ?? null,
      proximityScore: riskFactors.proximity ?? null,
      approachScore: riskFactors.approach ?? null,
      crossingScore: riskFactors.crossing ?? null,
      distanceM: kinematics.distanceM ?? null,
      closingMps: kinematics.closingMps ?? null,
      lanePosition: primary?.lanePosition ?? null,
      confidencePct: primary ? (num(primary.overallConfidence, 0) * 100) : null,
    };
  };

  const flattenEvent = (event, imagesByRef) => {
    if (!event) return event;
    const flat = flattenFrame(event);
    const ref = event.imageRef;
    const images = ref && imagesByRef && imagesByRef[ref] ? imagesByRef[ref] : null;
    return {
      ...flat,
      riskScore: num(event.primary?.score, null) ?? flat.riskScore ?? null,
      images: images || {},
    };
  };

  const stateClass = (stateOrBand) => {
    const v = String(stateOrBand || "").toLowerCase();
    if (v === "danger") return "danger";
    if (v === "caution") return "caution";
    if (v === "safe") return "safe";
    return "none";
  };
  const eventSeverityScore = (ev) => {
    return num(ev?.riskScore, 0);
  };
  const eventStateClass = (ev) => stateClass(ev?.riskState);
  const isActionableObject = (obj) => {
    const sc = stateClass(obj?.riskState ?? obj?.rawRiskState);
    return sc === "caution" || sc === "danger";
  };
  const eventTimestamp = (ev) => num(ev?.timestampSec, null);
  const eventLane = (ev) => titleCase(ev?.lane);
  const eventItemsFromEvents = (events) => (Array.isArray(events) ? events : [])
    .map((ev, sourceIndex) => ({
      ev,
      index: sourceIndex,
      sc: eventStateClass(ev),
      ts: eventTimestamp(ev),
      eta: ev?.collisionEta || null,
    }))
    .filter((item) => item.ts !== null && (item.sc === "caution" || item.sc === "danger"));
  const timelineEventItems = (source = state) => {
    const events = Array.isArray(source) ? source : (source?.events || []);
    return eventItemsFromEvents(events)
      .sort((a, b) => a.ts - b.ts || eventSeverityScore(b.ev) - eventSeverityScore(a.ev))
      .map((item, displayIndex) => ({ ...item, displayIndex }));
  };
  const eventTooltip = (item) => {
    const parts = [`#${(item.displayIndex ?? item.index) + 1}`, formatSeconds(item.ts), item.sc.toUpperCase()];
    parts.push(`ETA ${etaDisplay(item.eta)}`);
    const lane = eventLane(item.ev);
    if (lane) parts.push(lane);
    return parts.join(" | ");
  };
  const timelineSeverity = (row) => {
    const sc = stateClass(row?.riskState);
    if (sc === "danger") return 2;
    if (sc === "caution") return 1;
    return 0;
  };
  const isRiskRow = (row) => {
    const sc = stateClass(row?.riskState);
    return sc === "caution" || sc === "danger";
  };
  const preferTimelineRow = (candidate, current) => {
    if (!current) return true;
    const candidateScore = eventSeverityScore(candidate);
    const currentScore = eventSeverityScore(current);
    if (candidateScore !== currentScore) return candidateScore > currentScore;

    const candidateSeverity = timelineSeverity(candidate);
    const currentSeverity = timelineSeverity(current);
    if (candidateSeverity !== currentSeverity) return candidateSeverity > currentSeverity;

    const candidateEta = etaSeconds(candidate?.collisionEta);
    const currentEta = etaSeconds(current?.collisionEta);
    if (candidateEta !== null && currentEta !== null && candidateEta !== currentEta) {
      return candidateEta < currentEta;
    }
    if (candidateEta !== null && currentEta === null) return true;
    if (candidateEta === null && currentEta !== null) return false;

    const candidateApproach = num(candidate?.riskFactors?.approach, null);
    const currentApproach = num(current?.riskFactors?.approach, null);
    if (candidateApproach !== null && currentApproach !== null && candidateApproach !== currentApproach) {
      return candidateApproach > currentApproach;
    }
    if (candidateApproach !== null && currentApproach === null) return true;
    return false;
  };
  const normalizeTimelineRowsForChart = (rows) => {
    const grouped = new Map();
    (Array.isArray(rows) ? rows : []).forEach((row) => {
      const timeSec = roundedTimelineTime(row?.timeSec);
      if (timeSec === null) return;
      const key = timeSec.toFixed(2);
      const normalizedRow = { ...row, timeSec };
      if (preferTimelineRow(normalizedRow, grouped.get(key))) {
        grouped.set(key, normalizedRow);
      }
    });
    return Array.from(grouped.values()).sort((a, b) => a.timeSec - b.timeSec);
  };
  const timelinePointTooltip = (point) => {
    const type = shortType(point.objectType);
    const displayId = point.displayId ?? point.objectId;
    const id = isReal(displayId) ? ` #${displayId}` : "";
    const object = type === MISSING && !id ? MISSING : `${type === MISSING ? "Object" : type}${id}`;
    return [
      `Time: ${point.timeSec.toFixed(2).replace(/\.?0+$/, "")}s`,
      `State: ${point.riskState}`,
      `ETA: ${etaDisplay(point.collisionEta)}`,
      `Object: ${object}`,
      `Lane: ${shortLane(point.lane)}`,
    ].join("\n");
  };
  const riskClass = (sb) => {
    const c = stateClass(sb);
    if (c === "none") return "risk-none";
    if (c === "danger") return "risk-high";
    if (c === "caution") return "risk-medium";
    return "risk-low";
  };

  function safeFileStem(value) {
    const raw = isReal(value) ? String(value) : "spectra";
    return raw.replace(/\.[^.]+$/, "").replace(/[^a-z0-9_-]+/gi, "_").replace(/^_+|_+$/g, "") || "spectra";
  }

  function downloadBlob(blob, filename) {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function downloadDataUrl(dataUrl, filename) {
    if (!dataUrl) return;
    try {
      const response = await fetch(dataUrl);
      downloadBlob(await response.blob(), filename);
    } catch {
      const a = document.createElement("a");
      a.href = dataUrl;
      a.download = filename;
      a.click();
    }
  }

  const shortLane = (lane) => {
    if (!isReal(lane)) return MISSING;
    const v = String(lane).toLowerCase();
    if (v.includes("left")) return "Left";
    if (v.includes("right")) return "Right";
    if (v.includes("center")) return "Center";
    return titleCase(lane) || MISSING;
  };
  const shortType = (type) => {
    if (!isReal(type)) return MISSING;
    const v = String(type).toLowerCase();
    if (v === "none") return MISSING;
    if (v.includes("left lane")) return "Left";
    if (v.includes("right lane")) return "Right";
    if (v.includes("center lane")) return "Central";
    return titleCase(type) || MISSING;
  };
  // ─── preview / video ──────────────────────────────────────
  function setPreviewMedia(source) {
    const resolved = mediaSrc(source);
    if (!resolved) {
      previewVideo.hidden = true;
      previewVideo.removeAttribute("src");
      previewFrame.classList.remove("has-media", "is-playing");
      timeCurrent.textContent = "00:00";
      timeTotal.textContent = "00:00";
      seekBar.value = 0;
      seekBar.style.setProperty("--fill", "0%");
      hidePreviewOverlay();
      if (previewCanvas) { previewCanvas.hidden = true; clearOverlayCanvas(); }
      refreshEmptyStates();
      updateEvidenceControls();
      return;
    }
    previewVideo.src = resolved;
    previewVideo.hidden = false;
    previewFrame.classList.add("has-media");
    // Hide all empty-state labels so the center play button becomes visible
    ["empty-video"].forEach(id => {
      const el = byId(id);
      if (el) el.hidden = true;
    });
    updateOverlayVisibility();
    updateEvidenceControls();
  }

  function refreshEmptyStates() {
    const vEmpty = byId("empty-video");
    if (!vEmpty) return;
    const hasVideo = !!(previewVideo.src && previewVideo.src !== location.href);

    vEmpty.hidden = hasVideo;
    previewVideo.hidden = !hasVideo;
    previewBlend.hidden = !(hasVideo && previewBlend.src);
  }

  function clearPreviewOverlayTimer() {
    if (!state.previewOverlayTimer) return;
    clearTimeout(state.previewOverlayTimer);
    state.previewOverlayTimer = null;
  }

  function showPreviewOverlay(source, hideAfterMs = null) {
    clearPreviewOverlayTimer();
    const resolved = mediaSrc(source);
    if (!resolved) { hidePreviewOverlay(); return; }
    previewBlend.src = resolved;
    previewBlend.hidden = false;
    const delay = num(hideAfterMs, null);
    if (delay !== null && delay > 0) {
      state.previewOverlayTimer = setTimeout(() => hidePreviewOverlay(), delay);
    }
  }
  function hidePreviewOverlay() {
    clearPreviewOverlayTimer();
    previewBlend.hidden = true;
    previewBlend.removeAttribute("src");
  }

  // ─── live risk overlay (canvas) ───────────────────────────
  // Compute the displayed video rect inside the container, honoring
  // object-fit: contain (letterboxing). bbox/lane coords are normalized to the
  // processed frame and share the source aspect ratio, so they map onto this
  // rect regardless of render size.
  function containRect(boxW, boxH) {
    const intrinsicW = previewVideo?.videoWidth || state.currentResult?.frameWidth || boxW;
    const intrinsicH = previewVideo?.videoHeight || state.currentResult?.frameHeight || boxH;
    if (!intrinsicW || !intrinsicH || !boxW || !boxH) return { x: 0, y: 0, w: boxW, h: boxH };
    const scale = Math.min(boxW / intrinsicW, boxH / intrinsicH);
    const w = intrinsicW * scale;
    const h = intrinsicH * scale;
    return { x: (boxW - w) / 2, y: (boxH - h) / 2, w, h };
  }

  function resizeOverlayCanvas() {
    if (!previewCanvas || !previewFrame) return;
    const dpr = window.devicePixelRatio || 1;
    const cw = previewFrame.clientWidth;
    const ch = previewFrame.clientHeight;
    previewCanvas.width = Math.max(1, Math.round(cw * dpr));
    previewCanvas.height = Math.max(1, Math.round(ch * dpr));
    previewCanvas.style.width = `${cw}px`;
    previewCanvas.style.height = `${ch}px`;
    if (overlayCtx) overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function clearOverlayCanvas() {
    if (!overlayCtx || !previewCanvas) return;
    const dpr = window.devicePixelRatio || 1;
    overlayCtx.clearRect(0, 0, previewCanvas.width / dpr, previewCanvas.height / dpr);
  }

  function overlayActive() {
    return state.overlayEnabled
      && !!previewVideo?.src
      && !previewVideo.hidden
      && !previewFrame.classList.contains("is-analyzing");
  }

  function updateOverlayVisibility() {
    if (!previewCanvas) return;
    const active = overlayActive();
    previewCanvas.hidden = !active;
    if (!active) clearOverlayCanvas();
  }

  function drawOverlay(row) {
    if (!overlayCtx || !previewCanvas) return;
    if (!overlayActive()) { clearOverlayCanvas(); return; }
    clearOverlayCanvas();
    if (!row) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = containRect(previewCanvas.width / dpr, previewCanvas.height / dpr);
    const px = (nx) => rect.x + nx * rect.w;
    const py = (ny) => rect.y + ny * rect.h;

    // Ego-lane corridor.
    const lane = row.laneGeometry;
    const corridor = Array.isArray(lane?.corridor) ? lane.corridor : [];
    if (corridor.length >= 3) {
      const stateColor = RISK_COLORS[row.riskState] || "#8aa0b4";
      overlayCtx.beginPath();
      corridor.forEach(([nx, ny], i) => {
        const x = px(nx), y = py(ny);
        if (i === 0) overlayCtx.moveTo(x, y); else overlayCtx.lineTo(x, y);
      });
      overlayCtx.closePath();
      overlayCtx.fillStyle = lane?.detected ? "rgba(120,190,185,0.10)" : "rgba(120,140,140,0.06)";
      overlayCtx.fill();
      overlayCtx.lineWidth = 1.5;
      overlayCtx.strokeStyle = lane?.detected ? "rgba(214,228,222,0.65)" : "rgba(150,165,160,0.4)";
      if (!lane?.detected) overlayCtx.setLineDash([10, 7]);
      overlayCtx.stroke();
      overlayCtx.setLineDash([]);
    }

    // Per-object boxes — skip SAFE to match the baked overlay.
    const objects = Array.isArray(row.objects) ? row.objects : [];
    objects.forEach((obj) => {
      const st = obj?.rawRiskState ?? obj?.riskState;
      if (!obj?.bbox || st === "SAFE" || !st) return;
      const [x1, y1, x2, y2] = obj.bbox;
      const bx = px(x1), by = py(y1), bw = px(x2) - px(x1), bh = py(y2) - py(y1);
      const color = RISK_COLORS[st] || "#8aa0b4";
      overlayCtx.lineWidth = st === "DANGER" ? 2.5 : 1.5;
      overlayCtx.strokeStyle = color;
      overlayCtx.strokeRect(bx, by, bw, bh);

      const parts = [];
      const did = obj.displayId ?? obj.objectId;
      if (did !== null && did !== undefined) parts.push(`#${did}`);
      if (obj.objectType) parts.push(String(obj.objectType).toUpperCase());
      const eta = num(obj.collisionEta?.sec, null);
      if (eta !== null) parts.push(`ETA ${eta.toFixed(1)}s`);
      if (num(obj.riskFactors?.brake, 0) >= 0.5) parts.push("BRAKE");
      const label = parts.join(" ");
      if (label) {
        overlayCtx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
        const tw = overlayCtx.measureText(label).width;
        const lh = 15;
        const ly = Math.max(rect.y, by - lh);
        overlayCtx.fillStyle = color;
        overlayCtx.fillRect(bx, ly, tw + 8, lh);
        overlayCtx.fillStyle = "#0f1218";
        overlayCtx.fillText(label, bx + 4, ly + lh - 4);
      }
    });
  }

  function redrawOverlayAtCurrentTime() {
    drawOverlay(findTimelineRowAt(previewVideo?.currentTime ?? 0));
  }

  function overlayLoop() {
    if (!previewVideo || previewVideo.paused || previewVideo.ended) {
      state.overlayRafId = null;
      return;
    }
    redrawOverlayAtCurrentTime();
    state.overlayRafId = requestAnimationFrame(overlayLoop);
  }
  function startOverlayLoop() {
    if (state.overlayRafId === null) state.overlayRafId = requestAnimationFrame(overlayLoop);
  }
  function stopOverlayLoop() {
    if (state.overlayRafId !== null) { cancelAnimationFrame(state.overlayRafId); state.overlayRafId = null; }
  }

  function toggleOverlay(force = null) {
    state.overlayEnabled = force === null ? !state.overlayEnabled : !!force;
    const btn = byId("overlay-toggle");
    if (btn) {
      btn.classList.toggle("is-active", state.overlayEnabled);
      btn.setAttribute("aria-pressed", state.overlayEnabled ? "true" : "false");
    }
    updateOverlayVisibility();
    if (state.overlayEnabled) redrawOverlayAtCurrentTime();
  }

  // ─── player polish: frame step, speed, replay ─────────────
  function togglePlay() {
    if (!previewVideo || previewVideo.hidden || !previewVideo.src) return;
    if (previewVideo.paused) playWithinWindow(); else previewVideo.pause();
  }

  function stepFrame(dir) {
    if (!previewVideo || previewVideo.hidden || !previewVideo.src) return;
    const fps = num(state.currentResult?.fps, 0) || 30;
    const { start, end } = winBounds();
    previewVideo.pause();
    const target = clamp(previewVideo.currentTime + (dir / fps), start, end);
    try { previewVideo.currentTime = target; } catch {}
  }

  // Jump to the previous/next CAUTION/DANGER event relative to the playhead.
  function jumpToEvent(dir) {
    if (!previewVideo || previewVideo.hidden || !previewVideo.src) return;
    const items = timelineEventItems();
    if (!items.length) return;
    const t = previewVideo.currentTime ?? 0;
    let target;
    if (dir > 0) {
      target = items.find((it) => it.ts > t + 0.05) || items[items.length - 1];
    } else {
      const earlier = items.filter((it) => it.ts < t - 0.05);
      target = earlier.length ? earlier[earlier.length - 1] : items[0];
    }
    if (!target) return;
    previewVideo.pause();
    state.suppressVideoSyncCount = 1;
    seekPreviewVideo(target.ts);
    applyTimelineStateAt(target.ts, { switchMode: true });
    setActiveEventIndex(target.index, { scroll: true });
  }

  function setPlaybackRate(rate) {
    state.playbackRate = rate;
    if (previewVideo && state.playbackMode !== "slow-risk") previewVideo.playbackRate = rate;
    document.querySelectorAll("[data-speed]").forEach((b) => {
      b.classList.toggle("is-active", parseFloat(b.dataset.speed) === rate);
    });
  }

  function currentRiskSegment() {
    const segments = riskSegmentsFromTimelineRows(state.timelineRows);
    if (!segments.length) return null;
    const t = previewVideo?.currentTime ?? 0;
    return segments.find((s) => t >= s.start - 0.25 && t < s.end + 0.25)
      || segments.find((s) => s.start >= t)
      || segments[0];
  }
  function updateReplayButton() {
    const btn = byId("replay-event-btn");
    if (btn) {
      const on = !!state.loopSegment;
      btn.classList.toggle("is-active", on);
      btn.setAttribute("aria-pressed", on ? "true" : "false");
    }
  }
  function toggleReplayLoop() {
    if (state.loopSegment) { state.loopSegment = null; updateReplayButton(); return; }
    const seg = currentRiskSegment();
    if (!seg) return;
    state.loopSegment = seg;
    updateReplayButton();
    seekPreviewVideo(seg.start);
    playWithinWindow();
  }

  // ─── volume ───────────────────────────────────────────────
  function updateVolumeUI() {
    const btn = byId("volume-btn");
    const slider = byId("volume-slider");
    if (!previewVideo) return;
    const muted = previewVideo.muted || previewVideo.volume === 0;
    if (btn) {
      btn.classList.toggle("is-muted", muted);
      btn.setAttribute("aria-pressed", muted ? "true" : "false");
    }
    if (slider) {
      const pct = Math.round((muted ? 0 : previewVideo.volume) * 100);
      slider.value = String(pct);
      slider.style.setProperty("--vfill", `${pct}%`);
    }
  }
  function setVolume(fraction) {
    if (!previewVideo) return;
    const v = clamp(fraction, 0, 1);
    previewVideo.volume = v;
    previewVideo.muted = v === 0;
    if (v > 0) state.lastVolume = v;
    updateVolumeUI();
  }
  function toggleMute() {
    if (!previewVideo) return;
    if (previewVideo.muted || previewVideo.volume === 0) {
      setVolume(state.lastVolume || 1);
    } else {
      previewVideo.muted = true;
      updateVolumeUI();
    }
  }

  // ─── time / frame readout ─────────────────────────────────
  function fpsForReadout() {
    return num(state.currentResult?.fps, 0) || 30;
  }
  function renderTimeReadout(timeSec) {
    const t = num(timeSec, 0) || 0;
    const dur = (previewVideo?.duration && Number.isFinite(previewVideo.duration))
      ? previewVideo.duration
      : (currentTotalDuration() || 0);
    if (state.timeDisplayMode === "frames") {
      const fps = fpsForReadout();
      timeCurrent.textContent = `#${Math.round(t * fps)}`;
      timeTotal.textContent = `#${Math.round(dur * fps)}`;
    } else {
      timeCurrent.textContent = formatSeconds(t);
      timeTotal.textContent = formatSeconds(dur);
    }
  }
  function toggleTimeDisplay() {
    state.timeDisplayMode = state.timeDisplayMode === "frames" ? "time" : "frames";
    renderTimeReadout(previewVideo?.currentTime ?? 0);
  }

  // ─── seek-bar hover preview ───────────────────────────────
  function nearestEventWithImageAt(t) {
    const items = timelineEventItems().filter((it) => it.ev?.images?.blend);
    if (!items.length) return null;
    let best = null, bestDiff = Infinity;
    items.forEach((it) => { const d = Math.abs(it.ts - t); if (d < bestDiff) { bestDiff = d; best = it; } });
    const tol = clamp(currentTotalDuration() * 0.03, 0.3, 1.0);
    return best && bestDiff <= tol ? best.ev : null;
  }
  function showSeekTooltipAt(clientX) {
    const track = seekBar?.closest(".seek-track");
    const tip = byId("seek-tooltip");
    if (!track || !tip || !state.timelineRows.length) return;
    const rect = track.getBoundingClientRect();
    const ratio = clamp((clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    const { start, span } = winBounds();
    const t = start + ratio * span;
    byId("seek-tooltip-time").textContent = formatSeconds(t);
    const ev = nearestEventWithImageAt(t);
    const thumb = byId("seek-tooltip-thumb");
    const blend = ev?.images?.blend;
    if (blend) { thumb.src = mediaSrc(blend); thumb.hidden = false; }
    else { thumb.hidden = true; thumb.removeAttribute("src"); }
    tip.hidden = false;
    const half = tip.offsetWidth / 2;
    tip.style.left = `${clamp(clientX - rect.left, half, rect.width - half)}px`;
  }
  function hideSeekTooltip() {
    const tip = byId("seek-tooltip");
    if (tip) tip.hidden = true;
  }

  // ─── keyboard shortcuts help ──────────────────────────────
  function openShortcuts() {
    const m = byId("shortcuts-modal");
    if (!m) return;
    m.hidden = false;
    void m.offsetHeight;
    m.classList.add("is-open");
  }
  function closeShortcuts() {
    const m = byId("shortcuts-modal");
    if (!m) return;
    m.classList.remove("is-open");
    setTimeout(() => { if (!m.classList.contains("is-open")) m.hidden = true; }, 400);
  }

  function imageSetForEvent(ev) {
    const eventImages = ev?.images || {};
    const resultImages = state.currentResult?.images || {};
    return {
      original: eventImages.original || resultImages.original,
      blend: eventImages.blend || resultImages.blend,
    };
  }

  // ─── normalize ────────────────────────────────────────────
  function cleanResponsePayload(response) {
    return response?.payload && typeof response.payload === "object" ? response.payload : (response || {});
  }

  function sameEvent(a, b) {
    if (!a || !b) return false;
    return a.frameIndex === b.frameIndex && num(a.timestampSec, null) === num(b.timestampSec, null);
  }

  function normalizePayload(response) {
    const payload = cleanResponsePayload(response);
    const metadata = payload?.metadata || {};
    const imagesByRef = (payload?.images && typeof payload.images === "object") ? payload.images : {};
    const peakEventRaw = payload?.peakEvent || null;
    const peakEvent = peakEventRaw ? flattenEvent(peakEventRaw, imagesByRef) : null;
    const peakImages = peakEvent?.images || {};

    const eventKey = (ev) => `${ev?.frameIndex ?? ""}:${num(ev?.timestampSec, null) ?? ""}`;
    const events = [];
    const seenEvents = new Set();
    if (peakEvent?.frameIndex !== undefined) {
      events.push(peakEvent);
      seenEvents.add(eventKey(peakEvent));
    }
    if (Array.isArray(payload?.events)) {
      payload.events.forEach((evRaw) => {
        const flat = flattenEvent(evRaw, imagesByRef);
        const key = eventKey(flat);
        if (seenEvents.has(key)) return;
        events.push(flat);
        seenEvents.add(key);
      });
    }

    const frames = Array.isArray(payload?.frames) ? payload.frames.map(flattenFrame) : [];

    return {
      images: {
        original: peakImages.original,
        blend: peakImages.blend,
      },
      imagesByRef,
      elapsedSec: num(metadata.elapsedSec, null),
      fps: num(metadata.fps, null),
      frameCount: num(metadata.frameCount, null),
      processedFrames: num(metadata.processedFrames, null),
      frameWidth: num(metadata.frameWidth, null),
      frameHeight: num(metadata.frameHeight, null),
      peakEvent,
      events,
      timelineRows: frames,
      sourceName: metadata.sourceName || fileInput.files[0]?.name || null,
    };
  }

  // ─── header chip ──────────────────────────────────────────
  function setSelectedFile(file) {
    const meta = byId("selected-meta");
    if (!file) {
      selectedChip.hidden = true;
      uploadButton.style.display = "";
      if (meta) {
        meta.innerHTML = `<div>Duration: -</div><div>Est. Frames: -</div>`;
        meta.hidden = false;
      }
      return;
    }
    selectedChip.hidden = false;
    uploadButton.style.display = "none";
    byId("selected-name").textContent = file.name;
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    byId("selected-name").title = `${file.name} (${sizeMb} MB)`;
    if (meta) { meta.hidden = false; }
  }
  function updateSourceMetaFromVideo() {
    if (!previewVideo || !previewVideo.duration || !Number.isFinite(previewVideo.duration)) return;
    const dur = formatSeconds(previewVideo.duration);
    const w = previewVideo.videoWidth, h = previewVideo.videoHeight;
    
    state.sourceMeta = { durationSec: previewVideo.duration, width: w, height: h };
    renderTimeReadout(previewVideo.currentTime || 0);

    // Estimate frames assuming 30fps — backend clamps to actual frame count.
    const estFrames = Math.ceil(previewVideo.duration * 30);
    const meta = byId("selected-meta");
    if (meta) {
      meta.innerHTML = `<div>Duration: ${dur}</div><div>Est. Frames: ${estFrames}</div>`;
      meta.hidden = false;
    }

    // Enable Max Processed Frames shortcuts
    const framesInput = byId("max-frames-input");
    if (framesInput) {
      framesInput.placeholder = String(estFrames);
      framesInput.max = String(estFrames);
      framesInput.disabled = false;
    }
    byId("frames-min").disabled = false;
    byId("frames-max").disabled = false;

    // Update temporal window constraints
    const startInp = byId("start-time-input");
    const endInp = byId("end-time-input");
    if (startInp) {
      startInp.max = String(previewVideo.duration);
      startInp.disabled = false;
    }
    if (endInp) {
      endInp.max = String(previewVideo.duration);
      endInp.placeholder = previewVideo.duration.toFixed(1);
      endInp.disabled = false;
    }
    applyAnalysisWindowMode();

    renderTimeline(null);
  }

  // ─── stat row ─────────────────────────────────────────────
  function renderStatRow(result) {
    const fps = result?.fps ?? null;
    const dur = state.sourceMeta?.durationSec ?? null;
    const totalFrames = result?.frameCount ?? (fps && dur ? Math.round(fps * dur) : null);
    
    if (totalFrames && dur) {
      const meta = byId("selected-meta");
      if (meta) {
        meta.innerHTML = `<div>Duration: ${formatSeconds(dur)}</div><div>Frames: ${totalFrames}</div>`;
        meta.hidden = false;
      }
    }
  }

  // ─── LIVE / SUMMARY risk panel ───────────────────────────
  function objectLabel(source) {
    if (!source) return MISSING;
    const type = shortType(source.objectType);
    const displayId = source.displayId ?? source.objectId;
    const id = isReal(displayId) ? ` #${displayId}` : "";
    return type === MISSING && !id ? MISSING : `${type === MISSING ? "Object" : type}${id}`;
  }

  const distanceLabel = (value) => {
    const n = num(value, null);
    return n === null ? MISSING : `${n.toFixed(n < 10 ? 1 : 0)}m`;
  };
  // Compact signed closing speed for the narrow value column, e.g. "3.1 m/s"
  // (positive = closing, negative = receding).
  const closingShort = (value) => {
    const n = num(value, null);
    if (n === null) return MISSING;
    const sign = n < -0.05 ? "−" : "";
    return `${sign}${Math.abs(n).toFixed(1)} m/s`;
  };
  const riskScoreLabel = (value) => {
    const n = num(value, null);
    return n === null ? MISSING : String(Math.round(clamp(n, 0, 1) * 100));
  };

  function focusSummaryFrame(source) {
    if (state.uiMode !== "summary") return;
    const ts = eventTimestamp(source);
    if (ts === null) return;
    state.suppressVideoSyncCount = 2;
    seekPreviewVideo(ts);
    updateChartCursor(ts);
    updateVideoTimeControls(ts);
    setActiveEventIndex(nearestEventIndexAt(ts));

    const blendImage = imageSetForEvent(source).blend;
    if (blendImage) {
      showPreviewOverlay(blendImage, 5000);
    }
  }

  function stateLabel(value) {
    return isReal(value) ? titleCase(value) : MISSING;
  }

  function setModeButtons() {
    byId("toggle-mode-live")?.classList.toggle("is-active", state.uiMode === "live");
    byId("toggle-mode-summary")?.classList.toggle("is-active", state.uiMode === "summary");
    byId("toggle-mode-objects")?.classList.toggle("is-active", state.uiMode === "objects");
  }

  function panelSource() {
    if (state.uiMode === "summary") return state.selectedSummaryEvent || state.currentResult?.peakEvent || null;
    if (state.uiMode === "objects") {
      if (!state.currentTimelineRow) state.currentTimelineRow = findTimelineRowAt(previewVideo?.currentTime ?? 0);
      return state.selectedSummaryEvent || state.currentTimelineRow || null;
    }
    if (!state.currentTimelineRow) state.currentTimelineRow = findTimelineRowAt(previewVideo?.currentTime ?? 0);
    return state.currentTimelineRow;
  }

  function setUiMode(mode, { sourceEvent = null, timeSec = null } = {}) {
    const nextMode = mode === "summary" ? "summary" : mode === "objects" ? "objects" : "live";
    const previousMode = state.uiMode;
    state.uiMode = nextMode;

    if (state.uiMode !== "objects") {
      state.selectedObjectId = null;
      state.objectsMenuCollapsed = false;
    }
    if (state.uiMode === "summary") {
      state.selectedSummaryEvent = sourceEvent || state.currentResult?.peakEvent || null;
    } else if (state.uiMode === "objects") {
      state.objectsMenuCollapsed = false;
      if (sourceEvent) state.selectedSummaryEvent = sourceEvent;
      if (previousMode === "summary" && !state.selectedSummaryEvent) {
        state.selectedSummaryEvent = state.currentResult?.peakEvent || null;
      }
      if (!state.currentTimelineRow) {
        state.currentTimelineRow = findTimelineRowAt(timeSec ?? previewVideo?.currentTime ?? 0);
      }
    } else {
      state.selectedSummaryEvent = null;
      state.currentTimelineRow = findTimelineRowAt(timeSec ?? previewVideo?.currentTime ?? 0);
    }
    setModeButtons();
    renderRiskPanel();
  }

  const setText = (id, value) => { const el = byId(id); if (el) el.textContent = value; };
  const pctLabel = (v) => { const n = num(v, null); return n === null ? MISSING : `${Math.round(clamp(n, 0, 1) * 100)}%`; };
  const confidenceLabel = (source) => {
    const overall = num(source?.overallConfidence, null);
    if (overall !== null) return pctLabel(overall);
    const pct = num(source?.confidencePct, null);
    return pct === null ? MISSING : `${Math.round(clamp(pct, 0, 100))}%`;
  };

  // Detail Mode mirrors the How It Works pipeline sections for the selected object.
  function renderAdvanced(source, trafficLight) {
    const ev = source?.evidence || null;
    const conf = source?.confidence || {};
    const riskFactors = source?.riskFactors || {};
    const oid = source?.displayId ?? source?.objectId;
    // Decision + Delivery: delivered frame/object decision surfaced first.
    setText("ev-fusion-state", (() => { const rawState = source?.rawRiskState ?? source?.riskState; return isReal(rawState) ? titleCase(rawState) : MISSING; })());
    setText("ev-decision-primary", isReal(oid) ? `#${oid}` : MISSING);
    setText("ev-fusion-score", riskScoreLabel(source?.riskScore));
    setText("ev-fusion-eta", etaDisplay(source?.collisionEta));
    // Per-Object Risk Evaluation: score terms and gates.
    const etaSec = etaSeconds(source?.collisionEta);
    setText("ev-fusion-eta-pressure", etaSec === null ? MISSING : pctLabel(etaPressure(source?.collisionEta)));
    setText("ev-depth-proximity", pctLabel(riskFactors.proximity));
    setText("ev-fusion-approach", pctLabel(riskFactors.approach));
    setText("ev-risk-lane-relevance", pctLabel(riskFactors.crossing));
    setText("ev-advisory-brake", pctLabel(riskFactors.brake));
    setText("ev-fusion-confidence", confidenceLabel(source));
    setText("ev-fusion-agreement", pctLabel(source?.ttcAgreement));
    // Motion + Depth: distance, depth confidence and visual motion cues.
    setText("ev-depth-distance", distanceLabel(source?.kinematics?.distanceM));
    setText("ev-depth-closing", closingShort(source?.kinematics?.closingMps));
    setText("ev-depth-conf", pctLabel(conf.depth));
    const flow = ev?.flow || {};
    setText("ev-expansion-rate", pctLabel(flow.expansionScore));
    setText("ev-flow-radial", pctLabel(flow.radialScore));
    setText("ev-flow-conf", pctLabel(conf.flow));
    // Lane Pipeline: lane-relative object geometry.
    setText("ev-lane-bucket", isReal(source?.lane) ? titleCase(source.lane) : MISSING);
    const pos = num(source?.lanePosition, null);
    setText("ev-lane-pos", pos === null ? MISSING : pos.toFixed(2));
    setText("ev-lane-crossing", pctLabel(riskFactors.crossing));
    setText("ev-lane-conf", pctLabel(conf.lane));
    // Detection + Tracking: detector outputs plus the tracker-assigned object id.
    setText("ev-detector-class", isReal(source?.objectType) ? titleCase(source.objectType) : MISSING);
    setText("ev-detector-conf", pctLabel(conf.detection));
    setText("ev-tracking-id", isReal(oid) ? `#${oid}` : MISSING);
    // Advisory Context: non-object context.
    const tl = trafficLight;
    setText("ev-advisory-traffic", (tl === "red" || tl === "yellow" || tl === "green") ? titleCase(tl) : MISSING);
  }

  const confidenceBreakdown = (conf) => {
    if (!conf) return "Overall confidence scales the final Risk Score using detection and lane-geometry reliability.";
    return `Overall confidence scales the final Risk Score. Detection ${pctLabel(conf.detection)} · Lane ${pctLabel(conf.lane)} · Depth ${pctLabel(conf.depth)}.`;
  };

  // Frame-level traffic-light advisory (red/yellow/green); hidden otherwise.
  function renderTrafficLight(source) {
    const chip = byId("traffic-light-dot");
    if (!chip) return;
    const tl = source?.trafficLight;
    const show = tl === "red" || tl === "yellow" || tl === "green";
    chip.hidden = !show;
    if (!show) return;
    chip.dataset.state = tl;
    const label = chip.querySelector(".tl-label");
    if (label) label.textContent = titleCase(tl);
  }

  function applyRiskBannerState({ source, timeTag }) {
    const hasObject = source?.objectId !== null && source?.objectId !== undefined;
    const riskState = hasObject ? (source?.riskState || null) : null;
    const banner = byId("risk-banner");
    banner.classList.remove("risk-none", "risk-low", "risk-medium", "risk-high", "risk-critical");
    banner.classList.add(riskClass(riskState));
    byId("risk-band-main").textContent = riskState ? String(riskState).toUpperCase() : MISSING;
    byId("alert-ttc").textContent = etaDisplay(source?.collisionEta);
    setText("risk-score-main", hasObject ? riskScoreLabel(source?.riskScore) : MISSING);
    const subtitle = byId("risk-subtitle");
    if (subtitle) {
      const label = objectLabel(source);
      const present = hasObject && label !== MISSING;
      // Lane lives here now (object · lane) rather than as a standalone box.
      const laneText = present && source?.lane ? ` · ${laneWithPosition(source.lane, source.lanePosition)}` : "";
      subtitle.textContent = present ? `${label}${laneText}` : "";
      subtitle.hidden = !present;
    }
    setText("risk-distance", distanceLabel(source?.kinematics?.distanceM));
    setText("risk-approach", closingShort(source?.kinematics?.closingMps));
    const tag = byId("risk-time-tag");
    if (tag) {
      if (timeTag) { tag.innerHTML = timeTag; tag.hidden = false; }
      else tag.hidden = true;
    }
  }

  function setSignalBar(name, value) {
    const fill = byId(`signal-${name}`);
    const label = byId(`signal-${name}-value`);
    const n = num(value, null);
    const pct = n === null ? 0 : Math.round(clamp(n, 0, 1) * 100);
    if (fill) fill.style.width = `${pct}%`;
    if (label) label.textContent = n === null ? MISSING : `${pct}%`;
  }

  // ETA-input gauges: visualise the raw measurement on a fixed scale (these are
  // not 0–1 risk scores, so they get their own gauge, not proximity/approach).
  const setFillWidth = (id, frac) => {
    const el = byId(id);
    if (el) el.style.width = `${Math.round(clamp(num(frac, 0), 0, 1) * 100)}%`;
  };
  const closenessGauge = (m) => { const n = num(m, null); return n === null ? 0 : clamp(1 - n / 60, 0, 1); };
  const speedGauge = (mps) => { const n = num(mps, null); return n === null ? 0 : clamp(n / 12, 0, 1); };

  function objectsForSource(source) {
    return (Array.isArray(source?.objects) ? source.objects : []).filter((obj) => obj && isReal(obj.objectId));
  }

  function highestRiskObject(objects) {
    if (!objects.length) return null;
    return [...objects].sort((a, b) => eventSeverityScore(b) - eventSeverityScore(a))[0] || null;
  }

  function renderObjectList(source) {
    const objects = objectsForSource(source);
    const list = byId("objects-menu");

    if (list) {
      list.replaceChildren();
      list.hidden = state.uiMode !== "objects" || state.objectsMenuCollapsed || !objects.length;
      if (objects.length) {
        const sorted = [...objects].sort((a, b) => eventSeverityScore(b) - eventSeverityScore(a));

        sorted.forEach((item) => {
          const button = document.createElement("button");
          button.type = "button";
          const isSelected = item.objectId === state.selectedObjectId;
          const sClass = stateClass(item.riskState);
          button.className = `detection-row is-${sClass} ${isSelected ? 'is-selected' : ''}`;
          button.onclick = () => {
            state.selectedObjectId = item.objectId;
            state.objectsMenuCollapsed = true;
            focusSummaryFrame(source);
            renderRiskPanel();
          };
          button.innerHTML = `
            <span class="status-dot is-${sClass}"></span>
            <span class="detection-main">${objectLabel(item)}</span>
            <span class="detection-ttc"><span style="color:var(--muted); margin-right:4px; font-size:10.5px; font-weight:600;">ETA:</span>${etaDisplay(item.collisionEta)}</span>
          `;
          list.appendChild(button);
        });
      }
    }
  }

  function renderRiskPanel() {
    const source = panelSource();
    const isSummary = state.uiMode === "summary";
    const isObjectsMode = state.uiMode === "objects";
    const sourceTime = (isSummary || isObjectsMode) ? eventTimestamp(source) : num(source?.timeSec, null);
    const timeLabel = isSummary ? "Peak" : isObjectsMode ? "Objects" : "Live";
    const objects = objectsForSource(source);

    if (isObjectsMode) {
      if (state.selectedObjectId !== null && !objects.some(o => o.objectId === state.selectedObjectId)) {
        state.selectedObjectId = null;
      }
      if (state.selectedObjectId === null) {
        const firstObject = highestRiskObject(objects);
        state.selectedObjectId = firstObject?.objectId ?? null;
      }
    }

    const selectedObject = isObjectsMode && state.selectedObjectId !== null
      ? objects.find(o => o.objectId === state.selectedObjectId) || null
      : null;
    const displaySource = selectedObject || source;

    byId("risk-panel-title").textContent = isSummary ? "Peak Risk" : isObjectsMode ? "Object Risk" : "Current Risk";
    applyRiskBannerState({
      source: displaySource,
      timeTag: sourceTime === null ? `${timeLabel}: <span>00:00</span>` : `${timeLabel}: <span>${formatSeconds(sourceTime)}</span>`,
    });

    renderObjectList(source);
    renderTrafficLight(source);
    const activeObject = selectedObject || source;
      
    const factors = activeObject?.riskFactors || {};
    const km = activeObject?.kinematics || {};
    // Collision-ETA input gauges (raw measurements on a fixed scale).
    setFillWidth("fill-distance", closenessGauge(km.distanceM));
    setFillWidth("fill-closing", speedGauge(km.closingMps));
    // Additive weighted contributors (mirror score_raw): ETA + proximity + approach + brake.
    setSignalBar("eta", etaPressure(activeObject?.collisionEta));
    setSignalBar("near", factors.proximity);
    setSignalBar("closing", factors.approach);
    setSignalBar("brake", factors.brake);
    // Multipliers: lane relevance (crossing) and confidence — shown as bars too.
    setSignalBar("relevance", factors.crossing);
    const conf = num(activeObject?.confidencePct, null);
    setSignalBar("confidence", conf === null ? null : conf / 100);
    setText("confidence-breakdown", confidenceBreakdown(activeObject?.confidence));
    renderAdvanced(activeObject, source?.trafficLight);
  }

  // ─── Timeline + event strip ──────────────────────────────
  function renderTimeline(result) {
    const eventItems = timelineEventItems(result);
    const resultImages = result?.images || {};
    const timelineRows = Array.isArray(result?.timelineRows) ? result.timelineRows : [];
    // Scale the strip to the analyzed window so the data fills the timeline.
    const { start: winStart, span: winSpan } = winBounds();

    const totalEventsEl = byId("stat-total-events");
    if (totalEventsEl) totalEventsEl.textContent = String(eventItems.length);

    const axis = byId("timeline-axis");
    axis.replaceChildren();
    const ticks = 6;
    const tickStep = winSpan / (ticks - 1);
    for (let i = 0; i < ticks; i++) {
      const sp = document.createElement("span");
      sp.textContent = formatAxisTime(winStart + tickStep * i, tickStep);
      axis.appendChild(sp);
    }

    const track = byId("timeline-events");
    track.replaceChildren();
    renderFrameRiskSegments(timelineRows);
    renderTimelineHeat(timelineRows);
    const strip = byId("event-strip");
    strip.replaceChildren();

    if (!eventItems.length) {
      const e = document.createElement("div");
      e.className = "event-empty";
      e.textContent = "No events yet";
      strip.appendChild(e);
      return;
    }

    eventItems.forEach((item) => {
      const { ev, index: idx, sc, ts } = item;
      const left = clamp(((ts - winStart) / winSpan) * 100, 0, 100);
      const dot = document.createElement("button");
      dot.type = "button";
      dot.className = `timeline-event ev-${sc}`;
      dot.dataset.index = String(idx);
      dot.style.left = `${left}%`;
      dot.title = eventTooltip(item);
      dot.setAttribute("aria-label", dot.title);
      dot.addEventListener("click", (e) => {
        e.stopPropagation();
        seekPreviewVideo(ts);
        applyTimelineStateAt(ts);
        setActiveEventIndex(idx, { scroll: true });
      });
      track.appendChild(dot);
    });

    eventItems.forEach((item) => {
      const { ev, index: idx, sc, ts } = item;
      const card = document.createElement("button");
      card.type = "button";
      card.className = `event-card ev-${sc}`;
      card.dataset.index = String(idx);
      card.addEventListener("click", () => focusEvent(idx));

      const thumbImg = mediaSrc(ev?.images?.original || ev?.images?.blend || resultImages.original || resultImages.blend);

      card.innerHTML = `
        <div class="card-visual">
          ${thumbImg ? `<img src="${thumbImg}" alt="Event">` : '<div style="height:100%; display:grid; place-items:center; color:var(--muted); font-size:11px; font-weight:700;">—</div>'}
        </div>
        <div class="card-info">
          <div class="card-header">
            <span class="event-id">${shortType(ev.objectType)} ${isReal(ev.displayId ?? ev.objectId) ? `#${ev.displayId ?? ev.objectId}` : MISSING}</span>
            <span class="time">${formatSeconds(ts)}</span>
          </div>

          <div class="card-boxes">
            <div class="box-item">
              <span class="lbl">STATUS</span>
              <span class="val status-val">${sc.toUpperCase()}</span>
            </div>
            <div class="box-item">
              <span class="lbl">COLLISION ETA</span>
              <span class="val">${etaDisplay(ev.collisionEta)}</span>
            </div>
            <div class="box-item">
              <span class="lbl">RISK SCORE</span>
              <span class="val risk-score-val">${riskScoreLabel(ev.riskScore)}</span>
            </div>
            <div class="box-item">
              <span class="lbl">LANE</span>
              <span class="val">${laneWithPosition(ev.lane, ev.lanePosition)}</span>
            </div>
          </div>
        </div>
      `;
      strip.appendChild(card);
    });
  }

  function frameSegmentsFromTimelineRows(rows) {
    const normalized = normalizeTimelineRowsForChart(rows)
      .filter((row) => num(row.timeSec, null) !== null)
      .sort((a, b) => a.timeSec - b.timeSec);
    if (!normalized.length) return [];
    const { start: winStart, end: winEnd } = winBounds();
    return normalized.map((row, index) => {
      const prev = normalized[index - 1] || null;
      const next = normalized[index + 1] || null;
      const rowTime = num(row.timeSec, winStart);
      const start = index === 0 ? winStart : (num(prev?.timeSec, rowTime) + rowTime) / 2;
      const end = next ? (rowTime + num(next.timeSec, rowTime)) / 2 : winEnd;
      return {
        start: clamp(start, winStart, winEnd),
        end: clamp(Math.max(end, start), winStart, winEnd),
        sc: stateClass(row.riskState) === "none" ? "safe" : stateClass(row.riskState),
        row,
      };
    }).filter((segment) => segment.end > segment.start);
  }

  function riskSegmentsFromTimelineRows(rows) {
    const frameSegments = frameSegmentsFromTimelineRows(rows);
    const segments = [];
    frameSegments.forEach((segment) => {
      if (segment.sc !== "caution" && segment.sc !== "danger") return;
      const last = segments[segments.length - 1] || null;
      if (last && segment.start <= last.end + 0.06) {
        last.end = Math.max(last.end, segment.end);
        last.sc = last.sc === "danger" || segment.sc === "danger" ? "danger" : "caution";
        return;
      }
      segments.push({ start: segment.start, end: segment.end, sc: segment.sc });
    });
    return segments;
  }

  // Event-timeline heat band: one colored cell per frame segment, coloured by
  // risk state and faded by risk score so risk concentration reads at a glance.
  function renderTimelineHeat(rows) {
    const layer = byId("timeline-heat");
    if (!layer) return;
    layer.replaceChildren();
    const { start: winStart, span: winSpan } = winBounds();
    const segments = frameSegmentsFromTimelineRows(rows);
    segments.forEach((seg) => {
      const left = clamp(((seg.start - winStart) / winSpan) * 100, 0, 100);
      const right = clamp(((seg.end - winStart) / winSpan) * 100, 0, 100);
      const cell = document.createElement("span");
      cell.className = `heat-cell heat-${seg.sc}`;
      cell.style.left = `${left}%`;
      cell.style.width = `${Math.max(0.2, right - left)}%`;
      const score = clamp(num(seg.row?.riskScore, 0), 0, 1);
      const base = seg.sc === "safe" ? 0.16 : 0.4;
      cell.style.opacity = String(clamp(base + score * 0.6, 0.12, 1));
      layer.appendChild(cell);
    });
  }

  // ─── interactive heat band (click to seek, hover tooltip) ──
  function railTimeFromX(clientX) {
    const rail = byId("timeline-rail");
    if (!rail) return null;
    const rect = rail.getBoundingClientRect();
    const ratio = clamp((clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    const { start, span } = winBounds();
    return { t: start + ratio * span, rect };
  }
  function showTimelineTipAt(clientX) {
    const tip = byId("timeline-tip");
    const hit = railTimeFromX(clientX);
    if (!tip || !hit || !state.timelineRows.length) return;
    const row = findTimelineRowAt(hit.t);
    const sc = stateClass(row?.riskState);
    const score = Math.round(clamp(num(row?.riskScore, 0), 0, 1) * 100);
    tip.innerHTML = `<span class="tt-time">${formatSeconds(hit.t)}</span>`
      + `<span class="tt-state risk-${sc}">${sc === "none" ? "SAFE" : sc.toUpperCase()} · ${score}</span>`;
    tip.hidden = false;
    const half = tip.offsetWidth / 2;
    tip.style.left = `${clamp(clientX - hit.rect.left, half, hit.rect.width - half)}px`;
  }
  function hideTimelineTip() {
    const tip = byId("timeline-tip");
    if (tip) tip.hidden = true;
  }
  function seekFromRail(clientX) {
    if (!state.timelineRows.length) return;
    const hit = railTimeFromX(clientX);
    if (!hit) return;
    seekPreviewVideo(hit.t);
    applyTimelineStateAt(hit.t);
  }

  function renderFrameRiskSegments(rows) {
    const layer = byId("player-risk-segments");
    if (!layer) return;
    layer.replaceChildren();
    const { start: winStart, span: winSpan } = winBounds();
    const segments = frameSegmentsFromTimelineRows(rows);
    if (!segments.length) return;

    segments.forEach((segment) => {
      const left = clamp(((segment.start - winStart) / winSpan) * 100, 0, 100);
      const right = clamp(((segment.end - winStart) / winSpan) * 100, 0, 100);
      const width = Math.max(0.18, right - left);
      const el = document.createElement("span");
      el.className = `timeline-risk-segment seg-${segment.sc}`;
      el.style.left = `${left}%`;
      el.style.width = `${width}%`;
      el.title = `${segment.sc.toUpperCase()} ${formatSeconds(segment.start)} - ${formatSeconds(segment.end)}`;
      el.setAttribute("aria-label", el.title);
      layer.appendChild(el);
    });
  }

  function setActiveEventIndex(idx, { scroll = false } = {}) {
    const activeValue = idx === null || idx === undefined ? null : String(idx);
    let activeCard = null;
    document.querySelectorAll(".event-card, .timeline-event, .chart-event-marker").forEach((el) => {
      const isActive = activeValue !== null && el.getAttribute("data-index") === activeValue;
      el.classList.toggle("is-active", isActive);
      if (isActive && el.classList.contains("event-card")) activeCard = el;
    });
    if (scroll && activeCard) {
      activeCard.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
    }
  }

  function nearestEventIndexAt(timeSec, toleranceSec = null) {
    const t = num(timeSec, null);
    if (t === null) return null;
    const items = timelineEventItems();
    if (!items.length) return null;
    const tolerance = toleranceSec ?? clamp(currentTotalDuration() * 0.018, 0.25, 0.75);
    let best = null;
    let bestDiff = Infinity;
    items.forEach((item) => {
      const diff = Math.abs(item.ts - t);
      if (diff < bestDiff) {
        best = item;
        bestDiff = diff;
      }
    });
    return best && bestDiff <= tolerance ? best.index : null;
  }

  function timelineEventItemByIndex(idx) {
    const target = idx === null || idx === undefined ? null : String(idx);
    if (target === null) return null;
    return timelineEventItems().find((item) => String(item.index) === target) || null;
  }

  function focusEvent(idx) {
    const item = timelineEventItemByIndex(idx);
    const ev = item?.ev || (state.events || [])[idx];
    if (!ev) return;
    const ts = eventTimestamp(ev);
    if (ts !== null) {
      state.suppressVideoSyncCount = 2;
      seekPreviewVideo(ts);
      updateChartCursor(ts);
      updateVideoTimeControls(ts);
    }
    setUiMode("summary", { sourceEvent: ev });
    setActiveEventIndex(idx, { scroll: true });

    const blendImage = imageSetForEvent(ev).blend;
    if (blendImage) {
      showPreviewOverlay(blendImage, 5000);
    }
  }

  // ─── Risk state timeline from timelineRows ───────────────
  function renderRiskTimeline(result) {
    const lineGroup = byId("chart-line");
    const areaGroup = byId("chart-area");
    const pointCountEl = byId("stat-risk-points");
    if (!lineGroup || !areaGroup) return;

    const points = normalizeTimelineRowsForChart(result?.timelineRows)
      .map((row) => {
        const sc = stateClass(row.riskState);
        const riskState = sc === "none" ? "SAFE" : sc.toUpperCase();
        return {
          timeSec: row.timeSec,
          sc: sc === "none" ? "safe" : sc,
          riskState,
          collisionEta: row.collisionEta,
          objectType: row.objectType,
          objectId: row.objectId,
          displayId: row.displayId,
          lane: row.lane,
        };
      });

    // Scale the chart x-axis to the analyzed window, not the full media, so the
    // analysis fills the chart instead of being squeezed into a sub-range.
    const { start: winStart, span: winSpan } = winBounds();

    const W = 400;
    const yByState = {
      danger: 28,
      caution: 75,
      safe: 122,
    };
    const markerColorByState = {
      danger: "#ff4444",
      caution: "#ffb020",
      safe: "#12d492",
    };

    const svgNS = "http://www.w3.org/2000/svg";
    const createSvg = (tag, attrs) => {
      const el = document.createElementNS(svgNS, tag);
      Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
      return el;
    };
    areaGroup.replaceChildren();
    lineGroup.replaceChildren();

    const yAxis = document.querySelector(".chart-y-axis");
    if (yAxis) {
      yAxis.replaceChildren();
      ["DANGER", "CAUTION", "SAFE"].forEach((tick) => {
        const label = document.createElement("span");
        label.textContent = String(tick);
        yAxis.appendChild(label);
      });
    }

    const xForTime = (t) => clamp((t - winStart) / winSpan, 0, 1) * W;

    [
      { sc: "danger", y: 0, h: 50, color: "rgba(255, 68, 68, 0.07)" },
      { sc: "caution", y: 50, h: 50, color: "rgba(255, 176, 32, 0.06)" },
      { sc: "safe", y: 100, h: 50, color: "rgba(18, 212, 146, 0.055)" },
    ].forEach((band) => {
      areaGroup.appendChild(createSvg("rect", {
        x: 0,
        y: band.y,
        width: W,
        height: band.h,
        fill: band.color,
      }));
    });
    Object.values(yByState).forEach((y) => {
      areaGroup.appendChild(createSvg("line", {
        x1: 0,
        y1: y.toFixed(1),
        x2: W,
        y2: y.toFixed(1),
        stroke: "rgba(255,255,255,0.12)",
        "stroke-width": 1,
        "stroke-dasharray": "3 5",
      }));
    });

    if (!points.length) {
      if (pointCountEl) pointCountEl.textContent = "0";
      updateChartAxisX();
      return;
    }

    if (pointCountEl) pointCountEl.textContent = String(points.length);

    const chartPoints = points.map((point) => ({
      ...point,
      x: clamp(xForTime(point.timeSec), 0, W),
      y: yByState[point.sc] ?? yByState.safe,
    }));

    for (let i = 1; i < chartPoints.length; i++) {
      const prev = chartPoints[i - 1];
      const point = chartPoints[i];
      const color = markerColorByState[point.sc] || markerColorByState.safe;
      lineGroup.appendChild(createSvg("line", {
        x1: prev.x.toFixed(1),
        y1: prev.y.toFixed(1),
        x2: point.x.toFixed(1),
        y2: point.y.toFixed(1),
        stroke: color,
        "stroke-width": 5,
        "stroke-opacity": 0.13,
        "stroke-linecap": "round",
      }));
      lineGroup.appendChild(createSvg("line", {
        x1: prev.x.toFixed(1),
        y1: prev.y.toFixed(1),
        x2: point.x.toFixed(1),
        y2: point.y.toFixed(1),
        stroke: color,
        "stroke-width": 2.2,
        "stroke-linecap": "round",
      }));
    }

    chartPoints.forEach((point) => {
      const color = markerColorByState[point.sc] || markerColorByState.safe;
      const group = createSvg("g", {
        class: `chart-event-marker chart-event-${point.sc}`,
        "data-time": point.timeSec.toFixed(2),
        tabindex: 0,
        role: "button",
      });
      group.style.cursor = "pointer";
      group.addEventListener("click", () => {
        seekPreviewVideo(point.timeSec);
        applyTimelineStateAt(point.timeSec);
      });
      group.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        seekPreviewVideo(point.timeSec);
        applyTimelineStateAt(point.timeSec);
      });

      const title = createSvg("title", {});
      title.textContent = timelinePointTooltip(point);
      group.appendChild(title);

      group.appendChild(createSvg("rect", {
        x: (point.x - 8).toFixed(1),
        y: (point.y - 15).toFixed(1),
        width: 16,
        height: 30,
        fill: "rgba(0,0,0,0)",
        stroke: "none",
        "pointer-events": "all",
      }));
      group.appendChild(createSvg("circle", {
        cx: point.x.toFixed(1),
        cy: point.y.toFixed(1),
        r: point.sc === "danger" ? 4.6 : 4,
        fill: color,
        stroke: "rgba(255,255,255,0.92)",
        "stroke-width": 1.1,
      }));

      lineGroup.appendChild(group);
    });

    updateChartAxisX();
    setActiveEventIndex(nearestEventIndexAt(previewVideo?.currentTime ?? 0));
  }


  function currentTotalDuration() {
    return previewVideo?.duration
      || state.sourceMeta?.durationSec
      || (state.currentResult?.frameCount && state.currentResult?.fps ? state.currentResult.frameCount / state.currentResult.fps : null)
      || 1;
  }

  // Active analysis window the timeline/chart/player render against. Falls back
  // to the full media duration when no window has been set.
  function winBounds() {
    const w = state.analysisWindow || { startSec: 0, endSec: null };
    const total = currentTotalDuration();
    const start = clamp(num(w.startSec, 0), 0, total);
    const rawEnd = w.endSec === null || w.endSec === undefined ? total : num(w.endSec, total);
    const end = clamp(rawEnd, start, total);
    return { start, end, span: Math.max(end - start, 0.01) };
  }

  // Derive [min, max] timeSec from a set of timeline rows. Used to scale the
  // timeline/chart to the actually-analyzed span (both time and frames modes).
  function computeWindowFromRows(rows) {
    const times = (rows || []).map((row) => num(row.timeSec ?? row.timestampSec, null)).filter((v) => v !== null);
    if (!times.length) return { startSec: 0, endSec: null };
    return { startSec: Math.min(...times), endSec: Math.max(...times) };
  }

  function updateChartAxisX() {
    const axis = byId("chart-axis-x");
    if (!axis) return;
    const { start, span } = winBounds();
    axis.replaceChildren();
    const ticks = 6;
    const tickStep = span / (ticks - 1);
    for (let i = 0; i < ticks; i++) {
      const sp = document.createElement("span");
      sp.textContent = formatAxisTime(start + tickStep * i, tickStep);
      axis.appendChild(sp);
    }
  }

  function updateChartCursor(timeSec) {
    const cursor = byId("chart-cursor");
    if (!cursor) return;
    const { start, span } = winBounds();
    const ratio = clamp((timeSec - start) / span, 0, 1);
    if (cursor) {
      const x = ratio * 400;
      cursor.setAttribute("x1", String(x));
      cursor.setAttribute("x2", String(x));
      cursor.setAttribute("opacity", state.timelineRows.length || state.events.length ? "1" : "0");
    }
  }

  function updateVideoTimeControls(timeSec) {
    const t = num(timeSec, null);
    if (t === null) return;
    renderTimeReadout(t);
    // Seek bar spans the analyzed window (0% = winStart, 100% = winEnd).
    const { start, span } = winBounds();
    const ratio = clamp((t - start) / span, 0, 1);
    seekBar.value = String(Math.round(ratio * 1000));
    seekBar.style.setProperty("--fill", `${(ratio * 100).toFixed(1)}%`);
  }

  function seekPreviewVideo(timeSec) {
    const t = num(timeSec, null);
    if (t === null || !previewVideo || previewVideo.hidden || !previewVideo.src) return;
    const applySeek = () => {
      // Clamp seeks to the analyzed window so the player stays scoped to it.
      const { start, end } = winBounds();
      const target = clamp(t, start, end);
      try {
        previewVideo.pause();
        previewVideo.currentTime = target;
      } catch {
        try { previewVideo.currentTime = target; } catch {}
      }
      updateVideoTimeControls(target);
    };
    if (previewVideo.readyState >= 1) {
      applySeek();
    } else {
      previewVideo.addEventListener("loadedmetadata", applySeek, { once: true });
    }
  }

  function playWithinWindow() {
    if (previewVideo.hidden || !previewVideo.src) return;
    hidePreviewOverlay();
    const { start, end } = winBounds();
    if (state.playbackMode === "risk") {
      const segments = riskSegmentsFromTimelineRows(state.timelineRows);
      if (!segments.length) return;
      const current = previewVideo.currentTime;
      const active = segments.find((segment) => current >= segment.start - 0.04 && current < segment.end - 0.04);
      const next = active || segments.find((segment) => segment.start >= current - 0.04) || segments[0];
      if (!active) {
        try { previewVideo.currentTime = next.start; } catch {}
      }
    }
    // If parked at (or past) the window end, restart from the window start.
    if (previewVideo.currentTime >= end - 0.04 || previewVideo.currentTime < start - 0.04) {
      try { previewVideo.currentTime = start; } catch {}
    }
    previewVideo.play().catch(() => {});
  }

  function updatePlaybackModeButtons() {
    document.querySelectorAll("[data-playback-mode]").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.playbackMode === state.playbackMode);
    });
    const label = byId("playback-mode-label");
    if (label) {
      label.textContent = state.playbackMode === "risk"
        ? "Risk Only"
        : state.playbackMode === "pause-risk"
          ? "Pause On Risk"
          : state.playbackMode === "slow-risk"
            ? "Slow On Risk"
            : "Normal";
    }
  }

  function setPlaybackMode(mode) {
    const allowed = ["risk", "pause-risk", "slow-risk"];
    state.playbackMode = allowed.includes(mode) ? mode : "normal";
    state.pauseRiskLastKey = null;
    // Leaving slow-risk restores the user's chosen base rate.
    if (previewVideo && state.playbackMode !== "slow-risk") previewVideo.playbackRate = state.playbackRate;
    updatePlaybackModeButtons();
  }

  // Settings popup (gear) — holds playback mode, speed, overlay + loop toggles.
  function setSettingsMenuOpen(open) {
    const menu = byId("player-settings-menu");
    const button = byId("player-settings-button");
    if (!menu || !button) return;
    menu.hidden = !open;
    button.setAttribute("aria-expanded", open ? "true" : "false");
    button.classList.toggle("is-active", open);
  }

  function closeSettingsMenu() {
    setSettingsMenuOpen(false);
  }

  function toggleSettingsMenu() {
    const menu = byId("player-settings-menu");
    setSettingsMenuOpen(!!menu?.hidden);
  }

  function handlePlaybackModeTick(currentTime) {
    if (!previewVideo || previewVideo.paused) return;
    if (state.playbackMode === "normal") return;

    if (state.playbackMode === "slow-risk") {
      const inRisk = isRiskRow(findTimelineRowAt(currentTime));
      const desired = inRisk ? Math.min(0.4, state.playbackRate) : state.playbackRate;
      if (Math.abs(previewVideo.playbackRate - desired) > 1e-3) previewVideo.playbackRate = desired;
      return;
    }

    if (state.playbackMode === "risk") {
      const segments = riskSegmentsFromTimelineRows(state.timelineRows);
      if (!segments.length) {
        previewVideo.pause();
        return;
      }
      const active = segments.find((segment) => currentTime >= segment.start - 0.04 && currentTime < segment.end - 0.04);
      if (active) return;
      const next = segments.find((segment) => segment.start > currentTime + 0.04);
      if (next) {
        try { previewVideo.currentTime = next.start; } catch {}
      } else {
        previewVideo.pause();
      }
      return;
    }

    if (state.playbackMode === "pause-risk") {
      const row = findTimelineRowAt(currentTime);
      if (!isRiskRow(row)) return;
      const key = timelineKey(row) || String(Math.round(currentTime * 10) / 10);
      if (state.pauseRiskLastKey === key) return;
      state.pauseRiskLastKey = key;
      previewVideo.pause();
      const target = num(row?.timeSec, currentTime);
      state.suppressVideoSyncCount = 1;
      seekPreviewVideo(target);
      applyTimelineStateAt(target, { switchMode: true });
    }
  }

  function applyTimelineStateAt(timeSec, { switchMode = true } = {}) {
    const t = num(timeSec, 0);
    updateChartCursor(t);
    updateVideoTimeControls(t);
    const activeEventIndex = nearestEventIndexAt(t);
    setActiveEventIndex(activeEventIndex);

    const cursor = byId("timeline-cursor");
    if (cursor) {
      const { start, span } = winBounds();
      const left = clamp(((t - start) / span) * 100, 0, 100);
      cursor.style.left = `${left}%`;
    }

    const row = findTimelineRowAt(t);
    state.currentTimelineRow = row;
    drawOverlay(row);
    if (switchMode) setUiMode("live", { timeSec: t });
    else renderRiskPanel();
    updateEvidenceControls();

  }


  // ─── video-time sync to timelineRows ─────────────────────
  function findTimelineRowAt(timeSec) {
    const rows = state.timelineRows;
    if (!rows.length) return null;
    let best = 0, bestDiff = Infinity;
    for (let i = 0; i < rows.length; i++) {
      const t = num(rows[i].timeSec, null);
      if (t === null) continue;
      const diff = Math.abs(t - timeSec);
      if (diff < bestDiff) { bestDiff = diff; best = i; }
    }
    return rows[best];
  }

  function syncToVideoTime() {
    if (!state.syncFollowVideo) return;
    if (state.suppressVideoSyncCount > 0) {
      state.suppressVideoSyncCount -= 1;
      return;
    }
    const t = previewVideo?.currentTime ?? 0;
    applyTimelineStateAt(t);
  }

  // ─── full render ─────────────────────────────────────────
  function renderResult(payload) {
    const result = normalizePayload(payload);
    state.lastResult = { payload: cleanResponsePayload(payload) };
    state.currentResult = result;
    state.timelineRows = result.timelineRows || [];
    state.events = result.events || [];
    state.selectedSummaryEvent = null;
    // Scope all time-domain rendering + player control to the analyzed span.
    state.analysisWindow = computeWindowFromRows(state.timelineRows);

    hidePreviewOverlay();

    renderStatRow(result);
    renderTimeline(result);
    renderRiskTimeline(result);

    // Park the player at the window start so playback covers the analyzed
    // segment from the beginning (and stops at the window end).
    const { start: winStart } = winBounds();
    seekPreviewVideo(winStart);
    state.currentTimelineRow = findTimelineRowAt(winStart) || null;
    state.loopSegment = null;
    updateReplayButton();
    resizeOverlayCanvas();
    updateOverlayVisibility();
    redrawOverlayAtCurrentTime();
    setUiMode("summary");
    updateEvidenceControls();
  }

  function renderEmptyState() {
    state.lastResult = null;
    state.currentResult = null;
    state.timelineRows = [];
    state.events = [];
    state.liveEvents = [];
    state.selectedSummaryEvent = null;
    state.currentTimelineRow = null;
    state.analysisWindow = { startSec: 0, endSec: null };
    state.loopSegment = null;
    updateReplayButton();
    hidePreviewOverlay();
    if (previewCanvas) { previewCanvas.hidden = true; clearOverlayCanvas(); }
    stopOverlayLoop();
    renderStatRow(null);
    setUiMode("live");
    renderTimeline({ events: [] });
    renderRiskTimeline({ timelineRows: [] });
    updateEvidenceControls();

    updateChartCursor(0);
    const fMin = byId("frames-min"), fMax = byId("frames-max");
    if (fMin) fMin.disabled = true;
    if (fMax) fMax.disabled = true;
    const fInp = byId("max-frames-input"), sInp = byId("start-time-input"), eInp = byId("end-time-input");
    if (fInp) fInp.disabled = true;
    if (sInp) sInp.disabled = true;
    if (eInp) eInp.disabled = true;
  }

  // ─── live preview (WebSocket) ────────────────────────────
  function generateSessionId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    return `s-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  }

  function handleLivePreviewMessage(msg) {
    if (!msg || typeof msg !== "object") return;
    if (msg.type === "done") {
      state.livePreviewActive = false;
      return;
    }
    if (msg.type !== "preview") return;

    state.livePreviewActive = true;

    if (msg.frameImage) {
      showPreviewOverlay(msg.frameImage);
    }

    const flatFrame = msg.frame ? flattenFrame(msg.frame) : null;
    const flatFrames = Array.isArray(msg.frames) ? msg.frames.map(flattenFrame) : null;
    const ts = num(flatFrame?.timestampSec, 0);

    const status = byId("ap-status");
    if (status && msg.progress !== undefined && msg.progress !== null) {
      const pct = clamp(num(msg.progress, 0), 0, 100);
      status.textContent = `Processing… ${pct.toFixed(0)}%`;
    }

    const eventsChanged = flatFrame
      ? upsertLiveEventFromFrame(flatFrame, msg.frameImage)
      : false;
    const rowsChanged = appendLiveTimelineRows(flatFrames || flatFrame);
    if (rowsChanged) {
      state.timelineRows = state.liveTimelineRows;
      if (state.uiMode === "live") {
        state.currentTimelineRow = findTimelineRowAt(ts);
        renderRiskPanel();
      }
    }
    if (!state.analyzing && (rowsChanged || eventsChanged)) {
      // Keep the render domain in sync with the growing live span.
      state.analysisWindow = computeWindowFromRows(state.liveTimelineRows);
    }
    if (!state.analyzing && eventsChanged) {
      renderTimeline({ events: state.liveEvents });
    }
    if (!state.analyzing && (rowsChanged || eventsChanged)) {
      renderRiskTimeline({ timelineRows: state.liveTimelineRows, events: state.liveEvents });
    }
  }

  function startLivePreview() {
    return new Promise((resolve) => {
      closeLivePreview();
      const sessionId = generateSessionId();
      state.previewSessionId = sessionId;

      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        resolve(sessionId);
      };

      try {
        const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
        const ws = new WebSocket(`${proto}//${window.location.host}/ws/preview/${sessionId}`);
        ws.onopen = () => {
          state.previewWs = ws;
          finish();
        };
        ws.onmessage = (event) => {
          try { handleLivePreviewMessage(JSON.parse(event.data)); } catch {}
        };
        ws.onerror = () => finish();
        ws.onclose = () => {
          if (state.previewWs === ws) state.previewWs = null;
          finish();
        };
        // Safety: don't block analysis if the WS never opens
        setTimeout(finish, 1500);
      } catch {
        finish();
      }
    });
  }

  function closeLivePreview() {
    state.livePreviewActive = false;
    state.previewSessionId = null;
    const ws = state.previewWs;
    state.previewWs = null;
    if (!ws) return;
    try { ws.onmessage = null; ws.onerror = null; ws.onclose = null; } catch {}
    try { ws.close(); } catch {}
  }

  function appendLiveTimelineRows(rows) {
    const incoming = Array.isArray(rows) ? rows : (rows ? [rows] : []);
    if (!incoming.length) return false;
    let changed = false;
    incoming.forEach((row) => {
      const key = timelineKey(row);
      if (!key) return;
      const existingIndex = state.liveTimelineRows.findIndex((existing) => timelineKey(existing) === key);
      if (existingIndex < 0) {
        state.liveTimelineRows.push(row);
        changed = true;
        return;
      }
      if (preferTimelineRow(row, state.liveTimelineRows[existingIndex])) {
        state.liveTimelineRows[existingIndex] = row;
        changed = true;
      }
    });
    if (changed) {
      state.liveTimelineRows.sort((a, b) => {
        const ta = num(a.timeSec, 0);
        const tb = num(b.timeSec, 0);
        return ta - tb;
      });
    }
    return changed;
  }

  function upsertLiveEventFromFrame(frame, frameImage) {
    const sc = stateClass(frame?.riskState);
    if (sc !== "danger" && sc !== "caution") return false;
    const ts = num(frame.timestampSec, null);
    if (ts === null) return false;

    const candidate = {
      ...frame,
      riskState: String(frame.riskState || "").toUpperCase(),
      images: frameImage ? { blend: frameImage } : {},
    };

    let replaceIndex = -1;
    for (let i = 0; i < state.liveEvents.length; i++) {
      if (Math.abs(num(state.liveEvents[i].timestampSec, 0) - ts) <= 1.0) {
        replaceIndex = i;
        break;
      }
    }
    if (replaceIndex >= 0) {
      if (eventSeverityScore(candidate) <= eventSeverityScore(state.liveEvents[replaceIndex])) return false;
      state.liveEvents[replaceIndex] = candidate;
    } else {
      state.liveEvents.push(candidate);
    }
    state.liveEvents.sort((a, b) => num(a.timestampSec, 0) - num(b.timestampSec, 0));
    state.events = state.liveEvents.slice();
    return true;
  }

  // ─── analysis flow ───────────────────────────────────────
  function buildFormData() {
    const fd = new FormData(form);
    fd.set("mode", "video");
    if (state.analysisWindowMode === "time") {
      fd.set("max_processed_frames", "2000");
      fd.set("start_sec", String(num(fd.get("start_sec"), 0)));
      fd.set("end_sec", String(num(fd.get("end_sec"), 0)));
    } else {
      fd.set("max_processed_frames", String(num(fd.get("max_processed_frames"), 180)));
      fd.set("start_sec", "0");
      fd.set("end_sec", "0");
    }

    return fd;
  }
  async function parseResponse(response) {
    const text = await response.text();
    if (!text) return {};
    try { return JSON.parse(text); } catch { return { detail: text }; }
  }
  async function postAnalysis(sessionId) {
    const fd = buildFormData();
    if (sessionId) fd.set("session_id", sessionId);
    const response = await fetch("/api/analyze", { method: "POST", body: fd });
    const payload = await parseResponse(response);
    if (!response.ok) throw new Error(payload.detail || `Analysis failed (HTTP ${response.status}).`);
    return payload;
  }
  function showProgress() {
    const el = byId("analysis-progress");
    if (!el) return;
    el.hidden = false;
    el.classList.remove("is-complete", "is-error");
    byId("ap-label").textContent = "Analyzing";
    byId("ap-timer").textContent = "0.0s";
    byId("ap-status").textContent = "Sending request…";
    state.progressStart = Date.now();
    state.progressTimer = setInterval(() => {
      const sec = (Date.now() - state.progressStart) / 1000;
      byId("ap-timer").textContent = `${sec.toFixed(1)}s`;
      const status = byId("ap-status");
      if (status) {
        if (sec < 1) status.textContent = "Sending request…";
        else if (sec < 3) status.textContent = "Awaiting backend response…";
        else if (sec < 8) status.textContent = "Pipeline running…";
        else status.textContent = "Analysis in progress…";
      }
    }, 100);
  }
  function finishProgress({ label = "Completed", status = "Analysis complete.", isError = false } = {}) {
    const el = byId("analysis-progress");
    if (!el) return;
    if (state.progressTimer) { clearInterval(state.progressTimer); state.progressTimer = null; }
    el.hidden = false;
    el.classList.toggle("is-complete", !isError);
    el.classList.toggle("is-error", isError);
    byId("ap-label").textContent = label;
    byId("ap-status").textContent = status;
    window.setTimeout(() => {
      if (!state.analyzing) hideProgress();
    }, isError ? 3200 : 1800);
  }
  function hideProgress() {
    const el = byId("analysis-progress");
    if (el) el.hidden = true;
    if (el) el.classList.remove("is-complete", "is-error");
    if (state.progressTimer) { clearInterval(state.progressTimer); state.progressTimer = null; }
  }
  function setRunningUI(isRunning) {
    state.analyzing = isRunning;
    document.body.classList.toggle("is-analysis-running", isRunning);
    runButton.classList.toggle("is-running", isRunning);
    previewFrame.classList.toggle("is-analyzing", isRunning);
    updateOverlayVisibility();
    const label = runButton.querySelector("span");
    if (label) label.textContent = isRunning ? "Analyzing…" : "Start Analysis";
    runButton.disabled = isRunning;
    if (removeBtn) removeBtn.disabled = isRunning;
  }

  async function analyzeSelectedFile(event) {
    event?.preventDefault();
    if (!fileInput.files.length) {
      uploadButton?.focus();
      return;
    }
    if (state.analyzing) return;
    setRunningUI(true);
    showProgress();
    state.liveTimelineRows = [];
    state.liveEvents = [];
    state.events = [];
    state.currentTimelineRow = null;
    state.selectedSummaryEvent = null;
    setUiMode("live");
    renderTimeline({ events: [] });
    renderRiskTimeline({ timelineRows: [], events: [] });
    const sessionId = await startLivePreview();
    try {
      const payload = await postAnalysis(sessionId);
      closeLivePreview();
      renderResult(payload);
      finishProgress({ label: "Completed", status: "Analysis complete." });
    } catch (err) {
      closeLivePreview();
      renderEmptyState();
      finishProgress({ label: "Failed", status: err.message || "Analysis failed.", isError: true });
    } finally {
      setRunningUI(false);
    }
  }

  // ─── file selection ──────────────────────────────────────
  function releasePreviewUrl() {
    if (state.previewUrl) { URL.revokeObjectURL(state.previewUrl); state.previewUrl = ""; }
  }
  function handleFileSelection() {
    releasePreviewUrl();
    renderEmptyState();
    const file = fileInput.files[0];
    setSelectedFile(file);

    if (!file) { setPreviewMedia(""); state.sourceMeta = null; return; }
    state.previewUrl = URL.createObjectURL(file);
    setPreviewMedia(state.previewUrl);
  }
  function clearSelectedSource() {
    if (state.analyzing) return;
    releasePreviewUrl();
    fileInput.value = "";
    state.sourceMeta = null;
    setSelectedFile(null);
    setPreviewMedia("");
    renderEmptyState();

    // Reset frames input to empty
    const input = byId("max-frames-input");
    if (input) {
      input.value = "";
      input.max = "";
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }
    const fMin = byId("frames-min"), fMax = byId("frames-max");
    if (fMin) fMin.disabled = true;
    if (fMax) fMax.disabled = true;

    // Reset temporal window inputs
    const startInp = byId("start-time-input"), endInp = byId("end-time-input");
    if (startInp) { startInp.value = ""; startInp.max = ""; startInp.disabled = true; }
    if (endInp) { endInp.value = ""; endInp.max = ""; endInp.disabled = true; }
    applyAnalysisWindowMode();
  }

  // ─── settings drawer ─────────────────────────────────────
  function openDrawer() {
    if (!settingsDrawer) return;
    settingsDrawer.hidden = false;
    // force reflow so the transition plays
    void settingsDrawer.offsetHeight;
    settingsDrawer.classList.add("is-open");
  }
  function closeDrawer() {
    if (!settingsDrawer) return;
    settingsDrawer.classList.remove("is-open");
    setTimeout(() => { if (!settingsDrawer.classList.contains("is-open")) settingsDrawer.hidden = true; }, 400);
  }

  // "How Spectra Works" now lives on the standalone /how-it-works page;
  // the header's #open-help link navigates there directly (no modal JS).


  function setupSegmentedControls() {
    document.querySelectorAll(".segmented-control").forEach(ctrl => {
      const param = ctrl.dataset.param;
      if (!param) return;
      ctrl.querySelectorAll(".segmented-btn").forEach(btn => {
        btn.addEventListener("click", () => {
          const val = btn.dataset.value;
          const hidden = formField(param);
          if (hidden) hidden.value = val;
          
          ctrl.querySelectorAll(".segmented-btn").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
        });
      });
    });
  }

  function applyAnalysisWindowMode() {
    const mode = state.analysisWindowMode === "time" ? "time" : "frames";
    const hasSource = !!state.sourceMeta?.durationSec;

    document.querySelectorAll("[data-window-mode]").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.windowMode === mode);
    });

    const framesInput = byId("max-frames-input");
    const startInp = byId("start-time-input");
    const endInp = byId("end-time-input");
    const framesMin = byId("frames-min");
    const framesMax = byId("frames-max");

    if (framesInput) framesInput.disabled = !hasSource || mode !== "frames";
    if (framesMin) framesMin.disabled = !hasSource || mode !== "frames";
    if (framesMax) framesMax.disabled = !hasSource || mode !== "frames";
    if (startInp) startInp.disabled = !hasSource || mode !== "time";
    if (endInp) endInp.disabled = !hasSource || mode !== "time";
  }

  function setupAnalysisWindowMode() {
    document.querySelectorAll("[data-window-mode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.analysisWindowMode = btn.dataset.windowMode === "time" ? "time" : "frames";
        applyAnalysisWindowMode();
      });
    });
    applyAnalysisWindowMode();
  }

  function setupMaxSavedEventsClamp() {
    const input = byId("max-saved-events-input");
    if (!input) return;
    input.addEventListener("input", () => {
      const value = num(input.value, null);
      if (value !== null && value > 50) {
        input.value = "50";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      }
    });
  }

  function setSegmentedValue(param, value) {
    const control = document.querySelector(`.segmented-control[data-param="${param}"]`);
    const hidden = formField(param);
    if (hidden) hidden.value = String(value);
    if (!control) return;
    control.querySelectorAll(".segmented-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.value === String(value));
    });
  }

  function resetAdvancedSampling() {
    setSegmentedValue("depth_every", 10);
    setSegmentedValue("adaptive_depth", 1);
    setSegmentedValue("detect_every", 3);
    setSegmentedValue("lane_every", 3);
    setSegmentedValue("flow_every", 1);
    setSegmentedValue("resize_max_side", 512);

    const savedEvents = byId("max-saved-events-input");
    if (savedEvents) {
      savedEvents.value = "20";
      savedEvents.dispatchEvent(new Event("input", { bubbles: true }));
    }
  }

  function setRangeFill(input) {
    const min = num(input.min, 0), max = num(input.max, 100);
    const value = num(input.value, min);
    const fill = max === min ? 0 : ((value - min) / (max - min)) * 100;
    input.style.setProperty("--fill", `${clamp(fill, 0, 100)}%`);
    const key = input.dataset.param;
    if (!key) return;
    const out = document.querySelector(`output[data-for="${key}"]`);
    if (out) {
      const isInt = (value % 1 === 0);
      out.value = isInt ? String(Math.round(value)) : value.toFixed(2);
      out.textContent = out.value;
    }
    const hidden = formField(key);
    if (hidden) hidden.value = String(input.value);
  }

  // ─── preview controls ────────────────────────────────────
  function setupPreviewControls() {
    if (!previewVideo) return;
    previewVideo.addEventListener("loadedmetadata", () => {
      updateSourceMetaFromVideo();
      resizeOverlayCanvas();
      redrawOverlayAtCurrentTime();
      updateVolumeUI();
      updateEvidenceControls();
    });
    previewVideo.addEventListener("volumechange", updateVolumeUI);
    previewVideo.addEventListener("timeupdate", () => {
      const cur = previewVideo.currentTime;
      const { start, end } = winBounds();
      // Replay loop takes precedence: bounce back to the segment start.
      if (state.loopSegment) {
        if (cur >= state.loopSegment.end - 0.04 || cur < state.loopSegment.start - 0.04) {
          try { previewVideo.currentTime = state.loopSegment.start; } catch {}
          return;
        }
      }
      // Keep playback inside the analyzed window: stop at the end, never run
      // before the start.
      if (cur >= end - 0.04) {
        if (!previewVideo.paused) previewVideo.pause();
        if (cur > end) { try { previewVideo.currentTime = end; } catch {} }
      } else if (cur < start - 0.04) {
        try { previewVideo.currentTime = start; } catch {}
      }
      updateVideoTimeControls(previewVideo.currentTime);
      syncToVideoTime();
      handlePlaybackModeTick(previewVideo.currentTime);
    });
    previewVideo.addEventListener("seeked", syncToVideoTime);
    previewVideo.addEventListener("play", () => {
      previewFrame.classList.add("is-playing");
      if (previewVideo.playbackRate !== state.playbackRate && state.playbackMode !== "slow-risk") {
        previewVideo.playbackRate = state.playbackRate;
      }
      setUiMode("live", { timeSec: previewVideo.currentTime });
      startOverlayLoop();
    });
    previewVideo.addEventListener("pause", () => { previewFrame.classList.remove("is-playing"); stopOverlayLoop(); });
    previewVideo.addEventListener("ended", () => { previewFrame.classList.remove("is-playing"); stopOverlayLoop(); });

    playToggle.addEventListener("click", () => {
      if (previewVideo.hidden || !previewVideo.src) return;
      if (previewVideo.paused) {
        playWithinWindow();
      } else {
        previewVideo.pause();
      }
    });
    seekBar.addEventListener("input", () => {
      const ratio = num(seekBar.value, 0) / 1000;
      seekBar.style.setProperty("--fill", `${(ratio * 100).toFixed(1)}%`);
      if (previewVideo.duration) {
        // Seek bar maps onto the analyzed window, not the full video.
        const { start, span } = winBounds();
        const target = start + ratio * span;
        previewVideo.currentTime = target;
        setUiMode("live", { timeSec: target });
      }
    });
    byId("center-play-btn")?.addEventListener("click", () => {
      if (!previewVideo.src) return;
      if (previewVideo.paused) {
        playWithinWindow();
      } else {
        previewVideo.pause();
      }
    });
    byId("preview-expand")?.addEventListener("click", toggleFullscreen);
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    document.addEventListener("webkitfullscreenchange", handleFullscreenChange);

    byId("frame-step-back")?.addEventListener("click", () => stepFrame(-1));
    byId("frame-step-forward")?.addEventListener("click", () => stepFrame(1));
    byId("event-prev")?.addEventListener("click", () => jumpToEvent(-1));
    byId("event-next")?.addEventListener("click", () => jumpToEvent(1));
    const seekTrackEl = seekBar?.closest(".seek-track");
    seekTrackEl?.addEventListener("mousemove", (e) => showSeekTooltipAt(e.clientX));
    seekTrackEl?.addEventListener("mouseleave", hideSeekTooltip);
    const railEl = byId("timeline-rail");
    railEl?.addEventListener("mousemove", (e) => showTimelineTipAt(e.clientX));
    railEl?.addEventListener("mouseleave", hideTimelineTip);
    railEl?.addEventListener("click", (e) => seekFromRail(e.clientX));
    document.querySelectorAll("[data-shortcuts-close]").forEach((el) => el.addEventListener("click", closeShortcuts));
    byId("overlay-toggle")?.addEventListener("click", () => toggleOverlay());
    byId("replay-event-btn")?.addEventListener("click", toggleReplayLoop);
    byId("player-settings-button")?.addEventListener("click", (event) => {
      event.stopPropagation();
      toggleSettingsMenu();
    });
    document.querySelectorAll("[data-speed]").forEach((btn) => {
      btn.addEventListener("click", () => setPlaybackRate(parseFloat(btn.dataset.speed)));
    });
    byId("volume-btn")?.addEventListener("click", toggleMute);
    byId("volume-slider")?.addEventListener("input", (e) => setVolume(num(e.target.value, 0) / 100));
    byId("time-readout")?.addEventListener("click", toggleTimeDisplay);
    updateVolumeUI();

    // Keep the canvas matched to the displayed video rect on layout changes.
    if (typeof ResizeObserver !== "undefined" && previewFrame) {
      const ro = new ResizeObserver(() => { resizeOverlayCanvas(); redrawOverlayAtCurrentTime(); });
      ro.observe(previewFrame);
    }

    // Player keyboard shortcuts (ignored while typing in form controls).
    document.addEventListener("keydown", (e) => {
      const tag = (e.target?.tagName || "").toLowerCase();
      if (["input", "textarea", "select", "button", "a"].includes(tag) || e.target?.isContentEditable) return;
      // Shortcuts help works even before a video is loaded.
      if (e.key === "?") { e.preventDefault(); openShortcuts(); return; }
      if (!previewVideo?.src || previewVideo.hidden) return;
      switch (e.key) {
        case " ": e.preventDefault(); togglePlay(); break;
        case "ArrowLeft": case ",": e.preventDefault(); e.shiftKey ? jumpToEvent(-1) : stepFrame(-1); break;
        case "ArrowRight": case ".": e.preventDefault(); e.shiftKey ? jumpToEvent(1) : stepFrame(1); break;
        case "f": case "F": toggleFullscreen(); break;
        case "m": case "M": toggleOverlay(); break;
      }
    });
  }

  // Fullscreen the whole player (video + canvas overlay + controls) so the
  // risk overlay stays in sync, instead of cloning a bare <video>.
  function fullscreenElement() {
    return document.fullscreenElement || document.webkitFullscreenElement || null;
  }
  function toggleFullscreen() {
    if (!previewFrame || !previewVideo?.src) return;
    if (fullscreenElement()) {
      (document.exitFullscreen || document.webkitExitFullscreen)?.call(document);
    } else {
      (previewFrame.requestFullscreen || previewFrame.webkitRequestFullscreen)?.call(previewFrame);
    }
  }
  function handleFullscreenChange() {
    const on = fullscreenElement() === previewFrame;
    previewFrame.classList.toggle("is-fullscreen", on);
    byId("preview-expand")?.classList.toggle("is-active", on);
    // The frame just changed size; re-fit the overlay canvas.
    resizeOverlayCanvas();
    redrawOverlayAtCurrentTime();
  }

  function selectedEventAtCurrentTime() {
    const selectedTs = eventTimestamp(state.selectedSummaryEvent);
    if (selectedTs !== null) {
      const exact = timelineEventItems().find((item) => Math.abs(item.ts - selectedTs) <= 0.05);
      if (exact) return exact.ev;
      return state.selectedSummaryEvent;
    }
    const idx = nearestEventIndexAt(previewVideo?.currentTime ?? 0, 0.35);
    const item = timelineEventItemByIndex(idx);
    return item?.ev || null;
  }

  function selectedEvidenceSource() {
    return selectedEventAtCurrentTime()
      || state.currentTimelineRow
      || findTimelineRowAt(previewVideo?.currentTime ?? 0)
      || state.currentResult?.peakEvent
      || null;
  }

  function evidenceExportPayload() {
    const source = selectedEvidenceSource();
    if (!source) return null;
    const primaryObject = getPrimaryObject(source) || source;
    return {
      sourceName: state.currentResult?.sourceName || fileInput.files[0]?.name || null,
      frameIndex: source.frameIndex ?? null,
      timestampSec: num(source.timestampSec ?? source.timeSec, null),
      riskState: source.riskState || null,
      riskScore: num(source.riskScore, null),
      primaryObject: primaryObject ? {
        objectId: primaryObject.objectId ?? null,
        displayId: primaryObject.displayId ?? null,
        objectType: primaryObject.objectType ?? null,
        collisionEta: primaryObject.collisionEta ?? source.collisionEta ?? null,
        lane: primaryObject.lane ?? source.lane ?? null,
        lanePosition: primaryObject.lanePosition ?? source.lanePosition ?? null,
        riskFactors: primaryObject.riskFactors ?? source.riskFactors ?? null,
        kinematics: primaryObject.kinematics ?? source.kinematics ?? null,
        confidence: primaryObject.confidence ?? source.confidence ?? null,
      } : null,
    };
  }

  function updateEvidenceControls() {
    const hasVideo = !!(previewVideo?.src && !previewVideo.hidden && previewVideo.readyState >= 1);
    const hasEvidence = !!selectedEvidenceSource();
    const imageEvent = selectedEventAtCurrentTime();
    const hasImage = !!(imageEvent?.images?.blend);
    const snapshot = byId("export-snapshot");
    const image = byId("export-evidence-image");
    const json = byId("export-evidence-json");
    if (snapshot) snapshot.disabled = !hasVideo;
    if (image) image.disabled = !hasImage;
    if (json) json.disabled = !hasEvidence;
  }

  function exportSnapshot() {
    if (!previewVideo || !previewVideo.src || previewVideo.readyState < 2) return;
    const width = previewVideo.videoWidth;
    const height = previewVideo.videoHeight;
    if (!width || !height) return;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(previewVideo, 0, 0, width, height);
    canvas.toBlob((blob) => {
      const base = safeFileStem(state.currentResult?.sourceName || fileInput.files[0]?.name);
      const stamp = Math.round((previewVideo.currentTime || 0) * 1000);
      downloadBlob(blob, `${base}_snapshot_${stamp}ms.png`);
    }, "image/png");
  }

  async function exportEvidenceImage() {
    const event = selectedEventAtCurrentTime();
    if (!event) return;
    const blend = event?.images?.blend;
    if (!blend) return;
    const base = safeFileStem(state.currentResult?.sourceName || fileInput.files[0]?.name);
    const ts = Math.round(num(event?.timestampSec ?? event?.timeSec, previewVideo?.currentTime ?? 0) * 1000);
    await downloadDataUrl(mediaSrc(blend), `${base}_evidence_${ts}ms.jpg`);
  }

  function exportEvidenceJson() {
    const payload = evidenceExportPayload();
    if (!payload) return;
    const base = safeFileStem(payload.sourceName);
    const ts = Math.round(num(payload.timestampSec, previewVideo?.currentTime ?? 0) * 1000);
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    downloadBlob(blob, `${base}_evidence_${ts}ms.json`);
  }

  function telemetryExportPayload() {
    if (!state.lastResult) return null;
    return JSON.parse(JSON.stringify(state.lastResult, (key, value) => {
      if (key === "images" && value && typeof value === "object" && !Array.isArray(value)) {
        return Object.fromEntries(Object.keys(value).map((imageKey) => [imageKey, "[embedded image omitted]"]));
      }
      return value;
    }));
  }

  function openJsonView() {
    const view = byId("telemetry-json-view");
    if (!state.lastResult) {
      if (view) view.textContent = "No analysis telemetry available yet. Start an analysis to view result data.";
    } else {
      const json = JSON.stringify(telemetryExportPayload(), null, 2);
      if (view) view.textContent = json;
    }

    const drawer = byId("telemetry-drawer");
    if (drawer) {
      drawer.hidden = false;
      void drawer.offsetHeight;
      drawer.classList.add("is-open");
    }
  }

  function closeTelemetryDrawer() {
    const drawer = byId("telemetry-drawer");
    if (!drawer) return;
    drawer.classList.remove("is-open");
    setTimeout(() => { if (!drawer.classList.contains("is-open")) drawer.hidden = true; }, 400);
  }

  function openLogsView() {
    const payload = state.lastResult?.payload;
    const perfLogs = payload?.performance?.logs;
    if (!perfLogs || !perfLogs.length) {
      const view = byId("logs-view");
      if (view) view.textContent = "No performance logs available for this analysis.";
    } else {
      const logs = perfLogs.join("\n");
      const view = byId("logs-view");
      if (view) view.textContent = logs;
    }

    const drawer = byId("logs-drawer");
    if (drawer) {
      drawer.hidden = false;
      void drawer.offsetHeight;
      drawer.classList.add("is-open");
    }
  }

  function closeLogsDrawer() {
    const drawer = byId("logs-drawer");
    if (!drawer) return;
    drawer.classList.remove("is-open");
    setTimeout(() => { if (!drawer.classList.contains("is-open")) drawer.hidden = true; }, 400);
  }

  async function copyLogs() {
    const payload = state.lastResult?.payload;
    const perfLogs = payload?.performance?.logs;
    if (!perfLogs || !perfLogs.length) return;
    const text = perfLogs.join("\n");
    try {
      await navigator.clipboard.writeText(text);
      const btn = byId("copy-logs-btn");
      if (btn) {
        btn.style.color = "var(--safe)";
        setTimeout(() => { btn.style.color = ""; }, 2000);
      }
    } catch (err) {
      console.error("Failed to copy logs:", err);
    }
  }

  function downloadLogs() {
    const payload = state.lastResult?.payload;
    const perfLogs = payload?.performance?.logs;
    if (!perfLogs || !perfLogs.length) return;
    const sourceName = payload?.metadata?.sourceName || payload?.sourceName;
    const baseName = safeFileStem(sourceName);
    const dateStr = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 16).replace("T", "_");
    const blob = new Blob([perfLogs.join("\n")], { type: "text/plain;charset=utf-8" });
    downloadBlob(blob, `${baseName}_performance_logs_${dateStr}.txt`);
  }

  function downloadJson() {
    if (!state.lastResult) return;
    const blob = new Blob([JSON.stringify(telemetryExportPayload(), null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const payload = cleanResponsePayload(state.lastResult);
    const sourceName = payload?.metadata?.sourceName || payload?.sourceName;
    const baseName = sourceName ? sourceName.split('.')[0] : 'spectra';
    const dateStr = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 16).replace('T', '_');
    a.download = `${baseName}_telemetry_${dateStr}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function copyJson() {
    if (!state.lastResult) return;
    const text = JSON.stringify(telemetryExportPayload(), null, 2);
    try {
      await navigator.clipboard.writeText(text);
      const btn = byId("copy-json-btn");
      if (btn) {
        btn.style.color = "var(--safe)";
        setTimeout(() => { btn.style.color = ""; }, 2000);
      }
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  }



  // ─── init ────────────────────────────────────────────────
  function initialize() {
    document.querySelectorAll('input[data-param]').forEach((input) => {
      if (input.type === "range") {
        setRangeFill(input);
        input.addEventListener("input", () => setRangeFill(input));
      } else if (input.type === "number") {
        input.addEventListener("input", () => {
          const key = input.dataset.param;
          const hidden = formField(key);
          if (hidden) hidden.value = String(input.value);
        });
      } else if (input.type === "checkbox") {
        const key = input.dataset.param;
        const hidden = formField(key);
        if (hidden) input.checked = String(hidden.value) === "1" || String(hidden.value).toLowerCase() === "true";
        input.addEventListener("change", () => {
          const field = formField(key);
          if (field) field.value = input.checked ? "1" : "0";
        });
      }
    });


    fileInput.addEventListener("change", handleFileSelection);
    form.addEventListener("submit", analyzeSelectedFile);
    uploadButton.addEventListener("click", () => fileInput.click());
    removeBtn?.addEventListener("click", clearSelectedSource);

    byId("open-settings")?.addEventListener("click", openDrawer);
    byId("open-shortcuts")?.addEventListener("click", openShortcuts);
    byId("reset-advanced-sampling")?.addEventListener("click", resetAdvancedSampling);
    byId("view-json-btn")?.addEventListener("click", openJsonView);
    byId("download-json-btn")?.addEventListener("click", downloadJson);
    byId("copy-json-btn")?.addEventListener("click", copyJson);
    byId("export-snapshot")?.addEventListener("click", exportSnapshot);
    byId("export-evidence-image")?.addEventListener("click", exportEvidenceImage);
    byId("export-evidence-json")?.addEventListener("click", exportEvidenceJson);
    document.querySelectorAll("[data-playback-mode]").forEach((btn) => {
      btn.addEventListener("click", (event) => {
        event.stopPropagation();
        setPlaybackMode(btn.dataset.playbackMode);
      });
    });
    
    byId("view-logs-btn")?.addEventListener("click", openLogsView);
    byId("copy-logs-btn")?.addEventListener("click", copyLogs);
    byId("download-logs-btn")?.addEventListener("click", downloadLogs);
    
    byId("toggle-mode-live")?.addEventListener("click", () => setUiMode("live", { timeSec: previewVideo?.currentTime ?? 0 }));
    byId("toggle-mode-summary")?.addEventListener("click", () => setUiMode("summary"));
    byId("toggle-mode-objects")?.addEventListener("click", () => setUiMode("objects", { timeSec: previewVideo?.currentTime ?? 0 }));

    document.querySelectorAll("[data-telemetry-close]").forEach(el => el.addEventListener("click", closeTelemetryDrawer));
    document.querySelectorAll("[data-logs-close]").forEach(el => el.addEventListener("click", closeLogsDrawer));
    
    byId("frames-min")?.addEventListener("click", () => {
      const input = byId("max-frames-input");
      if (input) {
        input.value = "30";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      }
    });
    byId("frames-max")?.addEventListener("click", () => {
      const input = byId("max-frames-input");
      if (input) {
        let val = 360;
        if (state.sourceMeta && state.sourceMeta.durationSec) {
          val = Math.round(state.sourceMeta.durationSec * 30);
        }
        input.value = String(val);
        input.dispatchEvent(new Event("input", { bubbles: true }));
      }
    });

    document.querySelectorAll("[data-drawer-close]").forEach((el) => el.addEventListener("click", closeDrawer));
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
    document.addEventListener("click", (event) => {
      if (!event.target?.closest?.(".player-settings")) closeSettingsMenu();
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        closeSettingsMenu();
        closeShortcuts();
      }
    });

    setupPreviewControls();
    setupSegmentedControls();
    setupAnalysisWindowMode();
    setupMaxSavedEventsClamp();
    updatePlaybackModeButtons();

    document.querySelectorAll(".side-bar-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const toggleId = btn.dataset.toggle;
        if (!toggleId) return;
        const panel = byId(toggleId);
        if (panel) {
          panel.hidden = !panel.hidden;
          btn.classList.toggle("is-active", !panel.hidden);
        }
      });
    });

    renderEmptyState();
  }

  initialize();
}
