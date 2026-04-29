(() => {
  const MISSING = "—";


  const form = document.querySelector("#analysis-form");
  const fileInput = document.querySelector("#source-file");
  const runButton = document.querySelector("#run-analysis");
  const uploadButton = document.querySelector("#top-upload");
  const removeBtn = document.querySelector("#remove-source");
  const selectedChip = document.querySelector("#selected-chip");
  const settingsDrawer = document.querySelector("#settings-drawer");

  const previewVideo = document.querySelector("#visual-original-video");
  const previewBlend = document.querySelector("#visual-blend");
  const previewFrame = document.querySelector("#frame-original");
  const playToggle = document.querySelector("#play-toggle");
  const seekBar = document.querySelector("#seek-bar");
  const timeCurrent = document.querySelector("#time-current");
  const timeTotal = document.querySelector("#time-total");

  const previewStatus = document.querySelector("#preview-status");

  const state = {
    previewUrl: "",
    lastResult: null,
    sourceMeta: null,
    analyzing: false,

    progressTimer: null,
    progressStart: 0,
    timelineRows: [],
    events: [],
    syncFollowVideo: true,
    activeMainView: "video",
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
  const num = (value, fallback = null) => {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
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
  const ttcLabel = (v) => {
    const n = num(v, null);
    return n === null ? MISSING : `${n.toFixed(1)}s`;
  };
  const fmtNumber = (v, digits = 1, suffix = "") => {
    const n = num(v, null);
    return n === null ? MISSING : `${n.toFixed(digits)}${suffix}`;
  };

  const mediaSrc = (value) => {
    if (!isReal(value)) return "";
    if (/^(data:|blob:|https?:)/i.test(value)) return value;
    return `data:image/png;base64,${value}`;
  };

  const valueAt = (obj, path) => {
    if (!obj || !path) return undefined;
    return path.split(".").reduce((cur, key) => (cur && Object.prototype.hasOwnProperty.call(cur, key) ? cur[key] : undefined), obj);
  };
  const firstValue = (sources, paths) => {
    for (const src of sources) {
      for (const p of paths) {
        const v = valueAt(src, p);
        if (isReal(v)) return v;
      }
    }
    return undefined;
  };

  const stateClass = (stateOrBand) => {
    const v = String(stateOrBand || "").toLowerCase();
    if (v === "danger" || v === "high" || v === "critical") return "danger";
    if (v === "caution" || v === "medium") return "caution";
    if (v === "safe" || v === "low") return "safe";
    return "none";
  };
  const riskClass = (sb) => {
    const c = stateClass(sb);
    if (c === "none") return "risk-none";
    if (c === "danger") return "risk-high";
    if (c === "caution") return "risk-medium";
    return "risk-low";
  };

  const englishLane = (lane) => {
    if (!isReal(lane)) return MISSING;
    const v = String(lane).toLowerCase();
    if (v.includes("left")) return "Left Lane";
    if (v.includes("right")) return "Right Lane";
    if (v.includes("center")) return "Same Lane";
    return titleCase(lane) || MISSING;
  };
  const shortLane = (lane) => {
    if (!isReal(lane)) return MISSING;
    const v = String(lane).toLowerCase();
    if (v.includes("left")) return "L. Lane";
    if (v.includes("right")) return "R. Lane";
    if (v.includes("center")) return "C. Lane";
    return titleCase(lane) || MISSING;
  };
  const shortType = (type) => {
    if (!isReal(type)) return MISSING;
    const v = String(type).toLowerCase();
    if (v.includes("left zone")) return "L. Zone";
    if (v.includes("right zone")) return "R. Zone";
    if (v.includes("center zone")) return "C. Zone";
    return titleCase(type) || MISSING;
  };
  const subtitleFor = (sc, lane) => {
    if (sc === "none") return "No analysis yet";
    if (sc === "danger") {
      const l = englishLane(lane);
      if (l !== MISSING && l !== "Same Lane") return `Object approaching from ${l.replace(" Lane", "").toLowerCase()}`;
      if (l === "Same Lane") return "Object approaching in same lane";
      return "High collision risk detected";
    }
    if (sc === "caution") return "Drive with caution";
    return "Normal traffic flow";
  };

  // ─── preview / video ──────────────────────────────────────
  function setPreviewMedia(source) {
    const resolved = mediaSrc(source);
    if (!resolved) {
      previewVideo.hidden = true;
      previewVideo.removeAttribute("src");
      previewFrame.classList.remove("has-media", "is-playing");
      
      // Also clear overlays
      ["depth", "road", "motion"].forEach(name => {
        const img = byId(`visual-${name}-main`);
        if (img) { img.hidden = true; img.removeAttribute("src"); }
      });
      
      timeCurrent.textContent = "00:00";
      timeTotal.textContent = "00:00";
      seekBar.value = 0;
      seekBar.style.setProperty("--fill", "0%");
      previewStatus.hidden = true;
      hidePreviewOverlay();
      updateMapIndicators();
      refreshEmptyStates(true);
      return;
    }
    previewVideo.src = resolved;
    previewVideo.hidden = false;
    previewFrame.classList.add("has-media");
    // Hide all empty-state labels so the center play button becomes visible
    ["empty-video", "empty-depth", "empty-road", "empty-motion"].forEach(id => {
      const el = byId(id);
      if (el) el.hidden = true;
    });
  }

  function setMiniMedia(name, source) {
    const image = byId(`visual-${name}`);
    const frame = byId(`frame-${name}`);
    const mainImage = byId(`visual-${name}-main`);
    
    const resolved = mediaSrc(source);
    
    // Update Mini version if it exists
    if (image && frame) {
      if (!resolved) {
        image.hidden = true;
        image.removeAttribute("src");
        frame.classList.remove("has-media");
      } else {
        image.src = resolved;
        image.hidden = false;
        frame.classList.add("has-media");
      }
    }

    // Update Main version if it exists
    if (mainImage) {
      if (!resolved) {
        mainImage.hidden = true;
        mainImage.removeAttribute("src");
      } else {
        mainImage.src = resolved;
      }
    }
    refreshEmptyStates();
  }

  function refreshEmptyStates(switchingView) {
    const vEmpty = byId("empty-video");
    const dEmpty = byId("empty-depth");
    const rEmpty = byId("empty-road");
    const mEmpty = byId("empty-motion");
    if (!vEmpty || !dEmpty || !rEmpty || !mEmpty) return;

    const mode = state.activeMainView || "video";
    const hasVideo = !!(previewVideo.src && previewVideo.src !== location.href);

    // Always hide all empty labels first
    vEmpty.hidden = dEmpty.hidden = rEmpty.hidden = mEmpty.hidden = true;

    if (mode === "video") {
      if (!hasVideo) {
        vEmpty.hidden = false;
      }
      if (switchingView) {
        previewVideo.hidden = !hasVideo;
        previewBlend.hidden = !(hasVideo && previewBlend.src);
      }
    } else if (mode === "depth") {
      const img = byId("visual-depth-main");
      const hasSrc = img && img.getAttribute("src");
      if (!hasSrc) dEmpty.hidden = false;
      else img.hidden = false;
      if (switchingView) {
        previewVideo.hidden = true;
        previewBlend.hidden = true;
      }
    } else if (mode === "road") {
      const img = byId("visual-road-main");
      const hasSrc = img && img.getAttribute("src");
      if (!hasSrc) rEmpty.hidden = false;
      else img.hidden = false;
      if (switchingView) {
        // KEEP video visible for road mode so player controls work
        previewVideo.hidden = false;
        previewBlend.hidden = true;
      }
    } else if (mode === "motion") {
      const img = byId("visual-motion-main");
      const hasSrc = img && img.getAttribute("src");
      if (!hasSrc) mEmpty.hidden = false;
      else img.hidden = false;
      if (switchingView) {
        previewVideo.hidden = true;
        previewBlend.hidden = true;
      }
    }
  }

  function updateMapIndicators() {
    const depthIndicator = byId("depth-indicator");
    const motionIndicator = byId("motion-indicator");
    const roadIndicator = byId("road-indicator");
    const depthValue = byId("depth-indicator-value");
    const motionValue = byId("motion-indicator-value");
    const motionFill = byId("motion-indicator-fill");
    const mode = state.activeMainView || "video";
    const result = state.lastResult;
    const metrics = result?.metrics || {};
    const hasDepthMap = !!byId("visual-depth-main")?.getAttribute("src");
    const hasMotionMap = !!byId("visual-motion-main")?.getAttribute("src");
    const hasRoadMap = !!byId("visual-road-main")?.getAttribute("src");

    if (depthIndicator) depthIndicator.hidden = mode !== "depth" || !hasDepthMap;
    if (motionIndicator) motionIndicator.hidden = mode !== "motion" || !hasMotionMap;
    if (roadIndicator) roadIndicator.hidden = mode !== "road" || !hasRoadMap;

    if (depthValue) {
      const near = num(metrics.nearScore, null);
      depthValue.textContent = near === null ? MISSING : `${Math.round(clamp(near, 0, 1) * 100)}% Near`;
    }

    if (motionValue || motionFill) {
      const motion = num(metrics.motionMagnitude, null);
      const closing = num(metrics.closingSpeed, null);
      const signal = motion !== null ? motion : closing;
      const pct = signal === null ? 0 : Math.round(clamp(signal, 0, 1) * 100);
      if (motionValue) motionValue.textContent = signal === null ? MISSING : `${pct}% Motion`;
      if (motionFill) motionFill.style.width = `${pct}%`;
    }
  }

  function showPreviewOverlay(source) {
    const resolved = mediaSrc(source);
    if (!resolved) { hidePreviewOverlay(); return; }
    previewBlend.src = resolved;
    previewBlend.hidden = false;
  }
  function hidePreviewOverlay() {
    previewBlend.hidden = true;
    previewBlend.removeAttribute("src");
  }

  // ─── normalize ────────────────────────────────────────────
  function normalizePayload(payload) {
    const event = payload?.peakEvent || payload?.event || payload || {};
    const telemetry = event.telemetry || payload?.telemetry || {};
    const rawImages = payload?.images || event.images || telemetry.images || {};
    const sources = [event, payload, telemetry];

    const riskRaw = firstValue(sources, ["hazardScore", "riskScore", "risk_score"]);
    const riskNum = num(riskRaw, null);
    const riskScore = riskNum === null ? null : clamp(Math.round(riskNum <= 1 ? riskNum * 100 : riskNum), 0, 100);
    const explicitBand = firstValue(sources, ["hazardBand", "riskBand", "riskLevel", "band"]);
    const riskState = firstValue(sources, ["riskState", "risk_state"]);

    const metrics = {
      riskScore,
      band: titleCase(explicitBand),
      state: riskState ? String(riskState).toUpperCase() : null,
      ttc: num(firstValue(sources, ["estimatedTtcSec", "estimated_ttc_sec", "ttc"]), null),
      nearScore: num(firstValue(sources, ["nearScore", "near_score"]), null),
      closingSpeed: num(firstValue(sources, ["closingSpeed", "closing_speed"]), null),
      objectType: titleCase(firstValue(sources, ["objectType", "object_type"])),
      approach: titleCase(firstValue(sources, ["approach"])),
      lane: titleCase(firstValue(sources, ["lane", "primaryZone", "primary_zone"])),
      bbox: event.bbox || telemetry.bbox || null,
      motionMagnitude: num(firstValue(sources, ["velocityMagnitude", "velocity_magnitude", "motionMagnitude", "motion_magnitude"]), null),
    };

    return {
      payload, event, telemetry,
      images: {
        original: rawImages.original,
        depth: rawImages.depth,
        segmentation: rawImages.segmentation,
        road: rawImages.road,
        motion: rawImages.motion,
        blend: rawImages.blend,
      },
      metrics,
      elapsedSec: num(payload?.elapsedSec ?? payload?.elapsed_sec, null),
      fps: num(payload?.fps ?? telemetry?.fps, null),
      frameCount: num(payload?.frameCount, null),
      processedFrames: num(payload?.processedFrames, null),
      sampledFrames: num(payload?.sampledFrames, null),
      summary: event.summary || payload?.summary || null,
      reasons: Array.isArray(event.reasons) ? event.reasons : [],
      zoneMetrics: Array.isArray(event.zoneMetrics) ? event.zoneMetrics : [],

      events: Array.isArray(payload?.events) ? payload.events : [],
      timelineRows: Array.isArray(payload?.timelineRows) ? payload.timelineRows : [],
      sourceName: payload?.sourceName || fileInput.files[0]?.name || null,
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
    timeTotal.textContent = dur;
    timeCurrent.textContent = "00:00";
    
    // Estimate total frames (assume 30fps for UI placeholder)
    const estFrames = Math.round(previewVideo.duration * 30);
    const meta = byId("selected-meta");
    if (meta) {
      meta.innerHTML = `<div>Duration: ${dur}</div><div>Est. Frames: ${estFrames}</div>`;
      meta.hidden = false;
    }
    renderTimeline(null);

    // Auto-set the max frames input to the total estimated frames
    const framesInput = document.querySelector('input[data-param="max_processed_frames"]');
    if (framesInput) {
      framesInput.value = estFrames;
      framesInput.max = estFrames;
      const hidden = formField("max_processed_frames");
      if (hidden) hidden.value = String(estFrames);
    }
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

  // ─── risk banner (used both with peak and follow-mode) ───
  function applyRiskBannerState({ stateLabel, ttc, lane, sc, timeTag }) {
    const banner = byId("risk-banner");
    banner.classList.remove("risk-none", "risk-low", "risk-medium", "risk-high", "risk-critical");
    banner.classList.add(riskClass(stateLabel));
    byId("risk-band-main").textContent = stateLabel || MISSING;
    byId("alert-ttc").textContent = ttcLabel(ttc);
    byId("risk-subtitle").textContent = subtitleFor(sc, lane);
    const tag = byId("risk-time-tag");
    if (tag) {
      if (timeTag) { tag.innerHTML = timeTag; tag.hidden = false; }
      else tag.hidden = true;
    }
  }
  function renderRiskBannerFromMetrics(m) {
    const sc = stateClass(m?.state || m?.band);
    applyRiskBannerState({ stateLabel: m?.state || (m?.band ? m.band.toUpperCase() : null), ttc: m?.ttc, lane: m?.lane, sc });
  }


  // ─── Zone Risk bars ──────────────────────────────────────
  function renderZoneBars(result) {
    const zones = result?.zoneMetrics || [];
    const map = {};
    zones.forEach((z) => {
      const k = String(z.zone || "").toLowerCase();
      if (k) map[k] = z;
    });
    document.querySelectorAll(".zone-row").forEach((row) => {
      const key = row.dataset.zone;
      const z = map[key];
      const fill = row.querySelector(".zone-fill");
      const value = row.querySelector(".zone-value");
      row.classList.remove("is-safe", "is-caution", "is-danger");
      if (!z) {
        fill.style.width = "0%";
        value.textContent = MISSING;
        return;
      }
      const score = num(z.score, 0);
      const ttc = num(z.estimated_ttc_sec, null);
      const pct = clamp(Math.round(score * 100), 0, 100);
      fill.style.width = `${pct}%`;
      value.textContent = String(pct);
      let sc = "safe";
      if ((ttc !== null && ttc < 1.0) || score >= 0.75) sc = "danger";
      else if ((ttc !== null && ttc < 3.0) || score >= 0.45) sc = "caution";
      row.classList.add(`is-${sc}`);
    });
  }

  // ─── Timeline + event strip ──────────────────────────────
  function renderTimeline(result) {
    const events = result?.events || [];
    const dur = state.sourceMeta?.durationSec
      || (result?.frameCount && result?.fps ? result.frameCount / result.fps : null);
    const totalDur = dur || (events.length ? Math.max(...events.map((e) => num(e.timestampSec, 0))) : null) || 1;

    const axis = byId("timeline-axis");
    axis.replaceChildren();
    const ticks = 6;
    for (let i = 0; i < ticks; i++) {
      const sp = document.createElement("span");
      sp.textContent = formatSeconds((totalDur * i) / (ticks - 1));
      axis.appendChild(sp);
    }

    const track = byId("timeline-events");
    track.replaceChildren();
    const strip = byId("event-strip");
    strip.replaceChildren();

    if (!events.length) {
      const e = document.createElement("div");
      e.className = "event-empty";
      e.textContent = "No events yet";
      strip.appendChild(e);
      return;
    }

    events.forEach((ev, idx) => {
      const ts = num(ev.timestampSec, null);
      if (ts === null) return;
      const left = clamp((ts / totalDur) * 100, 0, 100);
      const dot = document.createElement("button");
      dot.type = "button";
      const sc = stateClass(ev.riskState || ev.hazardBand);
      dot.className = `timeline-event ev-${sc}`;
      dot.style.left = `${left}%`;
      dot.title = `#${idx + 1} · ${formatSeconds(ts)} · ${ev.riskState || ev.hazardBand || ""}`;
      dot.addEventListener("click", () => focusEvent(idx));
      

      
      track.appendChild(dot);
    });

    events.forEach((ev, idx) => {
      const sc = stateClass(ev.riskState || ev.hazardBand);
      const ts = num(ev.timestampSec, 0);
      const card = document.createElement("button");
      card.type = "button";
      card.className = `event-card ev-${sc}`;
      card.dataset.index = String(idx);
      card.addEventListener("click", () => focusEvent(idx));

      const thumbImg = mediaSrc(ev?.images?.original || ev?.images?.blend);
      const m = ev.metrics || ev || {};
      const ttcVal = m.ttc !== undefined ? m.ttc : m.estimatedTtcSec;
      const nearVal = num(m.nearScore, null);
      const speedVal = num(m.closingSpeed ?? m.velocityMagnitude, null);

      card.innerHTML = `
        <div class="card-visual">
          ${thumbImg ? `<img src="${thumbImg}" alt="Event">` : '<div style="height:100%; display:grid; place-items:center; color:var(--muted); font-size:11px; font-weight:700;">—</div>'}
        </div>
        <div class="card-info">
          <div class="card-metrics-grid">
            <div class="metric-box m-risk">
              <span class="m-label">Status</span>
              <span class="m-value" style="color:rgba(var(--evrgb),1);">${(ev.riskState || ev.hazardBand || "UNKNOWN").toUpperCase()}</span>
            </div>
            <div class="metric-box">
              <span class="m-label">Time</span>
              <span class="m-value">${formatSeconds(ts)}</span>
            </div>
            <div class="metric-box m-risk">
              <span class="m-label">TTC</span>
              <span class="m-value">${ttcLabel(ttcVal)}</span>
            </div>
            <div class="metric-box">
              <span class="m-label">Near</span>
              <span class="m-value">${nearVal !== null ? (nearVal * 100).toFixed(0) + '%' : '—'}</span>
            </div>
            <div class="metric-box">
              <span class="m-label">Speed</span>
              <span class="m-value">${speedVal !== null ? (speedVal * 100).toFixed(0) + '%' : '—'}</span>
            </div>
            <div class="metric-box">
              <span class="m-label">Type</span>
              <span class="m-value">${shortType(ev.objectType || m.objectType)}</span>
            </div>
          </div>
        </div>
      `;
      strip.appendChild(card);
    });
  }

  function focusEvent(idx) {
    const events = state.events || [];
    const ev = events[idx];
    if (!ev) return;
    state.syncFollowVideo = false;
    const ts = num(ev.timestampSec, null);
    if (ts !== null) {
      try { previewVideo.currentTime = ts; } catch {}
      const totalDur = state.sourceMeta?.durationSec || 1;
      const left = clamp((ts / totalDur) * 100, 0, 100);
      const cursor = byId("timeline-cursor");
      if (cursor) cursor.style.left = `${left}%`;
    }
    const sc = stateClass(ev.riskState || ev.hazardBand);
    applyRiskBannerState({
      stateLabel: ev.riskState || (ev.hazardBand ? String(ev.hazardBand).toUpperCase() : null),
      ttc: ev.estimatedTtcSec,
      lane: ev.lane,
      sc,
      timeTag: ts !== null ? `Event #${idx + 1}: <span>${formatSeconds(ts)}</span>` : null,
    });
    
    // Highlighting active card
    const strip = byId("event-strip");
    if (strip) {
      strip.querySelectorAll(".event-card").forEach((c, i) => {
        c.classList.toggle("is-active", i === idx);
      });
    }

    if (ev?.images?.blend) {
      // Auto switch back to video mode to see the overlay
      if (state.activeMainView !== "video") {
        const videoBtn = document.querySelector(".side-bar-btn[data-view='video']");
        if (videoBtn) videoBtn.click();
      }
      showPreviewOverlay(ev.images.blend);
      setTimeout(() => { if (!state.syncFollowVideo) hidePreviewOverlay(); }, 5000);
    }
    setTimeout(() => { state.syncFollowVideo = true; }, 5100);
  }

  // ─── TTC chart from timelineRows ─────────────────────────
  function renderTtcChart(result) {
    const path = byId("chart-line");
    const area = byId("chart-area");
    const avgTtcEl = byId("stat-avg-ttc");
    if (!path) return;

    const rows = result?.timelineRows || [];
    const points = rows
      .map((r) => ({ t: num(r["Time (s)"], null), ttc: num(r["TTC (s)"], null) }))
      .filter((p) => p.t !== null && p.ttc !== null && p.ttc >= 0)
      .sort((a, b) => a.t - b.t);

    const totalDur = state.sourceMeta?.durationSec
      || (points.length ? Math.max(...points.map((p) => p.t)) : 1);

    if (!points.length) {
      path.setAttribute("d", "");
      area.setAttribute("d", "");
      if (avgTtcEl) avgTtcEl.textContent = "—";
      updateChartAxisX(totalDur || 1);
      return;
    }

    // Calculate stats on raw points
    const avgTtc = points.reduce((acc, p) => acc + p.ttc, 0) / points.length;
    if (avgTtcEl) avgTtcEl.textContent = `${avgTtc.toFixed(1)}s`;

    // Update consolidated statistics grid
    const totalEventsEl = byId("stat-total-events");
    if (totalEventsEl) totalEventsEl.textContent = result?.events?.length || "0";

    // Smooth points for display
    const smoothed = points.map((p, i) => {
      // 5-point moving average window (2 before, 2 after)
      const window = points.slice(Math.max(0, i - 2), Math.min(points.length, i + 3));
      const avg = window.reduce((a, b) => a + b.ttc, 0) / window.length;
      return { ...p, ttc: avg };
    });

    const W = 400, H = 150;
    const maxTtc = 6;
    const xy = smoothed.map((p) => [
      (p.t / Math.max(totalDur, 0.01)) * W,
      H - clamp((p.ttc / maxTtc) * H, 0, H),
    ]);
    const d = xy.map((p, i) => (i === 0 ? `M ${p[0].toFixed(1)} ${p[1].toFixed(1)}` : `L ${p[0].toFixed(1)} ${p[1].toFixed(1)}`)).join(" ");
    path.setAttribute("d", d);
    area.setAttribute("d", `${d} L ${xy[xy.length - 1][0].toFixed(1)} ${H} L ${xy[0][0].toFixed(1)} ${H} Z`);
    updateChartAxisX(totalDur);
  }


  function updateChartAxisX(totalDur) {
    const axis = byId("chart-axis-x");
    if (!axis) return;
    axis.replaceChildren();
    const ticks = 6;
    for (let i = 0; i < ticks; i++) {
      const sp = document.createElement("span");
      sp.textContent = formatSeconds((totalDur * i) / (ticks - 1));
      axis.appendChild(sp);
    }
  }

  function updateChartCursor(timeSec) {
    const cursor = byId("chart-cursor");
    if (!cursor) return;
    const totalDur = state.sourceMeta?.durationSec || 1;
    const ratio = clamp(timeSec / Math.max(totalDur, 0.01), 0, 1);
    const x = ratio * 400;
    cursor.setAttribute("x1", String(x));
    cursor.setAttribute("x2", String(x));
    cursor.setAttribute("opacity", state.timelineRows.length ? "0.7" : "0");
  }


  // ─── video-time sync to timelineRows ─────────────────────
  function findTimelineRowAt(timeSec) {
    const rows = state.timelineRows;
    if (!rows.length) return null;
    let best = 0, bestDiff = Infinity;
    for (let i = 0; i < rows.length; i++) {
      const t = num(rows[i]["Time (s)"], null);
      if (t === null) continue;
      const diff = Math.abs(t - timeSec);
      if (diff < bestDiff) { bestDiff = diff; best = i; }
    }
    return rows[best];
  }

  function syncToVideoTime() {
    if (!state.syncFollowVideo) return;
    const t = previewVideo?.currentTime ?? 0;
    updateChartCursor(t);
    const cursor = byId("timeline-cursor");
    if (cursor) {
      const totalDur = state.sourceMeta?.durationSec || 1;
      const left = clamp((t / Math.max(totalDur, 0.01)) * 100, 0, 100);
      cursor.style.left = `${left}%`;
    }
    const row = findTimelineRowAt(t);
    if (!row) return;
    const stateLabel = row.State ? String(row.State).toUpperCase() : null;
    const sc = stateClass(stateLabel);
    applyRiskBannerState({
      stateLabel,
      ttc: row["TTC (s)"],
      lane: row.Zone,
      sc,
      timeTag: `Live: <span>${formatSeconds(num(row["Time (s)"], 0))}</span>`,
    });

    if (state.activeMainView !== "video") {
      updateMainImageFromTime(t, state.activeMainView);
    }
  }

  function updateMainImageFromTime(timeSec, viewMode) {
    if (!viewMode || viewMode === "video") return;
    const events = state.events || [];
    if (!events.length) return;

    let bestEvent = null;
    let minDiff = Infinity;
    for (const ev of events) {
      const diff = Math.abs(ev.timestampSec - timeSec);
      if (diff < minDiff) {
        minDiff = diff;
        bestEvent = ev;
      }
    }

    if (bestEvent && bestEvent.images && bestEvent.images[viewMode]) {
      const mainImg = byId(`visual-${viewMode}-main`);
      if (mainImg) {
        const src = mediaSrc(bestEvent.images[viewMode]);
        if (mainImg.src !== src) mainImg.src = src;
        mainImg.hidden = false;
      }
    }
  }

  // ─── full render ─────────────────────────────────────────
  function renderResult(payload) {
    const result = normalizePayload(payload);
    state.lastResult = result;
    state.timelineRows = result.timelineRows || [];
    state.events = result.events || [];

    setMiniMedia("depth", result.images.depth);
    setMiniMedia("segmentation", result.images.segmentation);
    setMiniMedia("road", result.images.road);
    setMiniMedia("motion", result.images.motion);

    showPreviewOverlay(result.images.blend);
    setTimeout(() => hidePreviewOverlay(), 5000);

    renderStatRow(result);
    renderRiskBannerFromMetrics(result.metrics);

    renderZoneBars(result);
    renderTimeline(result);
    renderTtcChart(result);
    updateMapIndicators();
  }

  function renderEmptyState() {
    state.lastResult = null;
    state.timelineRows = [];
    state.events = [];
    setMiniMedia("depth", "");
    setMiniMedia("segmentation", "");
    setMiniMedia("road", "");
    setMiniMedia("motion", "");
    hidePreviewOverlay();
    renderStatRow(null);
    renderRiskBannerFromMetrics({ state: null, band: null, ttc: null, lane: null });

    renderZoneBars({ zoneMetrics: [] });
    renderTimeline({ events: [] });
    renderTtcChart({ timelineRows: [] });

    updateChartCursor(0);
    updateMapIndicators();
  }

  // ─── analysis flow ───────────────────────────────────────
  function buildFormData() {
    const fd = new FormData(form);
    fd.set("mode", "video");

    return fd;
  }
  async function parseResponse(response) {
    const text = await response.text();
    if (!text) return {};
    try { return JSON.parse(text); } catch { return { detail: text }; }
  }
  async function postAnalysis() {
    const response = await fetch("/api/analyze", { method: "POST", body: buildFormData() });
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
    previewStatus.hidden = true;
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
    previewStatus.hidden = true;
    window.setTimeout(() => {
      if (!state.analyzing) hideProgress();
    }, isError ? 3200 : 1800);
  }
  function hideProgress() {
    const el = byId("analysis-progress");
    if (el) el.hidden = true;
    if (el) el.classList.remove("is-complete", "is-error");
    if (state.progressTimer) { clearInterval(state.progressTimer); state.progressTimer = null; }
    previewStatus.hidden = true;
  }
  function setRunningUI(isRunning) {
    state.analyzing = isRunning;
    runButton.classList.toggle("is-running", isRunning);
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
    try {
      const payload = await postAnalysis();
      renderResult(payload);
      finishProgress({ label: "Completed", status: "Analysis complete." });
    } catch (err) {
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
    
    // Auto switch to video mode on upload
    state.activeMainView = "video";
    document.querySelectorAll(".side-bar-btn[data-view]").forEach(b => {
      b.classList.toggle("is-active", b.dataset.view === "video");
    });
    updateMapIndicators();

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

  // ─── help modal ──────────────────────────────────────────
  function openHelpModal() {
    const modal = byId("help-modal");
    if (!modal) return;
    modal.hidden = false;
    void modal.offsetHeight;
    modal.classList.add("is-open");
  }
  function closeHelpModal() {
    const modal = byId("help-modal");
    if (!modal) return;
    modal.classList.remove("is-open");
    setTimeout(() => { if (!modal.classList.contains("is-open")) modal.hidden = true; }, 400);
  }



  function setupSegmentedControls() {
    document.querySelectorAll(".segmented-control").forEach(ctrl => {
      const param = ctrl.dataset.param;
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
    previewVideo.addEventListener("loadedmetadata", updateSourceMetaFromVideo);
    previewVideo.addEventListener("timeupdate", () => {
      const cur = previewVideo.currentTime;
      const dur = previewVideo.duration || 0;
      timeCurrent.textContent = formatSeconds(cur);
      if (dur > 0) {
        const ratio = clamp(cur / dur, 0, 1);
        seekBar.value = String(Math.round(ratio * 1000));
        seekBar.style.setProperty("--fill", `${(ratio * 100).toFixed(1)}%`);
      }
      syncToVideoTime();
    });
    previewVideo.addEventListener("seeked", syncToVideoTime);
    previewVideo.addEventListener("play", () => previewFrame.classList.add("is-playing"));
    previewVideo.addEventListener("pause", () => previewFrame.classList.remove("is-playing"));
    previewVideo.addEventListener("ended", () => previewFrame.classList.remove("is-playing"));

    playToggle.addEventListener("click", () => {
      if (previewVideo.hidden || !previewVideo.src) return;
      if (previewVideo.paused) {
        hidePreviewOverlay(); // Hide any event overlay when starting play
        previewVideo.play().catch(() => {});
      } else {
        previewVideo.pause();
      }
    });
    seekBar.addEventListener("input", () => {
      const ratio = num(seekBar.value, 0) / 1000;
      seekBar.style.setProperty("--fill", `${(ratio * 100).toFixed(1)}%`);
      if (previewVideo.duration) previewVideo.currentTime = ratio * previewVideo.duration;
    });
    byId("center-play-btn")?.addEventListener("click", () => {
      if (!previewVideo.src) return;
      if (previewVideo.paused) {
        hidePreviewOverlay();
        previewVideo.play().catch(() => {});
      } else {
        previewVideo.pause();
      }
    });
    byId("preview-expand")?.addEventListener("click", () => expandMedia(previewVideo.src, "video"));
  }

  function expandMedia(src, kind) {
    if (!src) return;
    const overlay = document.createElement("div");
    overlay.className = "media-overlay";
    const close = document.createElement("button");
    close.type = "button";
    close.className = "close-btn";
    close.innerHTML = '<svg><use href="#icon-x"></use></svg>';
    overlay.appendChild(close);
    const media = kind === "video" ? document.createElement("video") : document.createElement("img");
    media.src = src;
    if (kind === "video") {
      media.controls = true; media.autoplay = true; media.muted = true; media.playsInline = true;
    }
    overlay.appendChild(media);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay || e.target.closest(".close-btn")) overlay.remove();
    });
    document.body.appendChild(overlay);
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
      }
    });


    fileInput.addEventListener("change", handleFileSelection);
    form.addEventListener("submit", analyzeSelectedFile);
    uploadButton.addEventListener("click", () => fileInput.click());
    removeBtn?.addEventListener("click", clearSelectedSource);

    byId("open-settings")?.addEventListener("click", openDrawer);
    byId("open-help")?.addEventListener("click", openHelpModal);
    
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
        input.value = "360";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      }
    });

    document.querySelectorAll("[data-drawer-close]").forEach((el) => el.addEventListener("click", closeDrawer));
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });

    document.querySelectorAll("[data-help-close]").forEach(el => {
      el.addEventListener("click", closeHelpModal);
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        closeHelpModal();
      }
    });

    setupPreviewControls();
    setupSegmentedControls();

    document.querySelectorAll(".side-bar-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const toggleId = btn.dataset.toggle;
        const viewMode = btn.dataset.view;

        if (toggleId) {
          const panel = byId(toggleId);
          if (panel) {
            panel.hidden = !panel.hidden;
            btn.classList.toggle("is-active", !panel.hidden);
          }
        } else if (viewMode) {
          const isActive = btn.classList.contains("is-active");

          // Reset all view buttons and overlays
          document.querySelectorAll(".side-bar-btn[data-view]").forEach(b => b.classList.remove("is-active"));
          const mainDepth = byId("visual-depth-main");
          const mainRoad = byId("visual-road-main");
          const mainMotion = byId("visual-motion-main");
          if (mainDepth) mainDepth.hidden = true;
          if (mainRoad) mainRoad.hidden = true;
          if (mainMotion) mainMotion.hidden = true;

          if (viewMode === "video" || isActive) {
            // Switch to video (or toggle off active mode = back to video)
            const videoBtn = document.querySelector(".side-bar-btn[data-view='video']");
            if (videoBtn) videoBtn.classList.add("is-active");
            state.activeMainView = "video";
          } else {
            btn.classList.add("is-active");
            state.activeMainView = viewMode;
            // Stop video playback when in analysis modes
            if (previewVideo && !previewVideo.paused) {
              previewVideo.pause();
            }
          }

          // Center play button and player bar only in video and road mode
          const centerBtn = byId("center-play-btn");
          const playerBar = document.querySelector(".preview-bar");
          const isVideoOrRoad = (state.activeMainView === "video" || state.activeMainView === "road");

          if (centerBtn) {
            centerBtn.classList.toggle("force-hide", !isVideoOrRoad);
          }
          if (playerBar) {
            playerBar.classList.toggle("force-hide", !isVideoOrRoad);
          }

          refreshEmptyStates(true);
          updateMapIndicators();
        }
      });
    });

    renderEmptyState();
  }

  initialize();
})();
