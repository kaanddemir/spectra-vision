(() => {
  const MISSING = "—";
  const presets = {
    balanced:  { max_processed_frames: "180", depth_every: "3", detect_every: "3" },
    fast:      { max_processed_frames: "90",  depth_every: "5", detect_every: "5" },
    precision: { max_processed_frames: "300", depth_every: "2", detect_every: "2" },
  };

  const form = document.querySelector("#analysis-form");
  const fileInput = document.querySelector("#source-file");
  const runButton = document.querySelector("#run-analysis");
  const uploadButton = document.querySelector("#top-upload");
  const removeBtn = document.querySelector("#remove-source");
  const selectedChip = document.querySelector("#selected-chip");
  const settingsDrawer = document.querySelector("#settings-drawer");
  const presetPicker = document.querySelector("#preset-picker");
  const detectorButtons = document.querySelectorAll("[data-detector]");
  const previewVideo = document.querySelector("#visual-original-video");
  const previewBlend = document.querySelector("#visual-blend");
  const previewFrame = document.querySelector("#frame-original");
  const playToggle = document.querySelector("#play-toggle");
  const seekBar = document.querySelector("#seek-bar");
  const timeCurrent = document.querySelector("#time-current");
  const timeTotal = document.querySelector("#time-total");
  const previewQuality = null;

  const previewStatus = document.querySelector("#preview-status");

  const state = {
    previewUrl: "",
    lastResult: null,
    viewMode: "original",
    sourceMeta: null,
    analyzing: false,

    progressTimer: null,
    progressStart: 0,
    timelineRows: [],
    events: [],
    peakMetrics: null,
    syncFollowVideo: true,
    activeMainView: "video",
    bboxCrop: null,
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

  const englishApproach = (approach) => {
    if (!isReal(approach)) return MISSING;
    const v = String(approach).toLowerCase();
    if (v.includes("approach") || v.includes("clos")) return "Approaching";
    if (v.includes("stable")) return "Stable";
    return titleCase(approach) || MISSING;
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

  // ─── toast / status (stubs – currently no-op) ────────────
  function showToast() {}
  function setStatus() {}

  // ─── preview / video ──────────────────────────────────────
  function setPreviewMedia(source) {
    const resolved = mediaSrc(source);
    if (!resolved) {
      previewVideo.hidden = true;
      previewVideo.removeAttribute("src");
      previewFrame.classList.remove("has-media", "is-playing");
      
      // Also clear overlays
      ["depth", "motion"].forEach(name => {
        const img = byId(`visual-${name}-main`);
        if (img) { img.hidden = true; img.removeAttribute("src"); }
      });
      
      previewQuality && (previewQuality.hidden = true);
      timeCurrent.textContent = "00:00";
      timeTotal.textContent = "00:00";
      seekBar.value = 0;
      seekBar.style.setProperty("--fill", "0%");
      previewStatus.hidden = true;
      hidePreviewOverlay();
      refreshEmptyStates(true);
      return;
    }
    previewVideo.src = resolved;
    previewVideo.hidden = false;
    previewFrame.classList.add("has-media");
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
    const mEmpty = byId("empty-motion");
    if (!vEmpty || !dEmpty || !mEmpty) return;

    const mode = state.activeMainView || "video";
    const hasVideo = !!(previewVideo.src && previewVideo.src !== location.href);

    // Always hide all empty labels first
    vEmpty.hidden = dEmpty.hidden = mEmpty.hidden = true;

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
    if (previewQuality) previewQuality.hidden = true;
    
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


  function applyBboxCrop(img, container, bbox) {
    if (!container) return;
    container.classList.remove("is-cropped");
    img.style.cssText = "";
    if (!Array.isArray(bbox) || bbox.length < 4 || !img.naturalWidth || !img.naturalHeight) return;
    const [x1, y1, x2, y2] = bbox.map(Number);
    if (![x1, y1, x2, y2].every(Number.isFinite)) return;
    const bboxW = Math.max(1, x2 - x1);
    const bboxH = Math.max(1, y2 - y1);
    const cw = container.clientWidth || 110;
    const ch = container.clientHeight || 110;
    const pad = 0.2; // 20% padding around bbox so it's not too tight
    const targetW = bboxW * (1 + pad);
    const targetH = bboxH * (1 + pad);
    const scale = Math.min(cw / targetW, ch / targetH);
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;
    const left = cw / 2 - cx * scale;
    const top = ch / 2 - cy * scale;
    img.style.position = "absolute";
    img.style.width = `${img.naturalWidth * scale}px`;
    img.style.height = `${img.naturalHeight * scale}px`;
    img.style.left = `${left}px`;
    img.style.top = `${top}px`;
    img.style.maxWidth = "none";
    img.style.objectFit = "none";
    container.classList.add("is-cropped");
  }

  // ─── Analysis Modal Control ──────────────────────────────
  function openAnalysisModal(mode) {
    const res = state.lastResult;
    if (!res) return;
    const src = res.images?.[mode];
    if (!src) return;

    const modal = byId("analysis-modal");
    const modalImg = byId("modal-image");
    const modalTitle = byId("modal-title");

    const titles = {
      depth: "Depth Map Detail",
      motion: "Motion Flow Detail",
      segmentation: "Segmentation Detail"
    };

    modalTitle.textContent = titles[mode] || "Analysis Detail";
    modalImg.src = mediaSrc(src);
    modal.hidden = false;
  }

  function closeAnalysisModal() {
    const modal = byId("analysis-modal");
    if (modal) modal.hidden = true;
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
      if ((ttc !== null && ttc < 1.5) || score >= 0.75) sc = "danger";
      else if ((ttc !== null && ttc < 3.0) || score >= 0.45) sc = "caution";
      row.classList.add(`is-${sc}`);
    });
  }

  function renderMotionMagnitude(mag) {
    const textEl = byId("motion-current");
    const barEl = byId("motion-bar-fill");
    if (!textEl || !barEl) return;
    if (mag === null) {
      textEl.textContent = "0.0";
      barEl.style.width = "0%";
      return;
    }
    textEl.textContent = mag.toFixed(1);
    const pct = clamp((mag / 40) * 100, 0, 100);
    barEl.style.width = `${pct}%`;
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
    const countTag = byId("event-count");

    if (!events.length) {
      const e = document.createElement("div");
      e.className = "event-empty";
      e.textContent = "No events yet";
      strip.appendChild(e);
      if (countTag) countTag.textContent = "0";
      return;
    }
    if (countTag) countTag.textContent = events.length;

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
              <span class="m-label">Lane</span>
              <span class="m-value">${shortLane(ev.lane || m.lane)}</span>
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
    const peakRiskEl = byId("stat-peak-risk");
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
      if (peakRiskEl) peakRiskEl.textContent = "—";
      if (avgTtcEl) avgTtcEl.textContent = "—";
      updateChartAxisX(totalDur || 1);
      return;
    }

    // Calculate stats on raw points
    const minTtc = Math.min(...points.map(p => p.ttc));
    const avgTtc = points.reduce((acc, p) => acc + p.ttc, 0) / points.length;
    if (peakRiskEl) peakRiskEl.textContent = `${minTtc.toFixed(1)}s`;
    if (avgTtcEl) avgTtcEl.textContent = `${avgTtc.toFixed(1)}s`;

    // Update consolidated statistics grid
    const totalEventsEl = byId("stat-total-events");
    if (totalEventsEl) totalEventsEl.textContent = result?.events?.length || "0";

    // Smooth points for display
    const smoothed = points.map((p, i) => {
      const window = points.slice(Math.max(0, i - 1), Math.min(points.length, i + 2));
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
  }

  // ─── full render ─────────────────────────────────────────
  function renderResult(payload) {
    const result = normalizePayload(payload);
    state.lastResult = result;
    state.timelineRows = result.timelineRows || [];
    state.events = result.events || [];
    state.peakMetrics = result.metrics;

    setMiniMedia("depth", result.images.depth);
    setMiniMedia("segmentation", result.images.segmentation);
    setMiniMedia("motion", result.images.motion);

    showPreviewOverlay(result.images.blend);
    setTimeout(() => hidePreviewOverlay(), 5000);

    renderStatRow(result);
    renderRiskBannerFromMetrics(result.metrics);

    renderZoneBars(result);
    renderMotionMagnitude(result.metrics.motionMagnitude);
    renderTimeline(result);
    renderTtcChart(result);
  }

  function renderEmptyState() {
    state.lastResult = null;
    state.timelineRows = [];
    state.events = [];
    state.peakMetrics = null;
    setMiniMedia("depth", "");
    setMiniMedia("segmentation", "");
    setMiniMedia("motion", "");
    hidePreviewOverlay();
    renderStatRow(null);
    renderRiskBannerFromMetrics({ state: null, band: null, ttc: null, lane: null });

    renderZoneBars({ zoneMetrics: [] });
    renderMotionMagnitude(null);
    renderTimeline({ events: [] });
    renderTtcChart({ timelineRows: [] });

    updateChartCursor(0);
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
    detectorButtons.forEach((b) => (b.disabled = isRunning));
  }

  async function analyzeSelectedFile(event) {
    event?.preventDefault();
    if (!fileInput.files.length) {
      showToast("Please select a video first.", "warning");
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

    if (!file) { setPreviewMedia(""); state.sourceMeta = null; return; }
    state.previewUrl = URL.createObjectURL(file);
    setPreviewMedia(state.previewUrl);
    setStatus("Source ready. Run the analysis.", false);
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
    if (modal) modal.hidden = false;
  }
  function closeHelpModal() {
    const modal = byId("help-modal");
    if (modal) modal.hidden = true;
  }

  function applyPreset() {
    const values = presets[presetPicker?.value] || presets.balanced;
    Object.entries(values).forEach(([name, value]) => {
      const hidden = formField(name);
      if (hidden) hidden.value = value;
      
      const sliderEl = document.querySelector(`input[data-param="${name}"]`);
      if (sliderEl) {
        sliderEl.value = value;
        if (sliderEl.type === "range") setRangeFill(sliderEl);
      }

      const segmented = document.querySelector(`.segmented-control[data-param="${name}"]`);
      if (segmented) {
        segmented.querySelectorAll(".segmented-btn").forEach(btn => {
          const isActive = btn.dataset.value === String(value);
          btn.classList.toggle("active", isActive);
        });
      }
    });
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

  function setDetectorMode(mode) {
    const normalized = mode === "yolo" ? "yolo" : "zone";
    const hidden = formField("enable_yolo");
    if (hidden) hidden.value = normalized === "yolo" ? "true" : "false";
    detectorButtons.forEach((btn) => {
      const isActive = btn.dataset.detector === normalized;
      btn.classList.toggle("active", isActive);
      btn.setAttribute("aria-pressed", String(isActive));
    });

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
      if (previewVideo.hidden || !previewVideo.src) return;
      hidePreviewOverlay();
      previewVideo.play().catch(() => {});
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

    presetPicker?.addEventListener("change", applyPreset);
    fileInput.addEventListener("change", handleFileSelection);
    form.addEventListener("submit", analyzeSelectedFile);
    uploadButton.addEventListener("click", () => fileInput.click());
    removeBtn?.addEventListener("click", clearSelectedSource);
    detectorButtons.forEach((b) => b.addEventListener("click", () => setDetectorMode(b.dataset.detector)));

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

    document.querySelectorAll(".mini-expand-btn").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        openAnalysisModal(btn.dataset.view);
      });
    });

    document.querySelectorAll("[data-modal-close]").forEach(el => {
      el.addEventListener("click", closeAnalysisModal);
    });

    document.querySelectorAll("[data-help-close]").forEach(el => {
      el.addEventListener("click", closeHelpModal);
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        closeAnalysisModal();
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
          const mainMotion = byId("visual-motion-main");
          if (mainDepth) mainDepth.hidden = true;
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

          // Center play button and player bar only in video mode
          const centerBtn = byId("center-play-btn");
          const playerBar = document.querySelector(".preview-bar");
          const isVideo = (state.activeMainView === "video");

          if (centerBtn) {
            centerBtn.classList.toggle("force-hide", !isVideo);
          }
          if (playerBar) {
            playerBar.classList.toggle("force-hide", !isVideo);
          }

          refreshEmptyStates(true);
        }
      });
    });

    const centerPlayBtn = byId("center-play-btn");
    if (centerPlayBtn) {
      centerPlayBtn.addEventListener("click", () => {
        if (previewVideo.hidden || !previewVideo.src) return;
        if (previewVideo.paused) previewVideo.play().catch(() => {}); else previewVideo.pause();
      });
    }

    setDetectorMode(formField("enable_yolo")?.value === "true" ? "yolo" : "zone");

    renderEmptyState();
  }

  initialize();
})();
