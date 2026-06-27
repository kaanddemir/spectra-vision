"""Object-centric collision ETA and risk evaluation.

Spectra keeps two outputs separate:

    1. Collision ETA — physical time-to-collision from depth + longitudinal
       Kalman only.
    2. Risk evaluation — proximity, approach, lane crossing, detector
       confidence, brake lights and visual motion cues.

Expansion and optical flow can still influence risk, but they are not exposed
as user-facing collision time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .tracking import Track
from ..vision.brake_lights import brake_score
from ..vision.depth import DepthResult
from ..vision.detection import CLASS_RISK_WEIGHT
from ..vision.motion import FlowResult
from ..vision.road import LaneFrame, lane_corridor_relevance, lane_position


@dataclass
class TtcComponent:
    name: str
    value: Optional[float]
    confidence: float


@dataclass
class RiskEvent:
    frame_index: int
    timestamp_sec: float
    state: str
    # Physical collision ETA from depth + Kalman only. Visual TTC hints may
    # still be used internally for risk classification, but they must not be
    # surfaced through this field.
    ttc_sec: float | None
    direction: str
    lane: str
    object_type: str | None
    confidence: float
    near_score: float
    velocity_magnitude: float
    closing_speed: float
    bbox: tuple[int, int, int, int] | None
    reason: str
    object_id: int | None = None
    display_id: int | None = None
    expansion_rate: float = 0.0
    lateral_velocity_norm: float = 0.0
    crossing_risk: float = 0.0
    lane_position: float = 0.0
    ttc_components: tuple[TtcComponent, ...] = ()
    brake_score: float = 0.0
    distance_m: float | None = None
    closing_mps: float | None = None
    depth_ttc_sec: float | None = None
    detection_confidence: float = 0.0
    tracking_confidence: float = 0.0
    depth_confidence: float = 0.0


@dataclass(frozen=True)
class SpatialFields:
    depth: DepthResult
    flow: FlowResult
    lane: LaneFrame
    flow_dt_sec: float
    depth_is_fresh: bool
    bgr: "np.ndarray | None" = None  # raw frame for appearance cues (brake lights)


# Hard floors and ceilings on the reported TTC. The fusion can swing wildly
# when bboxes flicker by 1-2 px between frames; clamping keeps the downstream
# UI from showing nonsense values.
_TTC_MIN_EXPANSION_RATE = 0.01  # per-second; below this we treat as "stable"
_TTC_FLOOR_SEC = 0.15
_TTC_MAX_REPORTED_SEC = 15.0

# Expansion guards: tiny bboxes make the scale ratio jitter wildly (a 1-2 px
# size change is a huge relative change), so we don't trust expansion below
# this size, and we clamp the per-step scale ratio so single-frame detection
# noise cannot fabricate a giant approach rate.
_EXPANSION_MIN_BBOX_PX = 12.0
_EXPANSION_MAX_SCALE_RATIO = 2.0
_EXPANSION_MIN_SCALE_RATIO = 0.5
# Kinematic (depth) TTC loses reliability as the last measurement ages: the
# filter coasts on prediction between depth refreshes, so its fusion confidence
# decays with the gap (full trust up to ~0.5s, decaying to a floor for
# multi-second gaps).
_DEPTH_TTC_FRESH_DT_SEC = 0.5
_DEPTH_TTC_STALE_DECAY_SEC = 2.0
_DEPTH_TTC_MIN_FRESHNESS = 0.3

# ── Longitudinal constant-velocity Kalman (the physical TTC backbone) ─────────
# Each track's metric distance is filtered into a smooth (distance, range-rate)
# state. Range-rate ``s`` is d(distance)/dt (negative = approaching), so
# closing speed is ``-s`` and TTC = distance / closing. Because TTC is the
# ratio d/(-s), it is invariant to a constant scale error in the (uncalibrated)
# monocular depth — the right backbone even when absolute metres are biased.
_LON_PROCESS_ACCEL_STD = 4.0        # m/s^2 range-rate process noise (manoeuvre)
_LON_MEAS_REL_STD = 0.05            # measurement std as fraction of distance
_LON_MEAS_ABS_STD = 0.5             # measurement std floor, metres
_LON_INIT_VEL_VAR = 15.0 ** 2       # (m/s)^2 initial range-rate variance
_LON_GATE_SQ = 9.0                  # 3-sigma innovation gate (reject depth jumps)
_LON_MIN_CLOSING_MPS = 0.30         # below this closing we report no TTC
_LON_MAX_CLOSING_MPS = 60.0         # physical clamp on the exposed closing speed
_LON_MIN_UPDATES_FOR_TTC = 2        # need 2 measurements before trusting velocity

# State thresholds. Tuned for highway/urban dashcam: 1 s gives a driver one
# reaction-time before impact, 3 s is roughly the recommended following gap.
_DANGER_TTC_SEC = 1.0
_CAUTION_TTC_SEC = 3.0

# Smoothing constants for per-track expansion rate.
_EXPANSION_EMA_RISE = 0.55
_EXPANSION_EMA_FALL = 0.30

# Smoothing for the displayed per-track fused confidence. Display stability
# only — classify_state still uses raw detection confidence, so this never
# changes a risk decision. Rises fast so a newly-confident detection isn't
# held back; falls slower to ride out one-frame YOLO confidence dips.
_CONFIDENCE_EMA_RISE = 0.60
_CONFIDENCE_EMA_FALL = 0.30

# Maximum lateral velocity in lane-units per second that we still treat as
# real path-cross. Above this is bbox jitter or detection swap.
_LATERAL_MAX_LANE_PER_SEC = 2.5
_LANE_TRUST_FLOOR = 0.45

# Classes with rear brake lamps worth inspecting for a deceleration cue.
_BRAKE_LIGHT_CLASSES = {"car", "truck", "bus"}
# Brake-light confidence above which an in-path lead vehicle escalates risk.
_BRAKE_ESCALATE_SCORE = 0.5

# Approach score normalization. Metric depth closing is the primary approach
# cue; bbox expansion and optical flow remain corroborating image-space cues.
_APPROACH_MIN_CLOSING_MPS = 0.30
_APPROACH_FULL_CLOSING_MPS = 12.0

# Collision-cone: how much to trust the velocity-extrapolated crossing for a
# *far* object (near the horizon), whose lane_position is computed from few
# pixels and is jittery. Near objects get full trust (1.0).
_CROSSING_FAR_RELIABILITY_FLOOR = 0.4


def lane_bucket_from_position(pos: float) -> str:
    if pos < -0.7:
        return "left"
    if pos > 0.7:
        return "right"
    return "center"


def direction_from_lateral(lateral_velocity_lane_per_sec: float) -> str:
    if abs(lateral_velocity_lane_per_sec) < 0.1:
        return "center"
    return "left" if lateral_velocity_lane_per_sec < 0.0 else "right"


def _ema(prev: float, value: float, rise_alpha: float, fall_alpha: float) -> float:
    alpha = rise_alpha if value > prev else fall_alpha
    return float((alpha * value) + ((1.0 - alpha) * prev))


# ── TTC source 1: scale expansion ─────────────────────────────────────────────


def expansion_rate_from_track(track: Track, *, min_dt: float = 0.06) -> float:
    sample = track.previous_sample(min_dt=min_dt)
    if sample is None:
        return 0.0
    dt = float(track.timestamp_sec - sample.timestamp_sec)
    if dt <= 0.0:
        return 0.0

    px1, py1, px2, py2 = sample.bbox
    cx1, cy1, cx2, cy2 = track.bbox
    prev_w = max(1.0, float(px2 - px1))
    prev_h = max(1.0, float(py2 - py1))
    curr_w = max(1.0, float(cx2 - cx1))
    curr_h = max(1.0, float(cy2 - cy1))

    # Too small to trust: a couple-pixel size change dominates the ratio.
    if min(prev_w, prev_h, curr_w, curr_h) < _EXPANSION_MIN_BBOX_PX:
        return 0.0

    scale_ratio = math.sqrt((curr_w / prev_w) * (curr_h / prev_h))
    # Clamp so one noisy detection frame can't fabricate a huge approach rate.
    scale_ratio = float(
        np.clip(scale_ratio, _EXPANSION_MIN_SCALE_RATIO, _EXPANSION_MAX_SCALE_RATIO)
    )
    return float((scale_ratio - 1.0) / dt)


def ttc_from_expansion(expansion_rate: float, *, history_age: int) -> TtcComponent:
    if expansion_rate < _TTC_MIN_EXPANSION_RATE:
        return TtcComponent("expansion", None, 0.0)
    raw = 1.0 / float(expansion_rate)
    ttc = max(_TTC_FLOOR_SEC, raw)
    if ttc > _TTC_MAX_REPORTED_SEC:
        return TtcComponent("expansion", None, 0.0)
    # Confidence grows with how many samples back the history goes (a one-
    # sample history can be a single-frame jitter; >3 samples is reliable).
    # We allow a baseline confidence even for new tracks so the UI shows data early.
    confidence = float(np.clip(0.15 + (history_age / 4.0), 0.0, 1.0))
    return TtcComponent("expansion", round(float(ttc), 2), confidence)


# ── TTC source 2: radial flow toward the bbox ─────────────────────────────────


def ttc_from_flow(
    bbox: tuple[int, int, int, int],
    flow: np.ndarray,
    vanishing_point: tuple[float, float],
    flow_dt_sec: float,
) -> TtcComponent:
    """TTC inferred from radial outward flow inside the bbox.

    flow is (h, w, 2) ego-compensated optical flow in pixels/frame. The
    measured frame interval converts frames-to-impact into seconds.
    """

    h_full, w_full = flow.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w_full, x1))
    x2 = max(0, min(w_full, x2))
    y1 = max(0, min(h_full, y1))
    y2 = max(0, min(h_full, y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return TtcComponent("flow", None, 0.0)

    crop = flow[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    ys, xs = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    vx, vy = vanishing_point
    radial_x = xs - float(vx)
    radial_y = ys - float(vy)
    radial_dist = np.sqrt((radial_x * radial_x) + (radial_y * radial_y))
    radial_dist = np.maximum(radial_dist, 1.0)

    radial_unit_x = radial_x / radial_dist
    radial_unit_y = radial_y / radial_dist
    radial_velocity = (crop[..., 0] * radial_unit_x) + (crop[..., 1] * radial_unit_y)

    # Use the high-percentile outward velocity to suppress static background
    # pixels mixed into the bbox.
    radial_p75 = float(np.percentile(radial_velocity, 75))
    if radial_p75 <= 0.1:
        return TtcComponent("flow", None, 0.0)

    # Coherence gate: a genuinely approaching object expands outward across
    # most of its bbox. A receding or mixed object can still leave a positive
    # p75 from noise, so require a majority of pixels to flow outward and let
    # that fraction temper the confidence.
    outward_fraction = float(np.mean(radial_velocity > 0.0))
    if outward_fraction < 0.5:
        return TtcComponent("flow", None, 0.0)

    bbox_cx = (x1 + x2) / 2.0
    bbox_cy = (y1 + y2) / 2.0
    distance_from_vp = float(math.sqrt((bbox_cx - vx) ** 2 + (bbox_cy - vy) ** 2))
    distance_from_vp = max(distance_from_vp, 1.0)

    ttc_frames = distance_from_vp / radial_p75
    dt = float(np.clip(flow_dt_sec, 1.0 / 120.0, 1.0))
    ttc_sec = ttc_frames * dt
    if ttc_sec < _TTC_FLOOR_SEC:
        ttc_sec = _TTC_FLOOR_SEC
    if ttc_sec > _TTC_MAX_REPORTED_SEC:
        return TtcComponent("flow", None, 0.0)

    # Confidence: higher when the radial signal stands well clear of zero and
    # the outward motion is spatially coherent across the bbox.
    confidence = float(np.clip(radial_p75 / 6.0, 0.0, 1.0)) * outward_fraction
    return TtcComponent("flow", round(float(ttc_sec), 2), confidence)


# ── TTC source 3: longitudinal kinematic (constant-velocity Kalman) ──────────


@dataclass
class _LonState:
    """Per-track longitudinal Kalman state: distance ``d`` and range-rate ``s``.

    ``s`` is d(distance)/dt, so it is negative while the object approaches and
    closing speed is ``-s``. ``t`` is the timestamp of the last committed
    measurement; the filter coasts (predict-only) on frames between depth
    refreshes. ``P`` is the 2x2 state covariance; ``n`` counts measurements.
    """

    t: float
    d: float
    s: float
    P: np.ndarray
    n: int


def _lon_predict(state: _LonState, timestamp_sec: float) -> tuple[float, float, np.ndarray, float]:
    """Predict the state forward to ``timestamp_sec`` without committing.

    Returns ``(d_pred, s, P_pred, dt)``. Range-rate is constant under the CV
    model, so ``s`` is returned unchanged.
    """

    dt = max(0.0, float(timestamp_sec - state.t))
    d_pred = state.d + (state.s * dt)
    # F = [[1, dt], [0, 1]]; covariance propagation P' = F P F^T + Q.
    f01 = dt
    P = state.P
    fp00 = P[0, 0] + (f01 * P[1, 0])
    fp01 = P[0, 1] + (f01 * P[1, 1])
    p_pred = np.array(
        [
            [fp00 + (f01 * fp01), fp01],
            [P[1, 0] + (f01 * P[1, 1]), P[1, 1]],
        ],
        dtype=np.float64,
    )
    # Continuous white-acceleration process noise injected over dt.
    q = _LON_PROCESS_ACCEL_STD ** 2
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    p_pred[0, 0] += q * dt4 / 4.0
    p_pred[0, 1] += q * dt3 / 2.0
    p_pred[1, 0] += q * dt3 / 2.0
    p_pred[1, 1] += q * dt2
    return d_pred, float(state.s), p_pred, dt


def _ttc_from_lon(d: float, s: float, dt_since_meas: float, p_ss: float, n: int) -> TtcComponent:
    """Build the kinematic TTC component from a (filtered) longitudinal state."""

    closing = -s
    if n < _LON_MIN_UPDATES_FOR_TTC or closing < _LON_MIN_CLOSING_MPS or d <= 0.0:
        return TtcComponent("depth", None, 0.0)
    ttc_sec = d / closing
    if ttc_sec < _TTC_FLOOR_SEC:
        ttc_sec = _TTC_FLOOR_SEC
    if ttc_sec > _TTC_MAX_REPORTED_SEC:
        return TtcComponent("depth", None, 0.0)

    # Confidence: clear closing speed, recent measurement, and a velocity
    # estimate the filter is actually confident about (small range-rate
    # variance) all raise trust.
    freshness = float(
        np.clip(
            1.0
            - max(0.0, dt_since_meas - _DEPTH_TTC_FRESH_DT_SEC)
            / _DEPTH_TTC_STALE_DECAY_SEC,
            _DEPTH_TTC_MIN_FRESHNESS,
            1.0,
        )
    )
    vel_certainty = float(np.clip(1.0 - (p_ss / _LON_INIT_VEL_VAR), 0.0, 1.0))
    speed_term = float(np.clip((closing - _LON_MIN_CLOSING_MPS) / 8.0, 0.0, 1.0))
    confidence = speed_term * freshness * vel_certainty
    return TtcComponent("depth", round(float(ttc_sec), 2), confidence)


def ttc_from_depth_delta(
    track_id: int,
    bbox: tuple[int, int, int, int],
    depth_m: np.ndarray,
    timestamp_sec: float,
    history: dict[int, "_LonState"],
    *,
    update_history: bool,
) -> tuple[TtcComponent, float | None, float | None]:
    """Physical TTC from a per-track constant-velocity longitudinal Kalman.

    The metric distance sampled from the bbox is filtered into a smooth
    (distance, range-rate) state, and TTC = distance / closing-speed is read
    from that state every frame — coasting on prediction between depth
    refreshes, so the estimate is continuous instead of intermittent.

    ``history`` maps ``track_id`` to its :class:`_LonState`. The state is
    committed (advanced + corrected) only when ``update_history`` is True and a
    valid measurement is available — i.e. on a fresh-depth frame. Peek calls and
    stale-depth frames predict-and-return without mutating state, so the filter
    is never stepped twice per frame.

    Returns ``(component, filtered_distance_m, closing_mps)``.
    """

    state = history.get(track_id)

    # No committed state yet: initialise on the first valid measurement,
    # otherwise we have nothing to report.
    if state is None:
        if not update_history:
            return TtcComponent("depth", None, 0.0), None, None
        z = distance_m_for_bbox(depth_m, bbox)
        if z is None:
            return TtcComponent("depth", None, 0.0), None, None
        r = (_LON_MEAS_REL_STD * z) ** 2 + _LON_MEAS_ABS_STD ** 2
        history[track_id] = _LonState(
            t=float(timestamp_sec),
            d=float(z),
            s=0.0,
            P=np.array([[r, 0.0], [0.0, _LON_INIT_VEL_VAR]], dtype=np.float64),
            n=1,
        )
        # Velocity unknown after a single sample: distance is usable, TTC is not.
        return TtcComponent("depth", None, 0.0), float(z), None

    d_pred, s_pred, p_pred, dt = _lon_predict(state, timestamp_sec)
    closing_clamped = float(
        np.clip(-s_pred, -_LON_MAX_CLOSING_MPS, _LON_MAX_CLOSING_MPS)
    )

    # Predict-only (imminence peek, or a stale-depth frame): report the coasted
    # estimate without committing.
    if not update_history:
        component = _ttc_from_lon(d_pred, s_pred, dt, float(p_pred[1, 1]), state.n)
        return component, float(d_pred), (closing_clamped if state.n >= _LON_MIN_UPDATES_FOR_TTC else None)

    z = distance_m_for_bbox(depth_m, bbox)
    if z is None or dt <= 0.0:
        # Nothing to correct with: coast and keep the prior commit anchor.
        component = _ttc_from_lon(d_pred, s_pred, dt, float(p_pred[1, 1]), state.n)
        return component, float(d_pred), (closing_clamped if state.n >= _LON_MIN_UPDATES_FOR_TTC else None)

    r = (_LON_MEAS_REL_STD * z) ** 2 + _LON_MEAS_ABS_STD ** 2
    innovation = float(z - d_pred)
    s_innov = float(p_pred[0, 0]) + r
    # Innovation gate: a depth glitch (e.g. a 25th-pct sample jumping several
    # metres in one step) blows past 3-sigma; coast on prediction rather than
    # let it whip the velocity estimate to a non-physical value.
    if (innovation * innovation) / max(s_innov, 1e-6) > _LON_GATE_SQ:
        history[track_id] = _LonState(
            t=float(timestamp_sec), d=float(d_pred), s=float(s_pred), P=p_pred, n=state.n
        )
        component = _ttc_from_lon(d_pred, s_pred, 0.0, float(p_pred[1, 1]), state.n)
        return component, float(d_pred), (closing_clamped if state.n >= _LON_MIN_UPDATES_FOR_TTC else None)

    # Standard Kalman update with H = [1, 0].
    k0 = float(p_pred[0, 0]) / s_innov
    k1 = float(p_pred[1, 0]) / s_innov
    d_post = d_pred + (k0 * innovation)
    s_post = s_pred + (k1 * innovation)
    p_post = np.array(
        [
            [(1.0 - k0) * p_pred[0, 0], (1.0 - k0) * p_pred[0, 1]],
            [p_pred[1, 0] - (k1 * p_pred[0, 0]), p_pred[1, 1] - (k1 * p_pred[0, 1])],
        ],
        dtype=np.float64,
    )
    n_post = state.n + 1
    history[track_id] = _LonState(
        t=float(timestamp_sec), d=float(d_post), s=float(s_post), P=p_post, n=n_post
    )

    closing_post = float(np.clip(-s_post, -_LON_MAX_CLOSING_MPS, _LON_MAX_CLOSING_MPS))
    component = _ttc_from_lon(d_post, s_post, 0.0, float(p_post[1, 1]), n_post)
    return component, float(d_post), (closing_post if n_post >= _LON_MIN_UPDATES_FOR_TTC else None)


# ── Weighted-median fusion ───────────────────────────────────────────────────


def fuse_ttc(components: list[TtcComponent]) -> tuple[Optional[float], list[TtcComponent]]:
    """Robust fusion of the three TTC estimators.

    Sort components by TTC value and pick the one whose cumulative weight
    crosses 0.5 — the canonical weighted-median definition. Returns the
    fused TTC plus the same component list so the UI can show what each
    source contributed.
    """

    valid = [c for c in components if c.value is not None and c.confidence > 0.0]
    if not valid:
        return None, components

    if len(valid) == 1:
        return valid[0].value, components

    valid_sorted = sorted(valid, key=lambda c: c.value)  # type: ignore[arg-type]
    total_weight = sum(c.confidence for c in valid_sorted)
    if total_weight <= 0.0:
        return None, components

    cumulative = 0.0
    target = total_weight / 2.0
    chosen = valid_sorted[-1].value
    for component in valid_sorted:
        cumulative += component.confidence
        if cumulative >= target:
            chosen = component.value
            break

    return chosen, components


# ── Lane-relative crossing prediction ────────────────────────────────────────


def lane_lateral_velocity(
    track: Track,
    lane: LaneFrame,
    *,
    min_dt: float = 0.06,
    max_samples: int = 4,
    max_age_sec: float = 0.4,
) -> float:
    """Lateral velocity in lane-units per second, robust to bbox jitter.

    Builds pairwise (pos_now - pos_old)/dt instantaneous velocities from the
    current bbox and the most recent history samples within ``max_age_sec``,
    then returns the **median**. A single-frame bbox jitter that produces a
    phantom large velocity gets rejected by the median; sustained motion (real
    cut-in) survives because all pairwise slopes agree.

    Falls back to single-pair computation when history is too short.
    """

    # Collect (timestamp, lane_position) samples newest-first within the window.
    pos_now = lane_position(track.bbox, lane)
    samples: list[tuple[float, float]] = [(float(track.timestamp_sec), pos_now)]
    for sample in reversed(track.history):
        age = float(track.timestamp_sec - sample.timestamp_sec)
        if age > max_age_sec:
            break
        samples.append((float(sample.timestamp_sec), lane_position(sample.bbox, lane)))
        if len(samples) >= max_samples:
            break

    if len(samples) < 2 and track.history:
        # Sparse detections (high detect_every / low fps): the most recent
        # history sample can be older than the recency window. Rather than
        # collapsing to zero lateral velocity — which silently drops real
        # cut-ins at coarse sampling — fall back to that single most-recent
        # sample, mirroring how ``expansion_rate_from_track`` consumes history.
        recent = track.history[-1]
        dt = float(track.timestamp_sec - recent.timestamp_sec)
        if dt >= min_dt:
            samples.append((float(recent.timestamp_sec), lane_position(recent.bbox, lane)))

    if len(samples) < 2:
        return 0.0

    velocities: list[float] = []
    ts_new, pos_new = samples[0]
    for ts_old, pos_old in samples[1:]:
        dt = ts_new - ts_old
        if dt < min_dt:
            continue
        velocities.append((pos_new - pos_old) / dt)

    if not velocities:
        return 0.0

    velocities.sort()
    n = len(velocities)
    if n % 2 == 1:
        median_v = velocities[n // 2]
    else:
        median_v = 0.5 * (velocities[n // 2 - 1] + velocities[n // 2])

    return float(np.clip(median_v, -_LATERAL_MAX_LANE_PER_SEC, _LATERAL_MAX_LANE_PER_SEC))


def lane_crossing_risk(
    track: Track,
    lane: LaneFrame,
    fused_ttc: Optional[float],
) -> float:
    """How likely the object is to be inside the ego lane at impact time."""

    base = lane_corridor_relevance(track.bbox, lane)
    if fused_ttc is None:
        return base

    pos_now = lane_position(track.bbox, lane)
    lateral_v = lane_lateral_velocity(track, lane)
    horizon = float(min(fused_ttc, 3.0))
    predicted_pos = pos_now + (lateral_v * horizon)

    # Relevance falls off as |predicted_pos| grows past the lane boundary
    # (|pos|=1 in our half-lane convention).
    margin = max(0.0, abs(predicted_pos) - 0.6)
    predicted_relevance = float(np.exp(-(margin * margin) / 0.25))
    lane_trust = float(np.clip((lane.confidence - 0.25) / 0.60, 0.0, 1.0))
    if lane_trust < 1.0:
        predicted_relevance *= lane_trust

    # Collision-cone distance reliability: lane_position is normalised by the
    # lane width at the object's row, so far objects (near the horizon) yield a
    # jittery position and an unreliable velocity extrapolation. Damp the
    # *predicted* crossing toward the horizon; ``base`` below stays a floor so
    # genuine near cut-ins are untouched.
    vp_y = float(lane.vanishing_point[1])
    span = max(1.0, float(lane.height) - vp_y)
    depth_frac = float(np.clip((float(track.bbox[3]) - vp_y) / span, 0.0, 1.0))
    reliability = (
        _CROSSING_FAR_RELIABILITY_FLOOR
        + (1.0 - _CROSSING_FAR_RELIABILITY_FLOOR) * depth_frac
    )
    predicted_relevance *= reliability

    # Defense in depth against single-frame lateral_v outliers. The median
    # smoothing in lane_lateral_velocity is the primary line of defense, but
    # if a static off-corridor object briefly produces a high predicted
    # relevance we cap it so crossing cannot leap from ~0.2 to ~1.0 in one
    # frame. Sustained motion grows base too (overlap/proximity terms), so
    # genuine cut-ins are not capped.
    if base < 0.30 and predicted_relevance > base + 0.30:
        predicted_relevance = base + 0.30

    return float(np.clip(max(base, predicted_relevance), 0.0, 1.0))


# ── Per-track smoothers ──────────────────────────────────────────────────────


class ExpansionSmoother:
    def __init__(self) -> None:
        self._state: dict[int, float] = {}

    def update(self, track_id: int, raw_rate: float) -> float:
        prev = self._state.get(track_id)
        if prev is None:
            value = float(raw_rate)
        else:
            value = _ema(prev, float(raw_rate), _EXPANSION_EMA_RISE, _EXPANSION_EMA_FALL)
        self._state[track_id] = value
        return value

    def forget(self, active_track_ids: set[int]) -> None:
        for track_id in list(self._state.keys()):
            if track_id not in active_track_ids:
                self._state.pop(track_id, None)


class ConfidenceSmoother:
    """Per-track EMA of the displayed fused confidence.

    Smooths the confidence value shown in the UI / telemetry so it doesn't jump
    frame-to-frame with raw YOLO confidence. Purely cosmetic — the raw
    detection confidence still drives ``classify_state``.
    """

    def __init__(self) -> None:
        self._state: dict[int, float] = {}

    def update(self, track_id: int, raw_confidence: float) -> float:
        prev = self._state.get(track_id)
        if prev is None:
            value = float(raw_confidence)
        else:
            value = _ema(
                prev,
                float(raw_confidence),
                _CONFIDENCE_EMA_RISE,
                _CONFIDENCE_EMA_FALL,
            )
        self._state[track_id] = value
        return value

    def forget(self, active_track_ids: set[int]) -> None:
        for track_id in list(self._state.keys()):
            if track_id not in active_track_ids:
                self._state.pop(track_id, None)


class DepthDeltaSmoother:
    """Per-track longitudinal Kalman state bank for the kinematic TTC source.

    Holds one :class:`_LonState` per track id (see ``ttc_from_depth_delta``)
    and drops states for tracks that are no longer active.
    """

    def __init__(self) -> None:
        self._state: dict[int, _LonState] = {}

    @property
    def state(self) -> dict[int, "_LonState"]:
        return self._state

    def forget(self, active_track_ids: set[int]) -> None:
        for track_id in list(self._state.keys()):
            if track_id not in active_track_ids:
                self._state.pop(track_id, None)


class TtcImminenceSmoother:
    """Per-track sliding count of imminent-TTC frames within a short window.

    ``classify_state`` uses the returned count to require multi-frame
    confirmation before upgrading to DANGER on the TTC<1s rule. The window
    rule (2-of-last-3 imminent) survives a single intermediate frame where
    TTC briefly recovers above 1s — a common pattern for real cut-ins where
    expansion-rate noise causes one-frame TTC spikes — while still rejecting
    a single isolated TTC dip from bbox jitter.

    History is preserved across short tracker gaps (a YOLO frame where the
    track didn't match, or a propagate-only frame where it wasn't in the
    returned set) so a cut-in's imminent streak isn't wiped by an
    intermediate frame where the detector missed it. ``forget`` only drops a
    track after ``GRACE_FRAMES`` of consecutive inactivity, matching the
    tracker's ``max_misses`` tolerance.
    """

    WINDOW_SIZE = 3
    GRACE_FRAMES = 5

    def __init__(self) -> None:
        self._history: dict[int, list[bool]] = {}
        self._miss_count: dict[int, int] = {}

    def update(self, track_id: int, fused_ttc: float | None) -> int:
        # Frames where TTC is None usually mean we don't have a fresh
        # measurement (YOLO skipped, bbox unchanged on a propagated frame, or
        # no valid components fused). Skipping the window update preserves
        # the per-track signal across detection gaps so a cut-in's imminent
        # frames aren't diluted by intermediate propagation frames.
        self._miss_count[track_id] = 0
        window = self._history.get(track_id, [])
        if fused_ttc is None:
            return sum(window)
        window.append(bool(fused_ttc < _DANGER_TTC_SEC))
        if len(window) > self.WINDOW_SIZE:
            window = window[-self.WINDOW_SIZE:]
        self._history[track_id] = window
        return sum(window)

    def forget(self, active_track_ids: set[int]) -> None:
        for track_id in list(self._history.keys()):
            if track_id in active_track_ids:
                self._miss_count[track_id] = 0
                continue
            self._miss_count[track_id] = self._miss_count.get(track_id, 0) + 1
            if self._miss_count[track_id] > self.GRACE_FRAMES:
                self._history.pop(track_id, None)
                self._miss_count.pop(track_id, None)


# ── State machine ─────────────────────────────────────────────────────────────


def classify_state(
    *,
    fused_ttc: Optional[float],
    crossing: float,
    near_score: float,
    expansion_rate: float,
    lane_pos: float,
    confidence: float,
    lane_confidence: float = 1.0,
    ttc_imminent_streak: int = 2,
) -> str:
    if confidence < 0.20:
        return "SAFE"

    in_ego_lane = abs(lane_pos) < 0.7 and lane_confidence >= _LANE_TRUST_FLOOR

    if fused_ttc is not None:
        # The TTC<1s immediate-DANGER rule requires at least two consecutive
        # frames of imminent TTC so a single-frame bbox-expansion spike cannot
        # flip the state on its own. The streak counter is maintained per
        # track by ``TtcImminenceSmoother``. Callers without a smoother get
        # the default ``2`` (treated as already confirmed) for backward
        # compatibility.
        if (
            fused_ttc < _DANGER_TTC_SEC
            and ttc_imminent_streak >= 2
            and (in_ego_lane or crossing >= 0.55)
        ):
            return "DANGER"
        if fused_ttc < _CAUTION_TTC_SEC and (in_ego_lane or crossing >= 0.30):
            return "CAUTION"
        if fused_ttc < _CAUTION_TTC_SEC and near_score >= 0.55:
            return "CAUTION"

    if expansion_rate >= 0.40 and crossing >= 0.45 and near_score >= 0.40:
        return "DANGER"
    if expansion_rate >= 0.20 and (in_ego_lane or crossing >= 0.30):
        return "CAUTION"
    if near_score >= 0.78 and crossing >= 0.50:
        return "CAUTION"
    return "SAFE"


_STATE_SCORE = {"SAFE": 0.06, "CAUTION": 0.42, "DANGER": 0.68}


def _unit_interval(value: float | int | None) -> float:
    return float(np.clip(0.0 if value is None else float(value), 0.0, 1.0))


def eta_pressure(ttc_sec: float | None) -> float:
    if ttc_sec is None:
        return 0.0
    return _unit_interval((_CAUTION_TTC_SEC - float(ttc_sec)) / _CAUTION_TTC_SEC)


def score_raw(
    state: str,
    ttc_sec: float | None,
    near_score: float,
    closing_speed: float,
    crossing_risk: float = 0.0,
    brake: float = 0.0,
    confidence: float = 1.0,
) -> float:
    eta = eta_pressure(ttc_sec)
    signal = (
        (0.30 * eta)
        + (0.25 * _unit_interval(near_score))
        + (0.20 * _unit_interval(closing_speed))
        + (0.20 * _unit_interval(crossing_risk))
        + (0.05 * _unit_interval(brake))
    )
    confidence_gate = 0.65 + (0.35 * _unit_interval(confidence))
    floor = _STATE_SCORE.get(str(state or "").upper(), 0.0)
    return round(float(max(floor, signal * confidence_gate)), 3)


def score_event(event: RiskEvent) -> float:
    return score_raw(
        event.state,
        event.ttc_sec,
        event.near_score,
        event.closing_speed,
        crossing_risk=event.crossing_risk,
        brake=event.brake_score,
        confidence=event.confidence,
    )


def _risk_reason(state: str, ttc_sec: Optional[float], lane_pos: float) -> str:
    if state == "DANGER":
        if ttc_sec is not None:
            return f"imminent collision: TTC {ttc_sec:.1f}s"
        return "object expanding rapidly in lane"
    if state == "CAUTION":
        if abs(lane_pos) < 0.7:
            return "object in driving lane"
        return "object may be merging into lane"
    return "no immediate closing risk"


def distance_m_for_bbox(depth_m: np.ndarray, bbox: tuple[int, int, int, int]) -> float | None:
    """Robust object distance from the bbox lower-center crop, in meters."""

    height, width = depth_m.shape
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    bbox_w = x2 - x1
    bbox_h = y2 - y1
    cx1 = x1 + int(round(bbox_w * 0.15))
    cx2 = x2 - int(round(bbox_w * 0.15))
    cy1 = y1 + int(round(bbox_h * 0.30))
    cy2 = y2
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = depth_m[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    valid = crop[np.isfinite(crop) & (crop > 0.0)]
    if valid.size == 0:
        return None
    return round(float(np.percentile(valid, 25)), 2)


def near_score_from_distance(distance_m: float | None) -> float:
    if distance_m is None:
        return 0.0
    if distance_m <= 8.0:
        return 1.0
    if distance_m >= 60.0:
        return 0.0
    return float(np.clip((60.0 - distance_m) / 52.0, 0.0, 1.0))


def _flow_magnitude_for_bbox(magnitude_norm: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    height, width = magnitude_norm.shape
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = magnitude_norm[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    return float(np.clip(np.percentile(crop, 75), 0.0, 1.0))


def _approach_score(
    *,
    closing_mps: float | None,
    expansion_rate: float,
    flow_magnitude: float,
    class_weight: float,
) -> float:
    """Normalized approach risk, led by metric closing speed."""

    if closing_mps is None or closing_mps <= _APPROACH_MIN_CLOSING_MPS:
        metric_signal = 0.0
    else:
        metric_signal = float(
            np.clip(
                (closing_mps - _APPROACH_MIN_CLOSING_MPS)
                / (_APPROACH_FULL_CLOSING_MPS - _APPROACH_MIN_CLOSING_MPS),
                0.0,
                1.0,
            )
        )
    expansion_signal = float(np.clip(expansion_rate / 0.6, 0.0, 1.0))
    flow_signal = float(np.clip(flow_magnitude, 0.0, 1.0))
    return float(
        np.clip(
            class_weight
            * (
                (0.50 * metric_signal)
                + (0.30 * expansion_signal)
                + (0.20 * flow_signal)
            ),
            0.0,
            1.0,
        )
    )


# ── Main per-track risk computation ──────────────────────────────────────────


def calculate_track_risk(
    *,
    track: Track,
    depth_m: np.ndarray,
    near_map: np.ndarray,
    flow: np.ndarray,
    magnitude_norm: np.ndarray,
    lane: LaneFrame,
    expansion_rate: float,
    depth_history: dict[int, "_LonState"],
    flow_dt_sec: float,
    depth_is_fresh: bool,
    frame_index: int,
    timestamp_sec: float,
    ttc_imminent_streak: int = 2,
    bgr: "np.ndarray | None" = None,
    confidence_smoother: "ConfidenceSmoother | None" = None,
) -> RiskEvent:
    """Build a RiskEvent for a single tracked object using fused TTC.

    The three TTC sources are combined with a weighted median and lane-
    relative crossing prediction. ``depth_history`` is mutated in place by
    the depth-delta TTC source. ``bgr`` (raw frame) enables the brake-light
    cue for in-path lead vehicles.
    """

    bbox = track.bbox

    velocity_magnitude = _flow_magnitude_for_bbox(magnitude_norm, bbox)
    pos = lane_position(bbox, lane)

    # Three independent TTC estimators.
    expansion_component = ttc_from_expansion(expansion_rate, history_age=len(track.history))
    flow_component = ttc_from_flow(bbox, flow, lane.vanishing_point, flow_dt_sec)
    # The longitudinal Kalman is time-driven, so it must advance on the FRAME
    # clock (which ticks every processed frame), not ``track.timestamp_sec``
    # (which only advances on detection frames — frozen while a track coasts).
    # Using the track clock made the predict step dt=0 between detections, so
    # distance/closing/TTC stalled instead of coasting smoothly.
    depth_component, depth_distance_m, depth_closing_mps = ttc_from_depth_delta(
        track.track_id,
        bbox,
        depth_m,
        timestamp_sec,
        depth_history,
        update_history=depth_is_fresh,
    )
    # Prefer the Kalman-filtered distance; fall back to the raw bbox sample only
    # before the filter has been seeded (brand-new track).
    distance_m = depth_distance_m
    if distance_m is None:
        distance_m = distance_m_for_bbox(depth_m, bbox)
    near_score = near_score_from_distance(distance_m)

    risk_time_hint, components = fuse_ttc([expansion_component, flow_component, depth_component])
    physical_ttc = depth_component.value
    crossing = lane_crossing_risk(track, lane, risk_time_hint)
    lateral_v = lane_lateral_velocity(track, lane)

    class_weight = CLASS_RISK_WEIGHT.get(track.class_name, 1.0)
    # Approach = "is this object getting closer over time". Metric depth
    # closing speed is the lead signal now that depth is in meters; bbox
    # expansion and radial optical flow only corroborate the visual trend.
    closing_speed = _approach_score(
        closing_mps=depth_closing_mps,
        expansion_rate=expansion_rate,
        flow_magnitude=velocity_magnitude,
        class_weight=class_weight,
    )

    detection_confidence = float(np.clip(track.confidence, 0.0, 1.0))
    # Confidence = "how trustworthy is this measurement": YOLO detection
    # confidence weighted by lane geometry trust. Crossing/expansion are
    # risk-relevance signals, not trust signals, so they no longer feed in.
    fused_confidence = float(
        np.clip(
            (0.70 * detection_confidence) + (0.30 * lane.confidence),
            0.0,
            1.0,
        )
    )
    if confidence_smoother is not None:
        fused_confidence = confidence_smoother.update(track.track_id, fused_confidence)

    state = classify_state(
        fused_ttc=risk_time_hint,
        crossing=crossing,
        near_score=near_score,
        expansion_rate=expansion_rate,
        lane_pos=pos,
        confidence=detection_confidence,
        lane_confidence=lane.confidence,
        ttc_imminent_streak=ttc_imminent_streak,
    )

    # Brake-light cue: a lead vehicle braking in our lane is an early warning
    # even before TTC drops. Escalate one band when a rear-facing vehicle shows
    # a confident brake-lamp pair and is reasonably in-path. Corroborating only
    # — it lifts SAFE→CAUTION, and CAUTION→DANGER only with a closing TTC.
    brake = brake_score(bgr, bbox) if track.class_name in _BRAKE_LIGHT_CLASSES else 0.0
    in_ego_lane = abs(pos) < 0.7 and lane.confidence >= _LANE_TRUST_FLOOR
    braking_lead = brake >= _BRAKE_ESCALATE_SCORE and in_ego_lane
    if braking_lead:
        if state == "SAFE":
            state = "CAUTION"
        elif state == "CAUTION" and risk_time_hint is not None and risk_time_hint < _CAUTION_TTC_SEC:
            state = "DANGER"

    reason = _risk_reason(state, risk_time_hint, pos)
    if braking_lead and state != "SAFE":
        reason = (
            f"lead vehicle braking: approach hint {risk_time_hint:.1f}s"
            if risk_time_hint is not None
            else "lead vehicle braking ahead"
        )

    tracking_confidence = float(
        np.clip(
            (0.55 if track.confirmed else 0.30)
            + (0.10 * min(max(track.hits, 0), 4))
            - (0.18 * max(track.misses, 0)),
            0.0,
            1.0,
        )
    )
    depth_confidence = float(np.clip(depth_component.confidence, 0.0, 1.0))

    return RiskEvent(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        state=state,
        ttc_sec=physical_ttc,
        direction=direction_from_lateral(lateral_v),
        lane=lane_bucket_from_position(pos),
        object_type=track.class_name,
        confidence=round(fused_confidence, 3),
        near_score=round(near_score, 3),
        velocity_magnitude=round(velocity_magnitude, 3),
        closing_speed=round(closing_speed, 3),
        bbox=bbox,
        reason=reason,
        object_id=track.track_id,
        display_id=track.display_id if track.display_id is not None else track.track_id,
        expansion_rate=round(float(expansion_rate), 3),
        lateral_velocity_norm=round(float(lateral_v), 3),
        crossing_risk=round(float(crossing), 3),
        lane_position=round(float(pos), 3),
        ttc_components=tuple(components),
        brake_score=round(float(brake), 3),
        distance_m=None if distance_m is None else round(float(distance_m), 2),
        closing_mps=None if depth_closing_mps is None else round(float(depth_closing_mps), 2),
        detection_confidence=round(float(detection_confidence), 3),
        tracking_confidence=round(float(tracking_confidence), 3),
        depth_confidence=round(float(depth_confidence), 3),
        depth_ttc_sec=depth_component.value,
    )


# ── Hysteresis ───────────────────────────────────────────────────────────────


def is_imminent_danger(event: RiskEvent) -> bool:
    return (
        event.state == "DANGER"
        and event.ttc_sec is not None
        and event.ttc_sec <= _DANGER_TTC_SEC
    )


def stabilized_event_state(stabilizer: "StateStabilizer", event: RiskEvent) -> str:
    # Multi-frame TTC confirmation already happens per-track in
    # ``TtcImminenceSmoother`` before ``classify_state`` returns DANGER, so by
    # the time an imminent event reaches the stabilizer it has already
    # survived the ``ttc<1s for 2 consecutive frames`` filter. The stabilizer
    # still bypasses upgrade hysteresis for those confirmed imminent events
    # so true cut-ins are flagged without an extra frame delay.
    if is_imminent_danger(event):
        stabilizer.current_state = "DANGER"
        stabilizer.pending_state = "DANGER"
        stabilizer.counter = 0
        return "DANGER"

    return stabilizer.process(event.state)


class StateStabilizer:
    def __init__(self, upgrade_frames: int = 3, downgrade_frames: int = 7):
        self.current_state = "SAFE"
        self.pending_state = "SAFE"
        self.counter = 0
        self.upgrade_frames = upgrade_frames
        self.downgrade_frames = downgrade_frames

    def process(self, raw_state: str) -> str:
        if raw_state == self.current_state:
            self.pending_state = raw_state
            self.counter = 0
            return self.current_state

        if raw_state != self.pending_state:
            self.pending_state = raw_state
            self.counter = 1
        else:
            self.counter += 1

        r_curr = self._rank(self.current_state)
        r_pend = self._rank(self.pending_state)
        required = self.upgrade_frames if r_pend > r_curr else self.downgrade_frames

        if self.counter >= required:
            self.current_state = self.pending_state
            self.counter = 0
        return self.current_state

    def _rank(self, state: str) -> int:
        return {"SAFE": 0, "CAUTION": 1, "DANGER": 2}.get(state, 0)


def make_safe_event(
    *,
    frame_index: int,
    timestamp_sec: float,
) -> RiskEvent:
    return RiskEvent(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        state="SAFE",
        ttc_sec=None,
        direction="center",
        lane="center",
        object_type=None,
        confidence=0.0,
        near_score=0.0,
        velocity_magnitude=0.0,
        closing_speed=0.0,
        bbox=None,
        reason="no objects detected",
        object_id=None,
    )


def compute_quick_risk(flow: FlowResult, width: int, height: int) -> float:
    """Frame-level motion risk used to decide when to recompute depth."""

    motion_signal = (0.65 * flow.magnitude_norm) + (0.35 * flow.divergence_norm)
    if motion_signal.size == 0:
        return 0.0
    return float(np.percentile(motion_signal, 90))


def build_object_events(
    *,
    frame_index: int,
    timestamp_sec: float,
    tracks: list[Track],
    fields: SpatialFields,
    expansion_smoother: ExpansionSmoother,
    depth_smoother: DepthDeltaSmoother,
    ttc_imminence_smoother: TtcImminenceSmoother | None = None,
    confidence_smoother: ConfidenceSmoother | None = None,
) -> tuple[RiskEvent, list[RiskEvent]]:
    """Build per-object risk events plus a primary event for the frame."""

    if not tracks:
        expansion_smoother.forget(set())
        depth_smoother.forget(set())
        if ttc_imminence_smoother is not None:
            ttc_imminence_smoother.forget(set())
        if confidence_smoother is not None:
            confidence_smoother.forget(set())
        safe = make_safe_event(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
        )
        return safe, [safe]

    # Two-pass: fuse TTC first so the imminence streak per track is updated
    # before classify_state sees it. Per-track work is light, so this is
    # cheaper than carrying state across function calls.
    events: list[RiskEvent] = []
    active_ids: set[int] = set()
    for track in tracks:
        active_ids.add(track.track_id)
        raw_rate = expansion_rate_from_track(track)
        expansion_rate = expansion_smoother.update(track.track_id, raw_rate)

        if ttc_imminence_smoother is not None:
            # Peek fused TTC without mutating depth history: build the same
            # three components calculate_track_risk will, but pass
            # update_history=False here so the smoother doesn't consume the
            # depth delta sample twice.
            expansion_component = ttc_from_expansion(
                expansion_rate, history_age=len(track.history)
            )
            flow_component = ttc_from_flow(
                track.bbox,
                fields.flow.flow,
                fields.lane.vanishing_point,
                fields.flow_dt_sec,
            )
            depth_component, _distance_m, _closing_mps = ttc_from_depth_delta(
                track.track_id,
                track.bbox,
                fields.depth.depth_m,
                timestamp_sec,
                depth_smoother.state,
                update_history=False,
            )
            peek_ttc, _ = fuse_ttc(
                [expansion_component, flow_component, depth_component]
            )
            streak = ttc_imminence_smoother.update(track.track_id, peek_ttc)
        else:
            streak = 2  # backward-compat: no smoother → treat as confirmed

        event = calculate_track_risk(
            track=track,
            depth_m=fields.depth.depth_m,
            near_map=fields.depth.near_map,
            flow=fields.flow.flow,
            magnitude_norm=fields.flow.magnitude_norm,
            lane=fields.lane,
            expansion_rate=expansion_rate,
            depth_history=depth_smoother.state,
            flow_dt_sec=fields.flow_dt_sec,
            depth_is_fresh=fields.depth_is_fresh,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            ttc_imminent_streak=streak,
            bgr=fields.bgr,
            confidence_smoother=confidence_smoother,
        )
        events.append(event)

    expansion_smoother.forget(active_ids)
    depth_smoother.forget(active_ids)
    if ttc_imminence_smoother is not None:
        ttc_imminence_smoother.forget(active_ids)
    if confidence_smoother is not None:
        confidence_smoother.forget(active_ids)

    # Intent gate: off-corridor tracks (|lane_position| > 1.0) are admitted by
    # the depth-gated corridor filter so the tracker can build history before
    # they intrude, but they should NOT win primary-event selection while they
    # are still drifting away or sitting parallel. Only promote them when they
    # show inbound lateral motion toward the ego lane *and* their predicted
    # position within 1.5 s lands inside the corridor.
    eligible = [e for e in events if _primary_eligible(e)]
    if not eligible:
        safe = make_safe_event(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
        )
        return safe, events

    primary = max(
        eligible,
        key=lambda e: (
            score_event(e),
            {"SAFE": 0.0, "CAUTION": 1.0, "DANGER": 2.0}.get(e.state, 0.0),
            eta_pressure(e.ttc_sec),
        ),
    )
    return primary, events


_INBOUND_LANE_PER_SEC = 0.15
_INTENT_HORIZON_SEC = 1.5


def _primary_eligible(event: RiskEvent) -> bool:
    # A genuinely high-risk object always drives the primary/banner, even when
    # its bbox bottom-center sits more than a lane away. A car cutting across in
    # front at low TTC is dangerous regardless of where it lands laterally; the
    # intent gate below is only meant to keep *calm* (SAFE) off-corridor traffic
    # from winning primary selection, never to suppress a real threat.
    if event.state in ("CAUTION", "DANGER"):
        return True
    pos = float(event.lane_position)
    if abs(pos) <= 1.0:
        return True
    lat_v = float(event.lateral_velocity_norm)
    inbound_rate = -lat_v * (1.0 if pos > 0.0 else -1.0)
    if inbound_rate < _INBOUND_LANE_PER_SEC:
        return False
    predicted_pos = pos + (lat_v * _INTENT_HORIZON_SEC)
    return abs(predicted_pos) <= 1.0
