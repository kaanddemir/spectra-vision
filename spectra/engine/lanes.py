"""Road segmentation, lane geometry, the camera's ego lane, hood masking and track-to-lane relations.

Segmentation
    ``SegnetOnnxSegmenter`` runs an ONNX export of commaai/comma10k-segnet (U-Net, EfficientNet-B2; MIT
    code and weights, trained on comma10k) at 512 x 384. The published transform is a plain resize, so
    16:9 frames are squashed to 4:3 exactly as at training time. Classes: road, lane_marking,
    undrivable, movable, ego_vehicle. The model gives lane-marking pixels, not lane instances.

Geometry
    Under a pinhole camera above a flat road with negligible roll, a ground line parallel to the lanes at
    lateral distance X from the camera appears as

        x(y) = a * (y - y_h) + b + c / (y - y_h),     a = X / h (h = camera height)

    the hyperbola lane model: ``y_h`` is the horizon row, ``b`` the column the lanes run towards and ``c``
    a road-bend term; ``b`` and ``c`` are shared by all boundaries of one road. The near-field slope ``a``
    is the boundary's lateral position in camera heights and is independent of camera yaw, of the
    horizon, of where the camera sits across the windscreen and of the image centre. The camera's own
    lateral position is ``a = 0``. Road curvature enters ``c``, not ``a``, which is what separates a bend
    from a lane change. The model needs square pixels, so mask columns are rescaled by the frame's aspect
    ratio before fitting: the mask squashes every frame to 4:3. Roll is assumed negligible and the road
    flat; both are recorded with the estimator's parameters (``ASSUMPTIONS``). The camera's mounting offset within the vehicle is not observable, so positions refer
    to the camera, not the vehicle centre. Nothing here yields metric distances.

Evidence
    A boundary's evidence reaches as far up the image as markings are observed on its fitted curve: the fit
    rows plus mask rows where a marking run lies on that boundary's curve and nearer to it than to any
    other, with gaps no longer than those between dashes (``observed_far``). Boundaries are drawn at most
    a bounded distance beyond it, and track relations beyond it are reported as such.

Temporal lane tracking
    Boundaries are associated across frames by ``a`` after removing the common lateral shift, carried
    for a bounded time when not measured, and reported with ``measured`` and ``age_s``. A lane change is
    a tracked boundary whose ``a`` moves from one side of the camera to the other and stays there. A
    crossing that ends otherwise is reported once as ``cancelled``. Why an observation is ambiguous or
    limited is given as ``LaneReason`` codes.

Hood masking
    ``ego_overlap`` is the fraction of a detection box the segmenter labels as the camera's own vehicle.
    The camera vehicle's hood is fixed relative to the camera, so the most recent mask is used within a
    bounded age. A real vehicle in front of the hood is labelled ``movable`` over the pixels it occupies,
    so its overlap stays low however close it is.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from spectra.engine.adapters import ComponentFailure, Segmentation
from spectra.engine.contracts import (ComponentInfo, Detection, EgoLaneObservation, EgoLaneState, LaneBoundary,
                                      LaneChange, LaneReason, LaneRelation, Role, TrackedObject, TrackLaneRelation)
from spectra.engine.video import file_sha256
from spectra.engine.video import Frame

CLASSES = ("road", "lane_marking", "undrivable", "movable", "ego_vehicle")
ROAD, MARKING, UNDRIVABLE, MOVABLE, EGO = range(5)
INPUT_SIZE = (512, 384)  # width, height
ASSUMPTIONS = ("flat road", "roll negligible", "positions refer to the camera, not the vehicle centre")


class SegnetOnnxSegmenter:
    """``Segmenter`` adapter for the comma10k-segnet ONNX export (input RGB 0-255, NCHW 1x3x384x512)."""

    def __init__(self, model_path: Path | str, sha256: str, *, threads: int = 4, interval_s: Optional[float] = 0.2,
                 name: str = "comma10k-segnet") -> None:
        self.model_path, self.expected_sha256, self.threads = Path(model_path), sha256, threads
        self.info = ComponentInfo(role=Role.SEGMENTER, name=name, version="onnx-1", checkpoint_sha256=sha256,
                                  interval_s=interval_s,
                                  params={"input_size": list(INPUT_SIZE), "classes": list(CLASSES),
                                          "runtime": "onnxruntime-cpu"})
        self._session = None

    def load(self) -> None:
        import onnxruntime as ort
        if not self.model_path.is_file():
            raise FileNotFoundError(f"segmentation model not found: {self.model_path}")
        actual = file_sha256(self.model_path)
        if actual != self.expected_sha256:
            raise ValueError(f"segmentation model {self.model_path.name} has sha256 {actual}, expected "
                             f"{self.expected_sha256}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(self.model_path), options, providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        if [i.name for i in inputs] != ["image"]:
            raise ValueError(f"unexpected segmentation inputs {[i.name for i in inputs]}")

    def close(self) -> None:
        self._session = None

    def segment(self, frame: Frame) -> Segmentation:
        import cv2
        if self._session is None:
            raise RuntimeError("segmenter used before load()")
        rgb = cv2.cvtColor(cv2.resize(frame.image, INPUT_SIZE, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        logits = self._session.run(None, {"image": x})[0]
        return Segmentation(parse_logits(logits), CLASSES)


def parse_logits(logits: np.ndarray) -> np.ndarray:
    w, h = INPUT_SIZE
    if logits.shape != (1, len(CLASSES), h, w):
        raise ComponentFailure(f"invalid output: logits shape {logits.shape}")
    if not np.all(np.isfinite(logits)):
        raise ComponentFailure("invalid output: non-finite logits")
    return logits[0].argmax(0).astype(np.uint8)


# --- Hood masking ---------------------------------------------------------------------------------


def mark_ego_vehicle(detections: Sequence[Detection], ego: np.ndarray, width: int, height: int,
                     threshold: float = 0.5) -> tuple[Detection, ...]:
    """Attach ``ego_overlap`` (fraction of the box inside the boolean ego-vehicle mask ``ego``) to every
    detection and flag those at or above ``threshold``. Boxes are in source pixels of ``width`` x ``height``."""
    mh, mw = ego.shape
    out = []
    for d in detections:
        x1, x2 = int(np.floor(d.box.x1 / width * mw)), int(np.ceil(d.box.x2 / width * mw))
        y1, y2 = int(np.floor(d.box.y1 / height * mh)), int(np.ceil(d.box.y2 / height * mh))
        region = ego[max(0, y1):min(mh, y2), max(0, x1):min(mw, x2)]
        overlap = round(float(region.mean()), 4) if region.size else 0.0
        out.append(d.model_copy(update={"ego_overlap": overlap, "ego_vehicle": overlap >= threshold}))
    return tuple(out)


# --- Lane geometry from one mask ------------------------------------------------------------------


@dataclass
class _Piece:
    ys: np.ndarray  # rows (mask pixels), one sample per row
    xs: np.ndarray  # mean marking column on that row
    a: float = 0.0  # lateral position estimate (camera heights)


@dataclass
class RoadFit:
    horizon: float  # mask rows
    b: float
    c: float
    laterals: list[float]  # a per boundary, sorted
    support: list[int]  # rows of evidence per boundary
    far: list[float]  # furthest (smallest) row with evidence per boundary
    near_row: float  # lowest row above the ego vehicle
    residual: float


def _row_runs(row: np.ndarray) -> list[tuple[float, int]]:
    """(centre column, width) of each run of marking pixels in one mask row."""
    d = np.diff(np.r_[0, row.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [((s0 + e0 - 1) / 2, int(e0 - s0)) for s0, e0 in zip(starts, ends)]


def _pieces(mask: np.ndarray, near_row: int, min_rows: int, max_step: float = 3.0) -> list[_Piece]:
    """Marking pieces as chains of row runs, built from the near row upwards. A run continues a chain only
    when exactly one chain and one run match within ``max_step`` pixels (or the run width), so lines that
    merge towards the horizon end where they meet instead of fusing into one component."""
    marks = mask == MARKING
    open_chains: list[tuple[list, list]] = []
    done = []
    for y in range(near_row, -1, -1):
        runs = [r for r in _row_runs(marks[y]) if r[1] <= 40]
        claims: dict[int, list[int]] = {}
        for ci, (ys, xs) in enumerate(open_chains):
            hits = [ri for ri, (x, w) in enumerate(runs) if abs(x - xs[-1]) <= max(max_step, w / 2 + 1)]
            if len(hits) == 1:
                claims.setdefault(hits[0], []).append(ci)
        nxt, taken = [], set()
        for ri, chains in claims.items():
            if len(chains) == 1:
                ys, xs = open_chains[chains[0]]
                ys.append(y)
                xs.append(runs[ri][0])
                nxt.append((ys, xs))
                taken.add(chains[0])
        for ci, chain in enumerate(open_chains):
            if ci not in taken:
                done.append(chain)
        claimed = {ri for ri, chains in claims.items() if len(chains) == 1}
        nxt.extend(([y], [x]) for ri, (x, _) in enumerate(runs) if ri not in claimed)
        open_chains = nxt
    done.extend(open_chains)
    out = []
    for ys, xs in done:
        if len(ys) < min_rows:
            continue
        ys_a, xs_a = np.array(ys[::-1], float), np.array(xs[::-1], float)
        if np.ptp(xs_a) > 6 * len(ys_a):  # nearly horizontal: stop lines, crossings, text
            continue
        out.append(_Piece(ys_a, xs_a))
    return out


def _horizon_candidates(pieces: list[_Piece]) -> list[tuple[float, float]]:
    """(row, weight) of intersections between straight fits of pieces with clearly different slopes."""
    lines = []
    for p in pieces:
        if len(p.ys) >= 4:
            m, k = np.polyfit(p.ys, p.xs, 1)
            lines.append((m, k, len(p.ys)))
    out = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            (m1, k1, n1), (m2, k2, n2) = lines[i], lines[j]
            if abs(m1 - m2) > 0.3:
                out.append((-(k1 - k2) / (m1 - m2), float(n1 * n2)))
    return out


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    order = np.argsort(values)
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    return float(v[np.searchsorted(np.cumsum(w), w.sum() / 2)])


def fit_road(pieces: list[_Piece], horizon: float, near_row: float, gate: float = 0.35,
             bend_ridge: float = 1e5, iterations: int = 3, min_support: int = 8) -> Optional[RoadFit]:
    """Joint least-squares hyperbola fit: shared b and c, one a per boundary. Pieces are grouped into
    boundaries by their a and regrouped after each fit. Rows must lie below the horizon."""
    pieces = [p for p in pieces if p.ys.min() > horizon + 2]
    if not pieces:
        return None
    for p in pieces:  # initial a from each piece's own straight fit (the slope in image space)
        p.a = float(np.polyfit(p.ys, p.xs, 1)[0]) if len(p.ys) >= 3 else 0.0
    b = c = 0.0
    groups: list[list[_Piece]] = []
    residual = np.inf
    for _ in range(iterations):
        groups = []
        for p in sorted(pieces, key=lambda q: q.a):
            if groups and abs(p.a - np.mean([q.a for q in groups[-1]])) < gate * (1 + abs(p.a) / 3):
                groups[-1].append(p)
            else:
                groups.append([p])
        cols, rows, rhs = len(groups) + 2, [], []
        for g, members in enumerate(groups):
            for p in members:
                dy = p.ys - horizon
                block = np.zeros((len(dy), cols))
                block[:, g], block[:, -2], block[:, -1] = dy, 1.0, 1.0 / dy
                rows.append(block)
                rhs.append(p.xs)
        A, z = np.vstack(rows), np.concatenate(rhs)
        if len(z) < cols + 2:
            return None
        reg = np.zeros(cols)
        reg[-1] = 1.0 / bend_ridge  # c is weakly identified on short evidence: prefer a straight road
        sol = np.linalg.solve(A.T @ A + np.diag(reg) * len(z), A.T @ z)
        residual = float(np.sqrt(np.mean((A @ sol - z) ** 2)))
        b, c = sol[-2], sol[-1]
        for p in pieces:  # re-estimate each piece's a with the shared terms fixed
            dy = p.ys - horizon
            p.a = float(np.dot(dy, p.xs - b - c / dy) / np.dot(dy, dy))
    groups = [g for g in groups if sum(len(p.ys) for p in g) >= min_support]
    if not groups:
        return None
    laterals = [float(np.mean([p.a for p in g])) for g in groups]
    support = [int(sum(len(p.ys) for p in g)) for g in groups]
    far = [float(min(p.ys.min() for p in g)) for g in groups]
    order = np.argsort(laterals)
    return RoadFit(horizon, float(b), float(c), [laterals[i] for i in order], [support[i] for i in order],
                   [far[i] for i in order], near_row, residual)


def fit_road_search(pieces: list[_Piece], near_row: float, centre: float, half_range: float,
                    step: float = 2.0, margin: float = 0.06, height: int = 384) -> Optional[RoadFit]:
    """Choose the horizon row in ``centre +- half_range`` that minimises the joint-fit residual. Evidence
    within ``margin`` of the image height below a candidate horizon is excluded: it is a few pixels wide
    and dominates the error."""
    best = None
    for horizon in np.arange(centre - half_range, centre + half_range + step / 2, step):
        usable = [p for p in pieces if p.ys.max() > horizon + margin * height]
        trimmed = [_Piece(p.ys[p.ys > horizon + margin * height], p.xs[p.ys > horizon + margin * height])
                   for p in usable]
        trimmed = [p for p in trimmed if len(p.ys) >= 3]
        fit = fit_road(trimmed, float(horizon), near_row)
        if fit is not None and (best is None or fit.residual < best.residual):
            best = fit
    return best


def observed_far(mask: np.ndarray, fit: RoadFit, aspect: float, *, tolerance: float = 0.1, min_px: float = 1.5,
                 max_gap: int = 6, top_margin: float = 0.03, min_matches: int = 3) -> list[float]:
    """Per boundary, the farthest mask row reached by marking runs on its fitted curve.

    The fit leaves out evidence near the horizon and pieces that do not chain, so markings often continue
    beyond a boundary's ``far``. The fit itself is not changed: ``far`` moves up only through rows where a
    run centre lies within ``tolerance`` of the distance to the nearest other boundary (at least ``min_px``)
    and closer to this boundary than to any other. ``max_gap`` unmatched rows (gaps between dashes) end the
    walk; fewer than ``min_matches`` matched rows leave ``far`` unchanged. A lone boundary has no spacing to
    scale the tolerance and is left as fitted. ``aspect`` converts the fit's square-pixel columns to mask
    columns.
    """
    h = mask.shape[0]
    marks = mask == MARKING
    stop = int(np.ceil(fit.horizon + top_margin * h))
    out = []
    for i, far in enumerate(fit.far):
        best, gap, matches = far, 0, 0
        for y in range(int(far) - 1, max(stop, 0) - 1, -1):
            dy = y - fit.horizon
            xs = [(a * dy + fit.b + fit.c / dy) / aspect for a in fit.laterals]
            others = [abs(x - xs[i]) for j, x in enumerate(xs) if j != i]
            if not others:
                break
            tol = max(min_px, tolerance * min(others))
            hit = False
            for cx, width in _row_runs(marks[y]):
                if width <= 40:
                    d = [abs(cx - x) for x in xs]
                    if d[i] <= tol and d[i] == min(d):
                        hit = True
                        break
            if hit:
                matches, gap = matches + 1, 0
                if matches >= min_matches:
                    best = y
            else:
                gap += 1
                if gap > max_gap:
                    break
        out.append(float(min(far, best)))
    return out


def ego_near_row(mask: np.ndarray) -> int:
    """Lowest row above the camera vehicle's bonnet in the central half of the image."""
    h, w = mask.shape
    centre = mask[:, w // 4: 3 * w // 4] == EGO
    rows = np.flatnonzero(centre.mean(1) > 0.5)
    return int(rows[0]) - 1 if len(rows) else h - 1


# --- Temporal estimator ---------------------------------------------------------------------------


@dataclass
class _Boundary:
    id: int
    a: float
    last_t: float
    measured_a: Optional[float] = None
    support: int = 0
    far: float = 0.0
    side: int = 0  # last definite side: -1 left of the camera, +1 right, 0 not yet known
    crossing_from: Optional[int] = None  # side it entered the crossing band from
    far_since: Optional[float] = None  # when it reached the opposite side during a crossing
    last_far_t: Optional[float] = None  # last time it was at least 0.4 lane widths from the camera
    last_far_a: Optional[float] = None  # its lateral position then
    shift_since_far: float = 0.0  # common lateral shift of the whole road since then
    paused: bool = False  # lane-change decisions were paused (sharp bend) since its last decision
    unverified: bool = False  # its side changed while paused, without an observed crossing


@dataclass
class LaneGeometryEstimator:
    """``EgoLaneEstimator``: the camera's lane from comma10k-segnet masks, with temporal tracking.

    Parameters are in camera heights (lateral) and seconds. ``width_range`` bounds a plausible lane
    width; ``crossing_band`` is the fraction of a lane width around the camera inside which a boundary
    counts as being crossed; ``confirm_s`` is how long it must stay on the far side.
    """
    max_age_s: float = 1.0
    width_range: tuple[float, float] = (1.0, 5.0)
    crossing_band: float = 0.15
    confirm_s: float = 0.4
    horizon_window_s: float = 4.0
    max_bend: float = 0.015  # bend displacement at the nearest road row, in lane widths
    shift_support: float = 0.6  # share of the crossing explained by the whole road shifting
    min_depth: float = 0.18  # visible road between horizon and bonnet, as a fraction of image height
    min_rows: int = 4
    evidence_tolerance: float = 0.1  # paint beyond the fit: share of the spacing to the nearest other boundary
    evidence_max_gap: int = 6  # mask rows without paint before the evidence walk ends
    evidence_min_matches: int = 3
    name: str = "hyperbola-lane-tracker"
    info: ComponentInfo = field(init=False)
    _tracks: list = field(init=False, default_factory=list)
    _horizons: deque = field(init=False, default_factory=deque)
    _offsets: deque = field(init=False, default_factory=deque)
    _width: Optional[float] = field(init=False, default=None)
    _next_id: int = field(init=False, default=0)
    _lane_change: Optional[LaneChange] = field(init=False, default=None)
    _change_shown_until: float = field(init=False, default=-1.0)

    def __post_init__(self) -> None:
        self.info = ComponentInfo(role=Role.EGO_LANE, name=self.name, version="5", params={
            "assumptions": list(ASSUMPTIONS),
            "max_age_s": self.max_age_s, "width_range": list(self.width_range), "crossing_band": self.crossing_band,
            "confirm_s": self.confirm_s, "horizon_window_s": self.horizon_window_s, "min_rows": self.min_rows,
            "min_depth": self.min_depth, "shift_support": self.shift_support, "max_bend": self.max_bend,
            "model": "hyperbola lane model, shared bend, per-boundary lateral slope",
            "evidence": {"rule": "marking runs on the fitted curve", "tolerance": self.evidence_tolerance,
                         "max_gap_rows": self.evidence_max_gap, "min_matches": self.evidence_min_matches}})

    def load(self) -> None:
        self.close()

    def close(self) -> None:
        self._tracks, self._horizons, self._offsets = [], deque(), deque()
        self._width, self._next_id, self._lane_change, self._change_shown_until = None, 0, None, -1.0

    # -- per frame --

    def estimate(self, frame: Frame, segmentation: Segmentation) -> EgoLaneObservation:
        mask = segmentation.mask
        if segmentation.classes[:len(CLASSES)] != CLASSES:
            raise ComponentFailure("segmentation classes do not match the lane model")
        h, w = mask.shape
        t = frame.t
        near = ego_near_row(mask)
        pieces = _pieces(mask, near, self.min_rows)
        # The mask is a plain resize of the frame, so its pixels are not square unless the frame is 4:3. The
        # geometry needs square pixels: columns are rescaled by the frame's aspect relative to the mask's, and
        # ``w`` becomes the mask width in those units.
        ih, iw = frame.image.shape[:2]
        aspect = (iw / w) / (ih / h)
        for p in pieces:
            p.xs = p.xs * aspect
        w = w * aspect
        horizon = self._horizon(t, pieces, near, h)
        fit = (fit_road_search(pieces, near, horizon, 0.0, height=h) if horizon is not None else None)
        reasons: list[LaneReason] = []
        if horizon is not None and near - horizon < self.min_depth * h:
            # Too little road between the horizon and the bonnet: a boundary's lateral slope cannot be
            # separated from the road bend. Report the frame as ambiguous and keep tracks untouched.
            self._expire(t)
            return EgoLaneObservation(state=EgoLaneState.AMBIGUOUS, horizon_y=round(horizon / h, 4),
                                      reasons=(LaneReason.SHALLOW_ROAD,))
        if fit is not None and fit.laterals:
            fit.far = observed_far(mask, fit, aspect, tolerance=self.evidence_tolerance,
                                   max_gap=self.evidence_max_gap, min_matches=self.evidence_min_matches)
            self._associate(fit, t)
        self._expire(t)
        if horizon is None:
            reasons.append(LaneReason.NO_HORIZON)
        return self._observe(t, fit, horizon, w, h, reasons)

    def _horizon(self, t: float, pieces: list[_Piece], near: int, h: int) -> Optional[float]:
        """Per frame: the residual-minimising horizon near the running estimate (or near the intersection of
        the markings when there is none yet); over time: the median of the last ``horizon_window_s``."""
        if self._horizons:
            centre, half = float(np.median([y for _, y in self._horizons])), 0.06 * h
        else:
            cands = [(y, wt) for y, wt in _horizon_candidates(pieces) if -0.5 * h < y < 0.9 * h]
            centre, half = (_weighted_median(*zip(*cands)), 0.1 * h) if cands else (None, None)
        if centre is not None:
            fit = fit_road_search(pieces, near, centre, half, height=h)
            if fit is not None and len(fit.laterals) >= 2:
                self._horizons.append((t, fit.horizon))
        while self._horizons and self._horizons[0][0] < t - self.horizon_window_s:
            self._horizons.popleft()
        return float(np.median([y for _, y in self._horizons])) if self._horizons else None

    def _associate(self, fit: RoadFit, t: float) -> None:
        live = [b for b in self._tracks]
        meas = list(zip(fit.laterals, fit.support, fit.far))
        shift = 0.0
        if live:  # common lateral shift (the camera moving across the road) from nearest pairs
            diffs = [min((m - b.a for b in live), key=abs) for m, _, _ in meas]
            close = [d for d in diffs if abs(d) < 0.8]
            shift = float(np.median(close)) if close and len(meas) >= 2 else 0.0
        for b in live:
            b.shift_since_far += shift
        gate = 0.35
        used = set()
        for m, support, far in sorted(meas, key=lambda x: -x[1]):
            best = min((b for b in live if b.id not in used), key=lambda b: abs(b.a + shift - m), default=None)
            if best is not None and abs(best.a + shift - m) < gate:
                track = best
            else:
                track = _Boundary(self._next_id, m, t)
                self._next_id += 1
                self._tracks.append(track)
            used.add(track.id)
            track.a, track.measured_a, track.last_t, track.support, track.far = m, m, t, support, far
        for b in self._tracks:
            if b.id not in used:
                b.measured_a = None
                b.a += shift  # unmeasured boundaries move with the rest of the road

    def _expire(self, t: float) -> None:
        self._tracks = [b for b in self._tracks if t - b.last_t <= self.max_age_s]

    def _observe(self, t: float, fit: Optional[RoadFit], horizon: Optional[float], w: int, h: int,
                 reasons: list[LaneReason]) -> EgoLaneObservation:
        left = max((b for b in self._tracks if b.a < 0), key=lambda b: b.a, default=None)
        right = min((b for b in self._tracks if b.a > 0), key=lambda b: b.a, default=None)
        width = right.a - left.a if left and right else None
        sharp = False
        if fit is not None and width is not None and width > 0:
            dy = fit.near_row - fit.horizon
            sharp = dy > 0 and abs(fit.c / dy) / (width * dy) > self.max_bend
        if sharp:
            # On a tight bend the lateral position extrapolated to the camera depends on the bend being
            # modelled exactly (flat road, gentle curvature); lane-change decisions wait for straighter road.
            reasons.append(LaneReason.SHARP_BEND)
            for b in self._tracks:
                b.far_since, b.paused = None, True
        else:
            self._update_crossings(t)
        change = self._lane_change if t <= self._change_shown_until else None
        if fit is None and not self._tracks:
            return EgoLaneObservation(state=EgoLaneState.UNAVAILABLE, reasons=(*reasons, LaneReason.NO_EVIDENCE))
        lo, hi = self.width_range
        if width is None or not lo <= width <= hi:
            reasons.append(LaneReason.BOUNDARY_MISSING if width is None else LaneReason.IMPLAUSIBLE_WIDTH)
            return EgoLaneObservation(state=EgoLaneState.AMBIGUOUS, left=self._boundary(left, fit, t, w, h),
                                      right=self._boundary(right, fit, t, w, h), lane_change=change,
                                      horizon_y=None if horizon is None else horizon / h, reasons=tuple(reasons))
        if self._width is not None and abs(width - self._width) > 0.35 * self._width:
            reasons.append(LaneReason.WIDTH_CHANGED)
        in_band = min(-left.a, right.a) < self.crossing_band * width
        offset = (0.0 - (left.a + right.a) / 2) / width
        measured = sum(b.measured_a is not None for b in (left, right))
        consistent = self._width is None or abs(width - self._width) <= 0.35 * self._width
        self._width = width if self._width is None else 0.8 * self._width + 0.2 * width
        self._offsets.append((t, offset))
        while self._offsets and self._offsets[0][0] < t - 1.0:
            self._offsets.popleft()
        rate = None
        if len(self._offsets) >= 3 and self._offsets[-1][0] - self._offsets[0][0] >= 0.4:
            ts, os_ = np.array(self._offsets).T
            rate = float(np.polyfit(ts, os_, 1)[0])
        support = min(left.support, right.support)
        confidence = round(float((0.35 + 0.3 * (measured == 2) + 0.15 * consistent) * min(1.0, support / 40)), 3)
        state = EgoLaneState.AMBIGUOUS if in_band else EgoLaneState.ESTIMATED
        if in_band:
            reasons.append(LaneReason.ON_BOUNDARY)
        far = min(b.far for b in (left, right)) / h if fit is not None else None
        return EgoLaneObservation(
            state=state, left=self._boundary(left, fit, t, w, h), right=self._boundary(right, fit, t, w, h),
            offset=round(offset, 4), confidence=confidence if state is EgoLaneState.ESTIMATED else None,
            width=round(width, 4), horizon_y=round(horizon / h, 4), lateral_rate=None if rate is None else round(rate, 4),
            curvature=None if fit is None else round(fit.c / (w * h), 6), far_y=None if far is None else round(far, 4),
            lane_change=change, reasons=tuple(reasons))

    def _boundary(self, b: Optional[_Boundary], fit: Optional[RoadFit], t: float, w: int,
                  h: int) -> Optional[LaneBoundary]:
        if b is None or fit is None:
            return None if b is None else LaneBoundary(near=(0.0, 1.0), far=(0.0, 1.0), lateral=round(b.a, 4),
                                                       track_id=b.id, measured=False, age_s=round(t - b.last_t, 3))
        evidence = max(b.far, fit.horizon + 0.05 * h)
        top = max(fit.horizon + (evidence - fit.horizon) / 2, fit.horizon + 0.03 * h)  # at most twice as far
        ys = np.r_[np.linspace(fit.near_row, evidence, 6), np.linspace(evidence, top, 4)[1:]]
        dy = ys - fit.horizon
        xs = b.a * dy + fit.b + fit.c / dy
        pts = tuple((round(float(x / w), 4), round(float(y / h), 4)) for x, y in zip(xs, ys))
        return LaneBoundary(near=pts[0], far=pts[-1], points=pts, lateral=round(b.a, 4), track_id=b.id,
                            measured=b.measured_a is not None, age_s=round(t - b.last_t, 3),
                            evidence_y=round(float(evidence / h), 4))

    def _update_crossings(self, t: float) -> None:
        """Per measured boundary: left side (-1), crossing band (0) or right side (+1) of the camera. A lane
        change is entering the band from one side and then staying on the other side for ``confirm_s``;
        returning to the original side cancels it. A boundary that changed sides while decisions were paused
        on a sharp bend has an unobserved side: its next crossing is followed but not reported."""
        width = self._width
        if width is None:
            return
        band = self.crossing_band * width
        for b in self._tracks:
            if b.measured_a is None:
                continue
            if abs(b.a) >= 0.4 * width and b.crossing_from is None:
                b.last_far_t, b.last_far_a, b.shift_since_far = t, b.a, 0.0
            s = -1 if b.a < -band else 1 if b.a > band else 0
            paused, b.paused = b.paused, False
            if s == 0:
                if b.crossing_from is None and b.side != 0:
                    b.crossing_from, b.far_since = b.side, None
                    if not b.unverified:
                        self._lane_change = LaneChange(direction="left" if b.side < 0 else "right", phase="crossing",
                                                       t_start=round(b.last_far_t if b.last_far_t is not None else t, 3),
                                                       boundary_id=b.id)
                if b.crossing_from is not None and not b.unverified:
                    self._change_shown_until = t + self.max_age_s
                continue
            if b.crossing_from is None:
                if paused and b.side != 0 and s != b.side:
                    # it reached the other side while decisions were paused, without an observed crossing: its
                    # next crossing starts from a side that was never observed and cannot be a lane change
                    b.unverified = True
                b.side = s
            elif s == -b.crossing_from:
                b.far_since = t if b.far_since is None else b.far_since
                moved = b.a - (b.last_far_a if b.last_far_a is not None else b.a)
                road_moved = abs(moved) > 0 and b.shift_since_far * np.sign(moved) >= self.shift_support * abs(moved)
                if t - b.far_since >= self.confirm_s and not road_moved:
                    # the crossed boundary moved but the rest of the road did not: a change of which
                    # marking is tracked, not of the camera's lane
                    self._cancel(b, t)
                    b.side, b.crossing_from, b.far_since, b.unverified = s, None, None, False
                elif t - b.far_since >= self.confirm_s:
                    if self._lane_change is not None and self._lane_change.boundary_id == b.id:
                        self._lane_change = self._lane_change.model_copy(update={"phase": "completed"})
                        self._change_shown_until = t + self.max_age_s
                    b.side, b.crossing_from, b.far_since, b.unverified = s, None, None, False
            else:  # back on the side it came from: no lane change
                self._cancel(b, t)
                b.side, b.crossing_from, b.far_since, b.unverified = s, None, None, False


    def _cancel(self, b: _Boundary, t: float) -> None:
        """The crossing of ``b`` ended without a lane change: reported as cancelled on this frame only."""
        if self._lane_change is not None and self._lane_change.boundary_id == b.id:
            self._lane_change = self._lane_change.model_copy(update={"phase": "cancelled"})
            self._change_shown_until = t


# --- Track-to-lane relations ----------------------------------------------------------------------


def _x_at(boundary: LaneBoundary, y: float, extrapolate: float = 0.1) -> Optional[float]:
    pts = boundary.points or (boundary.near, boundary.far)
    ys = np.array([p[1] for p in pts])
    xs = np.array([p[0] for p in pts])
    lo, hi = ys.min(), ys.max()
    if not lo - 1e-6 <= y <= hi + extrapolate:  # never beyond the far evidence; a little below the near end
        return None
    order = np.argsort(ys)
    if y <= hi:
        return float(np.interp(y, ys[order], xs[order]))
    (y0, x0), (y1, x1) = (ys[order][-2], xs[order][-2]), (ys[order][-1], xs[order][-1])
    return float(x1 + (x1 - x0) / (y1 - y0) * (y - y1)) if y1 != y0 else float(x1)


def relate_tracks(tracks: Sequence[TrackedObject], lane: Optional[EgoLaneObservation], lane_t: float, width: int,
                  height: int, off_road: Optional[np.ndarray] = None,
                  margin: float = 0.1) -> Optional[tuple[TrackLaneRelation, ...]]:
    """Relate each track's ground contact point (bottom centre of its box) to the ego lane.

    Returns None unless the ego lane is ``estimated`` with both boundaries measured or recently carried.
    Contact points beyond the lane evidence or within ``margin`` lane widths of a boundary are ambiguous.
    ``off_road`` (boolean mask, optional) turns a contact point beyond the adjacent lanes into ``off_road``.
    """
    if lane is None or lane.state is not EgoLaneState.ESTIMATED or not (lane.left and lane.right):
        return None
    if not (lane.left.points and lane.right.points):
        return None
    out = []
    for o in tracks:
        x, y = (o.box.x1 + o.box.x2) / 2 / width, o.box.y2 / height
        xl, xr = _x_at(lane.left, y), _x_at(lane.right, y)
        if xl is None or xr is None or xr <= xl:
            out.append(TrackLaneRelation(track_id=o.track_id, relation=LaneRelation.AMBIGUOUS, lane_t=lane_t))
            continue
        lateral = (x - (xl + xr) / 2) / (xr - xl)
        d = abs(lateral)
        if d <= 0.5 - margin:
            rel = LaneRelation.EGO_LANE
        elif d < 0.5 + margin or 1.5 - margin < d < 1.5 + margin:
            rel = LaneRelation.AMBIGUOUS
        elif d <= 1.5 - margin:
            rel = LaneRelation.LEFT_ADJACENT if lateral < 0 else LaneRelation.RIGHT_ADJACENT
        else:
            rel = LaneRelation.BEYOND
        if rel is LaneRelation.BEYOND and off_road is not None:
            mh, mw = off_road.shape
            r, cl = min(mh - 1, max(0, int(y * mh))), min(mw - 1, max(0, int(x * mw)))
            if off_road[r, cl]:
                rel = LaneRelation.OFF_ROAD
        beyond = any(b.evidence_y is not None and y < b.evidence_y for b in (lane.left, lane.right))
        out.append(TrackLaneRelation(track_id=o.track_id, relation=rel, lateral=round(float(lateral), 4),
                                     lane_t=lane_t, beyond_evidence=beyond))
    return tuple(out)
