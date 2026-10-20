"""Event analysers that run on cached perception records, never on pixels or models.

The analysers read the records once, keep compact per-track series, and propose moments for human review.
A proposal is ``supported`` when every condition the analyser checks held on observed data and
``uncertain`` when the pattern was seen but a condition could not be checked; its reason codes say which.
Each evidence item says whether it was observed in the records, estimated under a stated model, or is
context only. Patterns an analyser saw but did not propose are counted by reason in its report. Missing
lanes, missing tracks and unobservable stretches never count as evidence that nothing happened.

``ApproachAnalyzer``: apparent time-to-contact from looming. For an object approaching at constant
closing speed the image width ``w`` of its box satisfies ``tau = w / (dw/dt)``, the time until it reaches
the camera plane (Lee 1976; used for monocular forward-collision warning by Dagan et al. 2004). It needs
no calibration, metric depth or ego speed, and it is a time to the camera plane relative to the camera,
whatever is moving. It is approximate: real closing speeds change, box widths are noisy, and a box that
stops covering the whole vehicle changes ``w`` without any approach. A rigid object approaching the camera
grows by the same factor horizontally and vertically, so growth of the box in one direction only
(emerging from behind another vehicle, the side of a passing vehicle coming into view, a track sliding
onto a neighbour) is not looming.

``CutInAnalyzer``: a tracked object's lane relation going from an adjacent lane to the camera's lane
while the camera's lane keeps the same boundary identities.

``ImpactAnalyzer``: a possible impact involving the camera vehicle, proposed only when two independent
signals coincide: an apparent approach that ends close to the camera, and a jolt in reliable estimates of
the camera's image motion. Neither signal alone is a proposal, and a missing or unreliable motion estimate
is recorded, never counted as a jolt.

``RoadUserContactAnalyzer`` (experimental, off by default): two tracked road users whose boxes meet at a
similar depth after converging, followed by an abrupt change in one's camera-compensated image motion.
Its proposals are always ``uncertain`` until it has been validated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Optional, Sequence

import numpy as np

from spectra.engine.adapters import AnalyzerOutput, AnalyzerUnavailable
from spectra.engine.contracts import (Basis, CameraMotion, ComponentInfo, EgoLaneState, EventDraft, EventSubtype,
                                      EventType, EvidenceItem, EvidenceState, FrameRecord, LaneRelation, OutputStatus,
                                      ReasonCode, Role, UnresolvedPattern)

# --- shared ---------------------------------------------------------------------------------------


def theil_sen(t: np.ndarray, y: np.ndarray, confidence: float = 0.9) -> tuple[float, float, float]:
    """Median of pairwise slopes and Sen's (1968) distribution-free confidence interval for it.

    Robust to up to about 29% outlying points and to irregular sample times. Returns (slope, low, high).
    """
    i, j = np.triu_indices(len(t), 1)
    dt = t[j] - t[i]
    keep = dt > 0
    slopes = np.sort((y[j] - y[i])[keep] / dt[keep])
    n, m = len(t), len(slopes)
    if m == 0:
        raise ValueError("theil_sen needs at least two distinct times")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    half = z * math.sqrt(n * (n - 1) * (2 * n + 5) / 18)  # Kendall's S standard deviation, no ties
    lo = max(int(round((m - half) / 2)) - 1, 0)
    hi = min(int(round((m + half) / 2)), m - 1)
    return float(np.median(slopes)), float(slopes[lo]), float(slopes[hi])


@dataclass(frozen=True)
class _Obs:
    t: float
    x1: float
    y1: float
    x2: float
    y2: float
    predicted: bool
    score: Optional[float]
    relation: Optional[LaneRelation]  # None: no relation computed for this frame
    lateral: Optional[float]
    beyond_evidence: bool
    lane_ids: Optional[tuple[int, int]]  # boundary identities of the ego lane the relation used


@dataclass
class _Perception:
    tracks: dict[int, list[_Obs]]
    tracker_frames: int
    lane_frames: int  # frames with any lane relations
    motion: list[tuple[float, float]]  # (t, log scale rate per second) of the camera-motion estimate
    lane_changes: list[tuple[float, float]]  # (start, last shown) of completed ego lane changes
    first_t: float
    last_t: float
    tracker_times: list[float] = field(default_factory=list)  # frames on which the tracker ran
    estimates: dict[float, CameraMotion] = field(default_factory=dict)  # camera motion by frame time


def _read(frames: Sequence[FrameRecord]) -> _Perception:
    tracks: dict[int, list[_Obs]] = {}
    lanes: dict[float, tuple[int, int]] = {}
    changes: dict[tuple, list] = {}  # crossing -> [start, last shown, completed]
    motion, tracker_frames, lane_frames = [], 0, 0
    tracker_times, estimates = [], {}
    first = last = None
    for f in frames:
        first = f.t if first is None else first
        last = f.t
        lane = f.ego_lane
        if lane is not None:
            if (lane.state is EgoLaneState.ESTIMATED and lane.left is not None and lane.right is not None
                    and lane.left.track_id is not None and lane.right.track_id is not None):
                lanes[f.t] = (lane.left.track_id, lane.right.track_id)
            c = lane.lane_change
            if c is not None:
                seen = changes.setdefault((c.boundary_id, c.t_start, c.direction), [c.t_start, f.t, False])
                seen[1] = f.t
                seen[2] = seen[2] or c.phase == "completed"
        if f.camera_motion is not None:
            m = f.camera_motion
            motion.append((f.t, math.log(m.scale) / (f.t - m.since_t)))
            estimates[f.t] = m
        if f.status.get(Role.TRACKER) is not OutputStatus.OK:
            continue
        tracker_frames += 1
        tracker_times.append(f.t)
        relations = {r.track_id: r for r in f.lane_relations or ()}
        lane_frames += f.lane_relations is not None
        for o in f.tracks:
            r = relations.get(o.track_id)
            tracks.setdefault(o.track_id, []).append(_Obs(
                f.t, o.box.x1, o.box.y1, o.box.x2, o.box.y2, o.predicted, o.score,
                None if r is None else r.relation, None if r is None else r.lateral,
                False if r is None else r.beyond_evidence, None if r is None else lanes.get(r.lane_t)))
    if first is None:
        raise AnalyzerUnavailable("no frame records to analyse")
    completed = sorted((start, end) for start, end, done in changes.values() if done)
    return _Perception(tracks, tracker_frames, lane_frames, motion, completed, first, last, tracker_times, estimates)


def _confident(o: _Obs, min_score: float) -> bool:
    """Observed in this frame by a detection the tracker itself treats as high confidence. Lower scores
    are matched only in its second association stage, to keep identities alive; their boxes often belong
    to another object or cover part of one, so they are not evidence for events."""
    return not o.predicted and o.score is not None and o.score >= min_score


def _info(name: str, params: dict, version: str = "1") -> ComponentInfo:
    return ComponentInfo(role=Role.EVENT_ANALYZER, name=name, version=version, params=params)


def _item(basis: Basis, name: str, t: float, **kw) -> EvidenceItem:
    return EvidenceItem(basis=basis, name=name, t=round(t, 3), **kw)


# --- rapid approach -------------------------------------------------------------------------------

TAU_NOTE = ("apparent time-to-contact from the growth of the box width, assuming constant closing speed; "
            "approximate and unverified; not a distance, a speed or a collision time")


class ApproachAnalyzer:
    """``EventAnalyzer`` for rapid apparent approach of a tracked object.

    At each observation of a track, the box over the trailing ``window_s`` gives Theil-Sen growth rates
    ``r = d ln(size)/dt`` of its width and of its height, each with a ``confidence`` interval; ``tau = 1/r``,
    attributed to the window's middle observation, which is the time a slope over the window represents
    (attributing it to the end overstates tau by about half the window while the object closes). An
    observation is approaching when, for width and for height alike, even the conservative end of the
    interval, ``1/r_low``, is at most ``tau_s``.
    Approaching observations of one track less than ``merge_gap_s`` apart form one episode, which is
    proposed or rejected with a reason.

    Rules and why each exists:

    * ``tau_s`` (3 s): the review threshold. Forward-collision-warning tests issue warnings at time-to-
      collision of about 2.0-2.4 s (US NCAP); 3 s keeps such situations and leaves ordinary following,
      whose apparent size changes slowly, out.
    * conservative bound instead of the point estimate: box jitter, partial occlusion and an identity
      switch to a differently sized object all widen the interval, so a noisy or stepped width series does
      not trigger, without a separate residual or jump threshold. Few or small boxes widen it too.
    * width and height must both show it: looming is isotropic. Growth in one direction only is a box
      that covers more of the scene, not an object coming closer. On the sample highway clip this was the
      mechanism behind vehicles emerging from occlusion, oncoming vehicles whose side came into view, and
      tracks sliding onto a neighbour. Edge-touching boxes are already excluded, so the frame bottom does
      not cut the height; the bonnet can still hide the bottom of a vehicle a few metres away.
    * boxes touching the frame edge, boxes the tracker predicted rather than observed, and boxes from
      detections below ``min_score`` (0.6, the tracker's own high-confidence threshold) are left out:
      their width does not reliably measure the object.
    * identity jumps: a step in box height between observations at most ``jump_window_s`` (0.5 s) apart
      that differs from what the episode's own fitted growth predicts by more than ``jump_ratio`` (2x)
      either way makes the proposal ``uncertain`` (``identity_jump``). It is what a track switching to a
      differently sized vehicle looks like. The comparison is with the episode's growth, not with a fixed
      ratio, because a real approach at under a second's time-to-contact doubles its box within 0.5 s too;
      a steady approach exceeds the limit only in the last ~0.15 s, when its box already touches the frame
      edge and is left out.
    * relevance from lanes: objects whose observed lane relation in the episode is mostly an adjacent,
      further or off-road lane are left out (``other_lane``). Mostly the camera's lane gives ``supported``.
    * relevance without lanes: ``tau`` is the time to reach the camera *plane*, which a vehicle being
      overtaken in the next lane also reaches. For a pinhole camera, an object at a constant lateral offset
      ``X`` has box centre ``x = c + (X / W) * w`` for its box width ``w`` and real width ``W``, whatever its
      depth, focal length or principal point ``c``. A Theil-Sen fit of ``x`` against ``w`` over the episode
      therefore estimates ``X / W``, its offset when it reaches the camera plane, in its own widths. When
      the lane relation is not established, an offset beyond ``path_half_width`` (1.5 widths: half the
      object plus half a similar vehicle, plus the unknown camera mounting offset) passes beside and is
      left out (``passes_beside``); otherwise the proposal is ``uncertain`` (``lane_not_established``). The relation needs a constant lateral offset and
      a box centred on the object; camera yaw shifts every box sideways and biases it, which is why it
      is not used to override an observed lane relation.

    Camera motion from the tracker is reported as context only. Its scale term mostly measures the
    parallax of the near road during forward driving, not zoom, so it cannot correct the object's growth.
    Camera pitch and roll move boxes but do not change their width.
    """

    name = "apparent-approach"

    def __init__(self, *, tau_s: float = 3.0, window_s: float = 1.0, min_observations: int = 5,
                 confidence: float = 0.9, merge_gap_s: float = 1.0, edge_px: float = 2.0,
                 path_half_width: float = 1.5, min_score: float = 0.6, jump_ratio: float = 2.0,
                 jump_window_s: float = 0.5) -> None:
        self.tau_s, self.window_s, self.min_obs = tau_s, window_s, min_observations
        self.jump_ratio, self.jump_window_s = jump_ratio, jump_window_s
        self.min_score = min_score
        self.path_half_width = path_half_width
        self.confidence, self.merge_gap_s, self.edge_px = confidence, merge_gap_s, edge_px
        self.info = _info(self.name, {"tau_s": tau_s, "window_s": window_s, "min_observations": min_observations,
                                      "confidence": confidence, "merge_gap_s": merge_gap_s, "edge_px": edge_px,
                                      "path_half_width": path_half_width, "min_score": min_score,
                                      "size": "box width and height", "fit": "Theil-Sen with Sen interval",
                                      "passing": "episode fit of box centre against width",
                                      "jump_ratio": jump_ratio, "jump_window_s": jump_window_s}, version="3")

    def load(self) -> None:
        pass

    def close(self) -> None:
        pass

    def analyze(self, frames: Sequence[FrameRecord], width: int, height: int) -> AnalyzerOutput:
        p = _read(frames)
        if not p.tracker_frames:
            raise AnalyzerUnavailable("no tracker output: apparent approach cannot be assessed")
        out = AnalyzerOutput()
        for track_id, obs in sorted(p.tracks.items()):
            out.extend(self._track(track_id, obs, p, width, height))
        return out

    def _usable(self, o: _Obs, width: int, height: int) -> bool:
        e = self.edge_px
        return _confident(o, self.min_score) and o.x1 > e and o.y1 > e and o.x2 < width - e and o.y2 < height - e

    def _track(self, track_id: int, obs: list[_Obs], p: _Perception, width: int, height: int):
        out = AnalyzerOutput()
        seen = [o for o in obs if self._usable(o, width, height)]
        if len(seen) < self.min_obs:
            return out
        t = np.array([o.t for o in seen])
        w = np.array([o.x2 - o.x1 for o in seen])
        x = np.array([(o.x1 + o.x2) / 2 for o in seen])
        logw, logh = np.log(w), np.log(np.array([o.y2 - o.y1 for o in seen]))
        points = []  # (index, tau, tau_conservative, window start, tau from height, middle observation)
        start = 0
        for k in range(len(seen)):
            while t[start] <= t[k] - self.window_s:
                start += 1
            n = k - start + 1
            if n < self.min_obs or t[k] - t[start] < 0.6 * self.window_s:
                continue
            rate, low, _ = theil_sen(t[start:k + 1], logw[start:k + 1], self.confidence)
            rate_h, low_h, _ = theil_sen(t[start:k + 1], logh[start:k + 1], self.confidence)
            if min(low, low_h) > 0 and max(1 / low, 1 / low_h) <= self.tau_s:
                points.append((k, 1 / rate, max(1 / low, 1 / low_h), start, 1 / rate_h, (start + k) // 2))
        episodes: list[list] = []
        for pt in points:
            if episodes and t[pt[0]] - t[episodes[-1][-1][0]] <= self.merge_gap_s:
                episodes[-1].append(pt)
            else:
                episodes.append([pt])
        for ep in episodes:
            span = slice(ep[0][3], ep[-1][0] + 1)
            c = self._propose(track_id, ep, seen, t, p, height, self._passing(w[span], x[span]))
            if isinstance(c, ReasonCode):
                out.reject(c)
            else:
                out.drafts.append(c)
        return out

    def _jump(self, window: list[_Obs]) -> Optional[tuple[float, float, float]]:
        """The largest box-height step against the episode's fitted growth, as (t, t_end, ratio), when it
        exceeds ``jump_ratio`` either way between observations at most ``jump_window_s`` apart."""
        t = np.array([o.t for o in window])
        logh = np.log(np.array([o.y2 - o.y1 for o in window]))
        rate = theil_sen(t, logh, self.confidence)[0]
        worst, limit = None, math.log(self.jump_ratio)
        for i in range(len(window) - 1):
            for j in range(i + 1, len(window)):
                dt = t[j] - t[i]
                if dt > self.jump_window_s:
                    break
                dev = logh[j] - logh[i] - rate * dt
                if abs(dev) > limit and (worst is None or abs(dev) > abs(worst[2])):
                    worst = (float(t[i]), float(t[j]), dev)
        return None if worst is None else (worst[0], worst[1], math.exp(worst[2]))

    def _passing(self, w: np.ndarray, x: np.ndarray) -> Optional[float]:
        """Slope of box centre against box width: the lateral offset on reaching the camera plane, in object
        widths (+ right). None when the widths do not vary."""
        try:
            return theil_sen(w, x, self.confidence)[0]
        except ValueError:
            return None

    def _propose(self, track_id, ep, seen, t, p: _Perception, height,
                   passing: Optional[float]) -> EventDraft | ReasonCode:
        first, last = ep[0], ep[-1]
        t0, t1 = float(t[first[3]]), float(t[last[0]])
        peak = min(ep, key=lambda e: e[2])
        tp = float(t[peak[5]])
        window = seen[first[3]:last[0] + 1]
        ego = sum(o.relation is LaneRelation.EGO_LANE and not o.beyond_evidence for o in window)
        other = sum(o.relation in (LaneRelation.LEFT_ADJACENT, LaneRelation.RIGHT_ADJACENT, LaneRelation.BEYOND,
                                   LaneRelation.OFF_ROAD) and not o.beyond_evidence for o in window)
        if other > ego:
            return ReasonCode.OTHER_LANE  # not approaching the camera's path
        reasons = []
        if not (ego > other and 2 * ego >= len(window)):
            if passing is not None and abs(passing) > self.path_half_width:
                return ReasonCode.PASSES_BESIDE
            reasons.append(ReasonCode.LANE_NOT_ESTABLISHED)
        jump = self._jump(window)
        if jump is not None:
            reasons.append(ReasonCode.IDENTITY_JUMP)
        tid = track_id
        evidence = [
            _item(Basis.OBSERVED, "approach_episode", t0, t_end=round(t1, 3), track_id=tid,
                  note="observations over which the box grew in width and height alike"),
            _item(Basis.OBSERVED, "ego_lane_fraction", t0, t_end=round(t1, 3), track_id=tid,
                  value=round(ego / len(window), 3), unit="fraction",
                  note="share of the episode's observations in the camera's lane, within the lane evidence"),
            _item(Basis.ESTIMATED, "apparent_tau_s", tp, track_id=tid, value=round(peak[1], 3), unit="s",
                  note=TAU_NOTE),
            _item(Basis.ESTIMATED, "apparent_tau_height_s", tp, track_id=tid, value=round(peak[4], 3), unit="s",
                  note="the same estimate from the box height"),
            _item(Basis.ESTIMATED, "apparent_tau_conservative_s", tp, track_id=tid, value=round(peak[2], 3),
                  unit="s", note=f"upper end of the {self.confidence:.0%} interval, the larger of width and height"),
            _item(Basis.CONTEXT, "box_height_fraction", tp, track_id=tid,
                  value=round((seen[peak[5]].y2 - seen[peak[5]].y1) / height, 4), unit="fraction_of_height",
                  note="box height over frame height; image size only, not a distance"),
        ]
        if passing is not None:
            evidence.append(_item(Basis.ESTIMATED, "passing_offset_widths", t0, track_id=tid,
                                  value=round(passing, 3), unit="object_widths",
                                  note="lateral offset from the camera on reaching the camera plane, + right, "
                                       "fitted over the episode; biased by camera yaw; approximate"))
        if jump is not None:
            evidence.append(_item(Basis.OBSERVED, "height_step_ratio", jump[0], t_end=round(jump[1], 3), track_id=tid,
                                  value=round(jump[2], 3), unit="ratio",
                                  note="box height step relative to the episode's fitted growth over the same time"))
        rates = [r for tm, r in p.motion if t0 <= tm <= t1]
        if rates:
            evidence.append(_item(Basis.CONTEXT, "image_log_scale_rate", tp, value=round(float(np.median(rates)), 4),
                                  unit="1/s", note="median global image scale rate from motion compensation; "
                                                   "mostly forward-motion parallax"))
        return EventDraft(type=EventType.RAPID_APPROACH, t_start=t0, t_peak=tp, t_end=t1,
                          state=EvidenceState.UNCERTAIN if reasons else EvidenceState.SUPPORTED,
                          reasons=tuple(reasons), rank=round(min(1.0, max(0.0, 1 - peak[2] / self.tau_s)), 4),
                          track_ids=(track_id,), evidence=tuple(evidence), analyzer=self.name)


# --- cut-in ---------------------------------------------------------------------------------------

_ADJACENT = {LaneRelation.LEFT_ADJACENT: "left", LaneRelation.RIGHT_ADJACENT: "right"}
_FAR = (LaneRelation.BEYOND, LaneRelation.OFF_ROAD)


def _side(o: _Obs) -> Optional[str]:
    """'ego', 'left', 'right', 'far', 'between' (straddling a boundary) or None when not observable here."""
    if o.relation is None or o.beyond_evidence:
        return None
    if o.relation is LaneRelation.EGO_LANE:
        return "ego"
    if o.relation in _ADJACENT:
        return _ADJACENT[o.relation]
    if o.relation in _FAR:
        return "far"
    return "between" if o.lateral is not None else None


@dataclass
class _Run:
    side: str
    obs: list

    @property
    def start(self) -> float:
        return self.obs[0].t

    @property
    def end(self) -> float:
        return self.obs[-1].t


class CutInAnalyzer:
    """``EventAnalyzer`` for another vehicle entering the camera's observed lane.

    Per track, observations are classified from the lane relation as the camera's lane, the left or right
    adjacent lane, further away, straddling a boundary, or not observable (no relation, or a contact
    point beyond the lane evidence where boundaries are extrapolated). A cut-in is an adjacent run of at
    least ``hold_s`` followed, after only straddling or unobservable observations, by a run in the camera's
    lane of at least ``hold_s``.

    Rules and why each exists:

    * ``hold_s`` (0.4 s, as for the camera's own lane changes) with ``min_observations`` on both sides:
      box jitter near a boundary flips single relations; a manoeuvre is sustained.
    * the camera's lane must keep the same two boundary identities from the end of the adjacent run to the
      confirming observation. When the camera changes lane, or the lane tracker switches which marking it
      follows, every vehicle's relation shifts by a lane without any vehicle moving; the identities are
      what tells these apart. If they changed and a completed ego lane change overlaps the entry, there is
      no proposal (``ego_lane_change``). If they changed without one, including during a crossing that
      was cancelled, the proposal is ``uncertain`` (``boundaries_reidentified``): a cancelled crossing did
      not move the camera into another lane.
    * relations beyond the lane evidence are unobservable: on curves and grades the extrapolated boundaries
      are where the lane model fails.
    * an entry that took longer than ``max_transition_s`` (3 s for crossing the straddling band of 0.2 lane
      widths; lane changes take about 4-6 s for a whole lane) was not observed as it happened, so the
      proposal is ``uncertain`` (``entry_not_observed``).
    * ``max_gap_s``: a run is only sustained if consecutive observations are no further apart than this.
    * observations from detections below ``min_score`` (the tracker's high-confidence threshold) or
      predicted by the tracker are not lane evidence.

    No proximity or risk condition: a proposal is an observation that a vehicle entered the lane, not a
    claim of danger or fault. The vehicle's apparent size is attached as context.
    """

    name = "lane-entry-cut-in"

    def __init__(self, *, hold_s: float = 0.4, min_observations: int = 3, max_transition_s: float = 3.0,
                 max_gap_s: float = 0.6, min_score: float = 0.6) -> None:
        self.hold_s, self.min_obs, self.min_score = hold_s, min_observations, min_score
        self.max_transition_s, self.max_gap_s = max_transition_s, max_gap_s
        self.info = _info(self.name, {"hold_s": hold_s, "min_observations": min_observations,
                                      "max_transition_s": max_transition_s, "max_gap_s": max_gap_s,
                                      "min_score": min_score, "suppressed_by": "completed ego lane changes"},
                          version="2")

    def load(self) -> None:
        pass

    def close(self) -> None:
        pass

    def analyze(self, frames: Sequence[FrameRecord], width: int, height: int) -> AnalyzerOutput:
        p = _read(frames)
        if not p.tracker_frames:
            raise AnalyzerUnavailable("no tracker output: cut-ins cannot be assessed")
        if not p.lane_frames:
            raise AnalyzerUnavailable("no lane relations were computed: cut-ins cannot be assessed")
        out = AnalyzerOutput()
        for track_id, obs in sorted(p.tracks.items()):
            out.extend(self._track(track_id, obs, p, height))
        return out

    def _sustained(self, run: _Run) -> bool:
        return len(run.obs) >= self.min_obs and run.end - run.start >= self.hold_s

    def _runs(self, obs: list[_Obs]) -> list[_Run]:
        """Sustained runs of one side, in time order. Short runs are relation noise and are dropped;
        consecutive sustained runs on the same side are one stay in that lane."""
        runs: list[_Run] = []
        for o in obs:
            side = _side(o) if _confident(o, self.min_score) else None
            if side is None or side == "between":
                continue
            if runs and runs[-1].side == side and o.t - runs[-1].end <= self.max_gap_s:
                runs[-1].obs.append(o)
            else:
                runs.append(_Run(side, [o]))
        kept: list[_Run] = []
        for r in runs:
            if not self._sustained(r):
                continue
            if kept and kept[-1].side == r.side:
                kept[-1].obs.extend(r.obs)
            else:
                kept.append(_Run(r.side, list(r.obs)))
        return kept

    def _track(self, track_id: int, obs: list[_Obs], p: _Perception, height: int) -> AnalyzerOutput:
        runs = self._runs(obs)
        out = AnalyzerOutput()
        for before, after in zip(runs, runs[1:]):
            if before.side in ("left", "right") and after.side == "ego":
                c = self._propose(track_id, before, after, p, height)
                if isinstance(c, ReasonCode):
                    out.reject(c)
                else:
                    out.drafts.append(c)
        return out

    def _confirming_index(self, run: _Run) -> int:
        """Index in ``run.obs`` of the first observation at which the run became sustained."""
        for k, o in enumerate(run.obs):
            if k + 1 >= self.min_obs and o.t - run.start >= self.hold_s:
                return k
        return len(run.obs) - 1

    def _propose(self, track_id: int, before: _Run, after: _Run, p: _Perception,
                   height: int) -> EventDraft | ReasonCode:
        last_adj, first_ego = before.obs[-1], after.obs[0]
        confirm = after.obs[self._confirming_index(after)]
        same_lane = last_adj.lane_ids is not None and last_adj.lane_ids == confirm.lane_ids
        ego_change = any(s <= confirm.t and e >= last_adj.t for s, e in p.lane_changes)
        if not same_lane and ego_change:
            return ReasonCode.EGO_LANE_CHANGE  # every relation shifted with the camera
        reasons = []
        if not same_lane:
            # a change of tracked marking cannot be told apart from the vehicle entering the lane
            reasons.append(ReasonCode.BOUNDARIES_REIDENTIFIED)
        if first_ego.t - last_adj.t > self.max_transition_s:
            reasons.append(ReasonCode.ENTRY_NOT_OBSERVED)
        h = (first_ego.y2 - first_ego.y1) / height
        tid = track_id
        lateral = "contact point from the camera's lane centre, + right; positions refer to the camera, whose " \
                  "mounting offset in the vehicle is unknown"
        evidence = (
            _item(Basis.OBSERVED, "lane_relation", before.start, t_end=round(last_adj.t, 3), track_id=tid,
                  value=f"{before.side}_adjacent", note=f"in the {before.side} adjacent lane"),
            _item(Basis.OBSERVED, "lane_relation", first_ego.t, t_end=round(confirm.t, 3), track_id=tid,
                  value="ego_lane", note="in the camera's lane, from the first observation there until sustained"),
            _item(Basis.OBSERVED, "transition_s", last_adj.t, t_end=round(first_ego.t, 3), track_id=tid,
                  value=round(first_ego.t - last_adj.t, 3), unit="s",
                  note="between the last adjacent and the first in-lane observation"),
            _item(Basis.OBSERVED, "lane_lateral_before", last_adj.t, track_id=tid, value=last_adj.lateral,
                  unit="lane_widths", note=lateral),
            _item(Basis.OBSERVED, "lane_lateral_after", confirm.t, track_id=tid, value=confirm.lateral,
                  unit="lane_widths", note=lateral),
            _item(Basis.CONTEXT, "box_height_fraction", first_ego.t, track_id=tid, value=round(h, 4),
                  unit="fraction_of_height", note="box height over frame height; image size only, not a distance"),
        )
        return EventDraft(type=EventType.CUT_IN, t_start=last_adj.t, t_peak=first_ego.t, t_end=confirm.t,
                          state=EvidenceState.UNCERTAIN if reasons else EvidenceState.SUPPORTED,
                          reasons=tuple(reasons), rank=round(min(1.0, 2 * h), 4), track_ids=(track_id,),
                          evidence=evidence, analyzer=self.name)


# --- impact involving the camera vehicle ------------------------------------------------------------

_COMPONENTS = ("rotation", "tx", "ty", "log_scale")  # camera-motion rate components, per second


@dataclass(frozen=True)
class _Jolt:
    t: float
    z: float
    component: str


class ImpactAnalyzer:
    """``EventAnalyzer`` for a possible impact involving the camera vehicle (``impact/camera_vehicle``).

    Decision rule. A proposal needs all of:

    1. **Approach**: an apparent-approach episode of a track, exactly as ``ApproachAnalyzer`` proposes it
       (same parameters), so objects in other lanes or passing beside are already left out.
    2. **Close**: at the episode's last observation the box height is at least ``close_height`` (0.25) of
       the frame; or within ``close_window_s`` (0.5 s) after it a confident box of the track touches the
       bottom or a side edge of the frame; or the track's last observation falls within that window while
       its box height is at least ``lost_height`` (0.1) and the tracker keeps running for at least
       ``close_window_s`` without it (lost while large). The first that holds sets the close time.
    3. **Jolt**: within ``jolt_window`` (-0.3 s to +1.0 s around the close time) the camera's image motion
       changes abruptly: for at least one frame, the change of one motion rate component (rotation rate,
       horizontal or vertical translation rate in frame sizes per second, log-scale rate) from the frame
       before exceeds the median change over the ``baseline_s`` (5 s) before the window by more than
       ``jolt_z`` (10) robust standard deviations (1.4826 times the median absolute deviation, at least
       the component's ``min_spread``).

    Only reliable motion estimates count: at least ``min_inliers`` RANSAC inliers making up at least
    ``min_inlier_ratio`` of the correspondences, on consecutive tracker frames at most ``max_step_s``
    apart. Blur, a scene dominated by one moving vehicle, or an empty scene make estimates fail or
    disagree; that is not a jolt.

    Missing signals are never evidence. When the jolt cannot be judged, the pattern is not proposed: it is
    rejected as ``motion_estimate_missing`` (an estimate in the window is missing or unreliable and the
    reliable ones show no jolt) or ``motion_baseline_insufficient`` (fewer than ``min_baseline`` reliable
    changes before the window), and listed with its time in the report's ``unresolved`` patterns. The
    same moment is still proposed by the apparent-approach analyser. With all estimates reliable and no
    jolt, it is rejected as ``no_jolt``; an approach that did not end close is ``not_close``.

    The proposal is ``supported`` only when its approach episode was (observed in the camera's lane, no
    identity jump); otherwise it is ``uncertain`` with the approach's reasons. It spans the episode start
    to the later of the close time and the jolt, peaking at the jolt. Proposals that share a track or
    overlap in time are merged into the one with the strongest jolt (``merged``).

    Why: a jolt alone is shake, a pothole or a bend, and every sample driving clip has jolts; an approach
    alone is a near miss, which stays a rapid-approach proposal; closeness alone is ordinary traffic. Known
    confound: a vehicle filling the image dominates the global motion fit, so when it leaves the view the
    estimate can step without any impact. Such estimates usually lose inliers, which the reliability test
    catches; the rest are why every proposal is for review. The settle of the camera afterwards is attached
    as context only. All thresholds are unvalidated starting values.
    """

    name = "camera-vehicle-impact"

    def __init__(self, *, close_height: float = 0.25, close_window_s: float = 0.5, lost_height: float = 0.1,
                 jolt_window: tuple[float, float] = (-0.3, 1.0), jolt_z: float = 10.0, baseline_s: float = 5.0,
                 min_baseline: int = 10, min_inliers: int = 20, min_inlier_ratio: float = 0.3,
                 max_step_s: float = 0.25, min_spread: tuple[float, float, float, float] = (0.005, 0.002, 0.002,
                                                                                            0.002),
                 settle_s: float = 2.0, approach: Optional[ApproachAnalyzer] = None) -> None:
        self.approach = approach or ApproachAnalyzer()
        self.close_height, self.close_window_s, self.lost_height = close_height, close_window_s, lost_height
        self.jolt_window, self.jolt_z, self.baseline_s = tuple(jolt_window), jolt_z, baseline_s
        self.min_baseline, self.min_inliers, self.min_inlier_ratio = min_baseline, min_inliers, min_inlier_ratio
        self.max_step_s, self.min_spread, self.settle_s = max_step_s, tuple(min_spread), settle_s
        self.info = _info(self.name, {
            "close_height": close_height, "close_window_s": close_window_s, "lost_height": lost_height,
            "jolt_window": list(self.jolt_window), "jolt_z": jolt_z, "baseline_s": baseline_s,
            "min_baseline": min_baseline, "min_inliers": min_inliers, "min_inlier_ratio": min_inlier_ratio,
            "max_step_s": max_step_s, "min_spread": dict(zip(_COMPONENTS, self.min_spread)), "settle_s": settle_s,
            "approach": {"name": self.approach.info.name, "version": self.approach.info.version,
                         "params": self.approach.info.params},
            "rule": "approach episode, ending close, with a jolt in reliable camera-motion estimates"})

    def load(self) -> None:
        pass

    def close(self) -> None:
        pass

    def analyze(self, frames: Sequence[FrameRecord], width: int, height: int) -> AnalyzerOutput:
        p = _read(frames)
        if not p.tracker_frames:
            raise AnalyzerUnavailable("no tracker output: impacts cannot be assessed")
        if not p.estimates:
            raise AnalyzerUnavailable("no camera-motion estimates were recorded: impacts cannot be assessed")
        changes = self._changes(p, width, height)
        out, found = AnalyzerOutput(), []
        for track_id, obs in sorted(p.tracks.items()):
            for episode in self.approach._track(track_id, obs, p, width, height).drafts:
                got = self._assess(episode, obs, p, changes, width, height)
                if isinstance(got, UnresolvedPattern):
                    out.unresolve(got)
                elif isinstance(got, ReasonCode):
                    out.reject(got)
                else:
                    found.append(got)
        merged, dropped = self._merge(found)
        out.drafts.extend(d for d, _ in merged)
        if dropped:
            out.reject(ReasonCode.MERGED, dropped)
        return out

    # -- signals --

    def _reliable(self, m: Optional[CameraMotion]) -> bool:
        return (m is not None and m.inliers >= self.min_inliers and m.matches > 0
                and m.inliers / m.matches >= self.min_inlier_ratio)

    def _changes(self, p: _Perception, width: int, height: int) -> dict[float, Optional[np.ndarray]]:
        """Per tracker frame: absolute change of the motion rate vector from the previous tracker frame, or
        None when either estimate is missing or unreliable or the frames are too far apart."""
        def rates(m: CameraMotion, t: float) -> np.ndarray:
            dt = t - m.since_t
            return np.array([m.rotation, m.tx / width, m.ty / height, math.log(m.scale)]) / dt
        out, prev = {}, None
        for t in p.tracker_times:
            m = p.estimates.get(t)
            v = rates(m, t) if self._reliable(m) else None
            ok = v is not None and prev is not None and prev[1] is not None and t - prev[0] <= self.max_step_s
            out[t] = np.abs(v - prev[1]) if ok else None
            prev = (t, v)
        return out

    def _close(self, episode: EventDraft, obs: list[_Obs], p: _Perception, width: int,
               height: int) -> Optional[tuple[float, str, float]]:
        """(close time, how, box height fraction) or None."""
        t1 = episode.t_end
        end = next(o for o in obs if o.t == t1)
        if (end.y2 - end.y1) / height >= self.close_height:
            return t1, "box_height", (end.y2 - end.y1) / height
        e = self.approach.edge_px
        later = [o for o in obs if t1 < o.t <= t1 + self.close_window_s and _confident(o, self.approach.min_score)]
        for o in later:
            if o.y2 >= height - e or o.x1 <= e or o.x2 >= width - e:
                return o.t, "frame_edge", (o.y2 - o.y1) / height
        last = max((o for o in obs if not o.predicted), key=lambda o: o.t)
        if (t1 <= last.t <= t1 + self.close_window_s and (last.y2 - last.y1) / height >= self.lost_height
                and any(t >= last.t + self.close_window_s for t in p.tracker_times)):
            return last.t, "track_lost", (last.y2 - last.y1) / height
        return None

    def _jolt(self, t_close: float, changes: dict) -> tuple[Optional[_Jolt], int, int, bool]:
        """(jolt in the window or None, judged frames, frames in the window, baseline sufficient). The jolt's
        time is its onset, the first frame over the threshold; its size is the largest z in the window (a
        one-frame jolt changes the rates twice: on the jolt and back)."""
        w0, w1 = t_close + self.jolt_window[0], t_close + self.jolt_window[1]
        base = np.array([d for t, d in changes.items() if w0 - self.baseline_s <= t < w0 and d is not None])
        window = [(t, d) for t, d in changes.items() if w0 <= t <= w1]
        judged = [(t, d) for t, d in window if d is not None]
        if len(base) < self.min_baseline:
            return None, len(judged), len(window), False
        med = np.median(base, axis=0)
        spread = np.maximum(1.4826 * np.median(np.abs(base - med), axis=0), self.min_spread)
        onset, best = None, None
        for t, d in judged:
            z = (d - med) / spread
            k = int(np.argmax(z))
            if z[k] >= self.jolt_z:
                onset = t if onset is None else onset
                best = (float(z[k]), _COMPONENTS[k]) if best is None or z[k] > best[0] else best
        jolt = None if onset is None else _Jolt(onset, *best)
        return jolt, len(judged), len(window), True

    def _settle(self, t: float, p: _Perception, width: int) -> Optional[str]:
        """'settled' when the camera's translation and scale rates over ``settle_s`` after ``t`` are within
        the clip's lowest decile, 'moving' otherwise; None without enough reliable estimates."""
        def level(tm: float, m: CameraMotion) -> float:
            return (math.hypot(m.tx, m.ty) / width + abs(math.log(m.scale))) / (tm - m.since_t)
        levels = {tm: level(tm, m) for tm, m in p.estimates.items() if self._reliable(m)}
        after = [v for tm, v in levels.items() if t + 0.5 < tm <= t + 0.5 + self.settle_s]
        if len(after) < 5 or len(levels) < 20:
            return None
        return "settled" if np.median(after) <= np.percentile(list(levels.values()), 10) else "moving"

    # -- proposals --

    def _assess(self, episode: EventDraft, obs: list[_Obs], p: _Perception, changes: dict, width: int,
                height: int) -> EventDraft | ReasonCode | UnresolvedPattern:
        close = self._close(episode, obs, p, width, height)
        if close is None:
            return ReasonCode.NOT_CLOSE
        t_close, how, frac = close
        jolt, judged, total, baseline = self._jolt(t_close, changes)
        span = dict(t_start=episode.t_start, t_end=round(t_close + self.jolt_window[1], 3),
                    track_ids=episode.track_ids)
        if not baseline:
            return UnresolvedPattern(reason=ReasonCode.MOTION_BASELINE_INSUFFICIENT, **span)
        if jolt is None:
            if judged < total:
                return UnresolvedPattern(reason=ReasonCode.MOTION_ESTIMATE_MISSING, **span)
            return ReasonCode.NO_JOLT
        tid = episode.track_ids[0]
        keep = {"approach_episode", "ego_lane_fraction", "apparent_tau_s", "height_step_ratio"}
        evidence = [e for e in episode.evidence if e.name in keep]
        evidence.append(_item(Basis.OBSERVED, f"close_{how}", t_close, track_id=tid, value=round(frac, 4),
                              unit="fraction_of_height",
                              note={"box_height": "box height over frame height at the end of the approach",
                                    "frame_edge": "the box reached the frame edge; height over frame height",
                                    "track_lost": "the track was lost while this large"}[how]))
        evidence.append(_item(Basis.ESTIMATED, "camera_jolt_z", jolt.t, value=round(jolt.z, 2), unit="z",
                              note=f"abrupt change of the camera's image {jolt.component} rate against the "
                                   f"{self.baseline_s:g} s before; global image motion, not measured vehicle "
                                   "motion"))
        evidence.append(_item(Basis.CONTEXT, "motion_estimates_judged", t_close + self.jolt_window[0],
                              t_end=round(t_close + self.jolt_window[1], 3), value=round(judged / total, 3),
                              unit="fraction", note="share of tracker frames in the jolt window with a reliable "
                                                    "motion change"))
        settle = self._settle(jolt.t, p, width)
        if settle is not None:
            evidence.append(_item(Basis.CONTEXT, "camera_after_jolt", jolt.t, value=settle,
                                  note=f"camera image motion over {self.settle_s:g} s afterwards against the "
                                       "clip's lowest decile"))
        t_end = max(t_close, jolt.t)
        return EventDraft(type=EventType.IMPACT, subtype=EventSubtype.CAMERA_VEHICLE, state=episode.state,
                          reasons=episode.reasons, t_start=episode.t_start, t_peak=jolt.t, t_end=t_end,
                          track_ids=episode.track_ids, evidence=tuple(evidence),
                          rank=round(min(1.0, jolt.z / (4 * self.jolt_z)), 4), analyzer=self.name)

    def _merge(self, found: list[EventDraft]) -> tuple[list[tuple[EventDraft, float]], int]:
        """Proposals sharing a track or overlapping in time become one: the strongest jolt's, spanning all
        of them, uncertain if any was."""
        def z(d: EventDraft) -> float:
            return next(e.value for e in d.evidence if e.name == "camera_jolt_z")
        groups: list[list[EventDraft]] = []
        for d in sorted(found, key=lambda d: d.t_start):
            g = next((g for g in groups if any(set(d.track_ids) & set(o.track_ids) or
                                               (d.t_start <= o.t_end and o.t_start <= d.t_end) for o in g)), None)
            (g.append(d) if g is not None else groups.append([d]))
        out, dropped = [], 0
        for g in groups:
            best = max(g, key=z)
            reasons = tuple(dict.fromkeys(r for d in g for r in d.reasons))
            merged = best.model_copy(update=dict(
                t_start=min(d.t_start for d in g), t_end=max(d.t_end for d in g),
                track_ids=tuple(sorted({i for d in g for i in d.track_ids})), reasons=reasons,
                state=EvidenceState.UNCERTAIN if reasons else EvidenceState.SUPPORTED))
            out.append((EventDraft.model_validate(dict(merged)), z(best)))
            dropped += len(g) - 1
        return out, dropped


# --- contact between other road users (experimental) --------------------------------------------------


class RoadUserContactAnalyzer:
    """``EventAnalyzer`` for possible contact between two other road users (``impact/road_users``).
    Experimental: off unless configured, and every proposal is ``uncertain`` (``not_validated``).

    For two confident tracks at a frame, all of:

    * **meeting**: their boxes overlap;
    * **similar depth**: their ground contact rows (box bottoms) differ by at most ``depth_rows`` (0.15) of
      the larger box height and their height ratio is within ``height_ratio`` (0.6-1.6). Boxes overlapping
      in the image are usually at different depths, one occluding the other; this is a proxy only;
    * **converging**: over the preceding ``converge_s`` (1 s) the horizontal gap between their box centres,
      in mean box widths, shrank: Theil-Sen slope below 0 with its ``confidence`` interval below 0;
    * **abrupt change**: within ``after_s`` (1 s) one of them changes its camera-compensated image velocity
      by more than ``jolt_z`` (10) robust standard deviations of its own changes over the second before
      (at least ``min_spread`` frame widths per second). Compensation maps the previous box centre through
      the camera-motion estimate, which must be reliable as for impacts.

    Patterns that met, at similar depth, while converging but showed no abrupt change are rejected as
    ``no_jolt``, or as ``motion_estimate_missing`` when the change could not be judged. A pair's frames
    less than ``merge_gap_s`` apart are one pattern. Proposal rates on ordinary traffic decide whether it
    is ever enabled.
    """

    name = "road-user-contact"

    def __init__(self, *, depth_rows: float = 0.15, height_ratio: tuple[float, float] = (0.6, 1.6),
                 converge_s: float = 1.0, after_s: float = 1.0, jolt_z: float = 10.0, min_spread: float = 0.01,
                 confidence: float = 0.9, merge_gap_s: float = 1.0, min_score: float = 0.6,
                 min_inliers: int = 20, min_inlier_ratio: float = 0.3, max_step_s: float = 0.25) -> None:
        self.depth_rows, self.height_ratio, self.converge_s = depth_rows, tuple(height_ratio), converge_s
        self.after_s, self.jolt_z, self.min_spread, self.confidence = after_s, jolt_z, min_spread, confidence
        self.merge_gap_s, self.min_score = merge_gap_s, min_score
        self.min_inliers, self.min_inlier_ratio, self.max_step_s = min_inliers, min_inlier_ratio, max_step_s
        self.info = _info(self.name, {
            "depth_rows": depth_rows, "height_ratio": list(self.height_ratio), "converge_s": converge_s,
            "after_s": after_s, "jolt_z": jolt_z, "min_spread": min_spread, "confidence": confidence,
            "merge_gap_s": merge_gap_s, "min_score": min_score, "min_inliers": min_inliers,
            "min_inlier_ratio": min_inlier_ratio, "max_step_s": max_step_s, "experimental": True})

    def load(self) -> None:
        pass

    def close(self) -> None:
        pass

    def analyze(self, frames: Sequence[FrameRecord], width: int, height: int) -> AnalyzerOutput:
        p = _read(frames)
        if not p.tracker_frames:
            raise AnalyzerUnavailable("no tracker output: contact between road users cannot be assessed")
        if not p.estimates:
            raise AnalyzerUnavailable("no camera-motion estimates were recorded: contact between road users "
                                      "cannot be assessed")
        seen = {k: {o.t: o for o in obs if _confident(o, self.min_score)} for k, obs in p.tracks.items()}
        seen = {k: v for k, v in seen.items() if v}
        velocity = {k: self._velocity(v, p, width) for k, v in seen.items()}
        meetings: dict[tuple[int, int], list[float]] = {}
        for t in p.tracker_times:
            here = sorted(k for k, v in seen.items() if t in v)
            for a_i, a in enumerate(here):
                for b in here[a_i + 1:]:
                    if self._meet(seen[a][t], seen[b][t]) and self._converging(seen[a], seen[b], t):
                        meetings.setdefault((a, b), []).append(t)
        out = AnalyzerOutput()
        for (a, b), ts in sorted(meetings.items()):
            for group in _groups(ts, self.merge_gap_s):
                got = self._assess(a, b, group, seen, velocity, height)
                if isinstance(got, UnresolvedPattern):
                    out.unresolve(got)
                elif isinstance(got, ReasonCode):
                    out.reject(got)
                else:
                    out.drafts.append(got)
        return out

    def _meet(self, a: _Obs, b: _Obs) -> bool:
        overlap = min(a.x2, b.x2) > max(a.x1, b.x1) and min(a.y2, b.y2) > max(a.y1, b.y1)
        ha, hb = a.y2 - a.y1, b.y2 - b.y1
        lo, hi = self.height_ratio
        return overlap and abs(a.y2 - b.y2) <= self.depth_rows * max(ha, hb) and lo <= ha / hb <= hi

    def _converging(self, a: dict, b: dict, t: float) -> bool:
        common = sorted(tm for tm in a if t - self.converge_s <= tm <= t and tm in b)
        if len(common) < 4 or common[-1] - common[0] < 0.6 * self.converge_s:
            return False
        gap = [abs((a[tm].x1 + a[tm].x2) - (b[tm].x1 + b[tm].x2)) / ((a[tm].x2 - a[tm].x1) + (b[tm].x2 - b[tm].x1))
               for tm in common]
        _, _, high = theil_sen(np.array(common), np.array(gap), self.confidence)
        return high < 0

    def _velocity(self, obs: dict[float, _Obs], p: _Perception, width: int) -> dict[float, Optional[np.ndarray]]:
        """Camera-compensated image velocity of the box centre, frame widths per second, per observation;
        None where the previous observation or a reliable motion estimate is missing."""
        out, ts = {}, sorted(obs)
        for t0, t1 in zip(ts, ts[1:]):
            m = p.estimates.get(t1)
            if t1 - t0 > self.max_step_s or m is None or m.since_t != t0 or m.matches == 0 or (
                    m.inliers < self.min_inliers or m.inliers / m.matches < self.min_inlier_ratio):
                out[t1] = None
                continue
            c0 = np.array([(obs[t0].x1 + obs[t0].x2) / 2, (obs[t0].y1 + obs[t0].y2) / 2])
            c1 = np.array([(obs[t1].x1 + obs[t1].x2) / 2, (obs[t1].y1 + obs[t1].y2) / 2])
            cos, sin = math.cos(m.rotation), math.sin(m.rotation)
            moved = m.scale * np.array([cos * c0[0] - sin * c0[1], sin * c0[0] + cos * c0[1]]) + [m.tx, m.ty]
            out[t1] = (c1 - moved) / width / (t1 - t0)
        return out

    def _change(self, v: dict, t_from: float, t_to: float, before: float) -> tuple[Optional[float], bool]:
        """(largest robust z of a velocity change in (t_from, t_to], judged everywhere?)"""
        ts = sorted(v)
        changes = {t1: (None if v[t0] is None or v[t1] is None else float(np.linalg.norm(v[t1] - v[t0])))
                   for t0, t1 in zip(ts, ts[1:])}
        base = [c for t, c in changes.items() if t_from - before <= t <= t_from and c is not None]
        window = [c for t, c in changes.items() if t_from < t <= t_to]
        if len(base) < 5 or not window:
            return None, False
        med = float(np.median(base))
        spread = max(1.4826 * float(np.median(np.abs(np.array(base) - med))), self.min_spread)
        judged = [c for c in window if c is not None]
        z = max(((c - med) / spread for c in judged), default=None)
        return z, len(judged) == len(window)

    def _assess(self, a: int, b: int, group: list[float], seen: dict, velocity: dict,
                height: int) -> EventDraft | ReasonCode | UnresolvedPattern:
        t0, t1 = group[0], group[-1]
        za, fa = self._change(velocity[a], t0, t1 + self.after_s, self.converge_s)
        zb, fb = self._change(velocity[b], t0, t1 + self.after_s, self.converge_s)
        z = max((x for x in (za, zb) if x is not None), default=None)
        span = dict(t_start=t0, t_end=round(t1 + self.after_s, 3), track_ids=(a, b))
        if z is None or z < self.jolt_z:
            if not (fa and fb):
                return UnresolvedPattern(reason=ReasonCode.MOTION_ESTIMATE_MISSING, **span)
            return ReasonCode.NO_JOLT
        oa, ob = seen[a][t0], seen[b][t0]
        evidence = (
            _item(Basis.OBSERVED, "boxes_meet", t0, t_end=round(t1, 3), note="the two boxes overlap at a similar "
                                                                             "ground row; a depth proxy only"),
            _item(Basis.OBSERVED, "height_ratio", t0, value=round((oa.y2 - oa.y1) / (ob.y2 - ob.y1), 3),
                  unit="ratio", note="box heights; similar heights suggest similar depth"),
            _item(Basis.ESTIMATED, "velocity_change_z", t0, t_end=round(t1 + self.after_s, 3), value=round(z, 2),
                  unit="z", note="largest change of a camera-compensated box velocity against its own "
                                 "preceding second"),
            _item(Basis.CONTEXT, "box_height_fraction", t0, track_id=a, value=round((oa.y2 - oa.y1) / height, 4),
                  unit="fraction_of_height", note="image size only, not a distance"),
        )
        return EventDraft(type=EventType.IMPACT, subtype=EventSubtype.ROAD_USERS, state=EvidenceState.UNCERTAIN,
                          reasons=(ReasonCode.NOT_VALIDATED,), experimental=True, t_start=t0, t_peak=t1,
                          t_end=t1, track_ids=(a, b), evidence=evidence,
                          rank=round(min(1.0, z / (4 * self.jolt_z)), 4), analyzer=self.name)


def _groups(ts: list[float], gap: float) -> list[list[float]]:
    out: list[list[float]] = []
    for t in ts:
        if out and t - out[-1][-1] <= gap:
            out[-1].append(t)
        else:
            out.append([t])
    return out


__all__ = ["ApproachAnalyzer", "CutInAnalyzer", "ImpactAnalyzer", "RoadUserContactAnalyzer", "theil_sen"]
