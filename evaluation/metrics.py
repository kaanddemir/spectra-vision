"""Event-level matching and metrics.

Matching rules (per clip, per event type, one-to-one, optimal assignment):
a candidate may match a labelled event when |peak difference| <= ``peak_tolerance_s``
or their [start, end] intervals have temporal IoU >= ``min_tiou``.
Labelled events marked ``ambiguous`` absorb a matching candidate without
counting it as a true or false positive, and never count as misses.
Only events whose origin is in ``origins`` (human-confirmed and synthetic by default) are ground truth;
imported and draft events are ignored, and a clip with nothing else is not scored for that type.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluation.annotations import GROUND_TRUTH_ORIGINS, ClipAnnotation, Event, EventType, Origin
from evaluation.candidates import Candidate


@dataclass(frozen=True)
class MatchConfig:
    peak_tolerance_s: float = 1.0
    min_tiou: float = 0.3


def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else float(a == b)


def _gt_interval(e: Event) -> Optional[tuple[float, float]]:
    if e.t_start is not None and e.t_end is not None:
        return (e.t_start, e.t_end)
    return None


def _compatible(c: Candidate, e: Event, cfg: MatchConfig) -> tuple[bool, float]:
    """Return (may match, cost) where lower cost is a better match."""
    peak_err = abs(c.t_peak - e.anchor())
    iv = _gt_interval(e)
    tiou = temporal_iou((c.t_start, c.t_end), iv) if iv else 0.0
    ok = peak_err <= cfg.peak_tolerance_s or tiou >= cfg.min_tiou
    return ok, peak_err - tiou


@dataclass
class Match:
    clip_id: str
    type: EventType
    event: Event
    candidate: Candidate


@dataclass
class TypeResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ambiguous_hits: int = 0
    hours: float = 0.0
    peak_errors: list[float] = field(default_factory=list)
    start_errors: list[float] = field(default_factory=list)
    end_errors: list[float] = field(default_factory=list)
    by_condition: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))

    @property
    def precision(self) -> Optional[float]:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> Optional[float]:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)

    @property
    def false_per_hour(self) -> Optional[float]:
        return self.fp / self.hours if self.hours > 0 else None

    def as_dict(self) -> dict:
        def stats(xs: list[float]) -> Optional[dict]:
            if not xs:
                return None
            a = np.abs(np.asarray(xs))
            return {"n": len(xs), "median_abs_s": float(np.median(a)), "p90_abs_s": float(np.percentile(a, 90)),
                    "mean_signed_s": float(np.mean(xs))}

        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "ambiguous_hits": self.ambiguous_hits,
            "hours_evaluated": round(self.hours, 6),
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "false_candidates_per_hour": self.false_per_hour,
            "timing": {"peak": stats(self.peak_errors), "start": stats(self.start_errors), "end": stats(self.end_errors)},
            "recall_by_condition": {k: {"tp": v[0], "fn": v[1], "recall": v[0] / (v[0] + v[1]) if sum(v) else None}
                                    for k, v in sorted(self.by_condition.items())},
        }


def _condition_keys(clip: ClipAnnotation, e: Event) -> list[str]:
    keys = []
    for name in ("light", "weather", "road", "occlusion"):
        v = getattr(e.conditions, name)
        if v is None:
            v = getattr(clip.conditions, name)
        if v is not None:
            keys.append(f"{name}={getattr(v, 'value', v)}")
    return keys


def evaluate(clips: list[ClipAnnotation], candidates: list[Candidate], cfg: MatchConfig = MatchConfig(),
             origins: frozenset[Origin] = GROUND_TRUTH_ORIGINS,
             ) -> tuple[dict[EventType, TypeResult], list[Match], int]:
    """Return per-type results, matches, and the number of candidates on clips not evaluated for their type."""
    by_clip: dict[tuple[str, EventType], list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_clip[(c.clip_id, c.type)].append(c)
    known_ids = {c.clip_id for c in clips}
    unscored = sum(1 for c in candidates if c.clip_id not in known_ids)
    results = {t: TypeResult() for t in EventType}
    matches: list[Match] = []

    for clip in clips:
        for etype in EventType:
            cands = sorted(by_clip.get((clip.clip_id, etype), []), key=lambda c: c.t_peak)
            if not clip.is_evaluated_for(etype, origins):
                unscored += len(cands)
                continue
            res = results[etype]
            res.hours += clip.duration_s / 3600.0
            gts = clip.ground_truth(etype, origins)
            matched_c: set[int] = set()
            matched_g: set[int] = set()
            if gts and cands:
                big = 1e9
                cost = np.full((len(gts), len(cands)), big)
                for i, g in enumerate(gts):
                    for j, c in enumerate(cands):
                        ok, k = _compatible(c, g, cfg)
                        if ok:
                            cost[i, j] = k
                rows, cols = linear_sum_assignment(cost)
                for i, j in zip(rows, cols):
                    if cost[i, j] >= big:
                        continue
                    matched_g.add(i)
                    matched_c.add(j)
                    g, c = gts[i], cands[j]
                    if g.ambiguous:
                        res.ambiguous_hits += 1
                        continue
                    res.tp += 1
                    matches.append(Match(clip.clip_id, etype, g, c))
                    if g.t_peak is not None:
                        res.peak_errors.append(c.t_peak - g.t_peak)
                    if g.t_start is not None:
                        res.start_errors.append(c.t_start - g.t_start)
                    if g.t_end is not None:
                        res.end_errors.append(c.t_end - g.t_end)
                    for key in _condition_keys(clip, g):
                        res.by_condition[key][0] += 1
            for i, g in enumerate(gts):
                if i not in matched_g and not g.ambiguous:
                    res.fn += 1
                    for key in _condition_keys(clip, g):
                        res.by_condition[key][1] += 1
            res.fp += len(cands) - len(matched_c)
    return results, matches, unscored
