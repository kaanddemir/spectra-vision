"""Evidence packages and the timeline, built from an ``AnalysisResult`` without re-judging anything.

Inputs are the result's proposals, analyser reports and perception records (read lazily from the cache
entry, one window at a time). Outputs depend only on those, the ``ReviewConfig`` and
``EVIDENCE_VERSION``, so the same inputs always give byte-identical JSON.

Review interval: the proposal span widened to cover every evidence item (cut-in evidence can begin before
the lane entry), plus ``pre_s`` before and ``post_s`` after, together capped at ``max_context_s`` by
scaling both, then clipped to the video. The same rule applies to every event type; the evidence decides
how far back an interval reaches. The proposal and its evidence are never cut.

Relations come only from what the engine recorded: an impact proposal names the approach analyser it
was built on in its parameters, and carries that approach episode as its ``approach_episode`` evidence,
so the approach proposal with the same episode (track and times) is linked exactly. Proposals about a
common track with overlapping spans are linked as ``shares_track``. Linked proposals stay separate entries.
"""
from __future__ import annotations

import bisect
from array import array
from collections import Counter
from itertools import groupby
from typing import Optional, Sequence

from spectra.engine.contracts import (AnalysisResult, AnalyzerReport, AnalyzerStatus, EgoLaneState, EventProposal,
                                      FrameRecord, OutputStatus, Role, UnresolvedPattern, digest)
from spectra.evidence.contracts import (EVIDENCE_VERSION, AnalysisState, AnalyzerCoverage, Coverage,
                                        EvidencePackage, Gap, LaneStateRun, LaneSummary, Note, Relation,
                                        RelationKind, RelationRun, ReviewConfig, ReviewInterval, ReviewSegment,
                                        Timeline, TimelineEntry, TrackSummary, entry_order)

EPS = 1e-6

TITLES = {
    ("rapid_approach", None): "Rapid approach",
    ("cut_in", None): "Cut-in",
    ("impact", "camera_vehicle"): "Possible impact involving the camera vehicle",
    ("impact", "road_users"): "Possible contact between other road users",
}


def build_key(analysis_key: str, config: ReviewConfig) -> str:
    return digest({"evidence_version": EVIDENCE_VERSION, "analysis_key": analysis_key,
                   "config": config.model_dump(mode="json")})


def package_path(event_id: str) -> str:
    return f"packages/{event_id}.json"


def title_key(p: EventProposal) -> str:
    return "event." + p.type.value + (f".{p.subtype.value}" if p.subtype else "")


def _r(t: float) -> float:
    return round(t, 6)


class Records:
    """Frame times of one perception result, read once, so any window is found by bisection and only the
    frames inside it are parsed."""

    def __init__(self, frames: Sequence[FrameRecord]) -> None:
        self.frames = frames
        self.times = array("d", (f.t for f in frames))

    @property
    def start(self) -> float:
        return self.times[0]

    @property
    def end(self) -> float:
        return self.times[-1]

    def window(self, t0: float, t1: float) -> Sequence[FrameRecord]:
        i = bisect.bisect_left(self.times, t0 - EPS)
        j = bisect.bisect_right(self.times, t1 + EPS)
        return self.frames[i:j]

    def gaps(self, gap_s: float, t0: float = 0.0, t1: float = float("inf")) -> tuple[Gap, ...]:
        i = max(bisect.bisect_left(self.times, t0 - EPS) - 1, 0)
        j = min(bisect.bisect_right(self.times, t1 + EPS) + 1, len(self.times))
        t = self.times
        return tuple(Gap(start_s=t[k], end_s=t[k + 1]) for k in range(i, j - 1)
                     if t[k + 1] - t[k] > gap_s + EPS and t[k + 1] > t0 and t[k] < t1)


# --- intervals ----------------------------------------------------------------------------------------


def review_interval(p: EventProposal, video_end: float, analysed: tuple[float, float],
                    config: ReviewConfig) -> tuple[ReviewInterval, set[Note]]:
    notes: set[Note] = set()
    ev0 = min([p.t_start, *(e.t for e in p.evidence)])
    ev1 = max([p.t_end, *(e.t_end if e.t_end is not None else e.t for e in p.evidence)])
    pre, post = config.pre_s, config.post_s
    if pre + post > config.max_context_s:
        k = config.max_context_s / (pre + post)
        pre, post = pre * k, post * k
        notes.add(Note.CONTEXT_CAPPED)
    end_limit = max(video_end, ev1)
    start, end = max(0.0, ev0 - pre), min(end_limit, ev1 + post)
    if pre > 0 and ev0 - pre < -EPS:
        notes.add(Note.CLIPPED_AT_VIDEO_START)
    if post > 0 and ev1 + post > end_limit + EPS:
        notes.add(Note.CLIPPED_AT_VIDEO_END)
    if start < analysed[0] - EPS or end > analysed[1] + EPS:
        notes.add(Note.OUTSIDE_ANALYSIS)
    return ReviewInterval(start_s=_r(start), end_s=_r(end), evidence_start_s=_r(ev0), evidence_end_s=_r(ev1)), notes


def video_end(result: AnalysisResult, records: Records) -> float:
    """The video's end for clipping: its declared duration, or the last analysed frame when undeclared."""
    declared = result.perception.video.duration_s
    return max(declared, records.end) if declared is not None else records.end


# --- per-package records ------------------------------------------------------------------------------


def _tracks(p: EventProposal, frames: Sequence[FrameRecord]) -> tuple[tuple[TrackSummary, ...], set[Note]]:
    notes, out = set(), []
    tracked = [f for f in frames if f.status.get(Role.TRACKER) is OutputStatus.OK]
    for tid in p.track_ids:
        boxes = [(f.t, o) for f in tracked for o in f.tracks if o.track_id == tid]
        observed = [t for t, o in boxes if not o.predicted]
        predicted = len(boxes) - len(observed)
        last = observed[-1] if observed else None
        ends_early = last is not None and any(f.t > last + EPS for f in tracked)
        category = Counter(o.category for _, o in boxes).most_common(1)[0][0] if boxes else None
        out.append(TrackSummary(track_id=tid, category=category, first_s=observed[0] if observed else None,
                                last_s=last, observed=len(observed), predicted=predicted, ends_early=ends_early))
        if ends_early:
            notes.add(Note.TRACK_ENDS_EARLY)
        if predicted:
            notes.add(Note.TRACK_PREDICTED)
    return tuple(out), notes


def _runs(items: list[tuple[float, tuple]], make) -> list:
    """Consecutive items with the same key become one run."""
    out = []
    for key, group in groupby(items, key=lambda x: x[1]):
        ts = [t for t, _ in group]
        out.append(make(key, ts[0], ts[-1], len(ts)))
    return out


def _lanes(p: EventProposal, frames: Sequence[FrameRecord], available: bool) -> tuple[LaneSummary, set[Note]]:
    notes: set[Note] = set()
    if not available:
        return LaneSummary(available=False), {Note.EGO_LANE_UNAVAILABLE}
    states = [(f.t, (f.ego_lane.state,)) for f in frames if f.ego_lane is not None]
    ego = _runs(states, lambda k, a, b, n: LaneStateRun(state=k[0], start_s=a, end_s=b, frames=n))
    relations = []
    for tid in p.track_ids:
        seen = []
        for f in frames:
            if f.tracks is None or not any(o.track_id == tid for o in f.tracks):
                continue
            rel = next((r for r in f.lane_relations or () if r.track_id == tid), None)
            seen.append((f.t, (rel.relation, rel.beyond_evidence) if rel else (None, False)))
        relations += _runs(seen, lambda k, a, b, n, tid=tid: RelationRun(
            track_id=tid, relation=k[0], beyond_evidence=k[1], start_s=a, end_s=b, frames=n))
        in_span = [k for t, k in seen if p.t_start - EPS <= t <= p.t_end + EPS]
        if not any(rel is not None and not beyond for rel, beyond in in_span):
            notes.add(Note.LANE_UNOBSERVED)
    if not any(p.t_start - EPS <= t <= p.t_end + EPS and k[0] is EgoLaneState.ESTIMATED for t, k in states):
        notes.add(Note.EGO_LANE_UNAVAILABLE)
    return LaneSummary(available=True, ego_lane=tuple(ego), relations=tuple(relations)), notes


def _motion_missing(p: EventProposal, frames: Sequence[FrameRecord], first_t: float) -> bool:
    """A tracker frame in the proposal's span without a camera-motion estimate (the run's first frame has
    none by construction)."""
    return any(f.status.get(Role.TRACKER) is OutputStatus.OK and f.camera_motion is None and f.t > first_t + EPS
               for f in frames if p.t_start - EPS <= f.t <= p.t_end + EPS)


# --- relations ----------------------------------------------------------------------------------------


def _episode(p: EventProposal) -> Optional[tuple]:
    e = next((e for e in p.evidence if e.name == "approach_episode"), None)
    return None if e is None else (e.track_id, e.t, e.t_end)


def relations(events: Sequence[EventProposal], reports: Sequence[AnalyzerReport]) -> dict[str, list[Relation]]:
    infos = {r.analyzer.name: r.analyzer for r in reports}
    out: dict[str, set[tuple]] = {e.event_id: set() for e in events}
    linked = set()
    for e in events:
        basis = infos[e.analyzer].params.get("approach")
        source = infos.get(basis.get("name")) if isinstance(basis, dict) else None
        if source is None or (source.version, source.params) != (basis.get("version"), basis.get("params")):
            continue
        ep = _episode(e)
        for q in events:
            if q.analyzer == source.name and q is not e and ep is not None and _episode(q) == ep:
                out[e.event_id].add((RelationKind.BASED_ON, q.event_id))
                out[q.event_id].add((RelationKind.BASIS_OF, e.event_id))
                linked.add(frozenset((e.event_id, q.event_id)))
    for i, a in enumerate(events):
        for b in events[i + 1:]:
            if (set(a.track_ids) & set(b.track_ids) and a.t_start <= b.t_end and b.t_start <= a.t_end
                    and frozenset((a.event_id, b.event_id)) not in linked):
                out[a.event_id].add((RelationKind.SHARES_TRACK, b.event_id))
                out[b.event_id].add((RelationKind.SHARES_TRACK, a.event_id))
    kinds = list(RelationKind)
    return {k: [Relation(kind=r, event_id=i) for r, i in sorted(v, key=lambda x: (kinds.index(x[0]), x[1]))]
            for k, v in out.items()}


# --- packages and timeline ----------------------------------------------------------------------------


def _overlaps(u: UnresolvedPattern, iv: ReviewInterval) -> bool:
    return u.t_start <= iv.end_s + EPS and iv.start_s <= u.t_end + EPS


def _sorted_notes(notes: set[Note]) -> tuple[Note, ...]:
    order = list(Note)
    return tuple(sorted(notes, key=order.index))


def package(result: AnalysisResult, p: EventProposal, records: Records, config: ReviewConfig,
            related: Sequence[Relation] = ()) -> EvidencePackage:
    perception = result.perception
    analysed = (records.start, records.end)
    iv, notes = review_interval(p, video_end(result, records), analysed, config)
    frames = records.window(iv.start_s, iv.end_s)
    tracks, n = _tracks(p, frames)
    notes |= n
    roles = {c.role for c in perception.provenance.components}
    lanes, n = _lanes(p, frames, Role.EGO_LANE in roles)
    notes |= n
    if Role.TRACKER in roles and _motion_missing(p, frames, records.start):
        notes.add(Note.MOTION_MISSING)
    gaps = records.gaps(config.gap_s, iv.start_s, iv.end_s)
    if gaps:
        notes.add(Note.FRAME_GAP)
    warnings = tuple(w for w in perception.warnings
                     if w.t is None or iv.start_s - EPS <= w.t <= iv.end_s + EPS)
    if warnings:
        notes.add(Note.RUN_WARNING)
    unresolved = tuple(sorted((u for r in result.reports for u in r.unresolved if _overlaps(u, iv)),
                              key=lambda u: (u.t_start, u.t_end, u.reason.value, u.track_ids)))
    if unresolved:
        notes.add(Note.UNRESOLVED_NEARBY)
    if p.experimental:
        notes.add(Note.EXPERIMENTAL)
    analyzer = next(r.analyzer for r in result.reports if r.analyzer.name == p.analyzer)
    return EvidencePackage(analysis_key=result.analysis_key, perception_key=result.perception_key,
                           build_key=build_key(result.analysis_key, config), proposal=p, analyzer=analyzer,
                           video=perception.video, analysed_start_s=records.start, analysed_end_s=records.end,
                           review=iv, tracks=tracks, lanes=lanes, gaps=gaps, warnings=warnings,
                           unresolved=unresolved, related=tuple(related), notes=_sorted_notes(notes))


def coverage(report: AnalyzerReport) -> AnalyzerCoverage:
    if report.status is AnalyzerStatus.OK:
        kind = Coverage.PROPOSED if report.proposals else Coverage.NO_PROPOSAL
    else:
        kind = Coverage.UNAVAILABLE if report.status is AnalyzerStatus.UNAVAILABLE else Coverage.FAILED
    info = report.analyzer
    return AnalyzerCoverage(name=info.name, version=info.version, experimental=info.params.get("experimental") is True,
                            status=report.status, coverage=kind, reason=report.reason, proposals=report.proposals,
                            considered=report.considered, rejected=report.rejected, unresolved=report.unresolved)


def segments(entries: Sequence[TimelineEntry]) -> list[ReviewSegment]:
    """Overlapping review intervals joined, in order of start; ties broken by the entry order."""
    spans = sorted(entries, key=lambda e: (e.review.start_s, entry_order(e)))
    out: list[list] = []
    for e in spans:
        if out and e.review.start_s <= out[-1][1] + EPS:
            out[-1][1] = max(out[-1][1], e.review.end_s)
            out[-1][2].append(e.event_id)
        else:
            out.append([e.review.start_s, e.review.end_s, [e.event_id]])
    return [ReviewSegment(start_s=a, end_s=b, event_ids=tuple(ids)) for a, b, ids in out]


def build(result: AnalysisResult, config: ReviewConfig = ReviewConfig()) -> tuple[Timeline, list[EvidencePackage]]:
    """The timeline and one evidence package per proposal, in timeline order."""
    records = Records(result.perception.frames)
    links = relations(result.events, result.reports)
    packages = [package(result, p, records, config, links[p.event_id]) for p in result.events]
    entries = sorted((_entry(pk) for pk in packages), key=entry_order)
    segs = segments(entries)
    index = {i: k for k, s in enumerate(segs) for i in s.event_ids}
    entries = [e.model_copy(update={"segment": index[e.event_id]}) for e in entries]
    by_id = {pk.proposal.event_id: pk for pk in packages}
    timing = result.perception.timing
    timeline = Timeline(
        state=AnalysisState.INCOMPLETE if timing.ended_early else AnalysisState.COMPLETE,
        message=f"decoding ended early: {timing.ended_early}" if timing.ended_early else None,
        analysis_key=result.analysis_key, perception_key=result.perception_key,
        build_key=build_key(result.analysis_key, config), config=config, video=result.perception.video,
        analysed_start_s=records.start, analysed_end_s=records.end,
        analyzers=tuple(coverage(r) for r in result.reports), entries=tuple(entries), segments=tuple(segs),
        gaps=records.gaps(config.gap_s), warnings=result.perception.warnings)
    return timeline, [by_id[e.event_id] for e in entries]


def _entry(pk: EvidencePackage) -> TimelineEntry:
    p = pk.proposal
    return TimelineEntry(
        event_id=p.event_id, type=p.type, subtype=p.subtype, title_key=title_key(p),
        title=TITLES[(p.type.value, p.subtype and p.subtype.value)], t_start=p.t_start, t_peak=p.t_peak,
        t_end=p.t_end, review=pk.review, state=p.state, reasons=p.reasons, experimental=p.experimental,
        analyzer=p.analyzer, track_ids=p.track_ids, rank=p.rank, package=package_path(p.event_id), segment=0,
        related=pk.related, notes=pk.notes)


def without_result(state: AnalysisState, message: Optional[str], config: ReviewConfig = ReviewConfig()) -> Timeline:
    """The timeline of a run that produced no result: no entries, and no claim about the video."""
    if state in (AnalysisState.COMPLETE, AnalysisState.INCOMPLETE):
        raise ValueError("a run with a result is built from it")
    return Timeline(state=state, message=message, config=config)


def from_outcome(outcome, config: ReviewConfig = ReviewConfig()) -> tuple[Timeline, list[EvidencePackage]]:
    """Timeline and packages for any ``RunOutcome``: built from the result, or saying why there is none."""
    from spectra.engine.runner import RunState, StopReason
    if outcome.result is not None:
        return build(outcome.result, config)
    if outcome.state is RunState.CANCELLED:
        return without_result(AnalysisState.CANCELLED, "the analysis was cancelled", config), []
    state = AnalysisState.RESOURCE_STOPPED if outcome.stop is StopReason.RESOURCES else AnalysisState.FAILED
    return without_result(state, outcome.error, config), []
