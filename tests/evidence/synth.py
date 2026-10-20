"""Synthetic analysis results with full control over frames, tracks, lanes, motion and proposals."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from spectra.engine.contracts import (AnalysisResult, AnalyzerReport, AnalyzerStatus, Basis, Box, CameraMotion,
                                      ComponentInfo, EgoLaneObservation, EgoLaneState, EventDraft, EventSubtype,
                                      EventType, EvidenceItem, EvidenceState, FrameRecord, LaneBoundary,
                                      LaneRelation, ObjectCategory, OutputStatus, PerceptionResult, Provenance,
                                      ReasonCode, Role, RunConfig, RunWarning, TimestampSource, TimingReport,
                                      TrackedObject, TrackLaneRelation, UnresolvedPattern, VideoInfo, analysis_key)

W, H = 640, 360
APPROACH = ComponentInfo(role=Role.EVENT_ANALYZER, name="apparent-approach", version="3", params={"tau_s": 2.0})
CUT_IN = ComponentInfo(role=Role.EVENT_ANALYZER, name="lane-entry-cut-in", version="2", params={})
IMPACT = ComponentInfo(role=Role.EVENT_ANALYZER, name="camera-vehicle-impact", version="1", params={
    "approach": {"name": APPROACH.name, "version": APPROACH.version, "params": APPROACH.params}})
CONTACT = ComponentInfo(role=Role.EVENT_ANALYZER, name="road-user-contact", version="1",
                        params={"experimental": True})
PERCEPTION = (ComponentInfo(role=Role.DETECTOR, name="det", version="1"),
              ComponentInfo(role=Role.TRACKER, name="trk", version="1"),
              ComponentInfo(role=Role.SEGMENTER, name="seg", version="1"),
              ComponentInfo(role=Role.EGO_LANE, name="lane", version="1"))
KEY = "a" * 64


def video(duration: Optional[float] = 20.0, **kw) -> VideoInfo:
    base = dict(file_name="clip.mp4", size_bytes=1000, sha256="b" * 64, container="mp4", codec="h264", width=W,
                height=H, coded_width=W, coded_height=H, rotation=0, time_base="1/1000", start_s=0.0,
                duration_s=duration, nominal_fps=10.0, timestamp_source=TimestampSource.CONTAINER)
    return VideoInfo(**{**base, **kw})


def box(i: int) -> Box:
    return Box(x1=100 + i, y1=100, x2=200 + i, y2=200)


Tracks = Callable[[float], Sequence[tuple[int, bool]]]  # t -> [(track_id, predicted)]
Relations = Callable[[float, int], Optional[LaneRelation]]  # (t, track_id) -> relation or None


def frames(times: Sequence[float], tracks: Tracks = lambda t: [(1, False)],
           relation: Relations = lambda t, i: LaneRelation.EGO_LANE, lanes: bool = True,
           lane_state: Callable[[float], EgoLaneState] = lambda t: EgoLaneState.ESTIMATED,
           motion: Callable[[float], bool] = lambda t: True) -> tuple[FrameRecord, ...]:
    out, prev = [], None
    for i, t in enumerate(times):
        objs = tuple(TrackedObject(track_id=k, box=box(i), score=None if p else 0.9, category=ObjectCategory.CAR,
                                   predicted=p) for k, p in tracks(t))
        status = {Role.DETECTOR: OutputStatus.OK, Role.TRACKER: OutputStatus.OK}
        kw = {}
        if lanes:
            status.update({Role.SEGMENTER: OutputStatus.NOT_SCHEDULED, Role.EGO_LANE: OutputStatus.OK})
            state = lane_state(t)
            b = LaneBoundary(near=(0.3, 0.9), far=(0.45, 0.5), evidence_y=0.6)
            kw["ego_lane"] = (EgoLaneObservation(state=state, left=b, right=b.model_copy(update={"near": (0.7, 0.9)}),
                                                 offset=0.0, confidence=0.8) if state is EgoLaneState.ESTIMATED
                              else EgoLaneObservation(state=state))
            rels = [(o.track_id, relation(t, o.track_id)) for o in objs]
            kw["lane_relations"] = tuple(TrackLaneRelation(track_id=k, relation=r, lane_t=t) for k, r in rels if r)
        if prev is not None and motion(t):
            kw["camera_motion"] = CameraMotion(since_t=prev, scale=1.0, rotation=0.0, tx=0.0, ty=0.0, matches=50,
                                               inliers=40, residual_px=0.5)
        out.append(FrameRecord(index=i, t=t, status=status, detections=(), tracks=objs, **kw))
        prev = t
    return tuple(out)


def perception(times: Sequence[float], *, info: Optional[VideoInfo] = None, lanes: bool = True,
               warnings: Sequence[RunWarning] = (), ended_early: Optional[str] = None, **kw) -> PerceptionResult:
    fr = frames(times, lanes=lanes, **kw)
    comps = PERCEPTION if lanes else PERCEPTION[:2]
    return PerceptionResult(video=info or video(), config=RunConfig(interval_s=0.1),
                            provenance=Provenance(engine_version="test", decoder={}, components=comps),
                            timing=TimingReport(frames_decoded=len(fr), frames_analysed=len(fr), first_t=times[0],
                                                last_t=times[-1], ended_early=ended_early),
                            frames=fr, warnings=tuple(warnings))


def item(name: str, t: float, t_end: Optional[float] = None, track_id: Optional[int] = 1,
         basis: Basis = Basis.OBSERVED, **kw) -> EvidenceItem:
    return EvidenceItem(basis=basis, name=name, t=t, t_end=t_end, track_id=track_id, **kw)


def approach(t0: float, t1: float, track: int = 1, peak: Optional[float] = None, **kw) -> EventDraft:
    return EventDraft(type=EventType.RAPID_APPROACH, state=EvidenceState.SUPPORTED, t_start=t0,
                      t_peak=peak if peak is not None else t1, t_end=t1, track_ids=(track,), analyzer=APPROACH.name,
                      evidence=(item("approach_episode", t0, t1, track),
                                item("apparent_tau_s", t1, None, track, Basis.ESTIMATED, value=1.2, unit="s"),
                                item("box_height_fraction", t1, None, track, Basis.CONTEXT, value=0.3,
                                     unit="fraction_of_height")), **kw)


def impact(t0: float, t1: float, jolt: float, track: int = 1, analyzer: ComponentInfo = IMPACT) -> EventDraft:
    return EventDraft(type=EventType.IMPACT, subtype=EventSubtype.CAMERA_VEHICLE, state=EvidenceState.SUPPORTED,
                      t_start=t0, t_peak=jolt, t_end=max(t1, jolt), track_ids=(track,), analyzer=analyzer.name,
                      evidence=(item("approach_episode", t0, t1, track),
                                item("camera_jolt_z", jolt, None, None, Basis.ESTIMATED, value=14.0, unit="z")))


def cut_in(t0: float, t1: float, track: int = 2, entry_from: Optional[float] = None) -> EventDraft:
    ev = [item("lane_relation", t1, None, track, value="ego_lane")]
    if entry_from is not None:
        ev.insert(0, item("lane_relation", entry_from, t0, track, value="left_adjacent"))
    return EventDraft(type=EventType.CUT_IN, state=EvidenceState.UNCERTAIN, reasons=(ReasonCode.ENTRY_NOT_OBSERVED,),
                      t_start=t0, t_peak=t1, t_end=t1, track_ids=(track,), analyzer=CUT_IN.name, evidence=tuple(ev))


def contact(t0: float, t1: float) -> EventDraft:
    return EventDraft(type=EventType.IMPACT, subtype=EventSubtype.ROAD_USERS, state=EvidenceState.UNCERTAIN,
                      reasons=(ReasonCode.NOT_VALIDATED,), experimental=True, t_start=t0, t_peak=t1, t_end=t1,
                      track_ids=(1, 2), analyzer=CONTACT.name)


def result(drafts: Sequence[EventDraft] = (), *, times: Sequence[float] = None,
           analyzers: Sequence[ComponentInfo] = (APPROACH, CUT_IN, IMPACT),
           status: dict[str, tuple[AnalyzerStatus, str]] = {},
           unresolved: dict[str, Sequence[UnresolvedPattern]] = {}, **kw) -> AnalysisResult:
    times = times if times is not None else [round(0.1 * i, 3) for i in range(201)]
    p = perception(times, **kw)
    akey = analysis_key(KEY, analyzers)
    reports = []
    for a in analyzers:
        st, reason = status.get(a.name, (AnalyzerStatus.OK, None))
        if st is AnalyzerStatus.OK:
            n = sum(d.analyzer == a.name for d in drafts)
            un = tuple(unresolved.get(a.name, ()))
            rejected = {}
            for u in un:
                rejected[u.reason] = rejected.get(u.reason, 0) + 1
            reports.append(AnalyzerReport(analyzer=a, status=st, proposals=n, considered=n + len(un),
                                          rejected=rejected, unresolved=un))
        else:
            reports.append(AnalyzerReport(analyzer=a, status=st, reason=reason))
    events = sorted((d.bind(akey) for d in drafts), key=lambda e: (e.t_start, e.event_id))
    return AnalysisResult(analysis_key=akey, perception_key=KEY, perception=p, reports=tuple(reports),
                          events=tuple(events))
