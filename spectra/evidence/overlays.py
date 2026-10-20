"""Optional drawing data for a review interval, read from the cached perception records.

Nothing is rendered into any video and no model runs. Each analysed frame's boxes and lane boundaries are
returned as recorded and are valid from that frame's time until the next analysed frame, or at most
``overlay_max_age_s`` later. Positions are never interpolated: between two analysed frames the records say
nothing, and an in-between box would be invented. A displayed video frame with no valid overlay frame
shows no overlay, which the interface should present as 'not analysed here', not as 'nothing there'.

Lane estimation may run at a coarser interval than tracking. A frame on which it was not scheduled shows the
most recent lane observation instead, with that observation's own time as ``observed_t``, but only while
the drawing stays within ``overlay_max_age_s`` of it, and only if lane estimation ran there and found a lane
(``estimated`` or ``ambiguous``). A failed, unavailable or missing observation is never carried forward.
"""
from __future__ import annotations

import bisect
from typing import Optional, Sequence

from spectra.engine.contracts import EgoLaneState, FrameRecord, OutputStatus, PerceptionResult, Role
from spectra.evidence.build import EPS, Records
from spectra.evidence.contracts import (EvidencePackage, LaneOverlay, Overlay, OverlayFrame, ReviewConfig,
                                        TrackOverlay)


def overlay(perception: PerceptionResult, pk: EvidencePackage, config: ReviewConfig = ReviewConfig(),
            records: Optional[Records] = None) -> Overlay:
    if perception.video != pk.video:
        raise ValueError("the perception records are not the ones this package was built from")
    records = records or Records(perception.frames)
    iv = pk.review
    frames = records.window(iv.start_s, iv.end_s)
    related = set(pk.proposal.track_ids)
    times = records.times
    out = []
    for f in frames:
        k = bisect.bisect_right(times, f.t + EPS)
        until = f.t + config.overlay_max_age_s
        if k < len(times):
            until = min(until, times[k])
        tracks = None
        if f.status.get(Role.TRACKER) is OutputStatus.OK:
            rel = {r.track_id: r.relation for r in f.lane_relations or ()}
            tracks = tuple(TrackOverlay(track_id=o.track_id, box=o.box, category=o.category, score=o.score,
                                        predicted=o.predicted, related=o.track_id in related,
                                        relation=rel.get(o.track_id)) for o in f.tracks)
        lane = None
        source = _lane_source(records, k - 1, until, config.overlay_max_age_s)
        if source is not None:
            e = source.ego_lane
            lane = LaneOverlay(state=e.state, left=e.left, right=e.right, observed_t=source.t)
        out.append(OverlayFrame(t=f.t, valid_until=round(until, 6), tracks=tracks, lane=lane))
    return Overlay(event_id=pk.proposal.event_id, width=perception.video.width, height=perception.video.height,
                   start_s=iv.start_s, end_s=iv.end_s, frames=tuple(out))


def _lane_source(records: Records, k: int, until: float, max_age: float) -> Optional[FrameRecord]:
    """The frame whose lane observation is drawn for frame ``k``, valid until ``until``: frame ``k`` itself
    when lane estimation was scheduled there, else the latest earlier frame on which it was, if that
    observation found a lane and remains within ``max_age`` for the whole drawing. None otherwise."""
    for j in range(k, -1, -1):
        f = records.frames[j]
        status = f.status.get(Role.EGO_LANE)
        if status is OutputStatus.NOT_SCHEDULED:
            if until - records.times[j] > max_age + EPS:
                return None  # any earlier observation would be drawn beyond its age limit
            continue
        if j == k:
            return f if f.ego_lane is not None else None
        ok = status is OutputStatus.OK and f.ego_lane is not None and \
            f.ego_lane.state in (EgoLaneState.ESTIMATED, EgoLaneState.AMBIGUOUS)
        return f if ok and until - f.t <= max_age + EPS else None
    return None


def frame_at(frames: Sequence[OverlayFrame], t: float) -> Optional[OverlayFrame]:
    """The overlay to draw at display time ``t``: the latest analysed frame at or before ``t`` that is still
    valid, or None."""
    times = [f.t for f in frames]
    k = bisect.bisect_right(times, t + EPS) - 1
    if k < 0 or t > frames[k].valid_until + EPS:
        return None
    return frames[k]
