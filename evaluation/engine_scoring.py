"""Score the engine's event proposals against annotations, over the time span the engine processed.

Usage:
    python -m evaluation.engine_scoring ANNOTATIONS.jsonl CLIP_ID ENGINE_OUTPUT.json \\
        --start S --end E --name NAME --out-root evaluation/reports [--origins human_confirmed ...]

``ENGINE_OUTPUT.json`` is what ``python -m spectra.engine analyze ... --events`` prints. Engine times are
source presentation times; the annotation's times must use the same base (check this once per source).
Only the window ``[start, end]`` is scored:
- events wholly inside it are kept;
- events that cross its edges are kept as ``ambiguous``, so they neither count as misses nor as hits;
- proposals whose peak lies outside it are dropped.

All proposals and ``supported`` proposals alone are scored separately. Impact proposals of either subtype
are scored as collisions. Results go to
``<out-root>/<name>/`` and are logged in ``<out-root>/results_log.jsonl``; an existing result is never
overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from evaluation.annotations import (GROUND_TRUTH_ORIGINS, ClipAnnotation, EventType, Origin,
                                    load_annotations)
from evaluation.candidates import Candidate, save_candidates
from evaluation.metrics import MatchConfig, evaluate

ENGINE_TYPES = {"rapid_approach": EventType.APPROACH, "cut_in": EventType.CUT_IN, "impact": EventType.COLLISION}


def _fmt(v) -> str:
    return "–" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def summarize(report: dict) -> str:
    lines = [f"# {report['experiment']} ({report['split']} split, {report['clips']} clips)", ""]
    unconfirmed = sorted(set(report.get("origins", [])) - {o.value for o in GROUND_TRUTH_ORIGINS})
    if unconfirmed:
        lines += [f"**Scored against labels that are not ground truth: {', '.join(unconfirmed)}.**", ""]
    if report.get("clips_licence_excluded"):
        lines += [f"Clips excluded by licence: {len(report['clips_licence_excluded'])}", ""]
    if report["clips_missing_cache"]:
        lines += [f"Clips without cache (not scored): {len(report['clips_missing_cache'])}", ""]
    lines += ["| algorithm | type | TP | FP | FN | precision | recall | F1 | false/h | median peak err (s) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, a in report["algorithms"].items():
        for t, r in a["by_type"].items():
            if r["hours_evaluated"] == 0:
                continue
            peak = r["timing"]["peak"]
            lines.append(f"| {name} | {t} | {r['tp']} | {r['fp']} | {r['fn']} | {_fmt(r['precision'])} | "
                         f"{_fmt(r['recall'])} | {_fmt(r['f1'])} | {_fmt(r['false_candidates_per_hour'])} | "
                         f"{_fmt(peak['median_abs_s'] if peak else None)} |")
    return "\n".join(lines) + "\n"


def _inside(a: float | None, start: float, end: float) -> bool:
    return a is None or start <= a <= end


def window(clip: ClipAnnotation, start: float, end: float) -> ClipAnnotation:
    """The clip restricted to ``[start, end]``, with times shifted to start at 0."""
    if not 0 <= start < end <= clip.duration_s + 1e-6:
        raise ValueError(f"window {start}-{end} is not within the clip (0-{clip.duration_s})")

    def shift(t):
        return None if t is None else min(max(t, start), end) - start

    def times(x):
        return [t for t in (x.t_start, getattr(x, "t_peak", None), x.t_end) if t is not None]

    events = []
    for e in clip.events:
        ts = times(e)
        if max(ts) < start or min(ts) > end:
            continue
        crosses = not all(_inside(t, start, end) for t in ts)
        events.append(e.model_copy(update=dict(
            t_start=shift(e.t_start), t_peak=shift(e.t_peak), t_end=shift(e.t_end),
            ambiguous=e.ambiguous or crosses,
            involved=tuple(b.model_copy(update=dict(t=b.t - start)) for b in e.involved if start <= b.t <= end),
            note="; ".join(filter(None, [e.note, "crosses the scored window" if crosses else None])) or None)))
    labels = []
    for s in clip.source_labels:
        ts = times(s)
        if ts and (max(ts) < start or min(ts) > end):
            continue
        labels.append(s.model_copy(update=dict(
            t_start=shift(s.t_start), t_end=shift(s.t_end),
            boxes=tuple(b.model_copy(update=dict(t=b.t - start)) for b in s.boxes if start <= b.t <= end))))
    data = clip.model_dump()
    data.update(clip_id=f"{clip.clip_id}@{start:g}-{end:g}", duration_s=end - start, frame_timestamps_s=None,
                events=[e.model_dump() for e in events], source_labels=[s.model_dump() for s in labels])
    return ClipAnnotation.model_validate(data)


def from_engine(clip_id: str, events: list[dict], start: float, end: float,
                supported_only: bool = False) -> list[Candidate]:
    """Engine ``EventProposal`` records as evaluation candidates, shifted into the window."""
    out = []
    for c in events:
        if c["type"] not in ENGINE_TYPES:
            raise ValueError(f"unknown engine event type {c['type']!r}")
        if not start <= c["t_peak"] <= end or (supported_only and c["state"] != "supported"):
            continue
        clamp = lambda t: min(max(t, start), end) - start
        out.append(Candidate(
            clip_id=clip_id, type=ENGINE_TYPES[c["type"]], t_start=clamp(c["t_start"]),
            t_peak=c["t_peak"] - start, t_end=clamp(c["t_end"]),
            score=c["rank"] if c.get("rank") is not None else 0.0, algorithm=c["analyzer"],
            algorithm_version="engine", track_ids=tuple(c.get("track_ids", ())),
            observations={"event_id": c["event_id"], "state": c["state"], "subtype": c.get("subtype"),
                          "reasons": list(c.get("reasons", ())), "t_peak_source_s": c["t_peak"]}))
    return out


def score(annotations: Path, clip_id: str, engine_output: Path, start: float, end: float, name: str,
          out_root: Path, origins: frozenset[Origin] = GROUND_TRUTH_ORIGINS,
          cfg: MatchConfig = MatchConfig()) -> dict:
    out = Path(out_root) / name
    if out.exists():
        raise FileExistsError(f"{out} exists; results are kept, choose a new name")
    clip = next((c for c in load_annotations(annotations) if c.clip_id == clip_id), None)
    if clip is None:
        raise ValueError(f"clip {clip_id!r} not in {annotations}")
    if clip.licence.evaluation_allowed is False:
        raise ValueError(f"the licence of {clip_id!r} does not allow evaluation")
    engine = json.loads(Path(engine_output).read_text())
    if engine.get("state") != "completed":
        raise ValueError(f"engine run did not complete: {engine.get('state')} {engine.get('error')}")
    w = window(clip, start, end)
    report = {
        "experiment": name, "split": clip.split.value, "origins": sorted(o.value for o in origins),
        "annotations_sha256": hashlib.sha256(Path(annotations).read_bytes()).hexdigest(),
        "engine_output_sha256": hashlib.sha256(Path(engine_output).read_bytes()).hexdigest(),
        "perception_key": engine.get("perception_key"), "engine_timing": engine.get("timing"),
        "clip": clip_id, "window_s": [start, end], "match": cfg.__dict__, "clips": 1,
        "clips_missing_cache": [], "algorithms": {},
    }
    out.mkdir(parents=True)
    for label, supported_only in (("engine_all", False), ("engine_supported", True)):
        cands = from_engine(w.clip_id, engine.get("events", []), start, end, supported_only)
        save_candidates(cands, out / f"{label}.candidates.jsonl")
        results, _, unscored = evaluate([w], cands, cfg, origins)
        report["algorithms"][label] = {
            "version": "engine", "candidates": len(cands), "unscored_candidates": unscored,
            "by_type": {t.value: results[t].as_dict() for t in EventType},
        }
    (out / "results.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "summary.md").write_text(summarize(report))
    with open(Path(out_root) / "results_log.jsonl", "a") as log:
        log.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "experiment": name,
                              "clip": clip_id, "window_s": [start, end], "output": str(out)}) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("annotations", type=Path)
    p.add_argument("clip_id")
    p.add_argument("engine_output", type=Path)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--origins", nargs="+", default=sorted(o.value for o in GROUND_TRUTH_ORIGINS))
    a = p.parse_args()
    report = score(a.annotations, a.clip_id, a.engine_output, a.start, a.end, a.name, a.out_root,
                   frozenset(Origin(o) for o in a.origins))
    print(summarize(report))


if __name__ == "__main__":
    main()
