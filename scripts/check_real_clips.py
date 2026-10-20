"""Strict, sequential cache-to-evidence verification; no execution at import time.

Use --cache-only to refuse any perception cache miss. --out must be a new
location outside the repository. Source media, models and existing caches are
read-only during cache-only checks. Model numerical equivalence is not tested.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

# Support direct execution from scripts/ without an editable project install.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.datasets import clips, load_manifest
from spectra.engine.contracts import TimestampSource, canonical_json
from spectra.engine.models import default_specs
from spectra.engine.resources import ResourceExhausted, ResourceGuard, memory_pressure_level, peak_rss_mb
from spectra.engine.runner import RunState, StopReason
from spectra.engine.video import file_sha256
from spectra.engine.worker import TrackerSpec, WorkerJob, run_in_worker
from spectra.evidence.build import from_outcome
from spectra.evidence.contracts import ReviewInterval
from spectra.evidence.media import ClipUnsupported, extract_clip, resolve
from spectra.evidence.overlays import overlay
from spectra.evidence.store import EvidenceStore


def _select(available, requested):
    if not requested:
        return available
    selected = []
    for term in requested:
        exact = [c for c in available if c.metadata.id == term]
        found = exact or [c for c in available if term in c.metadata.id]
        if len(found) != 1:
            raise ValueError(f'clip selector {term!r} matches {len(found)} clips; use an exact id')
        if found[0] in selected:
            raise ValueError(f'duplicate clip selection: {term}')
        selected.append(found[0])
    return selected


def _models_json(items):
    return canonical_json([item.model_dump(mode='json') for item in items])


def _check(clip, job, output, guard, timeout_s):
    started = time.perf_counter()
    before = file_sha256(clip.path)
    guard.check(force=True)
    outcome = run_in_worker(job, timeout_s=timeout_s)
    if outcome.state is not RunState.COMPLETED:
        if outcome.stop is StopReason.RESOURCES:
            raise ResourceExhausted(outcome.error)
        raise RuntimeError(f'{outcome.stop}: {outcome.error}')
    if job.cache_only and not outcome.cache_hit:
        raise RuntimeError('cache-only worker did not report a cache hit')
    result = outcome.result
    proposals = _models_json(result.events)
    guard.check(force=True)
    built = time.perf_counter()
    timeline, packages = from_outcome(outcome)
    again, again_packages = from_outcome(outcome)
    if canonical_json(timeline) != canonical_json(again) or _models_json(packages) != _models_json(again_packages):
        raise AssertionError('evidence is not deterministic')
    if _models_json(result.events) != proposals:
        raise AssertionError('evidence construction changed proposals')
    build_s = time.perf_counter() - built
    store = EvidenceStore(output / 'evidence')
    store.write(timeline, packages)
    if store.load(timeline.build_key, timeline.analysis_key).timeline != timeline:
        raise AssertionError('stored timeline failed verification')
    overlay_count = 0
    for package in packages:
        guard.check(force=True)
        if store.package(timeline.build_key, package.proposal.event_id) != package:
            raise AssertionError('stored package differs')
        overlay(result.perception, package)
        overlay_count += 1
    # Even a video with no proposals exercises interval resolution and media handling.
    intervals = [p.review for p in packages[:2]]
    if not intervals:
        timing = result.perception.timing
        start = timing.first_t
        end = min(timing.last_t, start + 5.0) if start is not None and timing.last_t is not None else None
        if end is None:
            raise ValueError('no analysed timestamps available for media validation')
        intervals = [ReviewInterval(start_s=start, end_s=end, evidence_start_s=start, evidence_end_s=end)]
    media = []
    for index, interval in enumerate(intervals):
        guard.check(force=True)
        began = time.perf_counter()
        resolved = resolve(clip.path, result.perception.video, interval)
        if resolved.frames < 1 or resolved.decode_errors:
            raise AssertionError('interval resolution has no frames or decode errors')
        record = {'resolved': resolved.model_dump(mode='json'), 'resolve_s': time.perf_counter() - began}
        began = time.perf_counter()
        try:
            if index < len(packages):
                copied = store.clip(packages[index], clip.path)
            else:
                copied = extract_clip(clip.path, result.perception.video, interval, output / 'clips', f'interval-{index}')
            record['clip'] = copied.model_dump(mode='json')
        except ClipUnsupported as exc:
            if result.perception.video.timestamp_source is TimestampSource.CONTAINER:
                raise
            # Assumed-rate streams must refuse copying rather than invent timestamps.
            record['clip_unsupported'] = str(exc)
        record['clip_s'] = time.perf_counter() - began
        media.append(record)
    if file_sha256(clip.path) != before:
        raise AssertionError('source content changed')
    return {'clip_id': clip.metadata.id, 'status': 'passed', 'source_sha256': before,
            'perception_key': result.perception_key, 'analysis_key': result.analysis_key,
            'cache_hit': outcome.cache_hit, 'build_s': build_s, 'packages': len(packages),
            'overlays_checked': overlay_count, 'no_proposals': not bool(packages), 'media': media,
            'worker_analyzer_seconds': dict(outcome.analyzer_seconds),
            'parent_peak_rss_mb': peak_rss_mb(), 'wall_s': time.perf_counter() - started}


def main(argv=None):
    guard = ResourceGuard(policy='strict')
    try:
        guard.check(force=True)
        if sys.platform == "darwin" and memory_pressure_level() is None:
            raise ResourceExhausted("macOS memory pressure is unavailable; refusing real-video validation")
    except ResourceExhausted as exc:
        print(f'refused: {exc}', file=sys.stderr)
        return 3
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, help='source root; defaults to evaluation/datasets/<manifest stem>')
    parser.add_argument('--clips', nargs='+', help='exact clip ids or unambiguous id substrings')
    parser.add_argument('--models', type=Path, default=REPO / 'models/engine')
    parser.add_argument('--cache', type=Path, default=REPO / 'evaluation/caches/engine_v1/cache')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache-only', action='store_true')
    parser.add_argument('--timeout-s', type=float, default=300.0)
    args = parser.parse_args(argv)
    output = args.out.resolve()
    owned_output = False
    current_clip = None
    try:
        root = (args.root or args.manifest.resolve().parents[1] / 'datasets' / args.manifest.stem).resolve()
        if output.is_relative_to(REPO) or any(output.is_relative_to(p.resolve()) for p in (root, args.models, args.cache)):
            raise ValueError('--out must be outside the repository, sources, models and cache')
        if args.timeout_s <= 0:
            raise ValueError('--timeout-s must be positive')
        manifest = load_manifest(args.manifest)
        selected = _select(list(clips(manifest, root)), args.clips)
        guard.check(force=True)
        detector, segmenter = default_specs(args.models)
        output.mkdir(parents=True, exist_ok=False)
        owned_output = True
        for index, clip in enumerate(selected):
            current_clip = clip.metadata.id
            guard.check(force=True)
            job = WorkerJob(str(clip.path), str(args.cache), config={'interval_s': 0.1,
                            'assumed_fps': clip.metadata.assumed_fps}, detector=detector,
                            tracker=TrackerSpec(reference_rate=10.0), segmenter=replace(segmenter, interval_s=0.2),
                            lanes=True, events=True, policy='strict', cache_only=args.cache_only)
            record = _check(clip, job, output / f'{index:03d}', guard, args.timeout_s)
            with (output / f'{index:03d}.json').open('x') as f:
                json.dump(record, f, indent=2, allow_nan=False)
        return 0
    except Exception as exc:
        record = {'status': 'failed', 'clip_id': current_clip, 'error': f'{type(exc).__name__}: {exc}',
                  'parent_peak_rss_mb': peak_rss_mb()}
        if owned_output:
            with (output / 'failure.json').open('x') as f:
                json.dump(record, f, indent=2)
        print(record['error'], file=sys.stderr)
        return 3 if isinstance(exc, ResourceExhausted) else 1


if __name__ == '__main__':
    raise SystemExit(main())
