"""Command-line entry point for the analysis engine.

    python -m spectra.engine probe VIDEO [--assumed-fps 20]
    python -m spectra.engine analyze VIDEO --cache DIR [--interval 0.1] [--start 0] [--end 60] [--max-width 640]
        [--detector MODEL.onnx --detector-sha256 HEX [--track]]
        [--segmenter MODEL.onnx --segmenter-sha256 HEX [--segment-interval 0.2] [--lanes]] [--events]
        [--strict-memory]
    python -m spectra.engine frame VIDEO --time 31.5 --out frame.jpg

Without models, ``analyze`` runs only decoding, sampling, timestamps and caching: the result records
which frames were analysed and when, and nothing about their content. With ``--detector`` (and tracking
with ``--track``) and/or ``--segmenter`` (and the ego lane with ``--lanes``) the run happens in a separate
worker process; models load only from the given files, whose sha256 must match. ``--events`` adds the
apparent-approach and camera-vehicle impact analysers (needs ``--track``) and, with ``--lanes``, the cut-in
analyser; they run on the cached perception, so changing them never repeats perception. Every analyser
reports what it did, including when it was unavailable or failed. Ctrl+C cancels cleanly. Progress goes to stderr, the outcome as JSON to stdout.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time

from spectra.engine.cache import PerceptionCache
from spectra.engine.contracts import RunConfig
from spectra.engine.resources import ResourceGuard
from spectra.engine.runner import AnalysisRun, CancelToken, Progress, RunState
from spectra.engine.video import VideoError, VideoSource
from spectra.engine.worker import DetectorSpec, SegmenterSpec, TrackerSpec, WorkerJob, run_in_worker


def _progress_printer():
    last = [0.0]

    def show(p: Progress) -> None:
        now = time.monotonic()
        if p.state is RunState.RUNNING and p.stage == "analysing frames" and now - last[0] < 1.0:
            return
        last[0] = now
        pct = f"{p.fraction * 100:5.1f}%" if p.fraction is not None else "   ? "
        t = f"t={p.t:8.3f}s" if p.t is not None else ""
        print(f"[{p.state.value:9}] {pct} frames={p.frames_analysed:6d} {t} {p.stage}", file=sys.stderr, flush=True)

    return show


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m spectra.engine", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "analyze", "frame"):
        s = sub.add_parser(name)
        s.add_argument("video")
        s.add_argument("--assumed-fps", type=float, help="only for streams without timestamps")
        if name == "analyze":
            s.add_argument("--cache", required=True)
            s.add_argument("--models", help="directory with a verified models.json runtime registry")
            s.add_argument("--interval", type=float, help="seconds between analysed frames (default: every frame)")
            s.add_argument("--start", type=float, default=0.0)
            s.add_argument("--end", type=float)
            s.add_argument("--max-width", type=int)
            s.add_argument("--detector", help="local D-FINE ONNX file")
            s.add_argument("--detector-sha256")
            s.add_argument("--track", action="store_true", help="track detections with BoT-SORT")
            s.add_argument("--segmenter", help="local comma10k-segnet ONNX file")
            s.add_argument("--segmenter-sha256")
            s.add_argument("--segment-interval", type=float, default=0.2, help="seconds between segmentations")
            s.add_argument("--lanes", action="store_true", help="estimate the ego lane from the segmentation")
            s.add_argument("--events", action="store_true", help="propose approach, cut-in and impact moments for review")
            s.add_argument("--strict-memory", action="store_true", help="stop above normal memory pressure")
        if name == "frame":
            s.add_argument("--time", type=float, required=True)
            s.add_argument("--out", required=True)
    a = p.parse_args(argv)

    if a.cmd == "probe":
        try:
            print(VideoSource(a.video, assumed_fps=a.assumed_fps).info.model_dump_json(indent=2))
        except VideoError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return 0

    if a.cmd == "frame":
        import cv2
        try:
            frame = VideoSource(a.video, assumed_fps=a.assumed_fps).frame_at(a.time)
        except (VideoError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        cv2.imwrite(a.out, frame.image)
        print(json.dumps({"requested_t": a.time, "frame_t": frame.t, "pts": frame.pts, "out": a.out}))
        return 0

    config = RunConfig(interval_s=a.interval, start_s=a.start, end_s=a.end, max_width=a.max_width,
                       assumed_fps=a.assumed_fps)
    cancel = CancelToken()
    signal.signal(signal.SIGINT, lambda *_: cancel.cancel())
    started = time.perf_counter()
    policy = "strict" if a.strict_memory else "standard"
    registered_detector = None
    if a.models:
        from spectra.engine.models import default_specs
        try:
            detector, segmenter = default_specs(a.models)
        except (OSError, ValueError) as exc:
            p.error(str(exc))
        if not a.detector:
            registered_detector = detector
            a.detector, a.detector_sha256 = detector.model_path, detector.sha256
        if not a.segmenter:
            a.segmenter, a.segmenter_sha256 = segmenter.model_path, segmenter.sha256
    if a.track and not a.detector:
        p.error("--track needs --detector")
    if a.lanes and not a.segmenter:
        p.error("--lanes needs --segmenter")
    if a.events and not a.track:
        p.error("--events needs --track")
    if a.detector and not a.detector_sha256:
        p.error("--detector needs --detector-sha256")
    if a.segmenter and not a.segmenter_sha256:
        p.error("--segmenter needs --segmenter-sha256")
    if a.detector or a.segmenter:
        job = WorkerJob(video=a.video, cache_root=a.cache, config=config.model_dump(exclude_defaults=True),
                        detector=(registered_detector or DetectorSpec(a.detector, a.detector_sha256))
                        if a.detector else None,
                        tracker=TrackerSpec(reference_rate=1 / a.interval if a.interval else 10.0) if a.track
                        else None,
                        segmenter=SegmenterSpec(a.segmenter, a.segmenter_sha256, interval_s=a.segment_interval)
                        if a.segmenter else None, lanes=a.lanes, events=a.events, policy=policy)
        outcome = run_in_worker(job, on_progress=_progress_printer(), cancel=cancel)
    else:
        outcome = AnalysisRun(a.video, config, cache=PerceptionCache(a.cache), cancel=cancel,
                              on_progress=_progress_printer(), guard=ResourceGuard(policy=policy)).run()
    summary = {"state": outcome.state.value, "error": outcome.error, "cache_hit": outcome.cache_hit,
               "notes": list(outcome.notes), "wall_s": round(time.perf_counter() - started, 3),
               "analyzer_seconds": dict(outcome.analyzer_seconds)}
    if outcome.result is not None:
        perception = outcome.result.perception
        summary.update(perception_key=outcome.result.perception_key,
                       timing=perception.timing.model_dump(mode="json"),
                       warnings=[w.model_dump(mode="json") for w in perception.warnings])
        if outcome.result.reports:
            summary.update(analysis_key=outcome.result.analysis_key,
                           reports=[r.model_dump(mode="json") for r in outcome.result.reports],
                           events=[e.model_dump(mode="json") for e in outcome.result.events])
    print(json.dumps(summary, indent=2))
    return 0 if outcome.state is RunState.COMPLETED else 1


if __name__ == "__main__":
    sys.exit(main())
