"""Command-line runner for the zone-based risk pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .fusion import fuse_frame_risk
from ..vision.depth_estimator import DepthResult, estimate_frame_depth
from ..vision.optical_flow import compute_velocity
from ..vision.preprocess import preprocess_frame
from .annotator import annotate_frame
from .video_loader import VideoLoader
from .video_writer import JsonlEventWriter, VideoWriter


_DEPTH_BLEND_ALPHA = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zone-based risk analysis on a video.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output", default="zone_output/annotated.mp4", help="Annotated output video path.")
    parser.add_argument("--events", default="zone_output/events.jsonl", help="JSONL event log path.")
    parser.add_argument("--max-side", type=int, default=720, help="Resize longest frame side before analysis.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame processing limit.")
    parser.add_argument("--depth-every", type=int, default=3, help="Run depth estimation every N frames.")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, int | str]:
    loader = VideoLoader(args.input, max_frames=args.max_frames)
    writer: VideoWriter | None = None
    event_writer = JsonlEventWriter(args.events)

    previous_gray = None
    last_depth: DepthResult | None = None
    processed = 0
    danger_count = 0
    caution_count = 0

    try:
        for video_frame in loader.frames():
            frame = preprocess_frame(video_frame.bgr, max_side=args.max_side)

            if writer is None:
                height, width = frame.bgr.shape[:2]
                writer = VideoWriter(args.output, fps=loader.fps, frame_size=(width, height))

            flow = compute_velocity(previous_gray, frame.gray)
            previous_gray = frame.gray

            if last_depth is None:
                last_depth = estimate_frame_depth(frame)
            elif video_frame.frame_index % max(args.depth_every, 1) == 0:
                new_depth = estimate_frame_depth(frame)
                blended_near = (
                    _DEPTH_BLEND_ALPHA * new_depth.near_map
                    + (1.0 - _DEPTH_BLEND_ALPHA) * last_depth.near_map
                ).astype(np.float32)
                blended_uint8 = np.clip(blended_near * 255.0, 0, 255).astype(np.uint8)
                last_depth = DepthResult(depth_map=blended_uint8, near_map=blended_near)

            primary_event, all_events = fuse_frame_risk(
                frame_index=video_frame.frame_index,
                timestamp_sec=video_frame.timestamp_sec,
                depth=last_depth,
                flow=flow,
            )

            annotated = annotate_frame(frame.bgr, primary_event, all_events)
            writer.write(annotated)

            if primary_event.state == "DANGER":
                danger_count += 1
            elif primary_event.state == "CAUTION":
                caution_count += 1

            event_writer.write(
                {
                    "primary": primary_event.to_dict(),
                    "regions": [event.to_dict() for event in all_events],
                }
            )
            processed += 1
    finally:
        if writer is not None:
            writer.close()
        event_writer.close()

    return {
        "processed_frames": processed,
        "danger_frames": danger_count,
        "caution_frames": caution_count,
        "output": str(Path(args.output)),
        "events": str(Path(args.events)),
    }


def main() -> None:
    summary = run(parse_args())
    print(summary)


if __name__ == "__main__":
    main()
