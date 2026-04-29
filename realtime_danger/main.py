"""Command-line runner for the real-time danger pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .analyst import GeminiDangerAnalyst
from .annotator import annotate_frame
from .depth_estimator import DepthResult, estimate_frame_depth
from .fusion import fuse_frame_risk
from .optical_flow import compute_velocity
from .preprocess import preprocess_frame
from .vehicle_detector import Detection, VehicleDetector
from .video_loader import VideoLoader
from .video_writer import JsonlEventWriter, VideoWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time danger detection on a video.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output", default="realtime_output/annotated.mp4", help="Annotated output video path.")
    parser.add_argument("--events", default="realtime_output/events.jsonl", help="JSONL event log path.")
    parser.add_argument("--max-side", type=int, default=720, help="Resize longest frame side before analysis.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame processing limit.")
    parser.add_argument("--depth-every", type=int, default=3, help="Run depth estimation every N frames.")
    parser.add_argument("--detect-every", type=int, default=3, help="Run YOLO detection every N frames.")
    parser.add_argument("--yolo-model", default="yolov8s.pt", help="YOLO model name/path.")
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO and use zone-level risk only.")
    parser.add_argument("--llm", action="store_true", help="Enable Gemini calls for DANGER events.")
    parser.add_argument("--gemini-api-key", default=os.getenv("GEMINI_API_KEY", ""), help="Gemini API key.")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash", help="Gemini model name.")
    parser.add_argument("--llm-cooldown", type=float, default=3.0, help="Minimum seconds between Gemini calls.")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, int | str]:
    loader = VideoLoader(args.input, max_frames=args.max_frames)
    detector = VehicleDetector(model_name=args.yolo_model, enabled=not args.no_yolo)
    analyst = GeminiDangerAnalyst(
        api_key=args.gemini_api_key,
        model=args.gemini_model,
        enabled=args.llm,
        cooldown_sec=args.llm_cooldown,
    )
    writer: VideoWriter | None = None
    event_writer = JsonlEventWriter(args.events)

    previous_gray = None
    last_depth: DepthResult | None = None
    last_detections: list[Detection] = []
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

            if last_depth is None or video_frame.frame_index % max(args.depth_every, 1) == 0:
                last_depth = estimate_frame_depth(frame)

            if video_frame.frame_index % max(args.detect_every, 1) == 0:
                last_detections = detector.detect(frame.bgr)

            primary_event, all_events = fuse_frame_risk(
                frame_index=video_frame.frame_index,
                timestamp_sec=video_frame.timestamp_sec,
                depth=last_depth,
                flow=flow,
                detections=last_detections,
            )

            analyst.submit_if_needed(primary_event)
            primary_event.llm_message = analyst.last_message
            annotated = annotate_frame(frame.bgr, primary_event, last_detections, analyst.last_message)
            writer.write(annotated)

            if primary_event.state == "DANGER":
                danger_count += 1
            elif primary_event.state == "CAUTION":
                caution_count += 1

            event_writer.write(
                {
                    "primary": primary_event.to_dict(),
                    "regions": [event.to_dict() for event in all_events],
                    "llm_error": analyst.last_error,
                }
            )
            processed += 1
    finally:
        if writer is not None:
            writer.close()
        event_writer.close()
        analyst.close()

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

