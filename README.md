# Spectra

<div align="center">
  <img src="spectra/web/static/assets/logo.png" alt="Spectra Logo" width="120" />
</div>

<div align="center">

**A video risk analysis dashboard for forward-facing driving footage.**

</div>

---

> **Status:** **Prototype Phase**. Model behavior, risk scoring, and UI details may change as the pipeline is refined.

## Overview

Spectra analyzes an uploaded driving video frame by frame and returns one object-centric risk payload: a `SAFE` / `CAUTION` / `DANGER` timeline, saved events with overlay imagery, per-object metrics, and performance logs. Detection, lane geometry, metric depth, and optical flow are fused into a lane-relative risk score for each tracked road participant. Everything runs on your machine — there is no database, no external AI service, and the uploaded video is deleted as soon as the analysis returns.

Built with FastAPI, vanilla browser UI assets, OpenCV, ONNX Runtime, and three local models: YOLOv8 for detection, UFLDv2 for lanes, and Depth Anything V2 for metric depth.

## Core Features

- **Browser Upload Flow**: Drop in forward-facing footage and get the whole analysis back in one request.
- **Live Preview Stream**: Watch analysis previews over a WebSocket while processing runs.
- **Risk Timeline**: `SAFE`, `CAUTION`, and `DANGER` states over time, with the peak event called out.
- **Saved Events**: Overlay snapshots and the metrics behind each event, deduplicated by frame.
- **Lane-Relative Reasoning**: UFLDv2 lane geometry places every participant relative to the ego corridor rather than to the frame.
- **TTC Fusion**: IoU tracking combined with metric depth, bbox expansion, and radial optical flow, with ego-motion compensated out.
- **Visual Advisories**: Brake-light and traffic-light cues, kept advisory rather than treated as collision participants.
- **Tunable Cadence**: Depth, detection, lane, and flow each run on their own frame interval, so speed and fidelity are traded off per run.
- **Provider Fallback**: ONNX Runtime prefers CoreML on macOS and falls back to CPU; compiled graphs are cached across restarts.

## Current Limitations

- **Not a Driver Assistance System**: An experimental tool for recorded footage. It is not designed for real-time warning, is not safe to operate while driving, and its output should not be relied on for driving, accident, or insurance decisions.
- **Prototype Risk Model**: Risk fusion is heuristic and still evolving.
- **All Three Models Required**: A missing or unloadable model is a hard failure before analysis starts — there is no weaker fallback, because a misleading one is worse than a clear stop.
- **Video-Only Input**: Uploaded video files only, up to 500 MB, in `mp4`, `mov`, `avi`, `mkv`, or `m4v`.
- **Hardware-Dependent Speed**: Runtime depends heavily on local CPU/GPU/CoreML support.
- **Single-User, Loopback Only**: The server binds `127.0.0.1` and holds analysis state in process memory. Do not expose it on a network.

## Tech Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/)
- **Computer Vision**: [OpenCV](https://opencv.org/) (detection filtering, DIS optical flow, overlays)
- **Inference**: [ONNX Runtime](https://onnxruntime.ai/), [PyTorch](https://pytorch.org/), [Ultralytics YOLO](https://docs.ultralytics.com/)
- **Models**: [YOLOv8n](https://docs.ultralytics.com/), [UFLDv2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) CULane R18, [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) Metric VKITTI Small
- **Frontend**: Vanilla HTML, CSS, and JavaScript modules
- **Testing**: [pytest](https://docs.pytest.org/)

## Getting Started

Requires Python 3.10+ and about 1 GB of model downloads, fetched once by the setup scripts.

**1. Install.**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**2. Fetch the models.** They live in `models/`, are ignored by git, and each script exits early if its file is already there.

```bash
python scripts/download_depth_model.py     # Depth Anything V2 Metric VKITTI → ONNX, ~95 MB
python scripts/download_lanenet_model.py   # UFLDv2 CULane ResNet-18 → ONNX, ~790 MB
python scripts/download_yolo_model.py      # YOLOv8n weights, ~6 MB
```

Optical flow is classical (OpenCV DIS), so there is no fourth model to install.

### Run

```bash
python run.py
```

It checks the setup before starting and prints where to open the app:

```text
  Spectra
  ────────────────────────────────────────────
  Open      http://localhost:8000
  Models    depth, lane, detector ready
  ONNX      CoreML, CPU fallback
  Stop      Ctrl+C
```

Ctrl+C stops it. Missing setup — a dependency, a model file, a port already taken — is reported before anything starts, with the command that fixes it. `.venv` does not have to be activated first: when it exists, `run.py` switches to that interpreter itself and says so.

### Configuration

Every setting is an environment variable and every one is optional. `PORT` (8000); `RELOAD` (`1` restarts on file changes, off by default because a reload discards the loaded models); `LOG_LEVEL` (`DEBUG` restores Uvicorn's startup notes and per-request access log). The model paths are `SPECTRA_DEPTH_MODEL` and `SPECTRA_LANENET_MODEL`, both read by `run.py` too, so a relocated model is reported as present rather than missing. The rest tune inference and must match the exported graphs: `SPECTRA_DEPTH_INPUT` (518), `SPECTRA_DEPTH_MAX_M` (80.0), `SPECTRA_LANENET_INPUT_H` (320), `SPECTRA_LANENET_INPUT_W` (1600), `SPECTRA_LANENET_CROP_RATIO` (0.6), and `SPECTRA_COREML_CACHE_DIR` (`models/.coreml_cache`).

## Project Structure

```text
Spectra/
├── models/               # Local model weights, ignored by git
├── scripts/              # Model setup helpers (depth, lane, YOLO)
├── spectra/
│   ├── app.py            # FastAPI app, routes, upload validation, serialization
│   ├── analysis/         # video (frame loop) → tracking → risk → overlay
│   ├── vision/           # detection, lanenet, depth, models (ONNX providers),
│   │                     # motion, road, brake_lights, traffic_light
│   └── web/static/       # Browser UI assets and logo
├── tests/                # API, backend, pipeline, and UI smoke tests
├── run.py                # Entrypoint: python run.py, Ctrl+C stops it
├── requirements.txt
└── LICENSE
```

## Data Storage

There is no database. The upload is written to a temporary directory that is removed as soon as the analysis returns, so nothing of the video outlives the request. Overlay and original frames come back inline: they are collected once under `payload.images` and referenced by events as `image_ref`. Model weights under `models/` are the only files the project keeps, and git ignores them as runtime artifacts.

## API

- `GET /` — analysis UI
- `GET /how-it-works` — algorithm explainer page
- `GET /api/health` — health check
- `POST /api/analyze` — `multipart/form-data`, field `file`: a video up to 500 MB
- `WS /ws/preview/{session_id}` — live preview stream for one analysis session

`POST /api/analyze` takes `mode` (`video` only) plus optional tuning fields. Out-of-range values are snapped to the nearest allowed one rather than rejected:

| Field | Default | Notes |
| --- | --- | --- |
| `resize_max_side` | `512` | One of `128`, `256`, `384`, `512`, `768`, `1024`. |
| `depth_every` | `10` | One of `1`, `2`, `3`, `5`, `10`, `15`. |
| `detect_every` / `lane_every` | `3` | One of `1`, `2`, `3`, `5`, `10`. |
| `flow_every` | `1` | One of `1`, `2`, `3`, `5`, `10`. |
| `adaptive_depth` | `1` | Truthy values enable motion-triggered depth refresh. |
| `sensitivity` | `balanced` | `conservative`, `balanced`, or `aggressive`; anything else falls back to `balanced`. |
| `max_saved_events` | `25` | Clamped to `1..50`. |
| `max_processed_frames` | unlimited | Minimum `1`. |
| `start_sec` / `end_sec` | `0.0` | Time window; `end_sec <= 0` means no end limit. |
| `start_frame` / `end_frame` | `0` | Frame window; `end_frame <= 0` means no end limit. |
| `session_id` | empty | When set, previews are pushed to `/ws/preview/{session_id}`. |

The response is one payload:

```json
{
  "payload": {
    "schema_version": 7,
    "metadata": {}, "frames": [], "peak_event": null,
    "events": [], "images": {}, "performance": {}
  }
}
```

## Testing

```bash
pytest                                                  # full suite
pytest tests/test_api_contract.py tests/test_static_ui_smoke.py   # API and UI contracts
```

## Runtime Notes

- Depth and UFLDv2 run through ONNX Runtime, which prefers CoreML on macOS and falls back to CPU; YOLO runs through Ultralytics/PyTorch and picks the best available local device.
- Every required model is loaded eagerly at the start of an analysis, so a corrupt file or a runtime mismatch surfaces there rather than mid-video.
- Hough-based lane detection was removed after benchmarking: it traced the road's outer edges instead of the ego corridor, and a misleading fallback is worse than a clear failure.
- Traffic-light detections are advisory only and are never tracked as collision participants.

## License

This project's own code is licensed under the [MIT License](LICENSE). No model weights are redistributed here; the setup scripts fetch them from their original sources, and the dependencies carry their own terms:

- **[Ultralytics](https://github.com/ultralytics/ultralytics)** (YOLOv8 and `yolov8n.pt`) is **AGPL-3.0**. Local use is unaffected, but distributing a derived work or offering it as a network service brings AGPL obligations; Ultralytics sells a commercial license for that case.
- **[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)** is Apache-2.0, as is the Metric VKITTI Small checkpoint used here. The larger V2 checkpoints are CC BY-NC-4.0, so swapping one in changes the terms.
- **[Ultra-Fast-Lane-Detection-v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)** is MIT; its CULane checkpoint follows the CULane dataset's research-use terms.

## Footer

<div align="center">
  <p>Built by <a href="https://heykaan.dev">heykaan.dev</a></p>
</div>
