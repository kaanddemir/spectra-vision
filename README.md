# Spectra

<div align="center">
  <img src="spectra/web/static/assets/logo.png" alt="Spectra Logo" width="120" />
</div>

<div align="center">

**A video risk analysis dashboard for forward-facing driving footage.**

</div>

---

> **Status:** **Prototype Phase**. This project is currently in active development. Model behavior, risk scoring, and UI details may change as the pipeline is refined.

## Overview

Spectra is a lane-relative driving risk analysis tool built with FastAPI, vanilla browser UI assets, OpenCV, ONNX Runtime, Ultralytics YOLO, and computer-vision models.

The app analyzes uploaded driving videos frame by frame and returns an object-centric risk payload with timeline rows, saved events, preview imagery, per-object metrics, and performance logs. It does not call external narrative or LLM services.

## Core Features

### Video Risk Dashboard

- **Browser Upload Flow**: Upload forward-facing driving footage from the browser UI.
- **Live Preview Stream**: Watch analysis previews over WebSocket while processing runs.
- **Risk Timeline**: Review `SAFE`, `CAUTION`, and `DANGER` risk states over time.
- **Saved Events**: Inspect peak events, overlay snapshots, and supporting metrics.

### Vision Pipeline

- **Road Participant Detection**: Uses YOLOv8 for vehicles and road-relevant participants.
- **Lane Geometry**: Uses UFLDv2 ONNX lane detection for ego-lane-relative positioning.
- **Metric Depth**: Uses Depth Anything V2 Metric VKITTI ONNX depth estimation.
- **Motion Cues**: Uses OpenCV DIS dense optical flow with ego-motion compensation.
- **Tracking & TTC Fusion**: Combines IoU tracking, metric depth, bbox expansion, and radial flow.
- **Visual Advisories**: Detects brake-light and traffic-light cues as advisory signals.

### Runtime Model

- **On-Device Processing**: Video analysis and model inference run on the user's machine.
- **No Backend Database**: The app does not require persistent server-side storage.
- **No External AI Calls**: Narrative generation and LLM services are not part of the runtime path.
- **Provider Fallbacks**: ONNX Runtime can prefer CoreML on macOS and fall back to CPU.

## Current Limitations

- **Prototype Risk Model**: Risk fusion is heuristic and still evolving.
- **Local Model Files Required**: Missing or unloadable model files are hard backend failures.
- **Video-Only Input**: The analysis endpoint currently supports uploaded video files only.
- **Traffic-Light Advisory Scope**: Traffic-light detections are advisory and are not tracked as collision participants.
- **Hardware-Dependent Speed**: Runtime performance depends heavily on local CPU/GPU/CoreML support.

## Tech Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **Server**: [Uvicorn](https://www.uvicorn.org/)
- **Computer Vision**: [OpenCV](https://opencv.org/)
- **Inference**: [ONNX Runtime](https://onnxruntime.ai/), [PyTorch](https://pytorch.org/), [Ultralytics YOLO](https://docs.ultralytics.com/)
- **Frontend**: Vanilla HTML, CSS, and JavaScript modules
- **Testing**: [pytest](https://docs.pytest.org/)

## Getting Started

### Requirements

- Python 3.8+
- Dependencies from `requirements.txt`
- A local virtual environment at `.venv` when using `start.sh`
- Required local model files:
  - `models/depth_anything_v2_metric_vkitti_vits.onnx`
  - `models/ufld_v2_culane_r18.onnx`
  - `models/yolov8n.pt`

Optical flow is computed with OpenCV DIS, so there is no neural flow model to install.

### Setup

1. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the depth model**
   ```bash
   .venv/bin/python scripts/download_depth_model.py
   ```

4. **Prepare the lane model**
   ```bash
   .venv/bin/python scripts/download_lanenet_model.py
   ```

5. **Ensure YOLO weights exist**
   ```text
   models/yolov8n.pt
   ```

   If Ultralytics is installed, it can download `yolov8n.pt`; place or copy the resulting file into the `models/` directory under that exact name.

### Run

1. **Start with the project script**
   ```bash
   ./start.sh
   ```

2. **Open the app**
   ```text
   http://localhost:8000
   ```

3. **Stop the server**
   ```bash
   ./stop.sh
   ```

Manual run:

```bash
python -m uvicorn spectra.app:app --host localhost --port 8000 --reload
```

## Routes

- `GET /` - main analysis UI
- `GET /how-it-works` - algorithm explainer page
- `GET /api/health` - health check
- `POST /api/analyze` - video analysis endpoint
- `WS /ws/preview/{session_id}` - live preview stream for an analysis session

## Analyze API

`POST /api/analyze` accepts `multipart/form-data`.

| Field | Default | Notes |
| --- | --- | --- |
| `file` | required | Video upload. Supported extensions: `mp4`, `mov`, `avi`, `mkv`, `m4v`. Max upload size is 500 MB. |
| `mode` | `video` | Only `video` is supported. |
| `max_processed_frames` | large internal default | Minimum `1`. Limits processed frames. |
| `max_saved_events` | `20` | Clamped to `1..50`. |
| `resize_max_side` | `512` | Snapped to one of `128`, `256`, `384`, `512`, `768`, `1024`. |
| `depth_every` | `10` | Snapped to one of `1`, `2`, `3`, `5`, `10`, `15`. |
| `adaptive_depth` | `1` | Truthy values enable motion-triggered depth refresh. |
| `detect_every` | `3` | Snapped to one of `1`, `2`, `3`, `5`, `10`. |
| `lane_every` | `3` | Snapped to one of `1`, `2`, `3`, `5`, `10`. |
| `flow_every` | `1` | Snapped to one of `1`, `2`, `3`, `5`, `10`. |
| `sensitivity` | `balanced` | One of `conservative`, `balanced`, `aggressive`; invalid values fall back to `balanced`. |
| `start_sec` | `0.0` | Start time window in seconds. |
| `end_sec` | `0.0` | Values `<= 0` mean no time end limit. |
| `start_frame` | `0` | Start frame index. |
| `end_frame` | `0` | Values `<= 0` mean no frame end limit. |
| `session_id` | empty | When provided, preview messages are sent to `/ws/preview/{session_id}`. |

The response shape is:

```json
{
  "payload": {
    "schema_version": 7,
    "metadata": {},
    "frames": [],
    "peakEvent": null,
    "events": [],
    "images": {},
    "performance": {}
  }
}
```

Images are stored once under top-level `payload.images`. Events reference them with `image_ref`.

## Project Structure

```text
Spectra/
├── models/               # Local model weights, ignored by git
├── scripts/              # Model setup helpers
├── spectra/
│   ├── app.py            # FastAPI app, routes, upload validation, serialization
│   ├── analysis/
│   │   ├── video.py      # Frame loop, model scheduling, event collection
│   │   ├── risk.py       # Risk scoring, TTC fusion, sensitivity bands
│   │   ├── tracking.py   # IoU tracking and coasting
│   │   └── overlay.py    # Rendered analysis overlays
│   ├── vision/
│   │   ├── detection.py  # YOLO wrapper
│   │   ├── lanenet.py    # UFLDv2 ONNX wrapper
│   │   ├── depth.py      # Depth map and nearness utilities
│   │   ├── models.py     # ONNX Runtime depth/provider setup
│   │   ├── motion.py     # DIS optical flow and ego-motion compensation
│   │   ├── road.py       # Lane geometry and road filtering
│   │   ├── brake_lights.py
│   │   └── traffic_light.py
│   └── web/static/       # Browser UI assets and logo
├── tests/                # API, backend, pipeline, and UI smoke tests
├── requirements.txt
├── start.sh
├── stop.sh
└── README.md
```

## Data & Model Storage

Spectra is local-first:

- **Uploaded Videos**: Processed locally by the FastAPI service.
- **Model Files**: Stored under `models/` and ignored by git because they are large runtime artifacts.
- **Runtime Images**: Returned in the API payload under `payload.images`; events reference them by `image_ref`.
- **No Backend Database**: The project does not include database-backed persistence.

## Basic Workflow

1. Prepare the virtual environment and install dependencies.
2. Download or place required model files under `models/`.
3. Start the app with `./start.sh`.
4. Upload a forward-facing driving video in the browser UI.
5. Review risk states, event snapshots, timeline rows, and performance logs.
6. Run tests before changing API contracts, risk logic, or UI behavior.

## Testing

Run the full test suite:

```bash
pytest
```

Useful focused checks while changing docs or API/UI contracts:

```bash
pytest tests/test_api_contract.py tests/test_static_ui_smoke.py
```

## Runtime Notes

- Depth and UFLDv2 run through ONNX Runtime. On macOS, provider setup prefers CoreML when available and falls back to CPU.
- YOLO runs through Ultralytics/PyTorch and prefers the best available local device.
- Missing or unloadable required models are hard backend failures before analysis starts.
- Traffic-light detections are advisory only; they are not tracked as collision participants.

## Footer

<div align="center">
  <p>Built by <a href="https://heykaan.dev">heykaan.dev</a></p>
</div>
