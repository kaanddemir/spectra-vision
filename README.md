# Spectra

Spectra is a lane-relative video risk analysis tool for forward-facing driving footage. It serves a FastAPI browser UI, analyzes uploaded videos frame by frame, and returns a schema v5 object-centric risk payload with timeline rows, saved events, preview imagery, and performance logs.

The pipeline combines YOLO road-participant detection, UFLDv2 lane geometry, Depth Anything V2 metric depth, OpenCV DIS optical flow, lightweight tracking, and TTC/risk fusion. It runs locally and does not call external narrative or LLM services.

## Features

- Browser-based video upload and analysis UI
- Live preview over WebSocket during analysis
- YOLOv8 road participant detection and IoU tracking
- UFLDv2 ego-lane detection with lane-relative object position
- Depth Anything V2 Metric VKITTI ONNX depth estimation
- OpenCV DIS dense optical flow with ego-motion compensation
- TTC fusion from metric depth, bbox expansion, and radial flow
- Brake-light and traffic-light visual advisories
- `SAFE`, `CAUTION`, and `DANGER` risk states
- Event snapshots, overlay imagery, per-object metrics, and performance logs

## Requirements

- Python 3.8+
- A local virtual environment at `.venv` when using `start.sh`
- Dependencies from `requirements.txt`
- Required local model files:
  - `models/depth_anything_v2_metric_vkitti_vits.onnx`
  - `models/ufld_v2_culane_r18.onnx`
  - `models/yolov8n.pt`

Optical flow is computed with OpenCV DIS, so there is no neural flow model to install.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the depth model:

```bash
.venv/bin/python scripts/download_depth_model.py
```

Prepare the lane model:

```bash
.venv/bin/python scripts/download_lanenet_model.py
```

Ensure the YOLO weights exist at:

```text
models/yolov8n.pt
```

If Ultralytics is installed, it can download `yolov8n.pt`; place or copy the resulting file into the `models/` directory under that exact name.

## Run

Using the project script:

```bash
./start.sh
```

The app is served at:

```text
http://localhost:8000
```

Stop the server:

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

Form fields:

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
spectra/
  app.py                  FastAPI app, routes, upload validation, serialization
  analysis/
    video.py              Frame loop, model scheduling, event collection
    risk.py               Risk scoring, TTC fusion, sensitivity bands
    tracking.py           IoU tracking and coasting
    overlay.py            Rendered analysis overlays
  vision/
    detection.py          YOLO wrapper
    lanenet.py            UFLDv2 ONNX wrapper
    depth.py              Depth map and nearness utilities
    models.py             ONNX Runtime depth/provider setup
    motion.py             DIS optical flow and ego-motion compensation
    road.py               Lane geometry and road relevance filtering
    brake_lights.py       Brake-light visual cue
    traffic_light.py      Traffic-light colour cue
    preprocessing.py      Resize and colour-space preparation
  web/static/             Browser UI assets
tests/                    API, backend, pipeline, and UI smoke tests
scripts/                  Model setup helpers
```

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

- Depth and UFLDv2 run through ONNX Runtime. On macOS, the provider setup prefers CoreML when available and falls back to CPU.
- YOLO runs through Ultralytics/PyTorch and prefers the best available local device.
- Missing or unloadable required models are hard backend failures before analysis starts.
- Traffic-light detections are advisory only; they are not tracked as collision participants.
