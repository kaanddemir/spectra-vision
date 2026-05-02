# Spectra

Spectra is a lane-relative video risk analysis tool. It processes uploaded driving-view videos, detects road geometry and YOLO traffic participants, fuses visual nearness with optical flow and track expansion, then reports risk state, TTC, and timeline events.

The current pipeline is object-centric and does not use external narrative services.

## Features

- Video upload and browser-based analysis UI
- YOLO-based road participant tracking
- Road/lane-relative risk scoring
- Required Depth Anything V2 ONNX depth estimation
- Required NeuFlow ONNX dense optical flow with ego-motion compensation
- Fused TTC from bbox expansion, radial flow, and depth delta
- `SAFE`, `CAUTION`, and `DANGER` states
- Timeline rows, lane metrics, event snapshots, depth view, motion view, and overlay view

## Requirements

- Python 3.8+
- A local virtual environment at `.venv` if using `start.sh`
- Dependencies listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Using the project script:

```bash
./start.sh
```

The app starts at:

```text
http://localhost:8000
```

Stop the server:

```bash
./stop.sh
```

Manual run:

```bash
python -m uvicorn zone_risk.app:app --host localhost --port 8000 --reload
```

## Project Structure

```text
zone_risk/
  app.py
  web/
    static/
      index.html
      app.js
      style.css
  pipeline/
    api.py
    annotator.py
    fusion.py
    risk_calculator.py
    tracker.py
    video_loader.py
  vision/
    depth_estimator.py
    image_preprocess.py
    object_detector.py
    optical_flow.py
    preprocess.py
    road_geometry.py
    road_roi.py
```

## Pipeline

1. `zone_risk/app.py`
   - FastAPI entry point
   - Serves the UI
   - Accepts video uploads through `/api/analyze`
   - Serializes analysis results for the frontend

2. `zone_risk/pipeline/api.py`
   - Web-facing analysis adapter
   - Runs the frame loop
   - Builds timeline rows, event payloads, and image outputs

3. `zone_risk/pipeline/video_loader.py`
   - Reads video frames with OpenCV
   - Provides frame index and timestamp metadata

4. `zone_risk/vision/preprocess.py`
   - Resizes frames
   - Converts BGR frames to RGB and grayscale
   - Calls image enhancement helpers

5. `zone_risk/vision/image_preprocess.py`
   - Applies denoising and CLAHE enhancement
   - Produces enhanced grayscale and denoised RGB images

6. `zone_risk/vision/optical_flow.py`
   - Uses required NeuFlow ONNX dense optical flow with ego-motion compensation
   - Produces motion magnitude, normalized motion, divergence, and RGB flow visualization

7. `zone_risk/vision/depth_estimator.py`
   - Coordinates per-frame nearness estimation
   - Uses required Depth Anything V2 ONNX for per-frame nearness maps

8. `zone_risk/pipeline/fusion.py`
    - Bundles depth, flow, lane geometry, and track history into spatial fields
    - Builds per-object risk events and selects the primary event

9. `zone_risk/pipeline/risk_calculator.py`
    - Calculates lane position, crossing risk, fused TTC, confidence, and risk state
    - Selects the primary risk event

10. `zone_risk/pipeline/annotator.py`
    - Draws lane corridor, object boxes, TTC components, and summary text onto frames

## API

Health check:

```text
GET /api/health
```

Analyze video:

```text
POST /api/analyze
```

Form fields:

- `file`: video file, required
- `mode`: must be `video`
- `max_processed_frames`: maximum number of frames to process
- `max_saved_events`: number of top events to keep
- `resize_max_side`: max frame side before processing
- `depth_every`: depth sampling interval

Supported video extensions:

```text
mp4, mov, avi, mkv, m4v
```

## Development Checks

Compile Python files:

```bash
python -m compileall zone_risk
```

Quick import check:

```bash
python -c "from zone_risk.app import app; print(app.title)"
```
