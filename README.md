# Spectra

Spectra is a zone-based video risk analysis tool. It processes uploaded driving-view videos, splits each frame into left, center, and right zones, estimates visual nearness and optical flow, then reports risk state, TTC, and timeline events.

The current pipeline is zone-only and does not use object detection or external narrative services.

## Features

- Video upload and browser-based analysis UI
- Left / center / right zone risk scoring
- Classical monocular depth cues from texture, edges, vertical position, and atmospheric contrast
- Dense optical flow for motion and closing-speed estimation
- TTC-based `SAFE`, `CAUTION`, and `DANGER` states
- Timeline rows, zone metrics, event snapshots, depth view, motion view, and overlay view

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
    main.py
    risk_calculator.py
    video_loader.py
    video_writer.py
  vision/
    depth_cues.py
    depth_estimator.py
    edge_detector.py
    image_preprocess.py
    optical_flow.py
    preprocess.py
    texture_analyzer.py
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
   - Computes dense Farneback optical flow
   - Produces motion magnitude, normalized motion, divergence, and RGB flow visualization

7. `zone_risk/vision/depth_estimator.py`
   - Coordinates per-frame nearness estimation
   - Uses edge, texture, and depth cue modules

8. `zone_risk/vision/depth_cues.py`
   - Fuses classical monocular cues into a normalized depth map

9. `zone_risk/vision/edge_detector.py`
   - Computes Sobel, Canny, and LoG edge cues

10. `zone_risk/vision/texture_analyzer.py`
    - Computes Gabor texture responses and local texture energy

11. `zone_risk/pipeline/fusion.py`
    - Splits the frame into left, center, and right zones
    - Sends each zone to risk scoring

12. `zone_risk/pipeline/risk_calculator.py`
    - Calculates zone, direction, near score, closing speed, pseudo-TTC, confidence, and risk state
    - Selects the primary risk event

13. `zone_risk/pipeline/annotator.py`
    - Draws zone risk boxes and summary text onto frames

14. `zone_risk/pipeline/video_writer.py`
    - Used by the optional CLI path for annotated video and JSONL event output

15. `zone_risk/pipeline/main.py`
    - Optional CLI runner for local video files

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

## CLI

Run the zone pipeline without the web UI:

```bash
python -m zone_risk.pipeline.main \
  --input input.mp4 \
  --output zone_output/annotated.mp4 \
  --events zone_output/events.jsonl
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
