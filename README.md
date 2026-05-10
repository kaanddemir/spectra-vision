# Spectra

Spectra is a lane-relative video risk analysis tool. It processes uploaded driving-view videos, detects road geometry and YOLO traffic participants, fuses visual nearness with optical flow and track expansion, then reports risk state, TTC, and timeline events.

The current pipeline is object-centric and does not use external narrative services.

## Features

- Video upload and browser-based analysis UI
- YOLO-based road participant tracking
- Road/lane-relative risk scoring
- Required Depth Anything V2 ONNX depth estimation
- Classical DIS dense optical flow with ego-motion compensation
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
python -m uvicorn spectra.app:app --host localhost --port 8000 --reload
```

## Project Structure

```text
spectra/
  app.py
  analysis/
    video.py
    risk.py
    tracking.py
    overlay.py
  vision/
    models.py
    preprocessing.py
    depth.py
    motion.py
    detection.py
    road.py
  web/
    static/
      index.html
      js/
        main.js
        controls.js
      css/
        main.css
```

## Pipeline

1. `spectra/app.py`
   - FastAPI entry point
   - Serves the UI
   - Accepts video uploads through `/api/analyze`
   - Serializes analysis results for the frontend

2. `spectra/analysis/video.py`
   - Web-facing analysis adapter
   - Runs the frame loop
   - Builds timeline rows, event payloads, and image outputs
   - Reads video frames with OpenCV
   - Provides frame index and timestamp metadata

3. `spectra/vision/preprocessing.py`
   - Resizes frames
   - Converts BGR frames to RGB and grayscale
   - Applies denoising and CLAHE enhancement
   - Produces enhanced grayscale and denoised RGB images

4. `spectra/vision/motion.py`
   - Uses OpenCV DIS dense optical flow with ego-motion compensation
   - Produces motion magnitude, normalized motion, divergence, and RGB flow visualization

5. `spectra/vision/depth.py`
   - Coordinates per-frame nearness estimation
   - Uses required Depth Anything V2 ONNX for per-frame nearness maps

6. `spectra/analysis/risk.py`
    - Bundles depth, flow, lane geometry, and track history into spatial fields
    - Builds per-object risk events and selects the primary event
    - Calculates lane position, crossing risk, fused TTC, confidence, and risk state

7. `spectra/analysis/overlay.py`
    - Draws lane corridor, object boxes, TTC components, and summary text onto frames

## How the Algorithm Works

This document explains Spectra's video risk analysis algorithm using the current `spectra` package structure. The system processes a driving-view video frame by frame, extracts road geometry and traffic participants, then combines nearness, motion, and object expansion signals to classify each situation as `SAFE`, `CAUTION`, or `DANGER`.

### 1. High-Level Goal

Spectra is designed to find risk-relevant objects in forward-facing driving footage relative to the ego lane. The algorithm relies only on visual signals:

- Object detection: YOLO detects road participants such as cars, people, bicycles, motorcycles, buses, and trucks.
- Tracking: The same object is linked across frames with IoU-based tracking.
- Road geometry: The road/lane corridor and vanishing point are estimated.
- Depth: Depth Anything V2 ONNX produces a relative nearness map.
- Motion: OpenCV DIS produces dense optical flow, with ego-motion compensation.
- Risk: TTC, lane relationship, nearness, closing speed, and confidence are fused into a risk state.

The backend returns timeline rows, the peak event, lane metrics, TTC components, and visual overlays to the frontend.

### 2. Main Entry Point

The request flow starts in `spectra/app.py`.

1. The frontend uploads a video to `/api/analyze`.
2. The API validates the file extension and upload size.
3. The uploaded file is written to a temporary directory.
4. The analysis is delegated to `spectra.analysis.video.analyze_spatial_video`.
5. If a `session_id` is provided, a WebSocket preview queue is created.
6. During analysis, preview frames and telemetry are sent through `/ws/preview/{session_id}`.
7. When analysis finishes, `_serialize_result` converts the output into the frontend JSON contract.

The public routes remain:

- `GET /api/health`
- `POST /api/analyze`
- `WS /ws/preview/{session_id}`

### 3. Required Models and Runtime

Before analysis starts, `_ensure_required_models()` in `spectra/analysis/video.py` verifies that required vision backends can load.

Required local model files:

- `models/depth_anything_v2_small.onnx`
- `models/ufld_v2_culane_r18.onnx`
- `models/yolov8n.pt`

If Depth Anything, UFLDv2, YOLO weights, or their runtime dependencies cannot load, analysis fails before frame processing begins. Optical flow is computed classically with OpenCV DIS, so there is no flow model to install.

### 4. Video Frame Loop

The main frame loop lives in `spectra/analysis/video.py`.

For each frame:

1. `VideoLoader` reads the frame with OpenCV.
2. `spectra/vision/preprocessing.py` resizes and enhances the frame.
3. RGB, grayscale, CLAHE-enhanced grayscale, and denoised RGB variants are created.
4. `spectra/vision/road.py` estimates the road corridor.
5. The vanishing point is computed and smoothed with EMA.
6. Optical flow is computed between the previous and current frames.
7. A cheap motion score decides whether depth should be refreshed early.
8. YOLO detects road participants.
9. The IoU tracker links detections to existing tracks.
10. Risk is computed for every active track.
11. The highest-risk event is selected.
12. State transitions are stabilized with hysteresis.
13. Original and risk-overlay images are generated for saved events.
14. A timeline row and event payload are produced.

### 5. Preprocessing

Preprocessing is implemented in `spectra/vision/preprocessing.py`.

Its purpose is to make downstream models more stable across different resolutions and lighting conditions.

Steps:

- Resize the image while preserving aspect ratio according to `resize_max_side`.
- Convert BGR to RGB.
- Create a grayscale image.
- Denoise the grayscale image.
- Improve contrast using CLAHE.
- Reapply the enhanced luminance channel through LAB color space.

The output is a `PreprocessedFrame`:

- `bgr`: frame used by OpenCV drawing and detection
- `gray`: grayscale frame
- `enhanced_gray`: contrast-enhanced grayscale frame
- `denoised_rgb`: cleaned RGB frame used by the ONNX models

### 6. Road and Lane Geometry

Road geometry is computed in `spectra/vision/road.py`.

There are two lane geometry paths:

- UFLDv2 ROI: the required lane model estimates the ego-lane boundaries.
- Default ROI: if a scheduled lane frame is weak and no cached ROI exists, a fixed perspective polygon is used.

The main output is a `LaneFrame`.

`LaneFrame` contains:

- Vanishing point
- Left and right lane lines
- Lane width at the bottom of the image
- Lane center at the bottom of the image
- Detection confidence
- Image width and height

Object lateral position is normalized relative to the lane, not raw pixels:

- `0.0`: ego-lane center
- `-1.0`: left lane boundary
- `+1.0`: right lane boundary
- `-2.0` or `+2.0`: roughly one lane away

This keeps the same thresholds usable across different camera angles and image sizes.

### 7. Optical Flow and Ego-Motion Compensation

Motion analysis is implemented in `spectra/vision/motion.py`.

OpenCV DIS produces dense optical flow between two grayscale frames. Raw flow contains both object motion and camera motion, so Spectra subtracts estimated ego-motion.

Ego-motion compensation:

1. Detect good trackable corners in the previous grayscale frame.
2. Track those points in the current frame with Lucas-Kanade optical flow.
3. Estimate a homography with RANSAC.
4. Convert the homography into a camera-motion flow field.
5. Subtract that camera-motion field from raw dense flow.
6. If homography fitting is unreliable, fall back to median translation compensation.

The output is a `FlowResult`:

- `flow`: dense optical flow after ego-motion subtraction
- `magnitude_norm`: normalized motion magnitude
- `divergence_norm`: positive divergence signal

This signal is used both for TTC estimation and the frontend motion visualization.

### 8. Depth and Nearness Map

Depth estimation is implemented in `spectra/vision/depth.py`.

Depth Anything V2 produces relative depth. Spectra does not interpret it as metric distance. Instead, it converts model output into a normalized `near_map`.

Nearness calibration:

- Normalize model output to `[0, 1]`.
- Compute a median baseline per image row.
- Treat row-relative excess above that baseline as obstacle evidence.
- Combine absolute nearness and row-relative excess.

This reduces false risk from normal road perspective. The bottom of the road naturally appears closer; Spectra gives stronger weight to regions that stand out relative to nearby pixels on the same row.

Depth is not recomputed every frame. In `spectra/analysis/video.py`, depth refresh happens when:

- It is the first frame.
- The frame index matches the `depth_every` interval.
- Cheap motion risk exceeds `0.15`.

This balances performance with freshness.

### 9. Object Detection

Object detection is implemented in `spectra/vision/detection.py`.

YOLO detections are filtered to road-relevant COCO classes:

- person
- bicycle
- car
- motorcycle
- bus
- train
- truck

Each detection is stored as `Detection`:

- `bbox`: `(x1, y1, x2, y2)`
- `class_name`: normalized class name
- `confidence`: YOLO confidence

Class contribution is weighted through `CLASS_RISK_WEIGHT`. Larger and more stable traffic participants receive stronger trust in expansion signals.

### 10. Object Tracking

Tracking is implemented in `spectra/analysis/tracking.py`.

The tracker uses a lightweight IoU matcher:

1. Compare active tracks with new detections.
2. Only detections of the same class can match a track.
3. Candidate pairs above the IoU threshold are sorted by descending IoU.
4. Greedy matching updates existing tracks.
5. Unmatched detections create new tracks.
6. Tracks missing for several frames are removed.

Each track stores short history. That history is required for:

- Bounding-box expansion rate
- Lane-relative lateral velocity

### 11. TTC Estimation

TTC means "time to collision." Spectra estimates TTC from three independent sources rather than trusting a single signal.

#### 11.1 Bounding-Box Expansion TTC

Code: `expansion_rate_from_track()` and `ttc_from_expansion()`

Logic:

- Compare the previous and current bounding-box sizes for the same tracked object.
- Compute scale growth from width and height ratios.
- If the object grows in the image, it may be approaching the camera.
- If expansion rate is high enough, estimate TTC with `1 / expansion_rate`.

Very small growth is treated as jitter and does not produce TTC.

#### 11.2 Flow TTC

Code: `ttc_from_flow()`

Logic:

- Extract optical flow inside the object's bounding box.
- Compute each pixel's radial direction relative to the vanishing point.
- Measure outward radial flow.
- If the object expands away from the vanishing point, it may be approaching.
- Use the 75th percentile to suppress static background pixels and noise.

This provides additional evidence when bounding-box size changes are weak.

#### 11.3 Depth Delta TTC

Code: `ttc_from_depth_delta()`

Logic:

- Compute median nearness inside the object's bounding box.
- Compare it with the previous nearness for the same track.
- If nearness increases, the object may be approaching.
- Estimate TTC from nearness growth rate and remaining-distance proxy.

This component updates history only when depth is fresh. If an old depth map is reused, history is not mutated, preventing false deltas.

### 12. TTC Fusion

Code: `fuse_ttc()`

The three TTC components are not averaged. Spectra uses weighted median fusion.

Why weighted median:

- A single bad signal cannot dominate the final TTC.
- Expansion, flow, and depth can fail under different conditions.
- Higher-confidence components have more influence.

Each component has:

- `name`: expansion, flow, or depth
- `value`: TTC in seconds, or `None`
- `confidence`: value in `[0, 1]`

If there are no valid components, fused TTC is `None`.

### 13. Lane Relationship and Crossing Risk

Code: `lane_position()`, `lane_lateral_velocity()`, `lane_crossing_risk()`

Risk is not based only on whether an object is approaching. Spectra also estimates whether the object is relevant to the ego lane.

Signals:

- Object bottom-center position relative to the ego lane
- Object lateral velocity in lane-units per second
- Predicted lane position within the TTC horizon
- How much the bounding-box bottom edge overlaps the road corridor

For example, a vehicle in the right lane may stay low risk even if it is approaching. If it is laterally moving toward the ego lane, crossing risk rises.

### 14. Closing Speed and Confidence

For every object, Spectra computes a normalized closing-speed-like signal:

- 50% bounding-box expansion
- 30% crossing risk
- 20% optical-flow magnitude

This signal is multiplied by class risk weight. People and bicycles are slightly more sensitive; traffic lights and stop signs are down-weighted.

Fused confidence combines:

- Detection confidence
- Crossing risk
- Expansion strength
- Lane geometry confidence

Very low detection confidence can force the state to `SAFE`.

### 15. Risk State Classification

Code: `classify_state()`

There are three states:

- `SAFE`
- `CAUTION`
- `DANGER`

Main thresholds:

- TTC < 1.0s and the object is in or entering the ego lane: `DANGER`
- TTC < 3.0s and the object is lane-relevant: `CAUTION`
- TTC < 3.0s and nearness is high: `CAUTION`
- Strong expansion with meaningful crossing and nearness: `DANGER`
- Moderate expansion with lane relevance: `CAUTION`
- Very high nearness with meaningful crossing: `CAUTION`
- Otherwise: `SAFE`

This raw decision is later stabilized.

### 16. State Stabilization

Code: `StateStabilizer` and `stabilized_event_state()`

Video detections can jitter from frame to frame. Spectra smooths state transitions with hysteresis.

Default behavior:

- Upgrading to a higher-risk state requires 3 consecutive frames.
- Downgrading to a safer state requires 7 consecutive frames.
- True imminent danger with TTC <= 1s is not delayed; it becomes `DANGER` immediately.

This reduces flicker in the frontend risk banner.

### 17. Multiple Objects in One Frame

Each active track produces its own `RiskEvent`. Then `build_object_events()` selects the primary event.

Primary event selection considers:

- Risk state severity
- How far TTC falls below 3 seconds
- Closing speed
- Crossing risk

If there are no active tracks, Spectra emits a synthetic `SAFE` event. This keeps the frontend payload consistent for every frame.

### 18. Event Payload and Timeline

For each frame, `spectra/analysis/video.py` produces:

- `timeline_rows`: frame-level risk history
- `events`: saved high-risk moments
- `peak_event`: highest-risk event in the analysis
- `lane_metrics`: scores per lane/object bucket
- `ttc_components`: expansion, flow, and depth TTC details
- `original_rgb`: original frame
- `overlay_rgb`: frame with boxes, lane lines, and risk text

Saved events are deduplicated within a 1-second window. If a stronger event appears in the same window, it replaces the previous saved event.

### 19. Visual Overlay

Overlay rendering is implemented in `spectra/analysis/overlay.py`.

It draws:

- Ego lane corridor
- Object bounding boxes
- Risk-state colors
- TTC components
- Object class
- Lane position
- Summary risk text

This overlay is used for live preview and saved event imagery.

### 20. Frontend Role

The frontend does not compute risk. `spectra/web/static/js/controls.js` and related modules visualize backend payloads.

Frontend responsibilities:

- File selection and upload
- WebSocket preview listening
- Timeline and event strip rendering
- Risk banner updates
- Depth, road, motion, and overlay views
- Telemetry JSON export

Risk scores and risk states are determined by the backend. The frontend presents them in an understandable interface.

### 21. Short Summary Flow

The algorithm runs in this order:

1. Upload video.
2. Read and preprocess a frame.
3. Estimate road/lane geometry.
4. Compute optical flow and subtract camera motion.
5. Run depth estimation when needed.
6. Detect traffic participants with YOLO.
7. Track objects across frames.
8. Estimate expansion, flow, and depth TTC for each object.
9. Fuse TTC components with weighted median.
10. Compute ego-lane relationship and crossing risk.
11. Compute closing speed, confidence, and near score.
12. Classify the object as `SAFE`, `CAUTION`, or `DANGER`.
13. Stabilize state transitions.
14. Select the most critical event.
15. Send timeline, event payload, and visual outputs to the frontend.

The core design is to avoid relying on a single visual cue. Spectra combines three independent approach signals with lane relevance before declaring risk.

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
python -m compileall spectra
```

Quick import check:

```bash
python -c "from spectra.app import app; print(app.title)"
```
