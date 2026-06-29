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
- Brake-light deceleration cue (early warning for in-path lead vehicles)
- Traffic-light colour advisory (red/yellow/green)
- Collision-cone distance reliability on lane-crossing risk
- `SAFE`, `CAUTION`, and `DANGER` states
- Timeline rows, per-object metrics, event snapshots, live preview, and overlay imagery

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
   - Resizes frames preserving aspect ratio
   - Converts BGR frames to RGB and grayscale
   - Skips heavy denoise/CLAHE (Depth Anything V2 and DIS flow are robust to raw frames; denoise was the biggest CPU sink)

4. `spectra/vision/motion.py`
   - Uses OpenCV DIS dense optical flow with ego-motion compensation
   - Produces motion magnitude, normalized motion, divergence, and RGB flow visualization

5. `spectra/vision/depth.py`
   - Coordinates per-frame nearness estimation
   - Uses required Depth Anything V2 Metric VKITTI ONNX for estimated metric distance maps

6. `spectra/analysis/risk.py`
    - Bundles depth, flow, lane geometry, and track history into spatial fields
    - Builds per-object risk events and selects the primary event
    - Calculates lane position, crossing risk, fused TTC, confidence, and risk state

7. `spectra/analysis/overlay.py`
    - Draws lane corridor, object boxes (with TTC and `BRAKE` labels), the state/TTC card, and a traffic-light advisory indicator onto frames

8. `spectra/vision/brake_lights.py` and `spectra/vision/traffic_light.py`
    - Heuristic OpenCV appearance cues: lead-vehicle brake-light detection and traffic-light colour state

## How the Algorithm Works

This document explains Spectra's video risk analysis algorithm using the current `spectra` package structure. The system processes a driving-view video frame by frame, extracts road geometry and traffic participants, then combines nearness, motion, and object expansion signals to classify each situation as `SAFE`, `CAUTION`, or `DANGER`.

### 1. High-Level Goal

Spectra is designed to find risk-relevant objects in forward-facing driving footage relative to the ego lane. The algorithm relies only on visual signals:

- Object detection: YOLO detects road participants such as cars, people, bicycles, motorcycles, buses, and trucks.
- Tracking: The same object is linked across frames with IoU-based tracking.
- Road geometry: The road/lane corridor and vanishing point are estimated.
- Depth: Depth Anything V2 Metric VKITTI ONNX estimates metric distance maps in meters.
- Motion: OpenCV DIS produces dense optical flow, with ego-motion compensation.
- Appearance cues: heuristic brake-light detection (early deceleration warning) and traffic-light colour classification (advisory).
- Risk: TTC, lane relationship (with a collision-cone distance reliability term), nearness, closing speed, brake-light, and confidence are fused into a risk state.

The backend returns timeline rows, the peak event, per-object risk metrics, and deferred visual overlays to the frontend.

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

- `models/depth_anything_v2_metric_vkitti_vits.onnx`
- `models/ufld_v2_culane_r18.onnx`
- `models/yolov8n.pt`

If Depth Anything, UFLDv2, YOLO weights, or their runtime dependencies cannot load, analysis fails before frame processing begins. Optical flow is computed classically with OpenCV DIS, so there is no flow model to install.

### 4. Video Frame Loop

The per-frame logic is encapsulated in `SpatialFrameAnalyzer.process_frame()` in `spectra/analysis/video.py`, which holds all cross-frame state so any source can drive it. `analyze_spatial_video()` is a thin orchestrator that reads frames with `VideoLoader` and handles video-level concerns (event dedup, deferred rendering, progress).

For each frame:

1. `VideoLoader` reads the frame with OpenCV.
2. `spectra/vision/preprocessing.py` resizes the frame and produces RGB and grayscale views.
3. UFLDv2 estimates the ego-lane corridor on the configured lane interval; cached lane geometry is reused between runs and smoothed with Kalman coasting.
4. DIS optical flow is computed on the configured flow interval; skipped frames reuse the previous flow.
5. A cheap motion score decides whether depth should be refreshed early.
6. Depth Anything V2 Metric VKITTI estimates metric depth when scheduled or motion-triggered.
7. YOLO detects road participants on the configured detection interval.
8. Detections are filtered by ego-lane relevance before tracking.
9. The IoU tracker links detections to existing tracks or propagates tracks on skipped detection frames.
10. Risk is computed for every active track and the primary object is selected.
11. State transitions are stabilized with hysteresis.
12. A metadata-only event payload and timeline row are produced for every processed frame.
13. Saved events are deduplicated, ranked, and trimmed to `max_saved_events`.
14. Heavy RGB render outputs are generated only for saved events and on-demand previews.

### 5. Preprocessing

Preprocessing is implemented in `spectra/vision/preprocessing.py`.

Its purpose is to make downstream models more stable across different resolutions and lighting conditions.

Steps:

- Resize the image while preserving aspect ratio according to `resize_max_side`.
- Convert BGR to RGB.
- Create a grayscale image.
- Skip heavy denoising and CLAHE; Depth Anything V2 and DIS flow operate reliably on the resized frame, and avoiding denoise keeps CPU cost lower.

The output is a `PreprocessedFrame`:

- `bgr`: frame used by OpenCV drawing and detection
- `gray`: grayscale frame
- `rgb`: plain RGB view consumed by the depth model

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

Depth Anything V2 Metric VKITTI produces estimated metric depth in meters (`depth_m`, capped at 80m). Spectra derives a normalized `near_map` from that metric map for compatibility with existing filtering and scoring code.

Metric depth usage:

- Object distance is read from the lower-center portion of each bbox using a 25th-percentile distance sample. This raw sample is the *measurement* fed to a per-track constant-velocity Kalman filter.
- Depth TTC is the physical backbone: the Kalman filters distance into a smooth `(distance, range-rate)` state and computes `TTC = distance / closing_speed` every frame (coasting between depth refreshes). Innovation gating rejects single-step metric glitches, so closing speed stays physical.
- The UI exposes the filtered object distance in meters while preserving normalized proximity bars.

Because `TTC = d / (-ṡ)`, a constant scale bias in the uncalibrated monocular depth cancels in the ratio. The depth estimate is still not a calibrated sensor, so expansion and optical-flow TTC remain part of the final fusion as corroborating cues.

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
- traffic light (advisory only)

Each detection is stored as `Detection`:

- `bbox`: `(x1, y1, x2, y2)`
- `class_name`: normalized class name
- `confidence`: YOLO confidence

Class contribution is weighted through `CLASS_RISK_WEIGHT`. Larger and more stable traffic participants receive stronger trust in expansion signals.

Traffic lights are detected but are not collision participants: they are split out before the corridor filter and tracker (so they never get track IDs or TTC) and feed only the advisory traffic-light colour cue in `spectra/vision/traffic_light.py`. `frame_light_state()` returns a frame-level `(state, confidence)` pair (`red`/`yellow`/`green`/`unknown`), surfaced per timeline row as `trafficLight: {state, confidence}` and as a coloured dot in the overlay. It stays frame-level and never gates collision logic.

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

#### 11.3 Longitudinal Kinematic TTC (the physical backbone)

Code: `ttc_from_depth_delta()` — a per-track constant-velocity Kalman.

Logic:

- The metric distance sampled from the bbox lower-center is the measurement.
- A 2-state Kalman filters it into a smooth `(distance, range-rate)` state; closing speed is `-range-rate`.
- `TTC = distance / closing_speed`, read from the filtered state **every frame** — the filter predicts (coasts) between depth refreshes, so the estimate is continuous rather than intermittent.
- Innovation gating rejects a depth sample whose error exceeds ~3σ of the predicted distance (the filter coasts instead), so a single metric glitch cannot whip the velocity to a non-physical value.

The filter is committed only on a fresh-depth frame with a valid measurement; stale-depth frames predict-and-return without mutating state.

### 11.4 Approach Score

Approach is the normalized "getting closer" factor shown in the UI as a percentage (`_approach_score`). It is led by metric closing speed:

- 50% metric closing speed (depth Kalman range-rate)
- 30% bounding-box expansion
- 20% optical-flow magnitude

This signal is multiplied by class risk weight (people and bicycles are slightly more sensitive) and becomes the `approach` term in the Risk Score.

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

Collision-cone distance reliability: because `lane_position` is normalized by the lane width at the object's row, far objects (near the horizon) yield jittery positions. `lane_crossing_risk()` damps the velocity-extrapolated crossing for far objects while keeping the static corridor relevance as a floor, so phantom far cut-ins fade and genuine near cut-ins are untouched. (Curved-path / ego-yaw handling is not modeled; lane geometry is straight lines.)

### 14. Confidence

Fused confidence combines detection confidence and lane-geometry confidence; it drives the Risk Score's confidence gate (section 15). The per-cue reliabilities (detection, lane, depth, flow, expansion) and the TTC-cue agreement (`ttc_agreement`) are also surfaced per object. Very low detection confidence forces the status to `SAFE` (a trust guard).

### 15. Risk Score and State Classification

There are two distinct outputs, and **the state is derived from the score** (no circular dependency, no double-counted lane signal).

**Risk Score (how dangerous)** — `score_raw()` / `score_event()`. A single continuous `[0, 1]` value where lane relevance is a *multiplicative gate*, not an additive term, so an off-path object scores ~0 regardless of how close or fast it is:

```
signal     = 0.40·eta_pressure(ttc) + 0.30·proximity + 0.25·approach + 0.05·brake
gate       = 0.65 + 0.35·confidence
relevance  = crossing_risk                # probability of being in the ego lane at impact
Risk Score = gate · relevance · signal
```

`eta_pressure(ttc) = clamp((3 − ttc) / 3, 0, 1)`.

**Status (which band)** — `state_from_score()` bands the Risk Score:

- Risk Score ≥ 0.60: `DANGER`
- Risk Score ≥ 0.25: `CAUTION`
- otherwise: `SAFE`
- Imminent escalation: a confirmed TTC < 1.0s on an object already in at least the CAUTION band snaps to `DANGER`.
- Detection confidence < 0.20 forces `SAFE`.

The brake-light cue (`spectra/vision/brake_lights.py`) feeds the Risk Score directly (the 5% `brake` term above) rather than overriding the state band; it is corroborating only. This raw decision is later stabilized (section 16).

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

For each frame, `spectra/analysis/video.py` produces frame rows, saved events (deduplicated + top-N trimmed), the peak event, per-object metrics, and shared image payloads. Saved events are deduplicated within a 1-second window; a stronger event replaces the previous one in the same window.

The client-facing JSON contract is **schemaVersion 5** (`_serialize_result` in `spectra/app.py`), an object-centric nested shape. Each frame/event row carries `frameIndex`, `timestampSec`, `state`, a `primary` pointer (`{trackId, score, lane}`), a frame-level `trafficLight` (`{state, confidence}`), `laneGeometry`, and `objects[]`. Each object groups its metrics:

- `id` / `class` / `label` / `state`, `tracking.trackId`
- `risk`: `{score, factors{etaPressure, proximity, approach, crossing, brake}}`
- `eta`: `{collisionSec, display, agreement, sources{depth, flow, expansion → {etaSec, confidence}}}`
- `motion`: `{distanceM, closingSpeedMps, expansionScore, radialScore}`
- `lane`: `{bucket, position, crossing}`
- `confidence`: `{overall, detection, lane, depth, flow, expansion}`
- `bbox`: normalized `[x1, y1, x2, y2]`

The envelope also carries `metadata`, `images`, and `performance` (`{summary, logs}`). The frontend flattens this nested shape back to flat field names at a single seam (`flattenObject` / `flattenFrame` in `controls.js`); it does not compute risk.

### 19. Visual Overlay

Overlay rendering is implemented in `spectra/analysis/overlay.py`.

It draws:

- Ego lane corridor
- Object bounding boxes coloured by stabilized risk state, with a per-box `#id TYPE TTC` label (plus a `BRAKE` tag when the brake-light cue fires)
- A coloured traffic-light dot + label (advisory) when a light state is known
- A compact top-left card (CAUTION/DANGER only): a coloured `STATE` pill, the fused `TTC`, and an `object · lane` subtitle

The numeric risk factors (proximity, approach, crossing, confidence) live in the right-side dashboard panel, not as in-frame bars. This overlay is used for live preview and saved event imagery.

### 20. Frontend Role

The frontend does not compute risk. `spectra/web/static/js/controls.js` and related modules visualize backend payloads.

Frontend responsibilities:

- File selection and upload
- WebSocket preview listening
- Analysis settings for frame/time scope, sampling intervals, saved-event count, and processing resolution
- Timeline and event strip rendering
- Risk banner updates
- Active/detected object list with per-object TTC and risk factors
- Summary object clicks seek the preview back to that peak frame
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
7. Filter detections by ego-lane relevance.
8. Track objects across frames.
9. Estimate expansion, flow, and depth TTC for each object.
10. Fuse TTC components with weighted median.
11. Compute ego-lane relationship and crossing risk.
12. Compute closing speed, confidence, and near score.
13. Classify the object as `SAFE`, `CAUTION`, or `DANGER`.
14. Stabilize state transitions.
15. Select and deduplicate top risk events.
16. Send timeline, event payload, and saved-event visual outputs to the frontend.

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
- `detect_every`: YOLO detection interval
- `lane_every`: UFLDv2 lane detection interval
- `flow_every`: DIS optical-flow interval
- `start_sec`: optional analysis start time in seconds
- `end_sec`: optional analysis end time in seconds

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
