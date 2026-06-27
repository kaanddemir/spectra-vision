# How the Algorithm Works

This document explains Spectra's video risk analysis algorithm using the current `spectra` package structure. The system processes a driving-view video frame by frame, extracts road geometry and traffic participants, then combines nearness, motion, and object expansion signals to classify each situation as `SAFE`, `CAUTION`, or `DANGER`.

## 1. High-Level Goal

Spectra is designed to find risk-relevant objects in forward-facing driving footage relative to the ego lane. The algorithm relies only on visual signals:

- Object detection: YOLO detects road participants such as cars, people, bicycles, motorcycles, buses, and trucks.
- Tracking: The same object is linked across frames with IoU-based tracking.
- Road geometry: The road/lane corridor and vanishing point are estimated.
- Depth: Depth Anything V2 Metric VKITTI ONNX estimates metric distance maps in meters. On Apple Silicon the ONNX session prefers the CoreML execution provider so depth inference runs on the ANE/GPU.
- Motion: Classical DIS optical flow (OpenCV) with ego-motion compensation. There is no neural-flow ONNX dependency anymore — DIS is fast on CPU and accurate enough for bbox-level TTC and divergence signals.
- Appearance cues: a brake-light detector (early deceleration warning for in-path lead vehicles) and a traffic-light colour classifier (advisory). Both are heuristic OpenCV, no extra models.
- Risk: TTC, lane relationship (with a collision-cone distance reliability term), nearness, closing speed, brake-light, and confidence are fused into a risk state.

The backend returns timeline rows, the peak event, per-object risk metrics, and deferred visual overlays to the frontend.

## 2. Main Entry Point

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

## 3. Required Models and Runtime

Before analysis starts, `_ensure_required_models()` in `spectra/analysis/video.py` verifies that the required vision backends can load.

Required model files:

- `models/depth_anything_v2_metric_vkitti_vits.onnx`
- `models/ufld_v2_culane_r18.onnx`
- `models/yolov8n.pt`

Optical flow is now computed classically (OpenCV DIS), so there is no flow model to install. If Depth Anything, UFLDv2, YOLO weights, or their runtime dependencies cannot load, analysis fails before frame processing begins.

Runtime notes:

- Depth uses ONNX Runtime. The depth session prefers `CoreMLExecutionProvider` on macOS and falls back to `CPUExecutionProvider` on other platforms or when CoreML is unavailable.
- Depth Anything Metric VKITTI input resolution defaults to `518` and can be overridden with the `SPECTRA_DEPTH_INPUT` environment variable.
- YOLO runs through Ultralytics/PyTorch and prefers MPS, then CUDA, then CPU. Load or inference backend failures are hard errors; frames with no road participants remain empty results.

## 4. Video Frame Loop

The per-frame perception + risk logic is encapsulated in the `SpatialFrameAnalyzer` class in `spectra/analysis/video.py`. It holds all cross-frame state (tracker, lane Kalman, hysteresis stabilizer, the per-track smoothers, and the cached lane/flow/depth) so any source can drive it via `process_frame(frame_bgr, frame_index, timestamp_sec)`. `analyze_spatial_video()` is a thin orchestrator that reads frames with `VideoLoader`, calls `process_frame`, and handles video-level concerns (event dedup/top-N, deferred rendering, progress callbacks).

For each frame:

1. `VideoLoader` reads the frame with OpenCV.
2. `spectra/vision/preprocessing.py` resizes the frame and produces RGB and grayscale views.
3. UFLDv2 estimates the ego-lane corridor on the configured lane interval; cached lane geometry is reused between runs and smoothed with Kalman coasting.
4. DIS optical flow is computed on the configured flow interval; skipped frames reuse the previous flow.
5. A cheap motion score decides whether depth should be refreshed early.
6. Depth Anything V2 Metric VKITTI estimates metric depth when scheduled or motion-triggered.
7. YOLO detects road participants on the configured detection interval. Traffic-light detections are split out (advisory colour cue) and excluded from tracking.
8. Detections are filtered by ego-lane relevance before tracking.
9. The IoU tracker links detections to existing tracks or propagates tracks on skipped detection frames.
10. Risk is computed for every active track (including the brake-light cue) and the primary object is selected.
11. State transitions are stabilized with hysteresis.
12. A metadata-only event payload and a timeline row are produced.
13. Saved events are deduplicated, ranked, and trimmed to `max_saved_events`.
14. Heavy RGB visualisations are generated only for saved events and on-demand previews (see section 18).

## 5. Preprocessing

Preprocessing is implemented in `spectra/vision/preprocessing.py`.

Its purpose is to deliver consistent RGB and grayscale views to the downstream models without spending CPU on operations that the neural and classical components do not need.

Steps:

- Resize the image while preserving aspect ratio according to `resize_max_side`.
- Convert BGR to RGB.
- Create a grayscale image.

Heavy denoising (`fastNlMeansDenoising`) and CLAHE/LAB enhancement were removed: Depth Anything V2 and DIS flow are robust to raw frames, and `fastNlMeansDenoising` was the single biggest CPU sink on Apple Silicon.

The output is a `PreprocessedFrame`:

- `bgr`: frame used by OpenCV drawing and detection
- `gray`: grayscale frame (used by ego-motion and DIS flow)
- `rgb`: plain RGB view consumed by the depth model

## 6. Road and Lane Geometry

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

## 7. Optical Flow and Ego-Motion Compensation

Motion analysis is implemented in `spectra/vision/motion.py`.

Spectra computes dense optical flow with OpenCV's DIS algorithm (preset MEDIUM, spatial propagation enabled). For performance the flow is computed on a downscaled pair (long side capped at 320 px) and bilinearly upscaled back to the working frame resolution; the resulting flow magnitudes are rescaled accordingly. This is more than enough for bbox-level radial percentile and divergence signals — the consumers of flow only need approximate direction and magnitude.

Raw flow contains both object motion and camera motion, so Spectra subtracts estimated ego-motion.

Ego-motion compensation:

1. Detect good trackable corners in the previous grayscale frame.
2. Track those points in the current frame with Lucas-Kanade optical flow.
3. Estimate a homography with RANSAC.
4. Build a camera-motion flow field on a coarse grid (the field is sampled on a 1/8 grid and bilinearly upscaled to full resolution — visually identical to a dense `perspectiveTransform` but ~50× cheaper).
5. Subtract that camera-motion field from the dense flow.
6. If homography fitting is unreliable (low texture, abrupt scene changes, or too few inliers), fall back to median translation compensation.

The output is a `FlowResult`:

- `flow`: dense optical flow after ego-motion subtraction
- `magnitude_norm`: normalized motion magnitude
- `divergence_norm`: positive divergence signal

This signal is used both for TTC estimation and the frontend motion visualization.

## 8. Depth and Nearness Map

Depth estimation is implemented in `spectra/vision/depth.py` and the ONNX wrapper in `spectra/vision/models.py`.

Depth Anything V2 Metric VKITTI produces estimated metric depth in meters (`depth_m`, capped at 80m). Spectra derives a normalized `near_map` from that metric map for compatibility with existing filtering and scoring code.

Metric depth usage:

- Object distance is read from the lower-center portion of each bbox using a 25th-percentile distance sample. This raw sample is the *measurement* fed to the longitudinal Kalman filter (section 11.3); the distance and closing speed exposed to the UI come from the filtered state, not the raw sample.
- Depth TTC is the physical backbone: distance and closing speed are estimated by a per-track constant-velocity Kalman, and `TTC = distance / closing_speed`. Because TTC is the ratio `d / (-ṡ)`, it is invariant to a constant scale error in the (uncalibrated) monocular depth.
- The UI exposes the filtered object distance in meters while preserving normalized proximity bars.

Metric monocular depth is still an estimate, not a calibrated sensor measurement, so expansion and optical-flow TTC remain part of the final fusion as corroborating image-space cues.

Performance notes:

- The ONNX Runtime session is built with `CoreMLExecutionProvider` first and `CPUExecutionProvider` as a fallback.
- The default input resolution is `518`; this is configurable via the `SPECTRA_DEPTH_INPUT` environment variable.

Depth is not recomputed every frame. In `spectra/analysis/video.py`, depth refresh happens when:

- It is the first frame.
- The frame index matches the `depth_every` interval.
- Cheap motion risk exceeds `0.15`.

This balances performance with freshness.

## 9. Object Detection

Object detection is implemented in `spectra/vision/detection.py`.

YOLO detections are filtered to road-relevant COCO classes:

- person
- bicycle
- car
- motorcycle
- bus
- train
- truck
- traffic light (advisory only — see below)

Each detection is stored as `Detection`:

- `bbox`: `(x1, y1, x2, y2)`
- `class_name`: normalized class name
- `confidence`: YOLO confidence

Class contribution is weighted through `CLASS_RISK_WEIGHT`. Larger and more stable traffic participants receive stronger trust in expansion signals.

Traffic lights are detected but are **not collision participants**: `spectra/analysis/video.py` splits them out of the detection list before the corridor filter and tracker, so they never receive track IDs or TTC. They feed only the advisory traffic-light colour cue (section 9b). Their `CLASS_RISK_WEIGHT` is `0.0`.

## 9b. Traffic Light State (advisory)

Colour classification is implemented in `spectra/vision/traffic_light.py`.

For each detected traffic-light bbox, `classify_light_state()` reads the HSV histogram and returns `red`, `yellow`, `green`, or `unknown`. `frame_light_state()` picks the largest (nearest) light and returns one frame-level state, coasted on detection-skipped frames. This is surfaced as a frame-level advisory (`trafficLight` on each timeline row and a coloured dot in the overlay); it never gates collision logic, because "which light applies to my lane" is ambiguous from a single forward camera.

## 10. Object Tracking

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

## 11. TTC Estimation

TTC means "time to collision." Spectra estimates TTC from three independent sources rather than trusting a single signal.

### 11.1 Bounding-Box Expansion TTC

Code: `expansion_rate_from_track()` and `ttc_from_expansion()`

Logic:

- Compare the previous and current bounding-box sizes for the same tracked object.
- Compute scale growth from width and height ratios.
- If the object grows in the image, it may be approaching the camera.
- If expansion rate is high enough, estimate TTC with `1 / expansion_rate`.

Very small growth is treated as jitter and does not produce TTC.

### 11.2 Flow TTC

Code: `ttc_from_flow()`

Logic:

- Extract optical flow inside the object's bounding box.
- Compute each pixel's radial direction relative to the vanishing point.
- Measure outward radial flow.
- If the object expands away from the vanishing point, it may be approaching.
- Use the 75th percentile to suppress static background pixels and noise.

This provides additional evidence when bounding-box size changes are weak.

### 11.3 Longitudinal Kinematic TTC (the physical backbone)

Code: `ttc_from_depth_delta()` (a per-track constant-velocity Kalman; the `DepthDeltaSmoother` bank holds one `_LonState` per track id).

Logic:

- The metric distance sampled from the bbox lower-center (section 8) is the measurement.
- A 2-state Kalman filters it into a smooth `(distance, range-rate)` state. Range-rate `ṡ` is `d(distance)/dt` (negative while approaching); closing speed is `-ṡ`.
- `TTC = distance / closing_speed`, read from the filtered state **every frame** — the filter predicts (coasts) between depth refreshes, so the estimate is continuous rather than intermittent.
- **Innovation gating:** a depth sample whose error exceeds ~3σ of the predicted distance is rejected (the filter coasts on prediction instead), so a single-step metric glitch cannot whip the velocity to a non-physical value. This is what removes the old ±20–30 m/s `closing_mps` artifacts.
- The exposed `distance_m` and `closing_mps` come from the filtered state (closing clamped to a physical range); a velocity estimate is only trusted after ≥2 measurements.

The filter is committed (advanced + corrected) only on a fresh-depth frame with a valid measurement. The imminence peek and stale-depth frames predict-and-return without mutating state, so the filter is never stepped twice per frame. Because `TTC = d / (-ṡ)`, a constant scale bias in the monocular depth cancels in the ratio.

## 12. TTC Fusion

Code: `fuse_ttc()`

The three TTC components are not averaged. Spectra uses weighted median fusion.

Why weighted median:

- A single bad signal cannot dominate the final TTC.
- Expansion, flow, and depth can fail under different conditions.
- Higher-confidence components have more influence.

The kinematic (depth) component is the continuous backbone — once a track has two depth measurements it contributes a TTC on every frame, so the fused result has a stable anchor instead of collapsing to whichever single visual cue happened to fire. Expansion and flow corroborate and can outvote it when they agree with higher confidence.

Each component has:

- `name`: expansion, flow, or depth
- `value`: TTC in seconds, or `None`
- `confidence`: value in `[0, 1]`

If there are no valid components, fused TTC is `None`.

## 13. Lane Relationship and Crossing Risk

Code: `lane_position()`, `lane_lateral_velocity()`, `lane_crossing_risk()`

Risk is not based only on whether an object is approaching. Spectra also estimates whether the object is relevant to the ego lane.

Signals:

- Object bottom-center position relative to the ego lane
- Object lateral velocity in lane-units per second
- Predicted lane position within the TTC horizon
- How much the bounding-box bottom edge overlaps the road corridor

For example, a vehicle in the right lane may stay low risk even if it is approaching. If it is laterally moving toward the ego lane, crossing risk rises.

Collision-cone distance reliability: `lane_position` is normalized by the lane width at the object's image row, so far objects (near the horizon) are computed from very few pixels and their lateral-velocity extrapolation is jittery. `lane_crossing_risk()` damps the *predicted* (velocity-based) crossing toward the horizon using the object's vertical position; the static corridor `base` stays a floor, so genuine near cut-ins are untouched while phantom far cut-ins fade. (Ego-yaw / curved-path handling is intentionally not modeled — lane geometry is stored as straight lines.)

## 14. Approach Score and Confidence

For every object, Spectra computes a normalized `approach` signal (`_approach_score`), led by metric closing speed:

- 50% metric closing speed (depth Kalman range-rate)
- 30% bounding-box expansion
- 20% optical-flow magnitude

This signal is multiplied by class risk weight (people and bicycles are slightly more sensitive) and becomes the `approach` term in the Risk Score (section 15a).

Fused confidence combines detection confidence and lane geometry confidence; it drives the Risk Score's confidence gate. Very low detection confidence forces the status to `SAFE` (section 15b).

## 15. Risk Score and State Classification

There are two distinct outputs, and **the state is derived from the score** (no circular dependency, no double-counted lane signal).

### 15a. Risk Score (how dangerous)

Code: `score_raw()` / `score_event()` in `spectra/analysis/risk.py`.

A single continuous `[0, 1]` value. Lane relevance is a **multiplicative gate**, not an additive term, so an object that is not in our path scores ~0 regardless of how close or fast it is:

```
signal    = 0.40·eta_pressure(ttc) + 0.30·proximity + 0.25·approach + 0.05·brake
gate      = 0.65 + 0.35·confidence
relevance = crossing_risk            # probability of being in the ego lane at impact
Risk Score = gate · relevance · signal
```

`eta_pressure(ttc) = clamp((3 − ttc) / 3, 0, 1)`. The score has **no state floor**, so the score and the state are computed in one direction only (features → score → state). Because lane relevance gates the whole score, a near/fast off-path vehicle stays low (this replaced the old explicit corridor gate inside the state machine).

### 15b. Status (which band)

Code: `state_from_score()` in `spectra/analysis/risk.py`.

The status is purely a banding of the Risk Score plus one safety escalation:

- `Risk Score ≥ _DANGER_SCORE_BAND` (0.60, i.e. 60/100, tunable): `DANGER`
- `Risk Score ≥ _CAUTION_SCORE_BAND` (0.25, i.e. 25/100, tunable): `CAUTION`
- otherwise: `SAFE`
- **Imminent escalation:** a confirmed TTC < 1.0s (≥2 imminent frames via `TtcImminenceSmoother`) on an object already in at least the CAUTION band snaps straight to `DANGER`. The escalation is gated by the score (not by a separate lane input), so an imminent *off-path* object — which scores low — is not raised.
- Detection confidence < 0.20 forces `SAFE` (trust guard).

The band edges are initial values and are meant to be tuned against real footage. This raw decision is later stabilized (section 16).

### 15c. Brake-Light Cue

Code: `spectra/vision/brake_lights.py` (`brake_score`), computed in `calculate_track_risk()`.

A lit brake-lamp pair lights up *before* the gap visibly closes, so it is an early forward-collision cue no TTC signal captures. For rear-facing vehicle classes (car/truck/bus), `brake_score()` looks for a bright, symmetric, localized red pair in the lower band of the bbox (full-band red is treated as bodywork and suppressed). The brake score now feeds the **Risk Score directly** (the 5% `brake` term above) rather than overriding the state band; it is exposed as `RiskEvent.brake_score` and a `BRAKE` label in the overlay.

## 16. State Stabilization

Code: `StateStabilizer` and `stabilized_event_state()`

Video detections can jitter from frame to frame. Spectra smooths state transitions with hysteresis.

Default behavior:

- Upgrading to a higher-risk state requires 3 consecutive frames.
- Downgrading to a safer state requires 7 consecutive frames.
- True imminent danger with TTC <= 1s is not delayed; it becomes `DANGER` immediately.

This reduces flicker in the frontend risk banner.

TTC is preserved through stabilization. Even when the stabilized state is `SAFE`, the fused TTC value (when measurable) survives, so the timeline chart and the in-frame HUD continue to show a numeric TTC reading rather than a placeholder. A `SAFE` object's Risk Score stays low on its own because the score is the lane-gated weighted signal (no state floor), so showing a SAFE TTC does not inflate the saved-event ranking.

## 17. Multiple Objects in One Frame

Each active track produces its own `RiskEvent`. Then `build_object_events()` selects the primary event.

Primary event selection considers:

- Risk state severity
- How far TTC falls below 3 seconds
- Closing speed
- Crossing risk

If there are no active tracks, Spectra emits a synthetic `SAFE` event. This keeps the frontend payload consistent for every frame.

## 18. Event Payload, Timeline, and Deferred Rendering

For each frame, `spectra/analysis/video.py` produces:

- `frames`: frame-level risk history for the timeline
- `events`: saved high-risk moments after deduplication and top-N trimming
- `peak_event`: highest-risk saved event in the analysis
- `objects`: per-frame object metrics, including TTC, estimated distance, metric closing speed, lane, risk score, proximity, approach, crossing, and confidence
- `images`: shared image payloads referenced by saved events

For events that are kept in the saved-events list, the following RGB views are attached:

- `original_rgb`: original frame
- `overlay_rgb`: frame with boxes, lane lines, and risk text

Heavy RGB rendering is deferred. Every frame produces a metadata-only event payload first; the per-frame rendering inputs (frame BGR, tracked objects, and lane geometry) are stashed in a `_DeferredRender` reference. After the frame's score is compared against the saved-events top-N (and the per-second deduplication window), only events that survive the cut have their RGB views materialized. Frames that never make the saved list pay no rendering cost. Preview ticks render the annotated frame on demand for the JPEG sent over the WebSocket.

Saved events are deduplicated within a 1-second window. If a stronger event appears in the same window, it replaces the previous saved event.

## 19. Visual Overlay (in-frame HUD)

Overlay rendering is implemented in `spectra/analysis/overlay.py`.

It draws:

- Ego lane corridor
- Object bounding boxes coloured by stabilized risk state, with a per-box label (`#id TYPE TTC`, plus a `BRAKE` tag when the brake-light cue fires)
- A coloured traffic-light dot + label in the top-right corner when a light state is known (advisory)
- A compact card pinned to the top-left (CAUTION/DANGER only) containing:
  - a coloured `STATE` pill plus the fused `TTC` reading
  - an object · lane subtitle (e.g. `CAR | Same Lane`)

The card stays small (≤220 px wide). The numeric risk factors (proximity, approach, crossing, confidence) live in the right-side dashboard panel, not as in-frame bars; the per-component TTC breakdown (`Exp / Flow / Depth`) is not drawn on the frame either — the HUD shows the fused TTC and the object panel carries the rest.

This overlay is used for live preview and saved event imagery.

## 20. Frontend Role

The frontend does not compute risk. `spectra/web/static/js/controls.js` and related modules visualize backend payloads.

Frontend responsibilities:

- File selection and upload
- WebSocket preview listening
- Analysis settings for frame/time scope, sampling intervals, saved-event count, and processing resolution
- Timeline and event strip rendering
- Risk banner updates (alert state, TTC, lane)
- "Active/Detected Objects" panel: list of objects with type and TTC, with a detail pane that shows ID / Type / TTC / Lane and four signal bars (Proximity, Approach, Crossing, Confidence). In summary mode, object clicks keep the preview on that peak frame.
- "Temporal Analysis" chart: a single red TTC line with a light EMA applied to suppress single-frame jitter, two dashed threshold guides at 3.0s (CAUTION) and 1.0s (DANGER), and a legend strip below explaining the bands. The chart shows TTC for SAFE rows too; only the in-chart category labels were removed because the legend below already names the bands.
- Telemetry JSON export

Risk scores and risk states are determined by the backend. The frontend presents them in an understandable interface.

## 21. Short Summary Flow

The algorithm runs in this order:

1. Upload video.
2. Read and resize a frame.
3. Estimate road/lane geometry.
4. Compute DIS optical flow and subtract ego-motion.
5. Run depth estimation when needed (CoreML on Apple Silicon).
6. Detect traffic participants with YOLO.
7. Filter detections by ego-lane relevance.
8. Track objects across frames.
9. Estimate expansion, flow, and depth TTC for each object.
10. Fuse TTC components with weighted median.
11. Compute ego-lane relationship and crossing risk.
12. Compute closing speed, confidence, and near score.
13. Classify the object as `SAFE`, `CAUTION`, or `DANGER`.
14. Stabilize state transitions (TTC is preserved through SAFE).
15. Select and deduplicate top risk events.
16. Send timeline, event metadata, and saved-event visual outputs to the frontend.

The core design is to avoid relying on a single visual cue. Spectra combines three independent approach signals with lane relevance before declaring risk.
