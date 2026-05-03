# How the Algorithm Works

This document explains Spectra's video risk analysis algorithm using the current `spectra` package structure. The system processes a driving-view video frame by frame, extracts road geometry and traffic participants, then combines nearness, motion, and object expansion signals to classify each situation as `SAFE`, `CAUTION`, or `DANGER`.

## 1. High-Level Goal

Spectra is designed to find risk-relevant objects in forward-facing driving footage relative to the ego lane. The algorithm relies only on visual signals:

- Object detection: YOLO detects road participants such as cars, people, bicycles, motorcycles, buses, and trucks.
- Tracking: The same object is linked across frames with IoU-based tracking.
- Road geometry: The road/lane corridor and vanishing point are estimated.
- Depth: Depth Anything V2 ONNX produces a relative nearness map. On Apple Silicon the ONNX session prefers the CoreML execution provider so depth inference runs on the ANE/GPU.
- Motion: Classical DIS optical flow (OpenCV) with ego-motion compensation. There is no neural-flow ONNX dependency anymore — DIS is fast on CPU and accurate enough for bbox-level TTC and divergence signals.
- Risk: TTC, lane relationship, nearness, closing speed, and confidence are fused into a risk state.

The backend returns timeline rows, the peak event, lane metrics, TTC components, and visual overlays to the frontend.

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

Before analysis starts, `_ensure_required_models()` in `spectra/analysis/video.py` verifies that the required ONNX model file is present.

Required model files:

- `models/depth_anything_v2_small.onnx`

Optical flow is now computed classically (OpenCV DIS), so there is no flow model to install. If Depth Anything is missing, analysis fails before frame processing begins.

Runtime notes:

- Depth uses ONNX Runtime. The depth session prefers `CoreMLExecutionProvider` on macOS and falls back to `CPUExecutionProvider` on other platforms or when CoreML is unavailable.
- Depth Anything input resolution defaults to `392` and can be overridden with the `SPECTRA_DEPTH_INPUT` environment variable.
- YOLO is lazy-loaded in `spectra/vision/detection.py`; if Ultralytics or the detector model cannot be loaded, detections are empty and frames resolve to safe synthetic events.

## 4. Video Frame Loop

The main frame loop lives in `spectra/analysis/video.py`.

For each frame:

1. `VideoLoader` reads the frame with OpenCV.
2. `spectra/vision/preprocessing.py` resizes the frame and produces RGB and grayscale views.
3. `spectra/vision/road.py` estimates the road corridor.
4. The vanishing point is computed and smoothed with EMA.
5. DIS optical flow is computed between the previous and current grayscale frames; ego-motion is then subtracted.
6. A cheap motion score decides whether depth should be refreshed early.
7. YOLO detects road participants.
8. The IoU tracker links detections to existing tracks.
9. Risk is computed for every active track.
10. The highest-risk event is selected.
11. State transitions are stabilized with hysteresis.
12. A metadata-only event payload and a timeline row are produced; heavy RGB visualisations (overlay, depth, motion, road) are deferred (see section 18).

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
- `enhanced_gray`: kept as an alias of `gray` for backward compatibility
- `denoised_rgb`: kept as an alias of plain RGB for backward compatibility — consumed by the depth model

## 6. Road and Lane Geometry

Road geometry is computed in `spectra/vision/road.py`.

There are two modes:

- Dynamic ROI: Canny edges and Hough lines estimate left and right lane boundaries.
- Default ROI: if lane detection is weak, a fixed perspective polygon is used.

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

Depth Anything V2 produces relative depth. Spectra does not interpret it as metric distance. Instead, it converts model output into a normalized `near_map`.

Nearness calibration:

- Normalize model output to `[0, 1]`.
- Compute a median baseline per image row.
- Treat row-relative excess above that baseline as obstacle evidence.
- Combine absolute nearness and row-relative excess.

This reduces false risk from normal road perspective. The bottom of the road naturally appears closer; Spectra gives stronger weight to regions that stand out relative to nearby pixels on the same row.

Performance notes:

- The ONNX Runtime session is built with `CoreMLExecutionProvider` first and `CPUExecutionProvider` as a fallback.
- The default input resolution is `392` (instead of the upstream-recommended `518`); this is configurable via the `SPECTRA_DEPTH_INPUT` environment variable.

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
- traffic light
- stop sign

Each detection is stored as `Detection`:

- `bbox`: `(x1, y1, x2, y2)`
- `class_name`: normalized class name
- `confidence`: YOLO confidence

Class contribution is weighted through `CLASS_RISK_WEIGHT`. Larger and more stable traffic participants receive stronger trust in expansion signals. Static objects such as traffic lights and stop signs receive lower risk weights.

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

### 11.3 Depth Delta TTC

Code: `ttc_from_depth_delta()`

Logic:

- Compute median nearness inside the object's bounding box.
- Compare it with the previous nearness for the same track.
- If nearness increases, the object may be approaching.
- Estimate TTC from nearness growth rate and remaining-distance proxy.

This component updates history only when depth is fresh. If an old depth map is reused, history is not mutated, preventing false deltas.

## 12. TTC Fusion

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

## 13. Lane Relationship and Crossing Risk

Code: `lane_position()`, `lane_lateral_velocity()`, `lane_crossing_risk()`

Risk is not based only on whether an object is approaching. Spectra also estimates whether the object is relevant to the ego lane.

Signals:

- Object bottom-center position relative to the ego lane
- Object lateral velocity in lane-units per second
- Predicted lane position within the TTC horizon
- How much the bounding-box bottom edge overlaps the road corridor

For example, a vehicle in the right lane may stay low risk even if it is approaching. If it is laterally moving toward the ego lane, crossing risk rises.

## 14. Closing Speed and Confidence

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

## 15. Risk State Classification

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

## 16. State Stabilization

Code: `StateStabilizer` and `stabilized_event_state()`

Video detections can jitter from frame to frame. Spectra smooths state transitions with hysteresis.

Default behavior:

- Upgrading to a higher-risk state requires 3 consecutive frames.
- Downgrading to a safer state requires 7 consecutive frames.
- True imminent danger with TTC <= 1s is not delayed; it becomes `DANGER` immediately.

This reduces flicker in the frontend risk banner.

TTC is preserved through stabilization. Even when the stabilized state is `SAFE`, the fused TTC value (when measurable) survives, so the timeline chart and the in-frame HUD continue to show a numeric TTC reading rather than a placeholder. The risk score (`_risk_score`) ignores TTC pressure when the state is `SAFE`, so showing a SAFE TTC does not inflate the saved-event ranking.

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

- `timeline_rows`: frame-level risk history
- `events`: saved high-risk moments
- `peak_event`: highest-risk event in the analysis
- `lane_metrics`: scores per lane/object bucket
- `ttc_components`: expansion, flow, and depth TTC details

For events that are kept in the saved-events list, the following RGB views are attached:

- `original_rgb`: original frame
- `depth_rgb`: depth heatmap
- `road_rgb`: road/lane overlay
- `motion_rgb`: optical-flow visualization
- `overlay_rgb`: frame with boxes, lane lines, and risk text

Heavy RGB rendering is deferred. Every frame produces a metadata-only event payload first; the per-frame rendering inputs (frame BGR, flow, depth, lane, road ROI) are stashed in a `_DeferredRender` reference. After the frame's score is compared against the saved-events top-N (and the per-second deduplication window), only events that survive the cut have their RGB views materialized. Frames that never make the saved list pay no rendering cost. Preview ticks render the annotated frame on demand for the JPEG sent over the WebSocket.

Saved events are deduplicated within a 1-second window. If a stronger event appears in the same window, it replaces the previous saved event.

## 19. Visual Overlay (in-frame HUD)

Overlay rendering is implemented in `spectra/analysis/overlay.py`.

It draws:

- Ego lane corridor
- Object bounding boxes coloured by stabilized risk state
- A compact, fixed-width card pinned to the top-left of the frame containing:
  - a coloured `STATE` pill plus the fused `TTC` reading
  - an object · lane subtitle (e.g. `CAR · Same Lane`)
  - four stacked progress bars for `Prox`, `Appr`, `Cross`, and `Conf`

The card has a fixed footprint (≤220 px wide, ~116 px tall) so it does not overflow narrow previews. The four bars mirror the right-side "Tracked Objects" panel one-for-one (`closing_speed` drives both the panel `Approach` bar and the overlay `Appr` bar, and so on), so the HUD reads the same way as the dashboard. The per-component TTC breakdown (`Exp / Flow / Depth`) is no longer drawn on the frame; it is still emitted in the event payload for the dashboard to render.

This overlay is used for live preview and saved event imagery.

## 20. Frontend Role

The frontend does not compute risk. `spectra/web/static/js/controls.js` and related modules visualize backend payloads.

Frontend responsibilities:

- File selection and upload
- WebSocket preview listening
- Timeline and event strip rendering
- Risk banner updates (alert state, TTC, lane)
- "Tracked Objects" panel: list of active tracks with type and TTC, with a detail pane that shows ID / Type / TTC / Lane and four signal bars (Proximity, Approach, Crossing, Confidence)
- "Temporal Analysis" chart: a single red TTC line with a light EMA applied to suppress single-frame jitter, two dashed threshold guides at 3.0s (CAUTION) and 1.0s (DANGER), and a legend strip below explaining the bands. The chart shows TTC for SAFE rows too; only the in-chart category labels were removed because the legend below already names the bands.
- Depth, road, motion, and overlay views
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
7. Track objects across frames.
8. Estimate expansion, flow, and depth TTC for each object.
9. Fuse TTC components with weighted median.
10. Compute ego-lane relationship and crossing risk.
11. Compute closing speed, confidence, and near score.
12. Classify the object as `SAFE`, `CAUTION`, or `DANGER`.
13. Stabilize state transitions (TTC is preserved through SAFE).
14. Select the most critical event.
15. Send timeline, event metadata, and (for saved events) visual outputs to the frontend.

The core design is to avoid relying on a single visual cue. Spectra combines three independent approach signals with lane relevance before declaring risk.
