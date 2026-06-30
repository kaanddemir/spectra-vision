# Spectra Agent Guide

This file is a concise maintainer guide for Claude/Codex agents working in this repository. Keep it aligned with the current code before making behavioral changes.

## Project Shape

Spectra is a local FastAPI application for lane-relative video risk analysis. The user uploads forward-facing driving footage in the browser UI; the backend processes frames with local vision models and returns a schema v5 payload for the frontend.

Core paths:

- `spectra/app.py` - FastAPI routes, static UI serving, upload validation, preview WebSocket queues, and client serialization.
- `spectra/analysis/video.py` - video loading, frame loop, model scheduling, progress callbacks, event ranking, and deferred rendering.
- `spectra/analysis/risk.py` - per-object risk scoring, TTC fusion, confidence fields, hysteresis, and sensitivity bands.
- `spectra/analysis/tracking.py` - IoU-based object tracking, track propagation, coasting, and history.
- `spectra/analysis/overlay.py` - lane corridor, object boxes, TTC labels, risk card, and advisory overlay drawing.
- `spectra/vision/*` - preprocessing, YOLO detection, UFLDv2 lane detection, metric depth, DIS optical flow, road filtering, brake-light cue, and traffic-light cue.
- `spectra/web/static/*` - browser UI, controls, timeline, preview panels, and the `/how-it-works` page.
- `tests/*` - API contract, backend failure, pipeline, tracking, risk, vision cue, and UI smoke tests.

## Runtime Truths

Required local model files:

- `models/depth_anything_v2_metric_vkitti_vits.onnx`
- `models/ufld_v2_culane_r18.onnx`
- `models/yolov8n.pt`

Model setup helpers:

- `.venv/bin/python scripts/download_depth_model.py`
- `.venv/bin/python scripts/download_lanenet_model.py`

There is no neural optical-flow model. Motion uses OpenCV DIS dense optical flow with ego-motion compensation in `spectra/vision/motion.py`.

Depth and UFLDv2 use ONNX Runtime. The provider setup prefers CoreML on macOS when available and falls back to CPU. YOLO runs through Ultralytics/PyTorch and chooses the best available local device.

Missing or unloadable Depth Anything, UFLDv2, or YOLO backends are hard failures. Do not silently downgrade to a fake or narrative-only pipeline.

## Public Interfaces

Routes:

- `GET /` - main UI
- `GET /how-it-works` - static explainer page
- `GET /api/health` - health check
- `POST /api/analyze` - video analysis
- `WS /ws/preview/{session_id}` - live preview stream

`POST /api/analyze` accepts multipart form data with these current controls:

- `file`
- `mode`
- `max_processed_frames`
- `max_saved_events`
- `resize_max_side`
- `depth_every`
- `adaptive_depth`
- `detect_every`
- `lane_every`
- `flow_every`
- `sensitivity`
- `start_sec`
- `end_sec`
- `start_frame`
- `end_frame`
- `session_id`

The response is always wrapped as:

```json
{
  "payload": {
    "schemaVersion": 5,
    "metadata": {},
    "frames": [],
    "peakEvent": null,
    "events": [],
    "images": {},
    "performance": {}
  }
}
```

Client-facing JSON is camelCase. Keep internal snake_case diagnostics out of serialized event rows. Images belong under top-level `payload.images`; frame/event rows use `imageRef` instead of embedding RGB arrays or data URIs directly.

The old `timelineRows` contract is not current. Use `payload.frames`, `payload.peakEvent`, and `payload.events`.

## Pipeline Summary

1. `spectra/app.py` validates the upload, writes it to a temp path, and calls `analyze_spatial_video`.
2. `VideoLoader` reads frames with OpenCV.
3. `preprocessing.py` resizes while preserving aspect ratio and prepares BGR, RGB, and grayscale views.
4. UFLDv2 estimates the ego-lane corridor on the configured `lane_every` interval; cached/smoothed geometry is used between runs.
5. DIS optical flow runs on the configured `flow_every` interval; skipped frames reuse the latest motion field.
6. Depth Anything V2 Metric VKITTI refreshes on the configured `depth_every` interval and, when enabled, on motion spikes.
7. YOLO detects road participants on the configured `detect_every` interval.
8. Traffic-light detections are split out as frame-level advisories and are not collision participants.
9. Road/lane relevance filtering removes detections that should not enter the risk tracker.
10. The IoU tracker links detections to active tracks and propagates tracks through skipped detection frames.
11. Risk scoring fuses metric depth TTC, bbox expansion TTC, radial-flow TTC, lane relationship, proximity, brake-light cues, and confidence.
12. Events are deduplicated, ranked, trimmed, serialized, and rendered only when needed.

## Development Guardrails

- Preserve the schema v5 response shape unless intentionally changing the public API and tests.
- Keep frontend contract fields camelCase.
- Keep backend-only ranking/dedup fields internal.
- Do not make traffic lights tracked collision objects; they are advisory only.
- Do not add a neural optical-flow dependency unless the runtime and docs are changed together.
- Preserve the sampling option clamps in `spectra/app.py` unless changing API behavior intentionally.
- Keep model load failures visible and hard; tests expect backend failures to surface.
- Avoid broad refactors while changing risk math, serialization, or UI controls. These areas are tightly coupled through tests and browser state.
- When touching static UI contracts, check `tests/test_static_ui_smoke.py` for required names and assets.

## Common Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
./start.sh
```

Manual server:

```bash
python -m uvicorn spectra.app:app --host localhost --port 8000 --reload
```

Stop the app:

```bash
./stop.sh
```

Run all tests:

```bash
pytest
```

Focused contract checks:

```bash
pytest tests/test_api_contract.py tests/test_static_ui_smoke.py
```

## Verification Checklist

Use the narrowest useful tests for the touched area:

- API serialization or endpoint parameters: `pytest tests/test_api_contract.py`
- Static UI controls/routes/assets: `pytest tests/test_static_ui_smoke.py`
- Model load and backend failure behavior: `pytest tests/test_backend_failures.py`
- Pipeline behavior: `pytest tests/test_video_pipeline_smoke.py tests/test_risk_smoke.py`
- Tracking changes: `pytest tests/test_tracking_smoke.py`
- Vision cue changes: `pytest tests/test_vision_cues.py`

For documentation-only edits, read `README.md` and `CLAUDE.md` end to end and run the two focused contract/UI tests when practical.
