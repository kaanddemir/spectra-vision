"""FastAPI UI and API for zone-based video risk analysis."""

from __future__ import annotations

import base64
import io
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from zone_risk.pipeline.api import analyze_zone_video


VIDEO_TYPES = {"mp4", "mov", "avi", "mkv", "m4v"}

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web" / "static"

app = FastAPI(title="Spectra", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _image_to_png_bytes(image: Any) -> bytes:
    """Encode an RGB or grayscale array as PNG bytes."""

    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        pil_image = Image.fromarray(array)
    else:
        pil_image = Image.fromarray(array[:, :, :3])

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_data_uri(image: Any) -> str:
    encoded = base64.b64encode(_image_to_png_bytes(image)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _sanitize_for_json(value: Any) -> Any:
    """Remove display arrays and normalize NumPy values for telemetry export."""

    if isinstance(value, dict):
        return {
            key: _sanitize_for_json(item)
            for key, item in value.items()
            if not key.endswith("_rgb")
        }
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return "<array omitted>"
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _event_image_payload(event: dict[str, Any], fields: Iterable[tuple[str, str]]) -> dict[str, str]:
    images: dict[str, str] = {}
    for payload_key, event_key in fields:
        image = event.get(event_key)
        if image is not None:
            images[payload_key] = _image_data_uri(image)
    return images


def _serialize_event(
    event: dict[str, Any],
    *,
    include_images: bool = True,
    image_fields: Iterable[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    payload = {
        "frameIndex": event.get("frame_index"),
        "timestampSec": event.get("timestamp_sec"),
        "hazardScore": event.get("hazard_score"),
        "hazardBand": event.get("hazard_band"),
        "primaryZone": event.get("primary_zone"),
        "estimatedTtcSec": event.get("estimated_ttc_sec"),
        "uncertaintyPct": event.get("uncertainty_pct"),
        "summary": event.get("heuristic_summary"),
        "reasons": event.get("reasons", []),
        "zoneMetrics": event.get("zone_metrics", []),
        "riskState": event.get("risk_state"),
        "objectType": event.get("object_type"),
        "approach": event.get("approach"),
        "lane": event.get("lane"),
        "bbox": event.get("bbox"),
        "nearScore": event.get("near_score"),
        "closingSpeed": event.get("closing_speed"),
        "velocityMagnitude": event.get("velocity_magnitude"),

        "telemetry": _sanitize_for_json(event.get("payload", {})),
    }

    if include_images:
        fields = image_fields or (
            ("original", "original_rgb"),
            ("depth", "depth_rgb"),
            ("segmentation", "segmentation_rgb"),
            ("motion", "motion_rgb"),
            ("blend", "overlay_rgb"),
        )
        payload["images"] = _event_image_payload(event, fields)

    return payload


def _serialize_result(result: dict[str, Any], *, elapsed_sec: float, source_name: str) -> dict[str, Any]:
    base_payload: dict[str, Any] = {
        "mediaType": result.get("media_type"),
        "sourceName": source_name,
        "summary": result.get("summary"),
        "elapsedSec": round(elapsed_sec, 3),
        "telemetry": _sanitize_for_json(result),
    }

    peak_event = result.get("peak_event")
    base_payload.update(
        {
            "fps": result.get("fps"),
            "frameCount": result.get("frame_count"),
            "processedFrames": result.get("processed_frames"),
            "sampledFrames": result.get("sampled_frames"),

            "timelineRows": result.get("timeline_rows", []),
            "peakEvent": None if peak_event is None else _serialize_event(peak_event),
            "events": [
                _serialize_event(
                    event,
                    include_images=True,
                    image_fields=(("blend", "overlay_rgb"), ("original", "original_rgb")),
                )
                for event in result.get("events", [])
            ],
        }
    )
    return base_payload


def _extension(filename: str | None) -> str:
    return Path(filename or "").suffix.lower().lstrip(".")


def _validate_video_upload(upload: UploadFile) -> None:
    ext = _extension(upload.filename)
    if ext not in VIDEO_TYPES:
        supported_list = ", ".join(sorted(VIDEO_TYPES))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '.{ext}'. Supported formats: {supported_list}.",
        )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.head("/", include_in_schema=False)
def index_head() -> Response:
    return Response(status_code=200, media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.head("/api/health", include_in_schema=False)
def health_head() -> Response:
    return Response(status_code=200, media_type="application/json")


@app.post("/api/analyze")
def analyze_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("video"),
    max_processed_frames: int = Form(180),
    max_saved_events: int = Form(6),
    resize_max_side: int = Form(640),
    depth_every: int = Form(10),
) -> dict[str, Any]:
    if mode.strip().lower() != "video":
        raise HTTPException(status_code=400, detail="Only video analysis is supported.")
    _validate_video_upload(file)

    upload_bytes = file.file.read()
    if not upload_bytes:
        raise HTTPException(status_code=400, detail="Upload is empty.")

    source_name = Path(file.filename or f"upload.{_extension(file.filename)}").name

    start_time = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / source_name
            source_path.write_bytes(upload_bytes)

            result = analyze_zone_video(
                video_path=source_path,
                max_processed_frames=max(1, int(max_processed_frames)),
                max_saved_events=max(1, int(max_saved_events)),
                resize_max_side=max(128, int(resize_max_side)),
                depth_every=max(1, int(depth_every)),
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    elapsed_sec = time.perf_counter() - start_time
    return _serialize_result(result, elapsed_sec=elapsed_sec, source_name=source_name)
