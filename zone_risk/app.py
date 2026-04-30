"""FastAPI UI and API for zone-based video risk analysis."""

from __future__ import annotations

import asyncio
import base64
import io
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from zone_risk.pipeline.api import analyze_zone_video


_PREVIEW_QUEUES: dict[str, asyncio.Queue] = {}
_PREVIEW_QUEUE_MAX = 24


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
        "riskScore": event.get("risk_score", event.get("hazard_score")),
        "riskBand": event.get("risk_band", event.get("hazard_band")),
        "zone": event.get("primary_zone"),
        "ttcSec": event.get("estimated_ttc_sec"),
        "uncertaintyPct": event.get("uncertainty_pct"),
        "summary": event.get("heuristic_summary"),
        "reasons": event.get("reasons", []),
        "zoneMetrics": event.get("zone_metrics", []),
        "riskState": event.get("risk_state"),
        "objectType": event.get("object_type"),
        "approach": event.get("approach"),
        "bbox": event.get("bbox"),
        "nearScore": event.get("near_score"),
        "closingSpeed": event.get("closing_speed"),
        "velocityMagnitude": event.get("velocity_magnitude"),
    }

    if include_images:
        fields = image_fields or (
            ("original", "original_rgb"),
            ("depth", "depth_rgb"),
            ("segmentation", "segmentation_rgb"),
            ("road", "road_rgb"),
            ("motion", "motion_rgb"),
            ("blend", "overlay_rgb"),
        )
        payload["images"] = _event_image_payload(event, fields)

    return payload


def _is_same_event(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if not left or not right:
        return False
    return (
        left.get("frame_index") == right.get("frame_index")
        and left.get("timestamp_sec") == right.get("timestamp_sec")
    )


def _serialize_result(result: dict[str, Any], *, elapsed_sec: float, source_name: str) -> dict[str, Any]:
    peak_event = result.get("peak_event")
    payload: dict[str, Any] = {
        "summary": result.get("summary"),
        "metadata": {
            "mediaType": result.get("media_type"),
            "sourceName": source_name,
            "fps": result.get("fps"),
            "frameCount": result.get("frame_count"),
            "processedFrames": result.get("processed_frames"),
            "sampledFrames": result.get("sampled_frames"),
            "elapsedSec": round(elapsed_sec, 3),
        },
        "timelineRows": result.get("timeline_rows", []),
        "peakEvent": None if peak_event is None else _serialize_event(peak_event),
        "events": [
            _serialize_event(
                event,
                include_images=True,
                image_fields=(("blend", "overlay_rgb"),),
            )
            for event in result.get("events", [])
            if not _is_same_event(event, peak_event)
        ],
    }
    return {"payload": payload}


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


def _get_or_create_preview_queue(session_id: str) -> asyncio.Queue:
    queue = _PREVIEW_QUEUES.get(session_id)
    if queue is None:
        queue = asyncio.Queue(maxsize=_PREVIEW_QUEUE_MAX)
        _PREVIEW_QUEUES[session_id] = queue
    return queue


@app.websocket("/ws/preview/{session_id}")
async def preview_websocket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    queue = _get_or_create_preview_queue(session_id)
    try:
        while True:
            message = await queue.get()
            if message is None:
                try:
                    await websocket.send_json({"type": "done"})
                except Exception:
                    pass
                break
            await websocket.send_json(message)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _PREVIEW_QUEUES.pop(session_id, None)
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/api/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("video"),
    max_processed_frames: int = Form(180),
    max_saved_events: int = Form(6),
    resize_max_side: int = Form(640),
    depth_every: int = Form(10),
    enable_road_roi: bool = Form(False),
    start_sec: float = Form(0.0),
    end_sec: float = Form(0.0),
    session_id: str = Form(""),
) -> dict[str, Any]:
    if mode.strip().lower() != "video":
        raise HTTPException(status_code=400, detail="Only video analysis is supported.")
    _validate_video_upload(file)

    upload_bytes = await file.read()
    if not upload_bytes:
        raise HTTPException(status_code=400, detail="Upload is empty.")

    source_name = Path(file.filename or f"upload.{_extension(file.filename)}").name

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue | None = None
    session_key = session_id.strip() if isinstance(session_id, str) else ""
    if session_key:
        queue = _get_or_create_preview_queue(session_key)

    def _push(payload: Any) -> None:
        if queue is None:
            return
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass

    def progress_callback(payload: dict[str, Any]) -> None:
        if queue is None:
            return
        try:
            loop.call_soon_threadsafe(_push, payload)
        except RuntimeError:
            pass

    start_time = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / source_name
            source_path.write_bytes(upload_bytes)

            result = await asyncio.to_thread(
                analyze_zone_video,
                video_path=source_path,
                max_processed_frames=max(1, int(max_processed_frames)),
                max_saved_events=max(1, int(max_saved_events)),
                resize_max_side=max(128, int(resize_max_side)),
                depth_every=max(1, int(depth_every)),
                enable_road_roi=bool(enable_road_roi),
                start_sec=float(start_sec),
                end_sec=float(end_sec) if float(end_sec) > 0 else None,
                progress_callback=progress_callback if queue is not None else None,
            )
    except HTTPException:
        if queue is not None:
            try:
                loop.call_soon_threadsafe(_push, None)
            except RuntimeError:
                pass
        raise
    except Exception as exc:
        if queue is not None:
            try:
                loop.call_soon_threadsafe(_push, None)
            except RuntimeError:
                pass
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    if queue is not None:
        try:
            loop.call_soon_threadsafe(_push, None)
        except RuntimeError:
            pass

    elapsed_sec = time.perf_counter() - start_time
    return _serialize_result(result, elapsed_sec=elapsed_sec, source_name=source_name)
