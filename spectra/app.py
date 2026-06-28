"""FastAPI UI and API for lane-relative spatial video risk analysis."""

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

from spectra.analysis.video import analyze_spatial_video


_PREVIEW_QUEUES: dict[str, asyncio.Queue] = {}
_PREVIEW_QUEUE_MAX = 24
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB

VIDEO_TYPES = {"mp4", "mov", "avi", "mkv", "m4v"}
DEPTH_EVERY_OPTIONS = (1, 2, 3, 5, 10, 15)
DETECT_EVERY_OPTIONS = (1, 2, 3, 5, 10)
LANE_EVERY_OPTIONS = (1, 2, 3, 5, 10)
FLOW_EVERY_OPTIONS = (1, 2, 3, 5, 10)
RESIZE_MAX_SIDE_OPTIONS = (128, 256, 384, 512, 768, 1024)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web" / "static"

app = FastAPI(title="Spectra", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _image_to_jpeg_bytes(image: Any, quality: int = 85) -> bytes:
    """Encode an RGB array as JPEG bytes."""

    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(array[:, :, :3] if array.ndim == 3 else array)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _image_data_uri(image: Any) -> str:
    encoded = base64.b64encode(_image_to_jpeg_bytes(image)).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


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
    image_ref: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "frameIndex": event.get("frame_index"),
        "timestampSec": event.get("timestamp_sec"),
        "stabilizedRiskState": event.get("stabilized_risk_state"),
        "primaryObjectId": event.get("primary_object_id"),
        "primaryRiskScore": event.get("primary_risk_score"),
        "primaryLane": event.get("primary_lane"),
        "trafficLight": event.get("traffic_light_state"),
        "laneGeometry": event.get("laneGeometry"),
        "objects": event.get("objects") or [],
    }
    if image_ref is not None:
        payload["imageRef"] = image_ref
    return payload


def _is_same_event(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if not left or not right:
        return False
    return (
        left.get("frame_index") == right.get("frame_index")
        and left.get("timestamp_sec") == right.get("timestamp_sec")
    )


_IMAGE_FIELDS: tuple[tuple[str, str], ...] = (
    ("original", "original_rgb"),
    ("blend", "overlay_rgb"),
)


def _pull_event_images(event: dict[str, Any], images: dict[str, dict[str, str]]) -> str | None:
    """Move RGB images from the event payload into the shared images dict.

    Returns the imageRef key, or None if the event has no images attached.
    Frame index is unique per saved event (deduplicated upstream), so it's
    a stable lookup key.
    """

    img_payload = _event_image_payload(event, _IMAGE_FIELDS)
    if not img_payload:
        return None
    ref = f"f{event.get('frame_index')}"
    images[ref] = img_payload
    return ref


def _serialize_result(result: dict[str, Any], *, elapsed_sec: float, source_name: str) -> dict[str, Any]:
    peak_event = result.get("peak_event")
    images: dict[str, dict[str, str]] = {}

    peak_ref = _pull_event_images(peak_event, images) if peak_event else None
    other_events = [
        event for event in (result.get("events") or []) if not _is_same_event(event, peak_event)
    ]
    serialized_events = [
        _serialize_event(event, image_ref=_pull_event_images(event, images))
        for event in other_events
    ]

    payload: dict[str, Any] = {
        "schemaVersion": 4,
        "metadata": {
            "sourceName": source_name,
            "fps": result.get("fps"),
            "frameCount": result.get("frame_count"),
            "processedFrames": result.get("processed_frames"),
            "frameWidth": result.get("frame_width"),
            "frameHeight": result.get("frame_height"),
            "elapsedSec": round(elapsed_sec, 3),
        },
        "frames": result.get("frames") or [],
        "peakEvent": None if peak_event is None else _serialize_event(peak_event, image_ref=peak_ref),
        "events": serialized_events,
        "images": images,
        "performance_summary": result.get("performance_summary") or {},
        "performance_logs": result.get("performance_logs") or [],
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


def _nearest_allowed(value: int, options: tuple[int, ...]) -> int:
    return min(options, key=lambda option: (abs(option - value), option))


def _form_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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
    max_saved_events: int = Form(20),
    resize_max_side: int = Form(512),
    depth_every: int = Form(10),
    adaptive_depth: str = Form("1"),
    detect_every: int = Form(3),
    lane_every: int = Form(3),
    flow_every: int = Form(1),
    start_sec: float = Form(0.0),
    end_sec: float = Form(0.0),
    session_id: str = Form(""),
) -> dict[str, Any]:
    if mode.strip().lower() != "video":
        raise HTTPException(status_code=400, detail="Only video analysis is supported.")
    _validate_video_upload(file)

    upload_bytes = await file.read(_MAX_UPLOAD_BYTES + 1)
    if not upload_bytes:
        raise HTTPException(status_code=400, detail="Upload is empty.")
    if len(upload_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

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
                analyze_spatial_video,
                video_path=source_path,
                max_processed_frames=min(2000, max(1, int(max_processed_frames))),
                max_saved_events=min(50, max(1, int(max_saved_events))),
                resize_max_side=_nearest_allowed(int(resize_max_side), RESIZE_MAX_SIDE_OPTIONS),
                depth_every=_nearest_allowed(int(depth_every), DEPTH_EVERY_OPTIONS),
                adaptive_depth=_form_bool(adaptive_depth),
                detect_every=_nearest_allowed(int(detect_every), DETECT_EVERY_OPTIONS),
                lane_every=_nearest_allowed(int(lane_every), LANE_EVERY_OPTIONS),
                flow_every=_nearest_allowed(int(flow_every), FLOW_EVERY_OPTIONS),
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
