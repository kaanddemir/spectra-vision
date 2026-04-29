"""Async Gemini analyst triggered only by DANGER events."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from typing import Callable

from depth_project.gemini_client import generate_hazard_narrative

from .risk_calculator import RiskEvent


class GeminiDangerAnalyst:
    """Non-blocking Gemini wrapper with cooldown and single-flight behavior."""

    def __init__(
        self,
        api_key: str | None,
        model: str = "gemini-2.5-flash",
        enabled: bool = False,
        cooldown_sec: float = 3.0,
        timeout_sec: float = 8.0,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.model = model
        self.enabled = enabled and bool(self.api_key)
        self.cooldown_sec = cooldown_sec
        self.timeout_sec = timeout_sec
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending: Future[str] | None = None
        self.last_submit_at = 0.0
        self.last_message: str | None = None
        self.last_error: str | None = None

    def submit_if_needed(
        self,
        event: RiskEvent,
        callback: Callable[[str], None] | None = None,
    ) -> Future[str] | None:
        if not self.enabled or event.state != "DANGER":
            return None
        if self.pending is not None and not self.pending.done():
            return None
        now = time.monotonic()
        if now - self.last_submit_at < self.cooldown_sec:
            return None

        self.last_submit_at = now
        payload = self._payload(event)
        self.pending = self.executor.submit(self._call_gemini, payload)
        self.pending.add_done_callback(lambda future: self._handle_done(future, callback))
        return self.pending

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def drain(self, timeout_sec: float = 0.0) -> str | None:
        """Collect a pending Gemini result if it finishes within the timeout."""

        if self.pending is None:
            return self.last_message
        try:
            self.last_message = self.pending.result(timeout=timeout_sec)
            self.last_error = None
        except TimeoutError:
            return self.last_message
        except Exception as exc:
            self.last_error = str(exc)
        return self.last_message

    def _payload(self, event: RiskEvent) -> dict[str, object]:
        return {
            "media_type": "video",
            "frame_index": event.frame_index,
            "timestamp_sec": event.timestamp_sec,
            "hazard_score": 1.0,
            "hazard_band": "critical",
            "primary_zone": event.zone,
            "estimated_ttc_sec": event.ttc_sec,
            "confidence_pct": round(event.confidence * 100.0, 1),
            "uncertainty_pct": round((1.0 - event.confidence) * 100.0, 1),
            "reasons": [
                event.reason,
                f"{event.object_type} moving from {event.direction}",
            ],
            "zone_metrics": [
                {
                    "zone": event.zone,
                    "object_type": event.object_type,
                    "direction_hint": event.direction,
                    "near_score": event.near_score,
                    "velocity_magnitude": event.velocity_magnitude,
                    "closing_speed": event.closing_speed,
                    "estimated_ttc_sec": event.ttc_sec,
                }
            ],
            "note": "Generate one short driving-safety warning. Values are visual estimates.",
        }

    def _call_gemini(self, payload: dict[str, object]) -> str:
        result = generate_hazard_narrative(
            payload,
            api_key=self.api_key,
            model=self.model,
            timeout_sec=self.timeout_sec,
        )
        return str(
            result.get("collision_risk")
            or result.get("headline")
            or result.get("summary")
            or "Danger detected."
        )

    def _handle_done(self, future: Future[str], callback: Callable[[str], None] | None) -> None:
        try:
            self.last_message = future.result()
            self.last_error = None
            if callback is not None:
                callback(self.last_message)
        except Exception as exc:
            self.last_error = str(exc)
