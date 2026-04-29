"""Gemini REST client for structured hazard narration."""

from __future__ import annotations

import json
from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

RISK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "headline": {
            "type": "string",
            "description": "Short hazard headline in English, max 12 words.",
        },
        "hazard_band": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
            "description": "The final hazard band.",
        },
        "summary": {
            "type": "string",
            "description": "A concise scene-risk summary using the numeric evidence.",
        },
        "collision_risk": {
            "type": "string",
            "description": "Short sentence describing collision risk and TTC if available.",
        },
        "recommended_action": {
            "type": "string",
            "description": "Single-line driving or monitoring recommendation.",
        },
        "uncertainty_note": {
            "type": "string",
            "description": "Must explicitly mention that the values are estimated and uncertain.",
        },
        "key_factors": {
            "type": "array",
            "description": "Main numeric contributors behind the decision.",
            "items": {"type": "string"},
        },
    },
    "required": [
        "headline",
        "hazard_band",
        "summary",
        "collision_risk",
        "recommended_action",
        "uncertainty_note",
        "key_factors",
    ],
}

SYSTEM_INSTRUCTION = (
    "You are a cautious autonomous-driving safety analyst. You receive structured telemetry derived from "
    "classical monocular depth estimation and optical flow. These inputs are approximate and not calibrated. "
    "Do not overclaim. Always preserve the supplied hazard band unless the telemetry is internally contradictory. "
    "Mention uncertainty clearly and keep the response concise."
)


def _extract_response_text(payload: Dict[str, Any]) -> str:
    """Extract the text field from a Gemini generateContent response."""

    candidates = payload.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini response did not contain any candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part.get("text", "") for part in parts if isinstance(part, dict) and "text" in part]
    response_text = "".join(texts).strip()
    if not response_text:
        raise RuntimeError("Gemini response did not contain any text parts.")
    return response_text


def generate_hazard_narrative(
    telemetry: Dict[str, Any],
    api_key: str,
    model: str = "gemini-2.5-flash",
    timeout_sec: float = 20.0,
) -> Dict[str, Any]:
    """Call Gemini with structured telemetry and receive a structured hazard brief."""

    if not api_key.strip():
        raise ValueError("Gemini API key is required.")

    prompt = (
        "Analyze the following hazard telemetry from an ego-vehicle view. "
        "Keep the answer grounded in the numbers. "
        "If motion is unavailable, say the current video frame has no reliable motion signal.\n\n"
        f"Telemetry JSON:\n{json.dumps(telemetry, ensure_ascii=True)}"
    )

    request_body = {
        "system_instruction": {
            "parts": [
                {
                    "text": SYSTEM_INSTRUCTION,
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 280,
            "responseMimeType": "application/json",
            "responseJsonSchema": RISK_SCHEMA,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    request = Request(
        GEMINI_ENDPOINT.format(model=model),
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API HTTP {exc.code}: {message}") from exc
    except URLError as exc:
        raise RuntimeError(f"Gemini API connection failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini API returned invalid JSON: {exc}") from exc

    response_text = _extract_response_text(payload)
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini structured output was not valid JSON: {response_text}") from exc

    return result
