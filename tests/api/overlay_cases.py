"""Authoritative overlay lookup cases, produced by ``spectra.evidence.overlays.frame_at``.

The JSON file next to this module is the shared reference for any other implementation of the lookup
(such as the application's TypeScript player). Regenerate it after a deliberate rule change with
``python -m tests.api.overlay_cases``; ``test_overlay_cases`` fails while it is out of date.
"""
from __future__ import annotations

import json
from pathlib import Path

from spectra.evidence.contracts import OverlayFrame
from spectra.evidence.overlays import frame_at

CASES_PATH = Path(__file__).with_name("fixtures") / "overlay_lookup_cases.json"
MAX_AGE_S = 0.25


def _frames(times: list[float]) -> list[OverlayFrame]:
    """Valid until the next analysed frame, or at most MAX_AGE_S later, as ``overlay`` computes it."""
    out = []
    for k, t in enumerate(times):
        until = t + MAX_AGE_S
        if k + 1 < len(times):
            until = min(until, times[k + 1])
        out.append(OverlayFrame(t=t, valid_until=round(until, 6)))
    return out


SCENARIOS = {
    "regular_10hz": [1.0, 1.1, 1.2, 1.3],
    "variable_rate_with_gap": [0.3, 0.34, 0.38, 1.0, 1.04, 2.5],
    "single_frame": [5.0],
    "stream_start_at_zero": [0.0, 0.1],
}


def _queries(times: list[float]) -> list[float]:
    probes = {round(times[0] - 0.1, 6), round(times[-1] + MAX_AGE_S + 0.1, 6)}
    for k, t in enumerate(times):
        probes |= {t, round(t + 1e-7, 7), round(t - 1e-5, 6), round(t + 0.05, 6), round(t + MAX_AGE_S, 6),
                   round(t + MAX_AGE_S + 1e-3, 6)}
        if k + 1 < len(times):
            probes |= {round((t + times[k + 1]) / 2, 6), round(times[k + 1] - 1e-4, 6)}
    return sorted(q for q in probes if q >= 0)


def build_cases() -> dict:
    scenarios = []
    for name, times in SCENARIOS.items():
        frames = _frames(times)
        lookups = []
        for t in _queries(times):
            found = frame_at(frames, t)
            lookups.append({"t": t, "frame": None if found is None else frames.index(found)})
        scenarios.append({"name": name, "frames": [{"t": f.t, "valid_until": f.valid_until} for f in frames],
                          "lookups": lookups})
    return {"description": "frame_at(frames, t): index of the overlay frame to draw at display time t, or null "
                           "for 'not analysed here'. Frames are never interpolated.",
            "epsilon": 1e-6, "max_age_s": MAX_AGE_S, "scenarios": scenarios}


if __name__ == "__main__":
    CASES_PATH.parent.mkdir(exist_ok=True)
    CASES_PATH.write_text(json.dumps(build_cases(), indent=1) + "\n")
