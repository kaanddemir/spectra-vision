"""Paths for new derived outputs; existing research reports are never moved or replaced."""
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re


def _component(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in (".", ".."):
        raise ValueError("output identifier must be a safe single path component")
    return value


def write_prediction(root: Path | str, dataset_id: str, analysis_key: str, clip_id: str, prediction: dict) -> Path:
    """Create <dataset>/<analysis_key>/<clip>.json exclusively, never overwriting."""
    if not re.fullmatch(r"[0-9a-f]{64}", analysis_key):
        raise ValueError("analysis_key must be a sha256 digest")
    if prediction.get("analysis_key", analysis_key) != analysis_key:
        raise ValueError("prediction analysis_key does not match the output path")
    root = Path(root).resolve()
    path = root / _component(dataset_id) / analysis_key / f"{_component(clip_id)}.json"
    if not path.resolve().is_relative_to(root):
        raise ValueError("prediction path escapes the output root")
    text = json.dumps(prediction, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(text)
    return path


def new_report_dir(root: Path | str, name: str, day: date | None = None) -> Path:
    """Reserve a dated report directory (UTC date by default); collisions refuse reuse."""
    day = day or datetime.now(timezone.utc).date()
    path = Path(root) / f"{day.isoformat()}_{_component(name)}"
    path.mkdir(parents=True, exist_ok=False)
    return path
