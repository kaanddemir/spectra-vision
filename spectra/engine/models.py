"""Verified local runtime model registry. Never downloads or loads model runtimes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spectra.engine.video import file_sha256
from spectra.engine.worker import DetectorSpec, SegmenterSpec


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["detector", "segmenter"]
    file: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input: list[int]
    licence: str = Field(min_length=1)
    export_script: str = Field(min_length=1)
    source_checkpoint: str = Field(min_length=1)
    source_url: str | None = Field(default=None, pattern=r"^https://")
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


def default_specs(models_dir: Path | str) -> tuple[DetectorSpec, SegmenterSpec]:
    """Resolve the registry's two runtime models, checking paths, shape and SHA256.

    Paths are relative to models_dir. Comparison exports remain outside this
    default runtime selection. An invalid or missing model refuses the shortcut.
    """
    root = Path(models_dir).resolve()
    document = json.loads((root / "models.json").read_text())
    if not isinstance(document, dict) or set(document) != {"models"} or not isinstance(document["models"], list):
        raise ValueError("registry must contain a models list")
    entries = {}
    for row in document["models"]:
        entry = ModelEntry.model_validate(row)
        if entry.role in entries:
            raise ValueError(f"duplicate model role: {entry.role}")
        relative = Path(entry.file)
        path = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
            raise ValueError(f"model path escapes registry directory: {entry.file}")
        shape = entry.input
        if len(shape) != 4 or shape[:2] != [1, 3] or min(shape) <= 0:
            raise ValueError(f"invalid input shape for {entry.role}")
        if entry.role == "detector" and shape[2] != shape[3]:
            raise ValueError("detector input must be square")
        if entry.role == "segmenter" and shape != [1, 3, 384, 512]:
            raise ValueError("segmenter input must be [1, 3, 384, 512]")
        if not path.is_file():
            raise FileNotFoundError(f"missing {entry.role} model: {path}")
        if file_sha256(path) != entry.sha256:
            raise ValueError(f"sha256 mismatch for {entry.role} model: {path}")
        entries[entry.role] = (entry, path)
    if set(entries) != {"detector", "segmenter"}:
        raise ValueError("registry requires one detector and one segmenter")
    detector, detector_path = entries["detector"]
    segmenter, segmenter_path = entries["segmenter"]
    return (DetectorSpec(str(detector_path), detector.sha256, input_size=detector.input[2]),
            SegmenterSpec(str(segmenter_path), segmenter.sha256))
