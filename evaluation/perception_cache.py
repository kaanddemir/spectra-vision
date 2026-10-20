"""Versioned on-disk contract for cached perception outputs.

A cache is a directory holding ``manifest.json`` and ``arrays.npz``. Event
algorithms read caches only; they never run models. Timestamps are decoder
presentation times in seconds, never ``frame_index / fps``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

CACHE_SCHEMA_VERSION = "1.0"

DETECTION_COLUMNS = ("frame", "x1", "y1", "x2", "y2", "score", "class_id", "track_id")
MOTION_COLUMNS = ("tx", "ty", "rotation", "scale", "inlier_ratio", "photometric_change")
PATH_COLUMNS = ("center_x", "half_width", "horizon_y", "confidence")


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    sha256: Optional[str] = None
    licence_ref: Optional[str] = None


@dataclass
class PerceptionCache:
    media_sha256: str
    frame_size: tuple[int, int]  # (width, height) of the analysed frames
    timestamps: np.ndarray  # (N,) float64 seconds, non-decreasing
    detections: np.ndarray  # (M, 8) float64, DETECTION_COLUMNS; track_id = -1 when untracked
    motion: Optional[np.ndarray] = None  # (N, 6) MOTION_COLUMNS, NaN where unavailable
    path: Optional[np.ndarray] = None  # (N, 4) PATH_COLUMNS, NaN where unavailable
    models: tuple[ModelInfo, ...] = ()
    config: dict = field(default_factory=dict)
    class_names: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        self.detections = np.asarray(self.detections, dtype=np.float64).reshape(-1, len(DETECTION_COLUMNS))
        n = len(self.timestamps)
        if n == 0:
            raise ValueError("cache has no frames")
        if np.any(np.diff(self.timestamps) < 0) or not np.all(np.isfinite(self.timestamps)):
            raise ValueError("timestamps must be finite and non-decreasing")
        if len(self.detections):
            frames = self.detections[:, 0]
            if frames.min() < 0 or frames.max() >= n or np.any(frames != np.floor(frames)):
                raise ValueError("detection frame index out of range")
            if np.any(self.detections[:, 3] <= self.detections[:, 1]) or np.any(
                self.detections[:, 4] <= self.detections[:, 2]
            ):
                raise ValueError("detection boxes must satisfy x2 > x1 and y2 > y1")
        for name, arr, cols in (("motion", self.motion, MOTION_COLUMNS), ("path", self.path, PATH_COLUMNS)):
            if arr is not None:
                arr = np.asarray(arr, dtype=np.float64)
                if arr.shape != (n, len(cols)):
                    raise ValueError(f"{name} must have shape ({n}, {len(cols)}), got {arr.shape}")
                setattr(self, name, arr)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()[:16]

    def track(self, track_id: int) -> np.ndarray:
        """Rows of one track sorted by frame."""
        rows = self.detections[self.detections[:, 7] == track_id]
        return rows[np.argsort(rows[:, 0], kind="stable")]

    def track_ids(self) -> list[int]:
        return sorted(int(t) for t in np.unique(self.detections[:, 7]) if t >= 0)

    def save(self, directory: Path) -> str:
        """Write atomically; returns the content hash of the arrays file."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {"timestamps": self.timestamps, "detections": self.detections}
        if self.motion is not None:
            arrays["motion"] = self.motion
        if self.path is not None:
            arrays["path"] = self.path
        tmp = directory / "arrays.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        tmp.replace(directory / "arrays.npz")
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "media_sha256": self.media_sha256,
            "frame_size": list(self.frame_size),
            "columns": {"detections": DETECTION_COLUMNS, "motion": MOTION_COLUMNS, "path": PATH_COLUMNS},
            "models": [m.__dict__ for m in self.models],
            "config": self.config,
            "config_hash": self.config_hash,
            "class_names": {str(k): v for k, v in self.class_names.items()},
            "arrays_sha256": digest,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return digest

    @classmethod
    def load(cls, directory: Path) -> "PerceptionCache":
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(f"unsupported cache schema {manifest.get('schema_version')!r}")
        raw = (directory / "arrays.npz").read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["arrays_sha256"]:
            raise ValueError("arrays.npz does not match manifest hash")
        with np.load(directory / "arrays.npz", allow_pickle=False) as data:
            return cls(
                media_sha256=manifest["media_sha256"],
                frame_size=tuple(manifest["frame_size"]),
                timestamps=data["timestamps"],
                detections=data["detections"],
                motion=data["motion"] if "motion" in data else None,
                path=data["path"] if "path" in data else None,
                models=tuple(ModelInfo(**m) for m in manifest["models"]),
                config=manifest["config"],
                class_names={int(k): v for k, v in manifest["class_names"].items()},
            )
