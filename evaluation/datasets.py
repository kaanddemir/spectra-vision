"""Manifest-based, read-only clip access; independent of dataset and model choice."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spectra.engine.video import file_sha256


class ClipMetadata(BaseModel):
    # Preserve provenance extensions rather than discarding dataset-specific fields.
    model_config = ConfigDict(extra="allow", frozen=True)
    id: str = Field(min_length=1)
    filename: str | None = Field(default=None, min_length=1)
    bytes: int = Field(gt=0, strict=True)
    role: str = Field(min_length=1)
    licence: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sha1: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    assumed_fps: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    split: Literal["dev", "validation", "test"] | None = None

    @model_validator(mode="after")
    def _pinned(self):
        if self.filename is not None and not (self.sha256 or self.sha1):
            raise ValueError("a local clip requires a pinned sha256 or sha1")
        if self.filename is not None:
            path = Path(self.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in self.filename or self.filename == ".":
                raise ValueError("filename must be a relative path inside the dataset root")
        return self


class Manifest(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    manifest_version: Literal["1.0"] = "1.0"
    clips: list[ClipMetadata] = Field(min_length=1)
    total_bytes: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def _consistent(self):
        ids = [c.id for c in self.clips]
        if len(ids) != len(set(ids)):
            raise ValueError("clip ids must be unique")
        names = [c.filename for c in self.clips if c.filename is not None]
        if len(names) != len(set(names)):
            raise ValueError("clip filenames must be unique")
        if self.total_bytes is not None and self.total_bytes != sum(c.bytes for c in self.clips):
            raise ValueError("total_bytes does not match the clip sizes")
        return self


@dataclass(frozen=True)
class Clip:
    path: Path
    metadata: ClipMetadata


def load_manifest(path: Path | str) -> Manifest:
    """Validate a downloaded or proposed manifest without touching media files."""
    return Manifest.model_validate_json(Path(path).read_text())


def clips(manifest: Manifest, root: Path | str, verify: bool = True) -> Iterator[Clip]:
    """Yield confined existing files and provenance; verify sizes and strongest hashes.

    Proposed manifests may omit filenames, but clip access requires one. verify=False
    skips only size/hash checks, never path confinement or the existence check.
    assumed_fps is metadata for callers, never inferred from a filename or dataset id.
    """
    root = Path(root).resolve()
    for meta in manifest.clips:
        if meta.filename is None:
            raise ValueError(f"{meta.id}: local filename not assigned")
        path = (root / meta.filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"{meta.id}: file escapes the dataset root")
        if not path.is_file():
            raise FileNotFoundError(f"{meta.id}: source clip missing: {path}")
        if verify:
            if path.stat().st_size != meta.bytes:
                raise ValueError(f"{meta.id}: size mismatch")
            if meta.sha256:
                actual, expected = file_sha256(path), meta.sha256
            else:
                with path.open("rb") as f:
                    actual = hashlib.file_digest(f, "sha1").hexdigest()
                expected = meta.sha1
            if actual != expected:
                raise ValueError(f"{meta.id}: hash mismatch")
        yield Clip(path, meta)
