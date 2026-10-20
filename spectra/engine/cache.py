"""Versioned on-disk perception cache.

One directory per perception key under the cache root::

    <root>/<key>/manifest.json      schema version, key and sha256 of every other file
    <root>/<key>/perception.json    canonical JSON of a PerceptionResult without its frames
    <root>/<key>/frames.jsonl       one canonical JSON FrameRecord per line, in order
    <root>/<key>/masks/*.png        lossless class-index masks referenced by the frames

The key hashes the video content hash, timestamp handling, run configuration, engine and decoder versions
and the identity of every perception component (model, version, checkpoint hash, parameters, interval),
so a change to any of them is a miss, never a reuse. An entry is written into a staging directory and
renamed into place only when complete; a reader still verifies every file against the manifest and the
key against the content, and reports why an entry was rejected. Frames are streamed to disk while a run
progresses and read back lazily by byte offset, so neither writing nor replaying an entry holds every
frame record in memory. Entries are derived data and may be
deleted at any time; they are never review decisions or project data.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from spectra.engine.contracts import (SCHEMA_VERSION, FrameRecord, FrameSequence, PerceptionResult, Provenance,
                                      RunConfig, SegmentationRef, VideoInfo, canonical_json, check_frames, digest)

from spectra.engine.video import file_sha256

STAGING = ".staging"
FRAMES = "frames.jsonl"


def perception_key(video: VideoInfo, config: RunConfig, provenance: Provenance) -> str:
    return digest({
        "schema_version": SCHEMA_VERSION,
        "video_sha256": video.sha256,
        "timestamps": {"source": video.timestamp_source.value, "assumed_fps": video.assumed_fps},
        "config": config.model_dump(mode="json"),
        "provenance": provenance.model_dump(mode="json"),
    })


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()



class StoredFrames(FrameSequence):
    """Frame records of one cache entry, parsed from ``frames.jsonl`` on access. Holds only byte offsets."""

    def __init__(self, path: Path, offsets: array, checked_roles: frozenset) -> None:
        self.path, self._offsets, self.checked_roles = path, offsets, checked_roles
        self._size = path.stat().st_size
        self._file = None

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return tuple(self[k] for k in range(*i.indices(len(self))))
        n = len(self._offsets)
        if not -n <= i < n:
            raise IndexError("frame index out of range")
        f = self._open()
        f.seek(self._offsets[i])
        return FrameRecord.model_validate_json(f.readline())

    def __iter__(self):
        f = self._open()
        f.seek(0)
        for _ in range(len(self._offsets)):
            yield FrameRecord.model_validate_json(f.readline())

    def _open(self):
        if self._file is None:
            if self.path.stat().st_size != self._size:
                raise ValueError("cache entry changed after it was verified")
            self._file = open(self.path, "rb")
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self):
        raise TypeError("stored frames are not picklable; pass the cache key instead")


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return _sha256(data)


@dataclass(frozen=True)
class Lookup:
    result: Optional[PerceptionResult]
    reason: Optional[str] = None  # why an existing entry was rejected; None for a hit or a plain miss


class PerceptionCache:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def entry(self, key: str) -> Path:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("cache key must be a sha256 hex digest")
        return self.root / key

    def lookup(self, key: str) -> Lookup:
        directory = self.entry(key)
        if not directory.exists():
            return Lookup(None)
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
        except (OSError, ValueError) as exc:
            return Lookup(None, f"unreadable manifest: {exc}")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            return Lookup(None, f"incompatible schema version {manifest.get('schema_version')!r}")
        if manifest.get("key") != key:
            return Lookup(None, "manifest key does not match the entry")
        files = manifest.get("files")
        if not isinstance(files, dict) or "perception.json" not in files or FRAMES not in files:
            return Lookup(None, "manifest lists no perception.json and frames.jsonl")
        for rel, expected in files.items():
            try:
                path = (directory / rel).resolve()
                if Path(rel).is_absolute() or ".." in Path(rel).parts or not path.is_relative_to(
                        self.root.resolve() / key):
                    return Lookup(None, "manifest file path escapes the cache entry")
                actual = file_sha256(path)
            except OSError:
                return Lookup(None, f"missing file {rel}")
            if actual != expected:
                return Lookup(None, f"file {rel} does not match its recorded hash")
        try:
            header = json.loads((directory / "perception.json").read_bytes())
            if not isinstance(header, dict) or "frames" in header:
                raise ValueError("perception.json must hold the result without frames")
            probe = PerceptionResult.model_validate({**header, "frames": (), "timing": {
                **header.get("timing", {}), "frames_analysed": 0}})
        except ValueError as exc:
            return Lookup(None, f"perception.json fails validation: {exc}")
        if perception_key(probe.video, probe.config, probe.provenance) != key:
            return Lookup(None, "content does not hash to the entry key")
        configured = frozenset(c.role for c in probe.provenance.components)
        offsets = array("q")
        try:
            check_frames(self._scan(directory / FRAMES, offsets, files), set(configured))
            frames = StoredFrames(directory / FRAMES, offsets, configured)
            result = PerceptionResult.model_validate({**header, "frames": frames})
        except ValueError as exc:
            return Lookup(None, f"frames fail validation: {exc}")
        return Lookup(result)

    @staticmethod
    def _scan(path: Path, offsets: array, files: dict):
        """Parse each record once, recording its offset; it is dropped as soon as it has been checked."""
        with open(path, "rb") as f:
            while True:
                at = f.tell()
                line = f.readline()
                if not line:
                    return
                frame = FrameRecord.model_validate_json(line)
                ref = frame.segmentation
                if ref is not None and files.get(ref.path) != ref.sha256:
                    raise ValueError(f"frame {frame.index} references an unlisted mask")
                offsets.append(at)
                yield frame

    def load_mask(self, key: str, ref: SegmentationRef) -> np.ndarray:
        import cv2
        path = (self.entry(key) / ref.path).resolve()
        if not path.is_relative_to(self.root.resolve() / key):
            raise ValueError("mask path escapes the cache entry")
        data = path.read_bytes()
        if _sha256(data) != ref.sha256:
            raise ValueError(f"mask {ref.path} does not match its recorded hash")
        mask = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape != (ref.height, ref.width):
            raise ValueError(f"mask {ref.path} is unreadable or has the wrong size")
        return mask

    def begin(self, key: str) -> "CacheWriter":
        return CacheWriter(self, key)


class CacheWriter:
    """Builds one entry in a private staging directory; nothing is visible until ``commit``."""

    def __init__(self, cache: PerceptionCache, key: str):
        self.cache, self.key = cache, key
        self.final = cache.entry(key)
        self.staging = cache.root / STAGING / f"{key}.{uuid.uuid4().hex}"
        self.staging.mkdir(parents=True)
        self.files: dict[str, str] = {}
        self.done = False
        self._frames = None
        self._frames_hash = hashlib.sha256()
        self._offsets, self._at, self._last = array("q"), 0, None
        self.frames_written = 0

    def append_frame(self, frame: FrameRecord) -> None:
        """Stream one record to disk; records must arrive in order and match the committed result."""
        if self.done:
            raise RuntimeError("cache entry already finished")
        if frame.index != self.frames_written or (self._last is not None and (
                frame.t <= self._last.t or set(frame.status) != set(self._last.status))):
            raise ValueError("frames must be appended in order with the same roles")
        if self._frames is None:
            self._frames = open(self.staging / FRAMES, "wb")
        line = (canonical_json(frame) + "\n").encode()
        self._frames.write(line)
        self._frames_hash.update(line)
        self._offsets.append(self._at)
        self._at += len(line)
        self._last = frame
        self.frames_written += 1

    def staged_frames(self) -> StoredFrames:
        """The appended records, readable from the staging file; valid until ``commit`` or ``abort``."""
        if self._frames is None:
            raise ValueError("no frames appended")
        self._frames.flush()
        return StoredFrames(self.staging / FRAMES, self._offsets, frozenset(self._last.status))

    def write_mask(self, frame_index: int, mask: np.ndarray, classes: tuple[str, ...]) -> SegmentationRef:
        import cv2
        mask = np.asarray(mask)
        if mask.ndim != 2 or mask.dtype != np.uint8 or mask.size == 0:
            raise ValueError("mask must be a non-empty 2-D uint8 array")
        if int(mask.max()) >= len(classes):
            raise ValueError("mask contains a class index without a name")
        ok, png = cv2.imencode(".png", mask)
        if not ok:
            raise ValueError("mask could not be encoded")
        rel = f"masks/{frame_index:06d}.png"
        self.files[rel] = _write(self.staging / rel, png.tobytes())
        return SegmentationRef(path=rel, sha256=self.files[rel], width=mask.shape[1], height=mask.shape[0],
                               classes=tuple(classes))

    def commit(self, result: PerceptionResult) -> Path:
        if self.done:
            raise RuntimeError("cache entry already finished")
        if perception_key(result.video, result.config, result.provenance) != self.key:
            raise ValueError("result does not hash to the key this entry was opened for")
        if self._frames is None:  # frames were not streamed: write them now
            for frame in result.frames:
                self.append_frame(frame)
        if self.frames_written != len(result.frames):
            raise ValueError("streamed frames do not match the result")
        if self._frames is None:
            self._frames = open(self.staging / FRAMES, "wb")
        self._frames.flush()
        os.fsync(self._frames.fileno())
        self._frames.close()
        self.files[FRAMES] = self._frames_hash.hexdigest()
        header = result.model_dump(mode="json", exclude={"frames"})
        self.files["perception.json"] = _write(self.staging / "perception.json", canonical_json(header).encode())
        manifest = {"schema_version": SCHEMA_VERSION, "key": self.key, "files": dict(sorted(self.files.items()))}
        _write(self.staging / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode())
        old = None
        if self.final.exists():  # replace a rejected or older entry; readers see a miss, never a mix
            old = self.cache.root / STAGING / f"old.{uuid.uuid4().hex}"
            os.replace(self.final, old)
        os.replace(self.staging, self.final)
        self.done = True
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
        return self.final

    def abort(self) -> None:
        if self._frames is not None and not self._frames.closed:
            self._frames.close()
        if not self.done:
            shutil.rmtree(self.staging, ignore_errors=True)
            self.done = True
