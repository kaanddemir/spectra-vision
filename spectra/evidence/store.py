"""On-disk evidence: one directory per build key, separate from perception caches and review decisions.

::

    <root>/<build_key>/manifest.json         evidence version, keys, configuration, sha256 of every file
    <root>/<build_key>/timeline.json         canonical JSON Timeline
    <root>/<build_key>/packages/<id>.json    canonical JSON EvidencePackage per proposal
    <root>/<build_key>/clips/<id>.<ext>      optional stream-copied clip, written on first request
    <root>/<build_key>/clips/<id>.json       its ClipFile record

The build key hashes the evidence version, the analysis key and the review configuration, so a changed
analysis, configuration or evidence version is a different directory and never a reuse; existing
directories are left alone. A build is written into a staging directory and renamed into place only when
complete. Readers verify every file against the manifest and the manifest against the requested keys, and
report why a directory was rejected. Overlays are not stored: they are read from the perception cache on
request. Everything here is derived data that can be rebuilt from the analysis; review decisions are kept
elsewhere and never written here.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from spectra.engine.contracts import canonical_json
from spectra.evidence.build import build_key, package_path
from spectra.evidence.contracts import EVIDENCE_VERSION, EvidencePackage, Timeline
from spectra.evidence.media import ClipFile, extract_clip, load_clip

STAGING = ".staging"


@dataclass(frozen=True)
class Lookup:
    timeline: Optional[Timeline]
    reason: Optional[str] = None  # why an existing build was rejected; None for a hit or a plain miss


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class EvidenceStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def entry(self, key: str) -> Path:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("build key must be a sha256 hex digest")
        return self.root / key

    def write(self, timeline: Timeline, packages: list[EvidencePackage]) -> Path:
        """Store a timeline built from a result; an existing valid build with the same key is kept as is."""
        if timeline.build_key is None:
            raise ValueError("only a timeline built from an analysis result is stored")
        if timeline.build_key != build_key(timeline.analysis_key, timeline.config):
            raise ValueError("timeline build key does not match its analysis and configuration")
        if sorted(p.proposal.event_id for p in packages) != sorted(e.event_id for e in timeline.entries) or any(
                p.build_key != timeline.build_key for p in packages):
            raise ValueError("packages must be exactly the timeline's, from the same build")
        final = self.entry(timeline.build_key)
        if self.load(timeline.build_key, timeline.analysis_key).timeline is not None:
            return final
        files = {"timeline.json": canonical_json(timeline).encode()}
        files.update({package_path(p.proposal.event_id): canonical_json(p).encode() for p in packages})
        manifest = {"evidence_version": EVIDENCE_VERSION, "build_key": timeline.build_key,
                    "analysis_key": timeline.analysis_key, "config": timeline.config.model_dump(mode="json"),
                    "files": {k: _sha(v) for k, v in sorted(files.items())}}
        staging = self.root / STAGING / f"{timeline.build_key}.{uuid.uuid4().hex}"
        try:
            for rel, data in [*files.items(), ("manifest.json", canonical_json(manifest).encode())]:
                path = staging / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
            if final.exists():  # a rejected build: kept for inspection, never silently deleted
                final.rename(final.with_name(f"{final.name}.rejected.{uuid.uuid4().hex[:8]}"))
            os.replace(staging, final)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return final

    def _manifest(self, key: str, analysis_key: Optional[str]) -> tuple[Optional[dict], Optional[str]]:
        directory = self.entry(key)
        if not directory.exists():
            return None, None
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
        except (OSError, ValueError) as exc:
            return None, f"unreadable manifest: {exc}"
        if manifest.get("evidence_version") != EVIDENCE_VERSION:
            return None, f"incompatible evidence version {manifest.get('evidence_version')!r}"
        if manifest.get("build_key") != key:
            return None, "manifest build key does not match the directory"
        if analysis_key is not None and manifest.get("analysis_key") != analysis_key:
            return None, "built from a different analysis"
        return manifest, None

    def _read(self, key: str, manifest: dict, rel: str) -> bytes:
        expected = manifest["files"].get(rel)
        if expected is None:
            raise LookupError(f"{rel} is not part of this build")
        path = (self.entry(key) / rel).resolve()
        if Path(rel).is_absolute() or ".." in Path(rel).parts or not path.is_relative_to(self.root.resolve() / key):
            raise ValueError("file path escapes the evidence entry")
        data = path.read_bytes()
        if _sha(data) != expected:
            raise ValueError(f"{rel} does not match its recorded hash")
        return data

    def load(self, key: str, analysis_key: Optional[str] = None) -> Lookup:
        """The stored timeline, verified; packages are read on request with ``package``."""
        manifest, reason = self._manifest(key, analysis_key)
        if manifest is None:
            return Lookup(None, reason)
        try:
            for rel in manifest["files"]:
                self._read(key, manifest, rel)
            timeline = Timeline.model_validate_json(self._read(key, manifest, "timeline.json"))
        except (OSError, ValueError, LookupError) as exc:
            return Lookup(None, f"build fails verification: {exc}")
        if (timeline.build_key, timeline.analysis_key) != (key, manifest["analysis_key"]) or (
                build_key(timeline.analysis_key, timeline.config) != key):
            return Lookup(None, "timeline does not hash to the build key")
        return Lookup(timeline)

    def package(self, key: str, event_id: str) -> EvidencePackage:
        manifest, reason = self._manifest(key, None)
        if manifest is None:
            raise LookupError(reason or "no such build")
        pk = EvidencePackage.model_validate_json(self._read(key, manifest, package_path(event_id)))
        if pk.build_key != key or pk.proposal.event_id != event_id:
            raise ValueError("package does not belong to this build")
        return pk

    def clip(self, pk: EvidencePackage, source: Path | str, *, cancel=None, **limits) -> ClipFile:
        """The package's clip, cut from ``source`` on first request and reused while it verifies."""
        directory = self.entry(pk.build_key) / "clips"
        if not (self.entry(pk.build_key) / "manifest.json").is_file():
            raise LookupError("store the build before requesting clips")
        found = load_clip(directory, pk.proposal.event_id, pk.video.sha256, pk.review)
        if found is not None:
            return found
        return extract_clip(source, pk.video, pk.review, directory, pk.proposal.event_id, cancel=cancel, **limits)

    def clip_path(self, pk: EvidencePackage, clip: ClipFile) -> Path:
        return self.entry(pk.build_key) / "clips" / clip.file
