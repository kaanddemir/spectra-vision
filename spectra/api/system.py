"""Model-registry and memory checks reported by the health endpoint. Nothing here loads a model."""
from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import Callable, Optional

from spectra.api.schemas import AnalysisReadiness, DatabaseHealth, MemoryHealth, ModelFile, ModelRegistryHealth
from spectra.api.store import Database, DatabaseError
from spectra.engine.models import default_specs

PressureReader = Callable[[], Optional[int]]


class ModelRegistryCheck:
    """Verify the runtime registry, hashing again only when a file in the models folder changes."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._lock = threading.Lock()
        self._signature: Optional[tuple] = None
        self._result: Optional[ModelRegistryHealth] = None

    def _current_signature(self) -> Optional[tuple]:
        try:
            with os.scandir(self.models_dir) as entries:
                return tuple(sorted((e.name, e.stat().st_size, e.stat().st_mtime_ns)
                                    for e in entries if e.is_file()))
        except OSError:
            return None

    def status(self) -> ModelRegistryHealth:
        with self._lock:
            signature = self._current_signature()
            if self._result is None or signature is None or signature != self._signature:
                self._result = self._verify()
                self._signature = signature
            return self._result

    def _verify(self) -> ModelRegistryHealth:
        try:
            detector, segmenter = default_specs(self.models_dir)
        except (OSError, ValueError) as exc:
            return ModelRegistryHealth(available=False, message=str(exc))
        return ModelRegistryHealth(available=True, models=[
            ModelFile(role="detector", file=Path(detector.model_path).name, sha256=detector.sha256),
            ModelFile(role="segmenter", file=Path(segmenter.model_path).name, sha256=segmenter.sha256)])


def memory_health(level: Optional[int]) -> MemoryHealth:
    """Map the macOS pressure level (1 normal, 2 warning, 4 critical); other platforms report none."""
    if level is None:
        return MemoryHealth(pressure_level=None, state="unavailable")
    state = "normal" if level <= 1 else "warning" if level < 4 else "critical"
    return MemoryHealth(pressure_level=level, state=state)


def database_health(database: Database) -> DatabaseHealth:
    try:
        return DatabaseHealth(available=True, schema_version=database.check())
    except (sqlite3.Error, DatabaseError) as exc:
        return DatabaseHealth(available=False, message=str(exc))


def analysis_readiness(database: DatabaseHealth, models: ModelRegistryHealth,
                       memory: MemoryHealth) -> AnalysisReadiness:
    """Whether a new analysis could start now. Analysis itself still runs under ResourceGuard."""
    reasons = []
    if not database.available:
        reasons.append("The application database is unavailable.")
    if not models.available:
        reasons.append("The analysis models are missing or failed verification.")
    if memory.state == "critical":
        reasons.append("System memory pressure is critical; close other applications and try again.")
    return AnalysisReadiness(available=not reasons, reasons=reasons)
