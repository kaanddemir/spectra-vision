"""Versioned registries of candidate datasets and models, with licence status.

A model may be marked ``distributable`` only when its code licence, checkpoint
licence and training-data terms are all confirmed; the loader enforces this.
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

REGISTRY_SCHEMA_VERSION = "1.0"
HERE = Path(__file__).parent


class Status(str, Enum):
    CONFIRMED = "confirmed"  # read in an official source on accessed date
    SECONDARY = "secondary"  # reported by a non-official source only
    UNKNOWN = "unknown"


class Use(str, Enum):
    DISTRIBUTABLE = "distributable"
    EVALUATION_ONLY = "evaluation_only"
    BLOCKED = "blocked"
    UNDECIDED = "undecided"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Term(_Strict):
    statement: str
    status: Status
    source: Optional[str] = None
    accessed: Optional[str] = None


class DatasetEntry(_Strict):
    id: str
    name: str
    url: str
    annotations: str
    event_coverage: list[str]
    clips: str
    clip_format: str
    access: str
    terms: Term
    evaluation_use: Use
    training_use: Use
    dashcam_relevance: str
    storage_estimate: str
    notes: str = ""


class ModelEntry(_Strict):
    id: str
    function: str
    repository: str
    version: str
    code_licence: Term
    checkpoint_licence: Term
    training_data: Term
    distribution_obligations: str
    use: Use
    measured_here: bool = False
    open_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clearance(self) -> "ModelEntry":
        if self.use is Use.DISTRIBUTABLE and any(
                t.status is not Status.CONFIRMED for t in (self.code_licence, self.checkpoint_licence, self.training_data)):
            raise ValueError(f"{self.id}: distributable requires confirmed code, checkpoint and training-data terms")
        return self


class Registry(_Strict):
    schema_version: str
    datasets: list[DatasetEntry]
    models: list[ModelEntry]

    @model_validator(mode="after")
    def _ids(self) -> "Registry":
        if self.schema_version != REGISTRY_SCHEMA_VERSION:
            raise ValueError("unsupported registry schema")
        for group in (self.datasets, self.models):
            ids = [x.id for x in group]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate registry id")
        return self


def load(path: Path = HERE / "registry.json") -> Registry:
    return Registry.model_validate(json.loads(Path(path).read_text()))
