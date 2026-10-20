"""Dataset annotation importers. Each converts one dataset's published label files into
``ClipAnnotation`` records whose ``source_labels`` keep the original values. They never produce
human-confirmed events or negatives.

Imported files belong in the git-ignored ``evaluation/restricted/`` workspace.
"""
from __future__ import annotations

from evaluation.annotations import Licence
from evaluation.registry import Use, load

RESTRICTED_DIR = "evaluation/restricted"


def registry_licence(dataset_id: str) -> Licence:
    """The licence record for a dataset, from the registry; evaluation is allowed only if it says so."""
    entry = next((d for d in load().datasets if d.id == dataset_id), None)
    if entry is None:
        raise ValueError(f"dataset {dataset_id!r} is not in the registry; record its terms first")
    allowed = {Use.EVALUATION_ONLY: True, Use.DISTRIBUTABLE: True, Use.BLOCKED: False}.get(entry.evaluation_use)
    return Licence(dataset_id=dataset_id, terms=entry.terms.statement, evaluation_allowed=allowed,
                   training_allowed=False if entry.training_use is Use.BLOCKED else None)
