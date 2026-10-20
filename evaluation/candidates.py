"""Candidate events in the scoring format: engine proposals converted for matching against annotations."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from evaluation.annotations import EventType


@dataclass(frozen=True)
class Candidate:
    clip_id: str
    type: EventType
    t_start: float
    t_peak: float
    t_end: float
    score: float
    algorithm: str
    algorithm_version: str
    track_ids: tuple[int, ...] = ()
    observations: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (self.t_start <= self.t_peak <= self.t_end):
            raise ValueError("candidate must satisfy t_start <= t_peak <= t_end")

    def to_json(self) -> str:
        d = asdict(self)
        d["type"] = self.type.value
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "Candidate":
        d = json.loads(line)
        d["type"] = EventType(d["type"])
        d["track_ids"] = tuple(d["track_ids"])
        return cls(**d)


def save_candidates(candidates: list[Candidate], path: Path) -> None:
    Path(path).write_text("".join(c.to_json() + "\n" for c in candidates))


def load_candidates(path: Path) -> list[Candidate]:
    return [Candidate.from_json(l) for l in Path(path).read_text().splitlines() if l.strip()]
