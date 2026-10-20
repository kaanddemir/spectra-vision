"""Deterministic, stratified selection of a small evaluation sample from dataset metadata.

Reads a metadata CSV (for example the Nexar ``train.csv``) and picks clips so that each
(label, stratum) group is represented, without looking at video content.

    python -m evaluation.sampling META.csv --label-col target --strata light_conditions weather \
        --per-label 8 --seed 0 --out sample.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def select(rows: list[dict], id_col: str, label_col: str, strata: list[str], per_label: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r[id_col]):
        by_label[r[label_col]].append(r)
    for label in sorted(by_label):
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in by_label[label]:
            groups[tuple(r.get(s, "") for s in strata)].append(r)
        for g in groups.values():
            rng.shuffle(g)
        keys = sorted(groups)
        rng.shuffle(keys)
        picked: list[dict] = []
        while len(picked) < per_label and any(groups[k] for k in keys):  # round-robin across strata
            for k in keys:
                if groups[k] and len(picked) < per_label:
                    picked.append(groups[k].pop())
        out.extend(sorted(picked, key=lambda r: r[id_col]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("metadata", type=Path)
    p.add_argument("--id-col", default="id")
    p.add_argument("--label-col", default="target")
    p.add_argument("--strata", nargs="*", default=[])
    p.add_argument("--per-label", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    with open(a.metadata, newline="") as f:
        rows = list(csv.DictReader(f))
    chosen = select(rows, a.id_col, a.label_col, a.strata, a.per_label, a.seed)
    a.out.write_text(json.dumps({"source": str(a.metadata), "seed": a.seed, "strata": a.strata,
                                 "clips": chosen}, indent=2))
    print(f"selected {len(chosen)} clips")


if __name__ == "__main__":
    main()
