"""Sequential, budgeted, hash-verified download of clips listed in a sample manifest.

    python -m evaluation.fetch evaluation/manifests/real_sample.json --dest evaluation/datasets/real_sample --max-mb 300

Each clip must pin ``bytes`` and ``sha1`` or ``sha256``. Files that already exist with the right
hash are skipped. A file that fails verification is deleted and the run stops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _expected(clip: dict) -> tuple[str, str]:
    for algo in ("sha256", "sha1"):
        if clip.get(algo):
            return algo, clip[algo]
    raise ValueError(f"{clip['id']}: no pinned hash")


def plan(manifest: dict, dest: Path, max_mb: float) -> list[dict]:
    clips = manifest["clips"]
    for c in clips:
        _expected(c)
    total = sum(c["bytes"] for c in clips) / 1e6
    if total > max_mb:
        raise ValueError(f"manifest totals {total:.1f} MB, above the {max_mb} MB budget")
    free = shutil.disk_usage(dest if dest.exists() else dest.parent).free / 1e6
    if free < 2 * total + 1024:
        raise ValueError(f"insufficient free disk: {free:.0f} MB")
    return clips


def fetch(manifest_path: Path, dest: Path, max_mb: float) -> list[dict]:
    manifest = json.loads(Path(manifest_path).read_text())
    dest.mkdir(parents=True, exist_ok=True)
    report = []
    for c in plan(manifest, dest, max_mb):
        target = dest / c["filename"]
        algo, want = _expected(c)
        if target.exists() and _digest(target, algo) == want:
            report.append({"id": c["id"], "status": "present"})
            continue
        part = target.with_suffix(target.suffix + ".part")
        subprocess.run(["curl", "-sSfL", "--max-time", "900", "-A", "SpectraEvaluation/0.1", "-o", str(part), c["url"]],
                       check=True)
        size, got = part.stat().st_size, _digest(part, algo)
        if size != c["bytes"] or got != want:
            part.unlink()
            raise ValueError(f"{c['id']}: verification failed (size {size}, {algo} {got})")
        part.replace(target)
        report.append({"id": c["id"], "status": "downloaded", "bytes": size, algo: got})
        print(json.dumps(report[-1]), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("manifest", type=Path)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--max-mb", type=float, required=True)
    a = p.parse_args()
    fetch(a.manifest, a.dest, a.max_mb)


if __name__ == "__main__":
    main()
