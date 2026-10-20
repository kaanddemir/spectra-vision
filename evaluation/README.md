# Evaluation tooling

Offline tools for dataset handling, scoring the engine's proposals against labels, comparing detectors
and trackers, and exporting the runtime models. The engine does not import this package; evaluation tools
consume engine contracts. None of it is needed to run the application.

| Module | Purpose |
|---|---|
| `datasets.py`, `outputs.py` | Manifest validation and verified clip access; exclusive new prediction/report paths. |
| `annotations.py` | Versioned ground-truth schema (JSON Lines, one clip per line). Unknown values stay `null`. Each event has an origin; only human-confirmed and synthetic events are scored by default. Dataset labels are kept verbatim as source labels. |
| `importers/` | DoTA and PREVENTION annotation importers. They produce source labels (and source-derived events where a dataset has a matching class), never ground truth. Output goes to the git-ignored `restricted/`. |
| `perception_cache.py` | Versioned cache contract: `manifest.json` + `arrays.npz` (timestamps, detections with track ids, motion features, optional travel path, model and config metadata). |
| `candidates.py` | Candidate-event format used for scoring. |
| `metrics.py` | One-to-one event matching (peak within 1.0 s or temporal IoU ≥ 0.3), precision/recall/F1, false candidates per hour, timing errors, recall by condition. |
| `engine_scoring.py` | Scores the engine CLI's event candidates against annotations over the processed window; supported and all candidates separately; never overwrites a result. |
| `detector_compare.py`, `tracker_compare.py`, `detector_bench.py`, `detectors/` | Detector and tracker comparison on identical frames and caches (D-FINE, RF-DETR adapters; BoT-SORT, McByte). Kept for re-evaluation once labels exist. |
| `export_dfine_onnx.py`, `export_segnet_onnx.py` | Runtime model exports; run only in `evaluation/.venv`. |
| `hardware.py` | Hardware probe plus isolated, sequential, time- and memory-bounded ONNX and decoding benchmarks. |
| `registry/` | Dataset and model licence registry; a model cannot be marked distributable unless code, checkpoint and training-data terms are all confirmed. |

## Tool entry points (not instructions to run benchmarks)

```bash
python -m pytest tests/evaluation
python -m evaluation.importers.dota ANNOTATIONS_DIR --out evaluation/restricted/dota/annotations.jsonl
python -m evaluation.importers.prevention RECORD4/DRIVE3/processed_data --drive RECORD4/DRIVE3 --out evaluation/restricted/prevention/annotations/RECORD4-DRIVE3.jsonl
python -m evaluation.engine_scoring ANNOTATIONS.jsonl CLIP_ID ENGINE_OUTPUT.json --start S --end E --name NAME --out-root evaluation/reports
python -m evaluation.hardware probe
python -m evaluation.hardware onnx --help
python -m evaluation.hardware decode --help
```

## Rules

- Tune on `dev`, check on `validation`, and score `test` only for a release decision.
- Report results per data source; do not pool synthetic, compilation and real dashcam footage.
- Benchmarks run one model at a time. Download no dataset or checkpoint without recording its terms in the registry first.
- Run the full test suite (`pytest -q`, including model tests), benchmarks and real-video checks one at a time,
  and only when `sysctl -n kern.memorystatus_vm_pressure_level` reports 1. Use `--strict-memory`, and never
  retry a refused run in a loop.
- Never modify, move or re-encode source media under `datasets/`; manifests take additive metadata only.
- The [evaluation protocol](EVALUATION_PROTOCOL.md) owns label semantics, metrics and reproducibility requirements.


## Verified datasets and separate derived outputs

Source media stays read-only under `datasets/<dataset_id>/`; manifests, registry entries and annotations
remain in their existing locations. `datasets.load_manifest(path)` validates provenance metadata without
opening media, and `datasets.clips(manifest, root)` yields a path and its metadata only after path
confinement, existence, size and SHA256 verification (SHA1 fallback for older records). `verify=False`
skips size/hash checks only. Proposed catalog entries may omit local filenames and hashes, but cannot
be accessed as local clips until both are assigned. The proposal catalog is not a verified dataset.
Raw-stream `assumed_fps` and optional dev/validation/test splits come from manifest fields; no dataset
names or timing exceptions belong in the engine. Sample clips are pinned by SHA-256; their SHA-1 values and
provenance are kept.

New predictions belong at `predictions/<dataset_id>/<analysis_key>/<clip_id>.json`; new reports at
`reports/<YYYY-MM-DD>_<name>/`. Both output roots are ignored and recreated by writers when needed.
`outputs.write_prediction` creates JSON exclusively and refuses path/key mismatch or overwrite;
`outputs.new_report_dir` reserves a unique dated directory (UTC date by default). Scoring already accepts
`--out-root evaluation/reports`; give each run a fresh dated `--name`. These output helpers do not migrate
existing files. Keep restricted annotations, reviews and historical comparison evidence in their original
restricted locations.

Run lightweight synthetic coverage with `pytest -m "not slow and not model and not real_video"`.
Real-video tests additionally require a positive `real_video` marker expression; this prevents a fast
`-m` override from inadvertently opting into real footage. See the evidence README for the still-pending
strict cache-only integration check. Report accuracy only against human-confirmed labels, per data source.

## Environments and reproducibility

Lightweight evaluation and scoring tools use the application environment with the `eval` extra
(`.venv/bin/python -m pip install -e ".[dev,eval]"` in a new environment). Installing optional packages
into an existing environment requires approval. [pyproject.toml](../pyproject.toml) holds all pins.

Model exports use **only** `evaluation/.venv`, isolated from the application and global site-packages.
Its 79 pinned distributions are in [requirements-export-isolated.lock.txt](requirements-export-isolated.lock.txt).
That list alone does not restore the local sources and checkpoints. Preserve:
- the D-FINE checkout (commit `956d1709314c2c6a4df6f34de232054578a7449f`)
- the RF-DETR and D-FINE checkpoints in `evaluation/.venv/weights/`
- the SegNet checkpoint and configuration in `evaluation/weights/comma10k-segnet/`
- the exporter adapters

Verify the checkpoints with:

```sh
shasum -a 256 -c evaluation/checkpoint_hashes.txt
```

Runtime export hashes and paths are in the [model registry](../models/engine/models.json). The existing
export scripts supply the export configuration; do not substitute new upstream versions or regenerate
weights casually. Both environments pass offline export imports, but numerical equivalence of the
exports in the repaired environment is still **pending**.

## Layout and adding a dataset

```text
evaluation/datasets/<dataset_id>/           immutable original media (ignored)
evaluation/manifests/                      provenance, pinned bytes/hashes, timing, optional split
evaluation/registry/                       licence and permitted-use records
evaluation/annotations/                    draft annotations (review steps in the protocol)
evaluation/restricted/                     restricted source labels, reviews and historical evidence
evaluation/caches/                         generated perception; engine_v1 and research formats coexist
evaluation/predictions/<dataset>/<key>/     new clip JSON predictions (ignored)
evaluation/reports/<YYYY-MM-DD>_<name>/     new reports (ignored)
```

1. Record provenance and permitted uses in the registry before acquiring anything. Downloads need
   approval; the proposal catalog is planning metadata, not a verified local dataset.
2. Add a manifest with unique clip IDs/relative filenames, exact bytes and SHA256, role/licence,
   optional `dev`/`validation`/`test` split and explicit `assumed_fps` for raw streams. Preserve all
   existing manifests; only additive changes are allowed to protected records.
3. Use `datasets.load_manifest` and `datasets.clips` for confined, verified local access. Never put a
   dataset name or timing exception in the engine. Keep original media and annotations unchanged.
4. Import source labels with the existing DoTA/PREVENTION importers into a new permitted location;
   source-derived labels are not human-confirmed ground truth. Follow the
   [protocol](EVALUATION_PROTOCOL.md), including its review steps.
5. Run sequentially under strict safeguards only at macOS pressure 1. Write new outputs, never
   overwrite or migrate existing reports/caches without an exact approved inventory.

## Labelled evaluation workflow

Use verified manifest paths and per-clip assumed timing when applicable. This is a template, not an
executed validation; replace placeholders and choose a new output filename/report name:

```sh
sysctl -n kern.memorystatus_vm_pressure_level
# Continue only at level 1; one clip at a time:
.venv/bin/python -m spectra.engine analyze CLIP --cache evaluation/caches/engine_v1/cache --models models/engine --interval 0.1 --segment-interval 0.2 --track --lanes --events --strict-memory > NEW_ENGINE_OUTPUT.json
.venv/bin/python -m evaluation.engine_scoring ANNOTATIONS.jsonl CLIP_ID NEW_ENGINE_OUTPUT.json --start S --end E --name YYYY-MM-DD_NAME --out-root evaluation/reports
```

Use `outputs.write_prediction` to publish validated JSON under the dataset/analysis-key/clip layout.
The CLI does not create that hierarchy automatically. `engine_scoring` refuses an existing run name.
The research NPZ cache format (`perception_cache.py`) is written by `build_cache.py` and consumed by
`review_frames.py` and `tracking.py`. New research caches record the detector checkpoint SHA-256 when
the adapter exposes its source `weights` path; otherwise the hash remains unknown (`null`). Historical
caches are unchanged. `detector_compare.py` runs through the engine and writes engine caches;
`tracker_compare.py` replays those engine detections.

Unconfirmed clips are unscored, not negative examples. Tune on dev, check validation and reserve test for release decisions; report each data source
separately. Do not pool synthetic and real footage or claim accuracy from the five unlabelled samples.

The reasons for the current model and method choices are summarized in
[ARCHITECTURE.md](../ARCHITECTURE.md#method-choices); real-video evidence checks remain
[pending](../ARCHITECTURE.md#validation-and-remaining-limits).
