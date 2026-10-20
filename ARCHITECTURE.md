# System architecture

Spectra is a local, CPU-based video analysis engine with a separate evidence layer, served by a local API
to a web application. It produces reviewable proposals, not incident findings.

## Responsibilities and boundaries

| Component | Owns | May depend on |
|---|---|---|
| `spectra/engine/` | Schema v1 contracts, decoding, perception, analysers, cache, resource guard, worker | runtime libraries only |
| `spectra/evidence/` | review intervals, deterministic timeline and packages, overlays, source references, stream copies | engine |
| `spectra/api/` | local HTTP server, analysis queue, SQLite application state, review decisions and edits | engine, evidence |
| `frontend/` | web application: library, videos, analyses, review workspace, export | API over HTTP only |
| `evaluation/` | manifests, dataset access, labels, scoring, detector comparison, model exports | engine |

Engine and evidence never import evaluation or API modules. Dataset-specific rules stay in evaluation.
Perception owns observations, analysers own interpretations, and people own review decisions.

## Data flow

```text
source file → VideoSource (PyAV, presentation timestamps)
  → fresh spawned worker + cancellation + ResourceGuard
  → D-FINE-N detector (10 Hz) → BoT-SORT + camera-motion quality
  → comma10k SegNet (5 Hz) → lane geometry + track/lane relations
  → atomic verified perception cache
  → configured analysers replay records → AnalysisResult
  → evidence build → timeline + packages → verified EvidenceStore → overlays, source intervals, clips
  → API (127.0.0.1, state in var/) → web application (same origin) → human review decisions and edits
```

`python3 run.py` starts the API, which serves the built web application from `frontend/dist`. The API
runs a persisted FIFO queue, one `run_in_worker` process at a time, and reports progress with Server-Sent
Events. Review decisions and edits are stored in their own tables and never change an analysis, its
result or its evidence. The JSON export is read-only.

Both models run on ONNX Runtime CPU with 4 intra-op threads and 1 inter-op thread. Lane geometry assumes a
locally flat road and negligible roll, and its coordinates are not metric. The
[runtime registry](models/engine/models.json) is the source for hashes, shapes and provenance.

## Method choices

Detector, tracker and lane choices were made by comparing candidates on unlabelled sample footage.
They are engineering choices, not measured accuracy. The comparison tooling is kept in `evaluation/`
(`detector_compare.py`, `tracker_compare.py`, `detector_bench.py`) for re-evaluation once labels exist.

- **Detector: D-FINE-N (COCO, 640 px).** Provisional choice.
  - ONNX output matched PyTorch exactly on identical decoded frames.
  - About 0.5–0.6 processing seconds per video second including tracking, with a 434–600 MB peak.
  - It found many more distant vehicles than RF-DETR Nano on highway and night-rain clips.
  - Known faults: it detects the camera vehicle's hood (the segmentation's ego-vehicle mask suppresses
    this) and gives false positives on night reflections.
  - D-FINE-S is the only viable challenger, kept as a local comparison export. Swapping detectors needs
    labelled data first.
  - Ultralytics YOLO was excluded (AGPL-3.0).
- **Tracker: BoT-SORT** (`trackers` 2.6.1) with camera-motion compensation. In comparison, McByte
  fragmented tracks less but produced 1.7–2.8× more abrupt size jumps, which are a sign of identity swaps.
  That trade-off is inconclusive without identity labels.
- **Lanes: comma10k SegNet plus geometry.**
  - It is the only candidate with permissively licensed weights and an ego-vehicle class.
  - Lane boundaries are fitted as a hyperbola model in image space and tracked over time, with evidence
    extended along observed paint.
  - Relations that cannot be observed stay unobservable rather than guessed. Occlusion by vehicles is
    the dominant cause of this.
  - Rejected alternatives:
    - UFLDv2 (CULane research-only terms)
    - TwinLiteNet+ (BDD100K terms, no ego-vehicle class)
    - the openpilot driving model (needs calibrated inputs)
- **Rapid approach:** monocular looming. Apparent time-to-contact comes from isotropic box growth, so the
  method needs no focal length, speed or depth. It excludes boxes at the frame edge, low scores and
  predicted boxes, and applies a lane-relevance rule.
- **Cut-in:** a sustained lane-membership transition of a tracked vehicle from an adjacent lane into the
  camera's lane. The camera lane's boundary identities separate a vehicle's movement from the camera's own:
  - a completed ego lane change yields no proposal;
  - re-identified boundaries without one make the proposal `uncertain`.
- **Impact (camera vehicle):** requires both a close approach and a reliable measured camera jolt. Missing
  motion produces an unresolved pattern, never a proposal.

## Contracts, identity and configuration

Schema `1.0` lives in `spectra/engine/contracts.py`. Frames distinguish unavailable, failed, unscheduled,
empty and observed results. Times derive from source presentation timestamps. An explicit assumed rate is
recorded for elementary streams. Predicted boxes never become observed event evidence.

`AnalysisResult` contains perception, proposals and a report per configured analyser. Reports include a
status, reconciled considered/proposed/rejected counts and unresolved patterns. Cancelled, failed or
resource-stopped runs have no analysis result. Decode truncation can leave incomplete coverage, which the
evidence displays.

Identities build on each other:
- **Perception identity:** the video hash, timing source, run configuration, versions and perception
  component identities.
- **Analysis identity:** adds the ordered analyser identities.
- **Evidence build identity:** adds the evidence version and review configuration.

Cache reads verify schema and hashes, and other formats are rejected, never migrated. Cache-only replay
fails before loading perception on a miss. Analyser defaults and cache keys are covered by fixed golden
fixtures.

Worker defaults are approach, then cut-in when lanes are enabled, then camera-vehicle impact. The
road-user-contact analyser is experimental and off by default.

## Interpretation and presentation

| Proposal | User-facing name | Limit |
|---|---|---|
| `rapid_approach` | Rapid approach (apparent) | no speed, distance or near-collision claim |
| `cut_in` | Vehicle entering your lane | no danger or fault claim |
| `impact/camera_vehicle` | Possible impact involving the camera vehicle | no confirmed collision claim |
| `impact/road_users` | Possible contact between other road users (experimental) | uncertain, off by default |

- `supported` means only that the analyser's own conditions held, not that an incident happened.
- `uncertain` displays its reasons.
- Rank only orders proposals; it is not a probability.
- Evidence packages embed proposals unchanged.
- An empty timeline carries the analysers' coverage, not an assertion that nothing happened.

## Application state

The API keeps all state in `var/` (`--data-dir`):
- `spectra.sqlite3`: SQLite in WAL mode, schema version 1
- `cache/`: the perception cache
- `evidence/`: the Evidence store
- `tmp/`

The database is created atomically on first start. A database at any other version, or whose tables or
indexes differ from the schema, is refused and left unchanged; there are no upgrades. See
[spectra/api/store.py](spectra/api/store.py) and the [API reference](spectra/api/README.md).

## Memory and execution

Each analysis runs in one fresh `run_in_worker` process, one at a time. The 8 GB machine is the reference.
The API calls this blocking function from a thread executor and never runs `AnalysisRun` in its own
process: `ru_maxrss` is a lifetime peak, so retained models would poison later budget checks. Scripts need
an import-safe `__main__` guard. Models load sequentially and close before the child exits, and cached
records are read lazily.

The guard checks macOS memory pressure and a peak-RSS budget, by default half of physical RAM.
- `standard` permits level 2 with a note and stops at critical level 4.
- `strict` refuses anything above normal level 1.

Benchmarks and real-video validation require level 1. Safeguards are never weakened or retried in a loop.
Unlabelled sample runs measured 1.1–1.2 processing seconds per video second and a 524–626 MB peak RSS.
These are observations, not guarantees.

## Validation and remaining limits

- Synthetic tests cover contracts, timestamps, cache failures, worker isolation, analyser logic, geometry,
  evidence, the API and the web application.
- Detection, tracking, lane and event accuracy are **unvalidated**: there are no human-confirmed labels.
  Real positive approach, cut-in and impact footage under clear licences is still needed.
- These checks are still pending:
  - an end-to-end real-video evidence check with the cache-only script
  - numerical equivalence of the ONNX exports in the repaired export environment
- Clips are stream copies. They keep the source codecs and begin at a preceding keyframe. Raw streams with
  an assumed rate refuse extraction. Browser HEVC support varies, and only videos the browser plays can be
  reviewed.

## Licensing and release blockers

| Item | Confirmed | Unresolved |
|---|---|---|
| Spectra code | MIT | — |
| D-FINE code and checkpoint | Apache-2.0; checkpoint trained on COCO train2017 | HGNetV2 backbone pretraining data; COCO images carry individual Flickr licences (annotations CC BY 4.0) |
| comma10k SegNet code and checkpoint | MIT (code, model card); comma10k images MIT per its README | contributor-consent terms; undocumented training recipe and encoder initialisation |
| ONNX Runtime, NumPy, OpenCV, `trackers`, `supervision`, FastAPI, Uvicorn, Pydantic | permissive (MIT / Apache-2.0 / BSD) | — |
| PyAV 19.0.1 wheel | PyAV BSD-3-Clause; bundled FFmpeg is LGPL-3.0-or-later, built without `--enable-gpl` | the wheel also bundles libx264 and libx265 (GPL-2.0+); H.264/HEVC patent licensing |

**Publication is blocked until these are resolved.** Spectra is published as a source repository without
model weights or FFmpeg binaries. Before publishing, the owner must still decide:
- whether and how users may obtain both model checkpoints, given their training-data terms;
- whether depending on the PyAV wheel is acceptable for a public project. Any bundled binary distribution
  would need a decode-only LGPL FFmpeg without GPL encoders.

Spectra never transcodes and calls no encoder. Details are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Dataset terms are recorded in the
[manifests](evaluation/manifests) and the [licensing registry](evaluation/registry).
