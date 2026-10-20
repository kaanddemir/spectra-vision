# Analysis engine

A modular video analysis engine with a CLI and worker integration interface.
It decodes video by presentation timestamp, runs perception components
through small adapter interfaces, caches the perception records, and runs event analysers on them.

| Module | Responsibility |
|---|---|
| `contracts.py` | Pydantic data contracts with `SCHEMA_VERSION`; canonical JSON and hashing |
| `video.py` | PyAV `VideoSource`: metadata, timestamp-grid sampling, verified `frame_at`, decode statistics |
| `adapters.py` | `Detector`, `Tracker`, `Segmenter`, `EgoLaneEstimator`, `EventAnalyzer` interfaces |
| `runner.py` | `AnalysisRun`: lifecycle, per-component schedules, progress, cancellation, cleanup |
| `cache.py` | `PerceptionCache`: content-addressed, atomically written, hash-verified entries |
| `detection.py` | `DFineOnnxDetector` (D-FINE-N through ONNX Runtime, CPU) and class-agnostic duplicate suppression |
| `tracking.py` | `BoTSortTracker` around Roboflow `trackers` (BoT-SORT with camera-motion compensation) |
| `events.py` | approach, cut-in, camera-vehicle impact and experimental road-user contact, on cached perception records |
| `lanes.py` | `SegnetOnnxSegmenter` (comma10k-segnet through ONNX Runtime), `LaneGeometryEstimator` (ego lane, boundary tracking, lane changes), hood marking, track-to-lane relations |
| `worker.py` | `run_in_worker`: one analysis in a spawned child process; progress, cancel and outcome over a queue |
| `resources.py` | Memory guard with `standard` and `strict` policies |
| `__main__.py` | `python -m spectra.engine probe|analyze|frame`; `analyze --detector ... --track --segmenter ... --lanes [--events]` runs a worker |

## Conventions

- **Time.** `t` is source presentation time in seconds since the stream's start time. It is never
  `index / fps`. Streams without timestamps, such as raw HEVC, need an explicit `assumed_fps`, and the
  result records that the times were assumed.
- **Coordinates.** Pixels are in the display-oriented source frame: rotation is applied, there is no
  analysis downscaling, the origin is top-left and y points down. Lane boundaries use normalised
  coordinates. Lateral lane positions are in camera heights (+ right); ego-lane offsets are in lane
  widths (+ right of the lane centre). Neither is metric.
- **Missing versus negative.** Each frame records a status per configured role: `ok`, `not_scheduled`,
  `input_unavailable` or `failed`. A payload is present only with `ok`, and an empty payload is a negative.
- **Proposals versus findings.** `EventProposal` is a reviewable moment, never a finding: nothing in a
  result states that a collision, near miss or cut-in happened. Its `state` is `supported` when every
  condition its analyser checks held on observed data and `uncertain` when one could not be checked;
  `reasons` (codes) say which. Each evidence item is `observed` (read from the records), `estimated`
  (computed under a stated model) or `context` (never decisive). `rank` orders proposals; it is not a
  probability. `ReviewDecision` belongs to project data and never goes into a result or the cache; it
  copies the proposal's type and interval so it can be matched again after re-analysis.
- **Codes, not prose.** Lane reasons (`LaneReason`), proposal reasons (`ReasonCode`) and run warnings
  (`WarningCode`) are enumerated; messages and notes are for people.
- **Results.** Only a completed run has an `AnalysisResult`: perception, one `AnalyzerReport` per
  configured analyser (`ok`, `unavailable` or `failed`, with counts of patterns rejected by reason) and
  the proposals. `event_id` is derived from `analysis_key` (perception key plus analyser identities), so
  the same records and analysers always give the same identifiers. Cancelled and failed runs report a
  state and an error. Analyser run times are on the run outcome, not in the result, so results are
  deterministic.

## Cache

The key covers:

- the video's sha256;
- the timestamp source;
- the `RunConfig`;
- the engine and decoder versions;
- each perception component's `ComponentInfo`: name, version, checkpoint sha256, parameters and interval.

Event analysers are excluded, so they can be re-run on cached perception without loading models. An
entry is assembled in `.staging/` and renamed into place. On lookup, every file is checked against the
manifest and the content is re-hashed to the key. An entry with another schema version is rejected with a
reason and recomputed, never migrated.

An entry holds `perception.json` (the result without frames), `frames.jsonl` (one record per line) and
`masks/*.png`. On lookup each line is validated once and only its byte offset kept;
`result.perception.frames` is a `StoredFrames` sequence that parses records on access. Neither a run nor
a replay holds every frame record in memory, and the worker returns only the key.

## Adding a real model

Implement the relevant protocol in `adapters.py`:

- map library classes to `ObjectCategory` or canonical mask class names;
- put the checkpoint sha256 and every output-affecting parameter in `info`;
- load only local files in `load()` and free them in `close()`;
- raise `ComponentFailure` for a recoverable error on a single frame.

Downstream contracts do not change.

## Runtime registry and worker configuration

`models/engine/models.json` is the runtime registry; the weights stay local and are ignored by Git. Each
entry records role, filename, SHA-256, input shape, licence notes, export script, source checkpoint and,
optionally, the checkpoint's upstream URL and SHA-256.
`spectra.engine.models.default_specs(models_dir)` verifies both model files and returns
`(DetectorSpec, SegmenterSpec)`. It refuses missing files, hash mismatches, duplicate roles and paths
outside the registry directory. It never downloads models or loads ONNX sessions.

```sh
python -m spectra.engine analyze VIDEO --cache CACHE --models models/engine --track --lanes --events --strict-memory
```

`--detector`/`--detector-sha256` and `--segmenter`/`--segmenter-sha256` override the registry defaults.
Tracking, lanes and events still need their switches.

- `WorkerJob.analyzers=None` selects the defaults: approach, cut-in when lanes are on, then impact. A
  tuple of `AnalyzerSpec(name, params)` overrides that selection; an empty tuple disables analysis.
  Names: `apparent-approach`, `lane-entry-cut-in`, `camera-vehicle-impact`. Parameters are constructor
  arguments. Impact takes an `approach` parameter dictionary, to match a customised approach analyser.
  The analysis key depends on the ordered analyser identities; the perception key does not.
- `WorkerJob.cache_only=True` replays cached perception read-only. A missing, incompatible or corrupt
  entry fails with `StopReason.ERROR` before any model loads. The flag is not part of the cache key.
- `run_in_worker` validates `RunConfig` in the parent before creating a child; invalid configuration
  fails with `StopReason.ERROR`. Scripts must guard worker calls with `if __name__ == "__main__":`,
  because spawn imports the script again. Use a fresh worker per analysis, one at a time.

## Detection and tracking

- **Detector.** `DFineOnnxDetector` loads an ONNX export of the official D-FINE-N COCO checkpoint,
  produced locally by `evaluation/export_dfine_onnx.py`. It refuses a file whose sha256 differs from
  the configured one.
  - Input is 640 × 640 px.
  - Detections score 0.1 or higher; low scores are kept for the tracker's second association.
  - Only person, bicycle, motorcycle, car, bus and truck are kept, mapped to `ObjectCategory`.
  - Vehicle boxes overlapping above IoU 0.7 are suppressed whatever their class.
- **Hood detections.** The detector does not remove the camera vehicle's hood. A geometric rule would
  also remove a real vehicle at the moment of impact; the segmentation model's ego-vehicle mask is the
  intended remedy.
- **Tracker.** `BoTSortTracker` passes each frame's source timestamp, so lost tracks expire after
  `lost_seconds` of real time (2 s by default) and Kalman steps follow the real intervals. Detections
  are scaled into the frame image's pixels for camera-motion compensation, and the stored boxes remain
  the detector's.
- **Camera motion.** Each frame records the similarity transform (scale, rotation, translation in source
  pixels) that the tracker's motion compensation estimated from the last frame it ran on (`since_t`),
  or none when it failed, had no previous frame, or was skipped because there was nothing to track. It
  aligns images; it is not vehicle motion. During forward driving its scale mostly reflects parallax of
  the near road (median 0.06–0.15 per second on the sample clips, up to about 1.2 on a mountain road),
  so it is not a zoom measurement.
- **Camera-motion quality.** Each estimate also records its correspondences (`matches`), RANSAC
  `inliers` and the inliers' median residual (`residual_px`, source pixels), so a degenerate fit can be
  told apart from real image motion. The `trackers` library discards the inlier mask, so the adapter
  replaces its estimator with a copy of the 2.6.1 sparse-optical-flow method that returns the same
  matrix and keeps the mask. Loading refuses any other `trackers` version while camera motion is on.
- **Process isolation.** Run models through `run_in_worker` in the application. A crash becomes a
  failed run, and all model memory returns to the system when the child exits.

## Lanes, camera geometry and the ego lane

- **Segmentation.** `SegnetOnnxSegmenter` runs an ONNX export of commaai/comma10k-segnet (produced
  locally by `evaluation/export_segnet_onnx.py`; the file's sha256 is checked on load) at 512 × 384,
  by default every 0.2 s. Classes: `road`, `lane_marking`, `undrivable`, `movable`, `ego_vehicle`.
  It labels marking pixels; it does not identify lanes.
- **Geometry.** For a pinhole camera over a flat road with negligible roll, a lane boundary at lateral
  distance X appears as `x = a·(y − y_h) + b + c/(y − y_h)` (the hyperbola lane model), with
  `a = X / camera height`. `LaneGeometryEstimator` chains marking runs row by row, fits all boundaries
  jointly with a shared `b` and `c`, and picks the horizon row `y_h` that minimises the fit residual.
  - The camera's position in its lane comes from the boundaries' `a`. That makes it independent of
    camera yaw, of where the camera sits across the windscreen and of the image centre; an
    image-centre rule is biased by all three.
  - Road bends enter `c`, not `a`, which is what separates a curve from a lane change.
  - Every observation records the assumptions: flat road, negligible roll, and positions that refer to
    the camera rather than the vehicle centre. The camera's mounting offset in the vehicle cannot be
    recovered from the video.
- **Temporal tracking.** Boundaries are associated across frames by `a` after removing the common
  lateral shift. A boundary not measured on a frame is carried for at most `max_age_s` (1 s) and marked
  `measured=False` with its `age_s`. Points beyond a boundary's `evidence_y` extrapolate the fitted
  model, to at most twice the furthest evidence depth.
- **Ego-lane state.**
  - `estimated`: both ego boundaries are present and the width is plausible (1–5 camera heights).
  - `ambiguous`: a boundary is missing, the width is implausible, or the camera is on a boundary.
  - `unavailable`: there is no lane evidence.

  Only `estimated` carries an offset and confidence. The confidence is a heuristic ranking, not a
  calibrated probability.
- **Lane changes.** A lane change is a tracked boundary entering the band within 0.15 lane widths of
  the camera from one side, then staying on the other side for `confirm_s` (0.4 s).
  - Returning to the original side cancels it.
  - The observation (`lane_change`, `crossing` then `completed`, or `cancelled` on the one frame where
    a crossing ends without a change) is not an event; only `completed` means a change was accepted.
  - `t_start` is the last time the crossed boundary was at least 0.4 lane widths away.

  Three conditions guard the decision. Each addresses a measured failure mechanism, not a particular
  clip:
  - **Visible road depth of at least 0.18 of the image height.** Below that, a boundary's lateral slope
    cannot be separated from the bend. On synthetic curves the offset error is at most 0.15 lane
    widths down to a depth of 0.21, then 0.23 at 0.16 and 0.36 at 0.12. Shallower frames are
    `ambiguous`.
  - **The whole road must have moved.** At least 60 % of the crossed boundary's displacement must be
    matched by the common lateral shift of all boundaries. When only the tracked boundary moves, the
    tracker has switched markings; the camera has not changed lane.
  - **No sharp bend.** While the bend term moves the boundaries by more than 0.015 lane widths at the
    nearest visible row, lane-change decisions pause. On tight bends and grades the flat,
    gently-curving road model no longer holds, and extrapolating to the camera turns heading change
    into apparent lateral movement. The 0.015 limit lies between the values seen at plausible lane
    changes (≤ 0.006) and at false ones on a mountain road (0.020–0.051). It is a calibration on five
    clips and needs labels to validate. A boundary that reaches the other side of the camera while
    decisions are paused was never observed crossing; its next crossing is followed but not reported.

  Consequence: lane changes made during tight bends (and a return crossing right after one), or in
  under about one second, can be missed
  (2 of 32 synthetic cases).
- **Hood.** With an `ego_vehicle` class, the runner gives every detection its `ego_overlap` against the
  latest mask, if that mask is at most 1 s old. Detections at 0.5 or above are flagged `ego_vehicle`,
  kept in the record and withheld from the tracker. A real vehicle in front of the bonnet is labelled
  `movable`, so its overlap stays low.
- **Track relations.** With a tracker and an ego-lane estimator, each frame's tracks are related to the
  latest `estimated` ego lane, if it is at most 0.5 s old. The ground contact point is the bottom
  centre of the box. Possible relations:
  - `ego_lane`, `left_adjacent`, `right_adjacent`, `beyond_adjacent` or `off_road`;
  - `ambiguous`, within 0.1 lane widths of a boundary or beyond the model's reach.
  A relation whose contact point lies above the boundaries' evidence (the farthest row with markings observed
  on the fitted curve, dash gaps bridged) is flagged `beyond_evidence`.

  `beyond_evidence` marks relations made on the extrapolated part of the boundaries. Without a reliable
  ego lane, no relations are recorded.
- **Not implemented.** There is no predicted future path, metric distance, speed or steering. The
  corridor is the current lane as observed, plus the bounded model extrapolation.

## Event analysers

They run once after perception through the `EventAnalyzer` interface, on the frame records only, and
return drafts plus counts of the patterns they saw but did not propose, by reason; the run binds the
drafts to the analysis. Their parameters are in their `ComponentInfo`, which identifies the analysis
(`analysis_key`) but not the perception cache entry, so changing them never repeats perception. Without
their inputs (tracks, lane relations, camera motion) they raise `AnalyzerUnavailable`, reported as an
`unavailable` analyser: unavailable is not the same as no events. Observations the tracker predicted, or
matched from detections below its high-confidence threshold (0.6), are not used as evidence.

| Analyser | Proposes | Default | User-facing name |
|---|---|---|---|
| `apparent-approach` | `rapid_approach` | on | "Rapid approach (apparent)" |
| `lane-entry-cut-in` | `cut_in` | on with lanes | "Vehicle entering your lane" |
| `camera-vehicle-impact` | `impact` / `camera_vehicle` | on | "Possible impact involving the camera vehicle" |
| `road-user-contact` | `impact` / `road_users`, `experimental` | off | "Possible contact between other road users (experimental)" |

- **Apparent approach.** For a box of width `w`, `tau = w / (dw/dt)` is the time to reach the camera
  plane at constant closing speed (looming; no calibration, depth or ego speed needed).
  - Over a trailing 1 s window, Theil-Sen fits of `ln w` and `ln h` give growth rates with Sen's 90 %
    intervals. An observation counts when, for width and height alike, even the conservative bound
    gives `tau` ≤ 3 s. Box jitter widens the interval rather than triggering, and a box growing in one
    direction only (emerging from occlusion, a passing vehicle's side coming into view, a track
    sliding onto a neighbour) is not looming, which is isotropic. The estimate refers to the window's
    middle observation.
  - Boxes touching the frame edge are left out: their width does not measure the object.
  - Approaching observations less than 1 s apart form one episode, proposed or rejected with a reason.
  - Relevance: mostly observed in the camera's lane gives `supported`; mostly in another lane is
    rejected (`other_lane`). Otherwise, for a pinhole camera an object at constant lateral offset `X`
    has box centre `x = c + (X / W) w`, so a Theil-Sen fit of centre against width over the episode
    estimates its offset at the camera plane in its own widths. Beyond 1.5 widths it passes beside
    (`passes_beside`); otherwise the proposal is `uncertain` (`lane_not_established`). A visible
    vehicle side makes the estimate too small (it keeps proposals); camera yaw can bias it either way.
  - Identity jumps: a box-height step between observations at most 0.5 s apart that differs from the
    episode's own fitted growth by more than 2× either way makes the proposal `uncertain`
    (`identity_jump`). The comparison is with the episode's growth because a real approach at under a
    second's time-to-contact also doubles its box within 0.5 s.
  - Evidence: `apparent_tau_s`, `apparent_tau_height_s`, the conservative bound and
    `passing_offset_widths` (estimated); the episode and `ego_lane_fraction` (observed);
    `box_height_fraction` and the camera-motion scale rate (context). No distance, speed or collision
    time is reported, and an approach is not a claim of danger.
- **Cut-in.** Per track, relations are reduced to sides: the camera's lane, left or right adjacent,
  further, straddling a boundary, or unobservable (no relation, or beyond the lane evidence). Runs
  shorter than 0.4 s or 3 observations are dropped as relation noise. A sustained adjacent run
  followed by a sustained run in the camera's lane is a proposal:
  - `t_start` is the last adjacent observation, `t_peak` the first in the camera's lane and `t_end`
    the observation that makes that run sustained.
  - The camera's lane must keep its two boundary identities across the entry. If they changed and a
    completed ego lane change overlaps the entry, every vehicle's relation shifted with the camera and
    it is rejected (`ego_lane_change`). If they changed without one, including during a crossing that
    was cancelled, the proposal is `uncertain` (`boundaries_reidentified`).
  - An entry longer than 3 s from the last adjacent to the first in-lane observation was not observed
    as it happened, so the proposal is `uncertain` (`entry_not_observed`).
  - There is no proximity or risk condition. A proposal means a vehicle appears to enter the camera's
    observed lane, not danger or fault.
- **Impact involving the camera vehicle.** Two independent signals must coincide, and a missing signal
  never stands in for one:
  1. an apparent-approach episode, as above, that ends **close**: box height at least 0.25 of the
     frame, the box reaching the bottom or a side edge within 0.5 s, or the track lost within 0.5 s
     while at least 0.1 of the frame high;
  2. a **jolt** in reliable camera-motion estimates from 0.3 s before to 1 s after the close time: a
     frame-to-frame change of the rotation, translation or log-scale rate more than 10 robust standard
     deviations above the median change over the 5 s before. Reliable means at least 20 inliers and an
     inlier share of at least 0.3, on consecutive tracker frames at most 0.25 s apart.

  When the jolt cannot be judged (an estimate in the window missing or unreliable and the reliable ones
  show none, or fewer than 10 reliable changes in the baseline), nothing is proposed: the pattern is
  rejected as `motion_estimate_missing` or `motion_baseline_insufficient` and listed with its time and
  track in the report's `unresolved` patterns. The moment is still proposed by the apparent-approach
  analyser. With every estimate reliable and no jolt it is `no_jolt`; an approach that ended far away is
  `not_close`. The state is the approach's (`supported` only for an approach observed in the camera's
  lane without an identity jump). Proposals sharing a track or overlapping in time merge (`merged`).
  A jolt alone (shake, potholes, bends: present on every sample clip), an approach alone (a near miss,
  which stays a rapid approach) and closeness alone are never proposals.
- **Contact between other road users** (experimental, off). Two confident tracks whose boxes overlap
  at a similar ground row (within 0.15 of the larger height, height ratio 0.6–1.6), whose centres
  converged over the preceding second (Theil-Sen slope and its interval below 0), followed within 1 s by
  a camera-compensated velocity change of one of them more than 10 robust standard deviations above its
  own preceding second. Every proposal is `uncertain` (`not_validated`) and `experimental`.

None of the analysers has been validated against human labels; all thresholds are unvalidated starting
values recorded in their `ComponentInfo`.

## Memory policy

| System pressure (macOS) | `standard` (product) | `strict` (benchmarks) |
|---|---|---|
| normal (1) | run | run |
| warning (2) | run, with a note on the result | refuse or stop |
| critical (4) | refuse or stop | refuse or stop |

Under both policies a run also stops if process peak memory exceeds the budget (half of RAM by default).

## Schema reference

The authoritative definitions are [contracts.py](contracts.py), Schema `1.0`. Research schema versions
are rejected, not migrated. `CameraMotion` records matches, inliers and residual error; lane observations
carry enumerated reasons and an explicit cancelled lane-crossing state.
`EventProposal` carries type/subtype, state, reasons, experimental flag, ordered times, track IDs,
evidence items, rank and analyzer name. Evidence basis is observed, estimated or context, with units
and optional interval bounds. Its ID is the first 16 hex characters of the digest of analysis identity,
analyzer, type/subtype, sorted tracks and start rounded to two decimals.

`AnalyzerReport` has component identity, status/reason, considered/proposed/rejected counts and timed
unresolved patterns. `considered = proposals + sum(rejected)` is validated. Missing-signal rejections
must be represented as unresolved patterns. `AnalysisResult` has keys, perception, reports and events;
there is no analyzer duration or separate warnings field on it. Warnings live on perception;
`RunOutcome.analyzer_seconds` carries timing. Review decisions record analysis/event identity, type,
interval, reviewer, verdict and optional corrected interval, separately from derived artifacts.

See [system architecture](../../ARCHITECTURE.md) for boundaries, user-facing names, validation and release
blockers.
