# Evaluation protocol

How Spectra components are compared. It applies to the baseline (D-FINE-N, BoT-SORT, comma10k-segnet,
the lane geometry, the approach and cut-in analysers) and to every later challenger.

## Ground truth

Annotations follow `annotations.py` (schema 1.1, JSON Lines; 1.0 files load unchanged). Each event has an
origin:

| Origin | Set by | Ground truth? |
|---|---|---|
| `human_confirmed` | clip `provenance.method = "manual"`, or an event with `origin` and `reviewed_by` | Yes |
| `synthetic` | `method = "synthetic"` (generated scenes) | Yes, for logic tests only; never pooled with real footage |
| `source_derived` | `method = "imported:<dataset>"` or `"converted:<dataset>"` | No |
| `draft` | anything else, e.g. model-assisted drafts | No |

- `metrics.evaluate` and `engine_scoring` score only ground-truth origins unless a run lists others in
  `--origins`; the summary then states that it was scored against labels that are not ground truth.
- `negative_for` (verified absence) is accepted only on manual or synthetic clips. Absence cannot be
  imported or drafted.
- `unmeasurable` records event types that cannot be judged on a clip (for example, no view of the lane);
  such a clip is not scored for that type and its candidates count as unscored.
- `source_labels` keep each dataset label verbatim (`label`, raw `fields`, `record` pointing at the file
  and key or line) with converted times, participant identities and optional boxes. `review_for` lists
  event types a reviewer should look for; it is not a label.
- A reviewer turns a source label into an event by adding it with `source_refs`, the agreed times and
  `origin: human_confirmed` plus `reviewed_by` (or by reviewing the whole clip and setting
  `method: manual`).
- Licence: every clip carries the registry's terms. `engine_scoring` refuses a clip with
  `evaluation_allowed: false`.

Restricted datasets are imported into `evaluation/restricted/` (git-ignored). Nothing under it is
committed or shipped, and the engine never imports from `evaluation/`.

## Importers

| Dataset | Importer | Verified against | Produces |
|---|---|---|---|
| DoTA | `importers/dota.py` | All 4,677 published annotation files | Source labels only (anomaly category, ego involvement, anomaly interval, participant tracklets) |
| PREVENTION | `importers/prevention.py` | The toolkit's parser and the `lane_changes.txt` and `log_camera1.txt` of all 11 drives (9 import; 2 have defective camera logs) | Source labels for every lane-change line, timed from the camera log; no events, because no drive uses the cut-in code |

### DoTA label semantics

- Categories (`accident_name`): `moving_ahead_or_waiting`, `start_stop_or_stationary`, `lateral`,
  `oncoming`, `turning`, `pedestrian`, `obstacle`, `leave_to_left`, `leave_to_right`, `unknown`. They are
  anomaly types, mostly collisions, and are not our events:
  - an ego-involved `moving_ahead_or_waiting` or `start_stop_or_stationary` clip may contain a rapid
    approach and a collision;
  - an ego-involved `lateral` clip may contain a cut-in.

  Each needs a reviewer, who records our definition's times and involved vehicle.
- `ego_involve` identifies camera-vehicle involvement (2,724 clips true, 1,953 false).
- Participant tracklets exist: `obj_track_id` with boxes, only on anomaly frames, 1–2 per clip in most
  clips (301 clips have none). They identify the involved road user. They do not score detection or
  tracking, because other road users are not boxed.
- Clips cannot be downloaded individually from an official source. The source is YouTube (by URL list)
  or 55 GB of frame archives.

### PREVENTION label semantics

Checked on the 852 lines of all 11 released drives, read by HTTP range requests from each
`processed_data.zip` without downloading the archives.

- `LC_type` (`type`): the toolkit names 1 cut-in, 2 cut-out, 3 left, 4 right, 5 hazardous and 6 pedestrian.
  The released files contain 3 (381), 4 (468), 5 (2) and 6 (1). **No drive has a cut-in (1) label.**
- A left or right change by a surrounding vehicle does not say whether it ends in the camera's lane.
  Each must be reviewed before it can count as a cut-in or as a non-cut-in.
- Only surrounding vehicles are annotated. The camera's own lane changes are not, and completeness is
  unknown: a reviewer must also watch for cut-ins with no line in the file.
- `f0 fe ff`: start, crossing (rear middle between the lanes) and end frame. `-1` is unknown. Many lines
  have `f0 = -1` and `fe = ff`, so only the end is known. One line (RECORD5/DRIVE1) is out of order and is
  kept without times.
- `id_l`, `id_d`: either may be -1. On RECORD4/DRIVE3 `labels.txt` is empty and `id_l` is always -1, which
  suggests that `id_l` indexes `labels.txt` and `id_d` indexes `detections_tracked.txt` (unconfirmed). They
  are kept as `l:<id>` and `d:<id>`. `blk` is 0 or 1, presumably the blinker.
- Time: `logs/log_camera1.txt` lists `cloud frame t0 t1` for frames 0..n-1. `t0` is in microseconds:
  RECORD4/DRIVE3 spans 430.7 s for a drive listed as 7 min. Frame rates are 10.7–11.7 fps, not 10.
  RECORD4/DRIVE1 and DRIVE2 have about 210 backward jumps of 0.91 s each, consistent with a missed
  seconds carry. The importer refuses them rather than guessing a repair.
- In RECORD4/DRIVE3 `labels.txt` and `trajectories.txt` are empty. Vehicle boxes exist only in
  `detections_tracked.txt`, which comes from the authors' detector, not manual labels.
- Engine times are mp4 presentation times. Before scoring, check that `video_camera1.mp4` has one frame
  per camera-log frame, so that frame `i` is at log time `i`.

## Scoring engine runs

`engine_scoring.py` scores the candidates that `python -m spectra.engine analyze ... --events` prints,
over the processed window only:
- events crossing the window edges become `ambiguous`;
- `supported` and all candidates are reported separately;
- a named result is never overwritten.

## Metrics

Compute a metric only where independent annotations support it. Report counts and sample sizes with every
rate. Report per data source and per condition, and never pool synthetic, compilation and real dashcam
footage.

| Area | Needs | Metrics | Status |
|---|---|---|---|
| Events (approach, cut-in) | Ground-truth events, or verified negatives | `metrics.evaluate`: one-to-one matching (peak within 1 s or tIoU ≥ 0.3), TP/FP/FN, precision, recall, false candidates per hour, start/peak/end error, recall by condition, ambiguous hits | Implemented |
| Involved vehicle | Event `involved` boxes or participant tracklets | Share of TPs whose candidate track overlaps the labelled participant (box IoU ≥ 0.5 on ≥ half the overlapping frames) | To implement with the first boxed events |
| Detection | Exhaustive boxes on sampled frames | Precision and recall at IoU 0.5, by box height (< 30 px, 30–80 px, > 80 px), occlusion and light; non-vehicle classes only if boxed | Needs labels. DoTA participants are not exhaustive |
| Tracking | Identities followed through a span | IDF1, identity switches, fragmentations, recovery after occlusion; flag spans with large camera motion (`camera_motion` in frame records) | Needs labels. `evaluation/tracking.py` fragment counts are label-free proxies only |
| Lanes | Boundary polylines on sampled frames; ego-lane and lane-change intervals | Boundary lateral error in lane widths at fixed image rows; ego-lane state agreement; vehicle-to-lane relation agreement (`ambiguous` and unobservable reported, not counted as errors); ego lane-change event matching | Needs labels |
| Unmeasurable | `unmeasurable` entries and analyser `ambiguous` statuses | Share of time and of events that could not be judged, per condition | Counted |

Do not quote an accuracy figure for unlabelled footage.

## Conditions

Tag `conditions` per clip or event. Coverage of the current real sample (Commons and comma clips) and what
remains:

| Condition | Represented now | Still needed |
|---|---|---|
| Off-centre or yawed camera, long bonnet | Kumamoto (hood visible) | Several mounts and vehicles |
| Curves, grades | Alboloduy hairpins | Urban curves with traffic |
| Wide-angle distortion | Not confirmed | Fisheye dashcam footage |
| Missing or occluded markings | I-270 dense traffic | Rural and unmarked roads |
| Night, rain | comma night/rain | Rain at day; glare |
| Vibration, unstable tracking | Kumamoto earthquake, Alboloduy | Ordinary rough roads |
| Variable frame rate | I-270 (about 14 fps average, gaps up to 1.47 s) | Phone dashcams |
| Real positives (approach, cut-in) | No sufficiently labelled positive set | See the smallest set below |

The separate [collision sample manifest](manifests/collision_sample.json) records one downloaded
collision-positive clip for internal evaluation. It is additional to the five-clip baseline above,
not part of that baseline or its draft annotations. Preview stills bracket an apparent impact at
20–21 s, but human-confirmed event labels and precise timing remain unavailable. This single sample
does not establish detector accuracy or provide sufficiently labelled positive events for reliable
precision/recall estimates. Preserve its provenance and internal-use restrictions; do not redistribute
the footage.

Synthetic scenes verify the geometry and logic only.

## Reproducible comparisons

A comparison is valid only if all candidates share:
- the same video files (sha256 in the manifest);
- the same annotation file (its sha256 is recorded by `engine_scoring`);
- the same split, match settings and origins.

Perception runs use the engine worker with the `strict` memory policy and one model at a time. Record each
component's `ComponentInfo` (checkpoint sha256, parameters, interval) through the perception cache key.
Analyse at the same sampling interval on presentation timestamps, never frame index divided by nominal
fps. When a model needs a different input size, record it as a parameter and leave the sampling
unchanged.

Resources: report seconds per video second and process peak memory from the run report, on the same
machine and under the same memory policy.

Keep baseline results: `engine_scoring` refuses an existing run name and appends to `results_log.jsonl`.
Never overwrite a baseline entry; rerun under a new name.

A model's native output format is not a reason to prefer it. Every detector goes through the same
adapter contract and the same class mapping before scoring.

## Smallest real-data set

| Part | Source | Size | Labels needed | Measures |
|---|---|---|---|---|
| Negatives | The 5 sample clips already on disk | 0 new | A full human review per clip, plus `negative_for` | False candidates per hour |
| Cut-in positives | 4 Commons CC BY-SA dense-traffic drives | About 614 MB | Every lane entry into the camera's lane, separated from the camera's own lane changes | Cut-in recall and timing |
| Cut-in candidates (private) | PREVENTION RECORD4/DRIVE3 | 193 MB front video (annotations already read) | Review of the 18 listed lane changes, plus a full watch for unlisted cut-ins | Cut-in recall and false candidates on a calibrated rig, if any change ends in the camera's lane |
| Approach and collision positives | 4 Commons PD-CCTV incident clips | About 60 MB | Approaching vehicle, start, contact | Approach recall |
| Approach positives (optional) | 20–40 ego-involved DoTA `moving_ahead_or_waiting` clips | 2–4 s each, from YouTube | As above | Only if the owner clears DoTA |

`engine_scoring`, the caches and the analysers need no change to score this set once the labels exist.

## Reviewing draft annotations

`annotations/real_sample.draft.jsonl` was prepared from a handful of rendered frames. Nothing in it is
ground truth until a person has watched each clip in full and confirmed or corrected it. Per clip:

1. Watch the whole clip; for long clips, at least the analysed first 150 s.
2. For each collision, rapid approach and cut-in, record `t_start`, `t_peak` and `t_end` in seconds of
   decoder presentation time. Use the player position, not a burned-in clock.
3. Mark an event `ambiguous` when reasonable reviewers could disagree.
4. Only after watching the whole analysed span, add types to `negative_for`.
5. Set `provenance.annotator` to the reviewer and `method` to `manual`.

Decoding caveats:
- `commons-i270-southbound` has a variable frame rate (about 14 fps on average, gaps up to 1.47 s), although
  the container reports 30 fps. Seeking by time in common players can land seconds away; check against the
  burned-in clock.
- `comma2k19-…` is a raw HEVC stream without timestamps (night driving in rain). Its times assume the
  documented 20 Hz camera rate.

Moments already inspected from frames only (not a review):

| Clip | Time (s) | Observation |
|---|---|---|
| commons-i270-southbound | 29.0–33.6 | Camera vehicle changes lane to the left; not a cut-in |
| commons-i270-southbound | 37.6–38.5, 63.9–64.8 | Looked like closing, but measured tau is 7–75 s; drafted as ambiguous approach for a reviewer to confirm or delete |
| commons-kumamoto-earthquake | 10.6–14.3 | Strong camera shake from the earthquake; no collision visible; D-FINE-N detects the hood as a car (score 0.74 at 10.6 s) |
| commons-alboloduy-a1075 | 10.5–78.7 | Mountain road, dashboard in view, strong vibration; no other vehicles at inspected times |
| commons-bolide-2024 | 5.8 | Night-sky flash near the end; no vehicles |

The smallest detection review takes about 30 minutes. For each frame below, mark every vehicle box from
each model as correct, false or duplicate, and circle clearly visible vehicles that neither model boxed.
Render a frame with both models with
`python3 -m evaluation.review_frames evaluation/datasets/real_sample/<file> evaluation/caches/dfine_n/<clip> evaluation/caches/rfdetr_n/<clip> --time <t> --out frame.jpg`
(add `--assume-fps 20` for the Comma clip).

| Clip | Time (s) | Why selected |
|---|---|---|
| commons-i270-southbound | 105.933, 56.733, 44.433 | Largest disagreements (D-FINE-N 21/22/18 vehicles, RF-DETR Nano 10/13/11) |
| comma2k19-b0c9d2329ad1606b-10 | 54.300, 29.900 | Night and rain (D-FINE-N 6, RF-DETR Nano 1 at 54.3 s) |
| commons-alboloduy-a1075 | 142.543 | Empty road: both models report one vehicle, in different places |
| commons-kumamoto-earthquake | 10.560 | Camera vehicle hood detected as a car |

Tracking identity (about 15 minutes): confirm that the lead vehicle keeps one identity across
commons-i270-southbound 55–70 s. Also confirm that the identity changes near 33 s come from the camera
vehicle's lane change rather than tracker errors.
