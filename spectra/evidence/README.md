# Review evidence and timeline

`spectra.evidence` turns an engine `AnalysisResult` into material a person can review: one evidence
package per proposal, a chronological timeline, optional overlay data, and references into the source
video, with an optional clip copied from the stream. It reads the engine's Schema v1 contracts and
does not judge anything itself:
- Proposals are embedded unchanged.
- `supported` still means only that the analyser's own conditions held.
- Human decisions (`ReviewDecision`) are kept with project data and never written here.

```
AnalysisResult (engine) ──► build() ──► EvidencePackage per proposal + Timeline
                                  │            └─► EvidenceStore  <root>/<build_key>/...
                                  ├─► overlay()   boxes, tracks, lanes, read from the perception cache
                                  └─► resolve()   interval checked against the decoded source
                                      extract_clip()  stream copy, on request only
RunOutcome without a result ──► from_outcome() ──► Timeline that states why, with no entries
```

| Module | Responsibility |
|---|---|
| `contracts.py` | `ReviewConfig`, `EvidencePackage`, `Timeline`, `TimelineEntry`, `ReviewSegment`, `Overlay`, `Note`, `Coverage`, `AnalysisState`, `EVIDENCE_VERSION` |
| `build.py` | review intervals, per-package track and lane summaries, relations, timeline, analyser coverage |
| `overlays.py` | per-frame drawing data for a review interval; a lookup by display time |
| `media.py` | source identity, interval resolution, stream-copied clips |
| `store.py` | atomic, verified persistence by build key |

## Evidence package

The package has one shape for every event type. It holds:
- **Identity:** `analysis_key`, `perception_key`, `build_key` and `evidence_version`.
- **The proposal:** the engine's `EventProposal`, unchanged, with its event ID, type and subtype, start, peak and end times, tracks, evidence items, state, reasons, `experimental` flag and rank.
- **The analyser:** its `ComponentInfo` (name, version, parameters).
- **The video:** the source's `VideoInfo` (hash, size, codec, time base, stream start, timestamp source) and the analysed span.
- **The review interval** (see below).
- **Track summaries:** what the records hold for each related track inside the interval: category, first and last observed box, observed and predicted counts, and whether the track ended early. Predicted boxes are never counted as observations.
- **Lanes:** runs of ego-lane state, and runs of lane relation per related track. A relation that was not computed is `None`, which means unknown, never "elsewhere". `available: false` means no lane estimator was configured.
- **Gaps:** stretches without analysed frames.
- **Run warnings:** warnings at a time inside the interval, or for the whole run.
- **Unresolved patterns:** patterns any analyser could not judge (missing signals) that overlap the interval.
- **Related proposals.**
- **Notes:** machine-readable limits of the material (`Note`), such as `track_ends_early`, `lane_unobserved`, `motion_missing`, `frame_gap`, `outside_analysis`, `clipped_at_video_start`, `experimental`. Notes describe the material; they never change the proposal.

## Review intervals

One rule applies to every event type:
1. **Start from the evidence span:** the proposal span, widened to cover every evidence item. A cut-in's adjacent-lane observations can begin before its `t_start`.
2. **Add context:** `pre_s` (default 3 s) before and `post_s` (default 2 s) after. If the two together exceed `max_context_s` (default 20 s), both are scaled down proportionally. The evidence itself is never cut.
3. **Clip to the video:** clip to `[0, duration]`, or to the last analysed frame when no duration is declared. Clipping is recorded as `clipped_at_video_*`.
4. **Mark unanalysed context:** context outside the analysed span is noted `outside_analysis`, because it has no records or overlays.

All times are source presentation times, as in the engine. A review interval is a span to show; it is not the time of any incident.

Overlapping intervals are joined into `ReviewSegment`s for playback. Every proposal stays its own timeline entry.

## Timeline

Ordering is deterministic. Entries sort by `(t_start, t_peak, type order, event_id)`: chronologically, then by peak time, then in a fixed order of types, then by ID.

Each entry carries:
- the event ID, type and subtype;
- `title_key` (for example `event.impact.camera_vehicle`) and an English fallback title;
- start, peak and end times, and the review interval;
- the state, reasons and `experimental` flag;
- the related track IDs and the rank;
- the package path and the segment index;
- related proposals and notes.

The timeline also states what each analyser covered, so an empty timeline is never read as "nothing happened".

| `Coverage` | Meaning |
|---|---|
| `proposed` | ran and proposed moments |
| `no_proposal` | ran over the analysed span and proposed nothing at its operating point |
| `unavailable` | its inputs were missing; says nothing either way |
| `failed` | raised an error; says nothing either way |

Patterns an analyser could not judge (`unresolved`, from the engine report) are listed with their times. They mean insufficient evidence, not absence.

| `AnalysisState` | Meaning |
|---|---|
| `complete` | result; the whole span was decoded |
| `incomplete` | result, but decoding stopped early; the tail was not analysed |
| `cancelled` | no result |
| `resource_stopped` | no result; the memory safeguards stopped the run (`RunOutcome.stop`) |
| `failed` | no result; the video could not be analysed, or an error occurred |

### Relations

Relations come only from what the engine recorded:
- **`based_on` / `basis_of`:** an impact proposal names the approach analyser it used in its parameters, and carries that approach episode as its `approach_episode` evidence. It links to the approach proposal with the same episode (track, start and end) only when the analyser's name, version and parameters match the configured approach analyser exactly.
- **`shares_track`:** two proposals about a common track whose spans overlap.

A relation is not a merged or confirmed incident. Both entries remain, and each is reviewed on its own.

## Overlays

Overlays are optional data for drawing over the source video. They are off unless the interface turns them on. Each analysed frame in the interval gives one `OverlayFrame`, read lazily from the perception cache with no model:
- the frame's tracked boxes, with a `predicted` flag, a `related` flag and the lane relation (`None` if not computed);
- the ego-lane state and boundaries, normalised, with `evidence_y` beyond which a boundary is extrapolated.

An overlay frame is valid from its time until the next analysed frame, or at most `overlay_max_age_s` (default 0.25 s). Positions are never interpolated. At a display time with no valid overlay frame, nothing is drawn, and the interface should say "not analysed here", not "nothing there". `frame_at(frames, t)` implements this lookup.

## Source video and clips

**Default: a reference, not a file.** A package refers to the source by its hash and to a span of presentation time. `check_source` reports whether the file at a path is the analysed video: `available`, `missing`, `changed` or `unreadable`. `resolve` decodes the interval's timestamps without converting pixels and reports:
- the first and last frame;
- the frame on screen at the start;
- the keyframe that decoding must start from;
- the frame count and the longest frame interval;
- gaps, decode errors and dropped timestamps;
- the stream start offset, to add for seeking at the container level.

**Optional: a stream-copied clip** (`extract_clip`, or `EvidenceStore.clip` to cache it):
- **How it is made:** packets are copied from the keyframe at or before the interval start up to the interval end, without decoding or re-encoding. No encoder is used, and the pictures are the source's own.
- **Timing:** the clip starts at that keyframe, `origin_s`, which can be up to one keyframe interval before the review interval. Clip time `c` corresponds to source time `origin_s + c`, and a player opens the clip at `start_s - origin_s`.
- **Verification:** every clip is decoded once after writing. Its frame times must match the source's frame for frame, and it must contain every source frame up to the interval end. Otherwise it is deleted and an error raised.
- **Container:** H.264, HEVC, MPEG-4 and AV1 go into MP4; VP8 and VP9 into WebM; anything else into Matroska. Audio is not copied.
- **Safety limits:** writes are atomic, through a partial file that is removed on any failure or cancellation. The interval length is bounded (default 120 s), and free disk space is checked against an estimate plus a reserve.
- **Refused:**
  - sources without container timestamps (times assume a rate, such as a raw `.hevc` stream), since a clip could not keep correct times; play the source interval instead;
  - missing or changed sources;
  - intervals that are too long.

**Not decided here: whether a browser can play the codec.** A stream copy keeps the source codec:
- VP9 in WebM and H.264 in MP4 play in current browsers.
- HEVC support varies.
- Ogg Theora, MPEG-4 Part 2 and raw elementary streams generally do not play.

### Release requirement for browser playback

See the authoritative [licensing and playback release blockers](../../ARCHITECTURE.md#licensing-and-release-blockers).
The installed PyAV wheel is not release-ready; no transcoding encoder has been approved.

## Storage

```
<root>/<build_key>/manifest.json         evidence version, keys, configuration, sha256 of every file
<root>/<build_key>/timeline.json
<root>/<build_key>/packages/<event_id>.json
<root>/<build_key>/clips/<event_id>.<ext>  +  <event_id>.json (ClipFile record)   on request only
```

- **Build key:** `sha256(evidence version, analysis key, review configuration)`. A change to the analysis, the configuration or `EVIDENCE_VERSION` gives a new directory; earlier ones are left alone.
- **Writes:** builds are written to `.staging/` and renamed into place.
- **Reads:** readers verify every file hash, the build key and the expected analysis key, and say why a build was rejected.
- **Repairs:** a damaged build that is rewritten is set aside as `<key>.rejected.*`, never deleted.
- **Clips:** reused only while their record, file hash, source hash and interval all match.
- **Not stored:** overlays; they are read from the perception cache on request.
- **Location:** everything here is derived data, kept apart from source videos, perception caches and review decisions. Builds are byte-for-byte reproducible from the same inputs.

## Real-footage check

`scripts/check_real_clips.py` is an import-safe, strict, sequential end-to-end check on real clips. It
verifies clips through the manifest and models through the runtime registry. With `--cache-only`, it refuses
missing or invalid perception caches before any model loads.

It checks:
- deterministic evidence and unchanged proposals
- store round-trips, overlays and interval resolution
- up to two stream-copied clips per source, and unchanged source hashes

Assumed-rate streams must refuse clip extraction rather than invent frame timestamps. Clips without
proposals still exercise a bounded media interval, and zero overlay coverage is reported.

Run it once, at normal memory pressure, into a new output directory outside the repository:

```sh
.venv/bin/python -B scripts/check_real_clips.py --manifest evaluation/manifests/real_sample.json --clips kumamoto bolide comma --cache-only --out /private/tmp/spectra-evidence-check-new
```

Outputs include build and media timings, worker analyser timings and the **parent** process's peak RSS; the
script does not measure the worker's peak RSS. It exits nonzero on any failure (3 for a resource refusal),
never retries, and refuses unknown macOS pressure. The `real_video` test wrapper is opt-in with
`pytest -m real_video`. This check has **not yet passed** on real footage.
