# API reference

`spectra/api/` is the local FastAPI backend for the review application. It orchestrates the
[engine](../engine/README.md) and [evidence](../evidence/README.md) layers and owns application state.
It never loads a model or runs `AnalysisRun` in its own process: analyses run one at a time through
`run_in_worker` from a dedicated thread, because RSS budgets use the lifetime peak.

## Running

`python3 run.py` starts this server and serves the web application; it passes these options through.
To run the server alone:

```sh
.venv/bin/python -m spectra.api --library ~/Movies/Dashcam
```

| Option | Default | Meaning |
|---|---|---|
| `--port` | `8765` | Port on `127.0.0.1`. There is no host option; the server is loopback-only. |
| `--data-dir` | `var/` in the repository (Git-ignored) | Database, perception cache, evidence store and temporary files. |
| `--models` | `models/engine` | Runtime model registry folder. |
| `--library DIR` | none | Folder offered from the start, in addition to folders added in the application. Repeat for several folders. |
| `--allow-origin URL` | none | Extra loopback page origin, such as `http://127.0.0.1:5173` for the web application's development server. |
| `--frontend DIR` | `frontend/dist` when it has been built | Web application build to serve at `/`. Without a build only the API is served. |

The OpenAPI document is served at `/api/openapi.json`. Interactive documentation pages are disabled
because they load scripts from a CDN.

## Data directory

| Path | Contents |
|---|---|
| `spectra.sqlite3` | Video registrations, probe results, browser checks, analyses with their stored reports and proposals, evidence state, human review decisions and event edits (WAL mode). |
| `cache/` | Perception cache written by analysis workers. |
| `evidence/` | Evidence store: timelines, packages and on-request stream copies. |
| `tmp/` | Temporary files owned by the server. |

Directories are created with mode `0700`. Source videos are never stored here.

[store.py](store.py) defines the only schema, version 1 (`PRAGMA user_version`). It has five tables:
- `videos`
- `analyses`, which references `videos`
- `library_folders`
- `review_decisions` and `event_edits`, both keyed by `(analysis_id, event_id)` and referencing `analyses`

All references use `ON DELETE RESTRICT`. A partial unique index allows one queued or running analysis per
video. CHECK constraints bound every enumerated value, hash length, revision and edit time.

An empty file is initialized atomically. On every start, the version and every table and index definition
are compared with the schema. A database at another version, or one that differs, is refused and left
unchanged: health reports the database unavailable, and the file is never reset, converted or upgraded.
Review decisions and edits never touch analysis rows, stored results, caches or evidence.

## Library folders and videos

Videos are read in place and are never copied, written, renamed or deleted. Only the server knows real paths;
a page can name a file only as `(folder id, path relative to that folder)`.

- **Folders.** There are two kinds:
  - Folders given with `--library` are offered on every start and cannot be removed (409).
  - `POST /api/library/folders` takes no path. It opens the operating system's folder dialog on the
    computer running the server (`osascript` on macOS, `zenity` elsewhere), one at a time; without one it
    answers 501. The folder must exist, be readable, not be a whole disk and not lie inside the data
    directory. It is remembered in `library_folders`.
- **Removing a folder** removes it from the allowlist only. Its registrations remain but report
  `outside_library` until the folder is added again. A folder that disappeared stays listed with
  `available: false`.
- **Browsing.** `GET /api/library/{folder_id}/entries?path=` lists subfolders and files with a video
  extension. Hidden, unreadable and out-of-folder entries are omitted.
- **References.** [`resolve_video_reference`](library.py) accepts only a relative path without `..` or NUL
  that stays inside the folder after symlinks, is outside the data directory, and is a readable regular
  file with a video extension. It opens nothing, so a FIFO or device cannot block the server. The format
  always comes from probing.
- **Probing.** Registration stores path, size, modification time and inode. [Probing](probe.py) then runs
  one file at a time in a helper process (`python -m spectra.api.probe PATH`) in its own session. The
  helper computes the full SHA-256, decodes the first frame, records audio codecs and prints one JSON report.
  - Crashes, invalid output and timeouts (60 s plus 1 s per 20 MB) become probe errors, and the helper's
    process group is killed. Decoder memory never enters the server.
  - A probe interrupted by shutdown resumes at the next start.
  - Registering an unchanged file again returns it (200). A changed file is probed again and loses its
    browser check. Corrupt and unsupported files are not re-read until they change.
- **Change detection.** Every read compares size, modification time and inode, and checks that the path
  still resolves to itself. Media is served only while the source is `available`. Analysis verifies the
  full hash again before it starts.

There is no upload: videos are only referenced in place, and their size is not limited.

## Playback eligibility

Review needs playback, so only videos the user's browser plays can be analysed.

1. **Assessment.** [`assess`](playback.py) classifies the probed container and codecs, never the extension:

   | Assessment | Formats |
   |---|---|
   | `likely` | H.264, VP9 or AV1 in MP4; VP8, VP9 or AV1 with Opus or Vorbis in WebM |
   | `uncertain` | HEVC; AV1 (not Safari on M1); QuickTime; other Matroska; Theora; unusual audio |
   | `unsupported` | MPEG-4 Part 2 and other codecs; AVI, MPEG-TS and elementary streams; missing timestamps |

2. **Serving.** `media_url` serves the file only when the probe is ready, the source is available and the
   assessment is not `unsupported`.
3. **Browser check.** The application loads `media_url` in a `<video>` element, waits for a decoded frame and
   posts `{playable, detail, sha256}` to `/api/videos/{id}/playback-check`. A report for other contents is
   refused (409). The result is stored with the SHA-256 and ignored once the file changes.
4. **Eligibility.** `analysis.eligible` requires a ready probe, an available source, an assessment other than
   `unsupported` and a successful browser check of the current contents. Otherwise `analysis.reasons` says why.

The browser check is a usability gate, not a security boundary. Nothing is transcoded.

## Analysis jobs

[`AnalysisService`](jobs.py) runs a persisted FIFO queue (the `analyses` table) on one runner thread. Each
analysis runs in a fresh `run_in_worker` process under the unchanged `standard` `ResourceGuard`; the API
process never loads a model.

- **Profile.** One fixed profile: detection and tracking every 0.1 s, segmentation every 0.2 s, lanes and the
  default analysers. Models are verified with `default_specs` for every job, and the profile and model hashes
  are stored with the analysis. Analysing an unchanged video again is a cache hit with the same keys.
- **Requesting** (`POST /api/analyses`) returns the video's active analysis if there is one (200); a partial
  unique index prevents a second. Otherwise it queues one (201) if the video is eligible, the models verify
  and memory pressure is below critical. Refusals are 409 (not eligible) or 503 (models, memory, shutdown)
  with `detail.reasons`.
- **Executing.** Before the worker starts, the runner re-reads the registration, refuses a changed SHA-256,
  runs `check_source` with a full hash and checks memory pressure again. A failure finishes the analysis
  without a worker.
- **Cancelling.** A queued analysis is cancelled at once; a running one stops cooperatively through its
  `CancelToken`, and atomic cache commits leave existing caches intact. Cancelling a finished analysis is
  409, except a cancelled one (no-op).
- **Restarting.** At startup, analyses left `running` become `failed`/`interrupted` and `queued` ones
  `cancelled`/`interrupted`. Work is never resumed silently.
- **Shutting down.** On Ctrl+C or SIGTERM, new requests are refused and open responses get 3 seconds
  (`GRACEFUL_SHUTDOWN_S`); event streams are then closed. The running analysis is cancelled
  (`cancelled`/`shutdown`) and shutdown waits for its worker.
- **Timeout.** 900 s plus 20 times the video duration (6 h when the duration is unknown).

| State | Meaning |
|---|---|
| `queued` | waiting; `queue_position` 1 is next |
| `running` | verifying the source or in the worker; `progress` has the stage, frames, source time and fraction (null when unknown, never estimated) |
| `completed` | finished; `result` has the cache keys, cache hit, proposal count and notes |
| `incomplete` | finished, but decoding ended early; the tail was not analysed |
| `cancelled` | `stop_reason` `cancelled`, `shutdown` or `interrupted` |
| `resource_stopped` | stopped or refused by memory safeguards; never retried automatically |
| `failed` | `stop_reason` `video`, `models`, `error` or `interrupted` |

**Progress** is persisted at most twice a second and at every stage change, and every change increases
`revision`. `GET /api/analyses/{id}` is the polling fallback. `GET /api/analyses/{id}/events` is a
Server-Sent Events stream: `analysis` events carry the revision as id and the `Analysis` as data, `end`
follows a final state, `Last-Event-ID` resumes, and a comment keeps idle connections alive every 15 s.

## Evidence and timelines

[`EvidenceService`](evidence.py) builds reviewable evidence from finished analyses. It reads only verified
caches and never runs a model.

- **Lifecycle.** A run with a result (`completed` or `incomplete`) stores its reports and proposals in
  `result_json`. `evidence.state` is `pending` while the timeline and packages are built, then `ready`
  (with the deterministic `build_key`) or `failed` with a message; the analysis state does not change.
  The event stream ends only after evidence has settled. Runs without a result are `not_applicable`; their
  timeline says why.
- **Reuse.** Identical analyses share one build directory.
- **Recovery.** At startup, `pending` evidence becomes `failed` ("interrupted"). A request for a missing,
  failed or unverifiable build rebuilds it from the verified perception cache and the stored result, one
  rebuild at a time. A missing or mismatched cache answers 409 and asks for a new analysis; nothing is rerun
  automatically.
- **Overlays.** `GET .../packages/{event_id}/overlay` returns the recorded frames inside the review interval,
  unchanged, or an optional `start_s`/`end_s` span of at most 120 s. Each frame is valid until the next
  analysed frame or for at most 0.25 s; outside that window the client shows "not analysed here" and never
  interpolates. `layers` says whether the tracker and ego lane ran, and `gaps` lists unanalysed stretches.
  [`overlay_lookup_cases.json`](../../tests/api/fixtures/overlay_lookup_cases.json) is the shared reference
  for the lookup; the web application's player is tested against every case.
- **Clips.** `POST .../packages/{event_id}/clip` stream-copies the review interval (no re-encoding; the
  source hash is verified first; atomic writes) and reuses a verified clip. A missing or changed source is
  409; an interval that cannot be clipped is 422. `GET` serves the clip with byte ranges.
  `browser_playback` is `unverified`.

## Review decisions

[`ReviewService`](reviews.py) stores a person's decision on each event proposal: `accepted`, `rejected` or
`not_reviewed` (the default). A decision is a human judgement of a machine proposal. It is independent of
the analyser's status (`supported`, `uncertain`), and accepting a proposal does not establish that an
incident occurred or who was responsible. Nothing decides automatically.

- **Identity.** A decision belongs to `(analysis_id, event_id)`. The event ID must be one of the proposals
  stored with that analysis's result (`result_json`); any other is 404, and an analysis without a result
  (queued, running, cancelled, stopped or failed) is 409. A new analysis of the same video starts with no
  decisions, even when its proposals have the same IDs.
- **Records.** A row exists only once a person has decided. Proposals without one are `not_reviewed`, so
  analyses with zero proposals have no records. Clearing a decision stores `not_reviewed` and keeps the row.
- **Revisions.** Every change increases the proposal's `revision`. A change carries `expected_revision`
  (0 when nothing was recorded); the check and the write happen in one `BEGIN IMMEDIATE` transaction, and a
  change based on an older revision is refused with 409 and the latest decision in `detail.current`.
  Requesting the decision already stored changes nothing and returns it, whatever revision was sent.

## Event edits

[`EditService`](edits.py) stores a person's corrections to a proposal: adjusted start and end times and a
plain-text description. The proposal's type, detected times, analyser status, evidence and review decision
are never changed, and edits and decisions are independent: editing does not decide, deciding does not
remove edits, and resetting an edit keeps the decision.

- **Identity and records** follow review decisions: `(analysis_id, event_id)`, membership in the stored
  result, 404/409 for unknown proposals and analyses without a result, no row until something is edited,
  and clearing keeps the row.
- **Times** are analysis times in seconds, like the proposal's, stored exactly as sent. Give both or neither
  (neither keeps the proposed times). They must satisfy `0 ≤ t_start < t_end ≤ duration`, where the duration
  is the analysed video's: the timeline's `video.duration_s`, or the end of the analysed span without one.
  The duration is read from the stored evidence before the write transaction (a missing build is rebuilt as
  for a timeline request). Violations are 422 with a sentence in `detail.message`; non-finite numbers are 422.
- **Description** is plain text, at most 2000 characters and without NUL, stored as written; an empty string
  means none. Nothing generates or rewrites it.
- **Revisions** work as for decisions: `expected_revision`, compare-and-set in one transaction, 409 with
  `detail.current`, and a request for the stored edit returns it unchanged.

The stored proposal (`result_json`), the evidence packages, `review_decisions` and `event_edits` join on
`(analysis_id, event_id)`.

## JSON export

`GET /api/analyses/{id}/export` ([export.py](export.py)) returns one JSON file per analysis, as an
attachment named `<video stem>_<first 8 characters of the analysis id>_spectra.json` (letters, digits,
`.`, `_` and `-` only), UTF-8, with `cache-control: no-store`. It is read-only: the analysis row, decisions
and edits are read in one read transaction (a consistent snapshot), and the timeline and packages come from
the stored, manifest-verified evidence build. Nothing is written, rebuilt, analysed or decoded. A completed
or incomplete analysis whose evidence is `ready` and verifies is exported; otherwise 409 says why (no
results, evidence not ready, or verification failed; opening the review prepares it), and 404 for an
unknown analysis.

Format `spectra.analysis-export`, version `1.0`:

| Field | Contents |
|---|---|
| `format`, `format_version`, `exported_at`, `application` | Identity of the file and the Spectra version that wrote it. |
| `disclaimer`, `units` | What machine output, statuses, ranks and decisions mean, and that times are seconds of analysis time (presentation time minus `video.start_s`). |
| `video` | The analysed file's name, size, SHA-256, container, codec, display size, rotation, `duration_s`, `nominal_fps`, `start_s`, `timestamp_source`. No paths. |
| `analysis` | `id`, `state`, timestamps, analysis, perception and evidence build keys, evidence version, `processing` (the fixed profile with model hashes), run `notes`, analysed span, review configuration, analyser coverage, gaps and run warnings. |
| `events[]` | In timeline (chronological) order. `machine.proposal` is the engine's `EventProposal` unchanged (type, subtype, detected `t_start`/`t_peak`/`t_end`, status, reasons, rank, evidence items with their basis); `machine.evidence` is the evidence package without the fields repeated at the top (review interval, analyser identity, track and lane summaries, gaps, warnings, unjudged patterns, related proposals, notes). `review` holds the person's `decision` (`not_reviewed` without a record), `adjusted_t_start_s`/`adjusted_t_end_s` (null unless adjusted), `description` and update times. |

Frame records, overlays, clips and caches are not exported. `allow_nan=False` guarantees standard JSON. As a
last check, the export is refused if any text outside the reviewer's own notes contains the source path or
an absolute local path.

## Request security

- Uvicorn binds to `127.0.0.1` with proxy headers disabled.
- [`LocalRequestGuard`](security.py) refuses:
  - requests whose `Host` is not `127.0.0.1` or `localhost` (400), which defeats DNS rebinding;
  - requests whose `Origin` is not the server's own origin or a configured `--allow-origin` (403);
  - requests without `Origin` whose `Sec-Fetch-Site` is `same-site` or `cross-site` (403).
- Clients without browser headers (curl, tests) are accepted. Any local process can reach the server.
- No CORS headers are sent.
- Every response carries `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer` and
  `Cross-Origin-Resource-Policy: same-origin`.

During frontend development, run the development server on a loopback origin and pass it with
`--allow-origin`. Proxy API calls through that server so pages stay same-origin; see the
[web application](../../frontend/README.md#commands).

## Web application

When a production build exists (`frontend/dist`, or `--frontend DIR`), [frontend.py](frontend.py) serves
it on the same origin as the API:
- files inside the build directory are served as they are; hashed `assets/` are cacheable, everything
  else is `no-cache`;
- hidden files and paths that resolve outside the build are never served;
- other paths outside `/api` return `index.html`, so application routes survive a reload; a missing
  file with an extension is 404, and `/api` paths never fall back to the page;
- the page carries a `Content-Security-Policy` that allows only this origin (no inline scripts or
  styles), in addition to the headers above.

The same host and origin checks apply to pages. API-only operation is unchanged when no build exists.

## Endpoints

| Method and path | Response |
|---|---|
| `GET /api/health` | Product version, database availability and schema version, model registry, memory pressure, analysis readiness, library folder count. Always 200; `status` is `ok` or `degraded`. |
| `GET /api/library` | Library folders `{id, name, path, source, available, registered_videos}`; `source` is `startup` or `added`. |
| `POST /api/library/folders` | Opens the native folder dialog; no request body. `{folder, created}`: 201 for a new folder, 200 when it was already present or (`folder: null`) the dialog was cancelled. Errors: 400 unsuitable folder, 409 dialog already open, 501 no supported dialog. |
| `DELETE /api/library/folders/{folder_id}` | 204. Removes an added folder from the library; files, registrations and analyses are kept. 404 unknown, 409 startup folder. |
| `GET /api/library/{folder_id}/entries?path=` | `{folder, path, parent, entries, truncated}`. Entries are `{name, path, kind, size_bytes, modified_at, video_id}`, at most 2000 per folder. Errors: 400 for paths outside the folder or not a folder, 404 for unknown entries and unavailable folders. |
| `POST /api/videos` `{folder_id, path}` | `Video`: 201 when created, 200 when already registered. Errors: 400 invalid path, 404 not found, 415 not a video file type, 422 malformed. Probing continues in the background; poll until `probe.state` is not `pending`. |
| `GET /api/videos`, `GET /api/videos/{id}` | `Video`, most recently registered first. |
| `DELETE /api/videos/{id}` | 204. Forgets the registration and never touches the file. 409 if analyses reference it. |
| `POST /api/videos/{id}/playback-check` `{playable, detail, sha256}` | Updated `Video`. 409 unless the video is servable, or when `sha256` is not the current contents. |
| `POST /api/analyses` `{video_id}` | `Analysis`: 201 when queued, 200 for the video's active analysis. 404 unknown video, 409 not eligible, 503 models, memory or shutdown. |
| `GET /api/analyses?video_id=` | Analyses, newest first. |
| `GET /api/analyses/{id}` | `Analysis`: state, `queue_position`, `progress`, `result`, `stop_reason`, `message`, `revision` and timestamps. |
| `GET /api/analyses/{id}/events` | Server-Sent Events stream of `Analysis` changes. |
| `POST /api/analyses/{id}/cancel` | Updated `Analysis`. 409 when it already finished. |
| `GET /api/analyses/{id}/timeline` | The evidence `Timeline`: state, analysed span, analyser coverage, proposals (`entries`) in review order, segments, gaps and warnings. 409 while queued or running, or when evidence cannot be rebuilt. |
| `GET /api/analyses/{id}/packages/{event_id}` | The `EvidencePackage`, containing the unchanged proposal. Event IDs are 16 hexadecimal characters. |
| `GET /api/analyses/{id}/packages/{event_id}/overlay?start_s=&end_s=` | `{overlay, layers, timing, gaps}`. |
| `POST` / `GET /api/analyses/{id}/packages/{event_id}/clip` | Create or reuse the stream-copied clip `{clip, media_url, mime_type, browser_playback}`, or serve it with byte ranges. |
| `GET /api/analyses/{id}/reviews` | `{analysis_id, decisions}`: recorded decisions `{event_id, decision, revision, updated_at}`; every other proposal is not reviewed. 404 unknown analysis. |
| `PUT /api/analyses/{id}/reviews/{event_id}` `{decision, expected_revision}` | The stored `ReviewDecision`. 404 unknown analysis or a proposal not in its result, 409 no result or a newer revision (`detail.current`), 422 malformed. |
| `GET /api/analyses/{id}/edits` | `{analysis_id, edits}`: recorded edits `{event_id, t_start, t_end, description, revision, updated_at}`; every other proposal is unedited. 404 unknown analysis. |
| `PUT /api/analyses/{id}/edits/{event_id}` `{t_start, t_end, description, expected_revision}` | The stored `EventEdit`. 404 unknown analysis or proposal, 409 no result, duration unavailable or a newer revision (`detail.current`), 422 malformed or times out of order or outside the video. |
| `GET /api/analyses/{id}/export` | The JSON export as an attachment (see JSON export). 404 unknown analysis, 409 no results or evidence not ready or not verifiable. |
| `GET` / `HEAD /api/videos/{id}/media` | The original bytes with `Accept-Ranges`. Range requests return 206; malformed ranges return 400; unsatisfiable ranges return 416. 409 when the source changed, is missing or is not servable. |

Response shapes are defined in [schemas.py](schemas.py) and published at `/api/openapi.json`.

Absolute paths appear only in library folder responses, for the folders the owner chose. Video IDs are
32-character hexadecimal strings, and other values are rejected with 422. Media responses are read-only
`FileResponse` streams of the stat that was just verified. A disconnected client ends the stream, and
no background work is started.

Health checks the registry with `default_specs`, which verifies paths, input shapes and SHA-256 but
loads no runtime. The result is cached until a file in the models folder changes size or modification
time. On macOS, pressure levels are 1 normal, 2 warning and 4 critical; other platforms report
`unavailable`. Readiness is false when the database is unavailable, models are missing or invalid, or
pressure is critical. A ready report is not a guarantee: every analysis still runs under the
unmodified `ResourceGuard` in its worker.

## Tests

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q tests/api
```

The tests use temporary data directories, a synthetic model registry and small synthetic videos; they load
no model and read no real footage. `test_boundaries.py` checks that the API never imports evaluation or
perception runtimes and that serving requests loads neither ONNX Runtime, OpenCV nor PyAV in the API
process. `test_frontend_fixtures.py` checks that the web application's fixtures match these contracts.
