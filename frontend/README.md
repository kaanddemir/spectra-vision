# Web application

`frontend/` is Spectra's review application: React and TypeScript (strict), built with Vite. It talks
to the [local API](../spectra/api/README.md) over HTTP only and never reads files or imports Python code.
Visual rules live in the [design system](DESIGN_SYSTEM.md).

## Commands

Node.js 22.12 or newer. Run from `frontend/`, except where noted.

```sh
npm ci                     # install the exact versions in package-lock.json
npm run dev                # development server on http://127.0.0.1:5173
npm test                   # Vitest and Testing Library, once
npm run typecheck          # TypeScript project check
npm run build              # type-check, then write the production build to dist/
```

`python3 run.py --dev` (from the repository root) starts the API and this development server together.
To start the API alone for development, allow the development server's origin:

```sh
.venv/bin/python -m spectra.api --library ~/Movies/Dashcam --allow-origin http://127.0.0.1:5173
```

Open `http://127.0.0.1:5173`. The development server forwards `/api` to `http://127.0.0.1:8765`
(override with `SPECTRA_API`). Forwarded requests keep the page's `Host` (`127.0.0.1:5173`, an accepted
loopback name), and the browser's `Origin` is accepted only because of `--allow-origin`. The development
server sends no CORS headers; nothing in the API's host, origin or Fetch Metadata checks is relaxed.

After `npm run build`, the API serves the application itself at `http://127.0.0.1:8765/` (it uses
`frontend/dist` automatically when it exists, or `--frontend DIR`). Without a build only the API is served.
`python3 run.py` builds it when it is missing or older than its sources. `dist/` is never committed.

## Dependencies

| Package | Why |
|---|---|
| `react`, `react-dom` | interface |
| `react-router` | addressable pages (filters and folders live in the URL) and history |
| `@tanstack/react-query` | server state: polling while probing, refetch, stale-data detection, cache updates after mutations |
| `lucide-react` | the single icon library |
| `vite`, `@vitejs/plugin-react`, `typescript` | build and type-check |
| `vitest`, `jsdom`, `@testing-library/*` | tests |

No UI kit, CSS framework or browser-automation tool is used. Versions are exact in `package.json` and
locked in `package-lock.json`. `.npmrc` sets `legacy-peer-deps`: npm 10's resolver fails on the build
plugin's optional peer dependencies, and every required peer is listed explicitly instead.

## Structure

| Path | Contents |
|---|---|
| `src/styles/` | `tokens.css` (all colours and scales), `base.css` (elements, typography roles), `layout.css` (page layouts) |
| `src/ui/` | shared components, `icons.ts` (the icon vocabulary), `BrandMark.tsx` |
| `src/theme/` | theme preference, resolution and provider; `public/theme-init.js` applies it before first paint |
| `src/api/` | typed contracts (`types.ts`), request client and errors, query keys, polling and mutations |
| `src/app/` | routes, application shell, server status, shared loading/error/stale-data handling |
| `src/features/` | screens: `videos`, `library`, `events`, `analyses` |
| `src/test/` | test helpers and `fixtures/api.json` |

## Screens

| Route | Purpose |
|---|---|
| `/videos` | Registered videos with status tabs, search and metadata. |
| `/library?folder=&path=` | Add folders through the native dialog, remove them (files and history are kept), browse them and register videos in place, one or several at a time. |
| `/videos/:id` | Metadata, probe progress, playback eligibility, the in-browser playback check, starting, following and cancelling the analysis, analysis history. |
| `/events` | Proposals from the stored timeline of each video's latest analysis with a result; each links to its review. |
| `/analyses/:id?event=` | One analysis: live progress and cancellation while active; the review workspace once evidence is ready; outcome, stop reason and evidence state otherwise. `event` selects a proposal. |


## API integration

- `src/api/types.ts` mirrors `spectra/api/schemas.py` and the evidence contracts field for field.
- Errors become `ApiError` (status, sentence-cased message, reasons) or `ServerUnavailableError` (no
  connection, or a non-JSON answer from something other than the API). A banner appears on every page
  while the server cannot be reached.
- `QueryStatus` is the single pattern for server data: loading, error with retry, or data. When a refresh
  fails, the earlier data stays visible under a warning naming its time, never as current data.
- Video lists and details poll every second while a probe is pending; health every 15 seconds.
- The Events page reads only stored timelines (`evidence.state == "ready"`), so browsing never triggers an
  evidence rebuild. An empty list is never presented as "nothing happened".

**Analyses.** Analysis starts only on an explicit action, with the server's fixed profile; a response for an
already-queued or running analysis (200) is reported as such. `useAnalysisLive` follows the active analysis
through the event stream (`/api/analyses/{id}/events`) until it settles (final state and evidence no longer
pending). Every update carries a revision and older ones are ignored, so reconnects never move the display
backwards. The browser reconnects by itself; after three consecutive stream errors, or without `EventSource`,
the analysis is polled every 2 s until it settles. A cancel request shows "Cancelling…" until the server
reports the final state. Progress shows only reported values: percentage and a determinate bar only with a
fraction, frames and video position only when reported. One toast announces an outcome observed on the page;
nothing is retried automatically.

**Playback check.** The detail page loads `media_url` in its `<video>` element and waits for a decoded
frame. A `HEAD` request first confirms the server will stream the file, so a refused or changed source is
never recorded as a browser failure. Only definite answers are reported (a decoded frame, or a decode or
format error); interrupted downloads and timeouts are shown and not recorded. The report carries the
SHA-256 of the contents that were loaded; the result is discarded if the video changed during the check,
and the server refuses it (409) if the file changed before the report arrived.

## Review workspace

`src/features/review/` opens when an analysis with a result has ready evidence. Pending evidence is waited for;
failed evidence is rebuilt only when the person chooses "Prepare evidence" (a timeline request), never by
browsing. The selected proposal is the `event` URL parameter, so links survive a reload; selecting one seeks
to the start of its review span.

- **Player.** The registered source is streamed read-only from `media_url`; nothing is transcoded. A missing,
  changed (hash differs from the analysis) or unplayable source is explained in the stage and the evidence
  stays available. Space or K plays and pauses, the arrow keys move five seconds.
- **Clock.** `MediaClock` holds the position in analysis time outside React state; components subscribe to
  derived values (tenths of a second, an overlay frame index, an overlay span), so playback re-renders only
  what changes. With `requestVideoFrameCallback` the clock takes each presented frame's media time; without
  it, `currentTime` is read on animation frames while playing and on seek and time updates while paused.
  Analysis time is media time minus the stream start (`timeline.video.start_s`): browsers present frames at
  their container times, which an end-to-end check confirmed in Chrome on an MP4 starting at 0.3 s. The
  recorded `duration_s` is the timeline's length, because Chrome reports such a file's `duration` without
  the offset.
- **Timeline.** Analysed and unanalysed stretches (before, gaps, after), review segments, proposal spans and
  peaks, and the playhead. It is a slider (arrows 1 s, Shift 5 s, Page keys 10 s, Home, End); clicking or
  dragging seeks; clicking a proposal selects it. Unanalysed stretches are labelled as unknown, never as empty.
- **Overlays** are off until switched on, then requested for the selected review interval, or for 20 s
  spans of the analysed span elsewhere; nothing is requested where no analysed frame can be valid.
  `frameIndexAt` is `frame_at` exactly (same tolerance, no interpolation) and is tested against all 96
  reference cases in `tests/api/fixtures/overlay_lookup_cases.json`. The SVG's view box is the recorded frame
  size with the video's letterboxing, so boxes keep their source-pixel coordinates; overlays are hidden when
  the displayed video's shape differs. Lane boundaries beyond their evidence row, and carried-over boundaries,
  are dashed. The corner note distinguishes "Not analysed here" from "No tracked objects", "Tracker did not run
  here" and "No lane observed".
- **Evidence.** The proposal's times, analyser and checks, uncertainty reasons, evidence items with basis,
  value and unit, related tracks and lane records, limits (notes, gaps, warnings, unjudged patterns) and
  related proposals. Switching proposals cancels the previous package request.
- **Decisions.** "Your review" above the summary shows the reviewer's decision (Not reviewed, Accepted,
  Rejected, drawn with a person icon) apart from the analyser's status badge, with Accept and Reject toggles
  (pressing the active one, or Clear, returns it to not reviewed). The list marks decided proposals and counts
  them. A change is sent with the revision on screen and shown only once the server stored it; repeated clicks
  send one request. A refusal because it changed elsewhere (409) loads and shows the latest decision; nothing
  is retried. Cached decisions only move forward by revision, so a refresh that was overtaken by a save never
  shows the older decision. Decisions refresh when the window regains focus.
- **Edits.** "Edit" in the event panel opens a compact form under the decision: Start and End, typed as
  `m:ss.d`, `h:mm:ss.d` or seconds with up to six decimals and filled with every recorded decimal
  (`formatExactTime`, `parseTime`: nothing is rounded), and an optional plain-text note of up to 2,000
  characters. The form names the detected interval and the wider playback context. Malformed times, an end
  not after the start, and times beyond the video are explained per field and never sent. "Use detected
  times" and "Clear note" reset the inputs; saving the detected times stores no adjustment. Saved times are
  shown as "Adjusted by you · detected …" in the panel and "Adjusted by you" in the list; the timeline's
  markers, the clip and the evidence keep the detected times. The note appears as "Your note", apart from
  "Why this event?", rendered as text. Edits never change the review decision. Saving, conflicts (409),
  refusals and offline failures behave as for decisions; the typed values stay in the form. Selecting a
  proposal seeks to its playback context, or to an adjusted start before it.
- **Export.** "Export JSON" in the page header of a reviewable analysis downloads `GET …/export` through
  `fetch`, then hands the file to the browser with the server's suggested name (`lib/download.ts`). No review
  work is required. The button shows "Exporting…" and repeated clicks make one download; the browser's
  download is the only confirmation. A refusal is one toast; an unreachable server is left to the banner.
- **Clips** are made only on request (`POST …/clip`), shown with their actual copied span; "Open clip" is
  offered only when the browser reports it may play the type, "Download" always.

## Tests

`npm test` runs focused tests of the theme, design-system rules, shared components, the API client and
every screen against a controlled `fetch`. Browsers' media loading and playback are simulated
(`src/test/video.ts`); nothing is downloaded.

`src/test/fixtures/api.json` holds real API responses generated from the server's own models, playback
rules and evidence builder by `tests/api/frontend_fixtures.py`. Regenerate it after a contract change:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m tests.api.frontend_fixtures   # from the repository root
```

`tests/api/test_frontend_fixtures.py` fails while the file is out of date or no longer validates.
