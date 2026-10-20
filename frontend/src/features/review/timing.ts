/**
 * Time rules of the review player. Analysis time `t` is source presentation time minus the stream's start
 * (`VideoInfo.start_s`), exactly as in the evidence contracts; the browser's media time is that presentation
 * time. Nothing here is derived from frame numbers or rates.
 */
import type { OverlayFrame, Span } from "../../api/types";

/** The tolerance `spectra.evidence` uses when comparing times. */
export const EPS = 1e-6;

/** First index whose time is greater than `t` (Python's `bisect.bisect_right`). */
function bisectRight(frames: readonly OverlayFrame[], t: number): number {
  let lo = 0;
  let hi = frames.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (t < frames[mid]!.t) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

/**
 * The overlay frame to draw at analysis time `t`: the latest analysed frame at or before `t` that is still
 * valid, or null for "not analysed here". The same rule as `spectra.evidence.overlays.frame_at`; frames
 * are never interpolated.
 */
export function frameIndexAt(frames: readonly OverlayFrame[], t: number): number | null {
  const k = bisectRight(frames, t + EPS) - 1;
  if (k < 0 || t > frames[k]!.valid_until + EPS) return null;
  return k;
}

/**
 * Browsers place a file's frames at their container presentation times (the HTML media timeline), so the
 * media time of analysis time zero is the stream's start time (`VideoInfo.start_s`, the overlay timing's
 * `stream_start_s`). A browser's earliest seekable position does not reveal this: Chrome reports 0 for an
 * MP4 whose first frame is presented at 0.3 s.
 */
export const toAnalysisTime = (mediaTime: number, streamStart: number) => Math.max(0, mediaTime - streamStart);
export const toMediaTime = (t: number, streamStart: number) => t + streamStart;

export interface CoverageRun extends Span {
  analysed: boolean;
}

export interface AnalysedSpan {
  analysed_start_s: number | null;
  analysed_end_s: number | null;
  gaps: Span[];
}

/** The video from 0 to `duration` split into analysed and unanalysed runs, in order. */
export function coverageRuns({ analysed_start_s, analysed_end_s, gaps }: AnalysedSpan, duration: number): CoverageRun[] {
  if (analysed_start_s === null || analysed_end_s === null || duration <= 0) {
    return duration > 0 ? [{ start_s: 0, end_s: duration, analysed: false }] : [];
  }
  const runs: CoverageRun[] = [];
  const push = (start_s: number, end_s: number, analysed: boolean) => {
    const start = Math.max(0, Math.min(start_s, duration));
    const end = Math.max(0, Math.min(end_s, duration));
    if (end > start) runs.push({ start_s: start, end_s: end, analysed });
  };
  push(0, analysed_start_s, false);
  let cursor = analysed_start_s;
  for (const gap of [...gaps].sort((a, b) => a.start_s - b.start_s)) {
    push(cursor, gap.start_s, true);
    push(gap.start_s, gap.end_s, false);
    cursor = Math.max(cursor, gap.end_s);
  }
  push(cursor, analysed_end_s, true);
  push(analysed_end_s, duration, false);
  return runs;
}

/** Whether `t` lies inside the analysed span and outside every gap. */
export function isAnalysedAt({ analysed_start_s, analysed_end_s, gaps }: AnalysedSpan, t: number): boolean {
  if (analysed_start_s === null || analysed_end_s === null) return false;
  if (t < analysed_start_s - EPS || t > analysed_end_s + EPS) return false;
  return !gaps.some((gap) => t > gap.start_s + EPS && t < gap.end_s - EPS);
}

/** Overlay data outside the selected proposal's review interval is requested in spans of this length. */
export const OVERLAY_CHUNK_S = 20;

/**
 * Which overlay span covers `t`: `"review"` (the proposal's review interval), a fixed chunk of the
 * analysed span, or `"outside"` when no analysed frame can be valid at `t` and nothing needs to be
 * requested. The last analysed frame stays valid for up to `maxAge` seconds, as in `frame_at`.
 * Returned as a string so playback re-renders only when the span changes.
 */
export function overlayWindow(t: number, review: Span | null, analysed: AnalysedSpan, maxAge = 0): string {
  if (review && t >= review.start_s && t <= review.end_s) return "review";
  const { analysed_start_s: start, analysed_end_s: end } = analysed;
  if (start === null || end === null || t < start - EPS || t > end + maxAge + EPS) return "outside";
  return `chunk:${Math.floor(Math.min(t, end) / OVERLAY_CHUNK_S)}`;
}

/** The span to request for an `overlayWindow` chunk, clipped to the analysed span. */
export function chunkSpan(window: string, analysed: AnalysedSpan): Span | null {
  if (!window.startsWith("chunk:") || analysed.analysed_start_s === null || analysed.analysed_end_s === null) return null;
  const index = Number(window.slice("chunk:".length));
  const start_s = Math.max(index * OVERLAY_CHUNK_S, analysed.analysed_start_s);
  const end_s = Math.min((index + 1) * OVERLAY_CHUNK_S, analysed.analysed_end_s);
  return end_s > start_s ? { start_s, end_s } : null;
}

/** Position of `t` on a timeline of `duration`, as a percentage clamped to 0–100. */
export function percent(t: number, duration: number): number {
  if (!(duration > 0)) return 0;
  return Math.min(Math.max(t / duration, 0), 1) * 100;
}

/** Whether the overlay's recorded frame size has the shape of the video as the browser displays it. */
export function geometryMatches(videoWidth: number, videoHeight: number, width: number, height: number): boolean {
  if (!(videoWidth > 0 && videoHeight > 0)) return true;
  return Math.abs(videoWidth / videoHeight - width / height) <= 0.01 * (width / height);
}
