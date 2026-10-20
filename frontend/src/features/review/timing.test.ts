import { describe, expect, it } from "vitest";

// The authoritative lookup cases, generated from spectra.evidence.overlays.frame_at.
import reference from "../../../../tests/api/fixtures/overlay_lookup_cases.json";
import type { OverlayFrame } from "../../api/types";
import {
  EPS,
  OVERLAY_CHUNK_S,
  chunkSpan,
  coverageRuns,
  frameIndexAt,
  geometryMatches,
  isAnalysedAt,
  overlayWindow,
  percent,
  toAnalysisTime,
  toMediaTime,
} from "./timing";

const frames = (times: { t: number; valid_until: number }[]): OverlayFrame[] =>
  times.map((f) => ({ ...f, tracks: null, lane: null }));

describe("overlay lookup matches the server's frame_at", () => {
  const cases = reference.scenarios.flatMap((scenario) =>
    scenario.lookups.map((lookup) => ({ name: scenario.name, frames: scenario.frames, ...lookup })),
  );

  it("uses the reference's tolerance and has every case", () => {
    expect(reference.epsilon).toBe(EPS);
    expect(cases).toHaveLength(96);
  });

  it.each(cases)("$name at t=$t draws frame $frame", ({ frames: recorded, t, frame }) => {
    expect(frameIndexAt(frames(recorded), t)).toBe(frame);
  });

  it("has no frame to draw before the first or without frames", () => {
    expect(frameIndexAt([], 3)).toBeNull();
    expect(frameIndexAt(frames([{ t: 1, valid_until: 1.1 }]), 0.5)).toBeNull();
  });
});

describe("media time", () => {
  it("is container presentation time: the stream start is removed", () => {
    expect(toAnalysisTime(15.9, 1.4)).toBeCloseTo(14.5, 9);
    expect(toMediaTime(14.5, 1.4)).toBeCloseTo(15.9, 9);
    expect(toAnalysisTime(12.5, 0)).toBe(12.5);
  });

  it("never reports a negative analysis time", () => {
    expect(toAnalysisTime(1.2, 1.4)).toBe(0);
  });
});

describe("coverage", () => {
  const analysed = { analysed_start_s: 2, analysed_end_s: 50, gaps: [{ start_s: 20, end_s: 23.5 }] };

  it("splits the video into analysed and unanalysed runs", () => {
    expect(coverageRuns(analysed, 60)).toEqual([
      { start_s: 0, end_s: 2, analysed: false },
      { start_s: 2, end_s: 20, analysed: true },
      { start_s: 20, end_s: 23.5, analysed: false },
      { start_s: 23.5, end_s: 50, analysed: true },
      { start_s: 50, end_s: 60, analysed: false },
    ]);
  });

  it("treats a run without a result as unanalysed throughout", () => {
    expect(coverageRuns({ analysed_start_s: null, analysed_end_s: null, gaps: [] }, 10)).toEqual([
      { start_s: 0, end_s: 10, analysed: false },
    ]);
  });

  it("knows where frames were analysed", () => {
    expect(isAnalysedAt(analysed, 1)).toBe(false);
    expect(isAnalysedAt(analysed, 2)).toBe(true);
    expect(isAnalysedAt(analysed, 20)).toBe(true); // the last analysed frame before the gap
    expect(isAnalysedAt(analysed, 21)).toBe(false);
    expect(isAnalysedAt(analysed, 55)).toBe(false);
  });
});

describe("overlay spans", () => {
  const analysed = { analysed_start_s: 0, analysed_end_s: 62.4, gaps: [] };
  const review = { start_s: 9, end_s: 16.5 };

  it("uses the selected review interval while the playhead is inside it", () => {
    expect(overlayWindow(12, review, analysed)).toBe("review");
  });

  it("requests short chunks elsewhere and nothing outside the analysed span", () => {
    expect(overlayWindow(30, review, analysed)).toBe("chunk:1");
    expect(chunkSpan("chunk:1", analysed)).toEqual({ start_s: OVERLAY_CHUNK_S, end_s: 2 * OVERLAY_CHUNK_S });
    expect(chunkSpan("chunk:3", analysed)).toEqual({ start_s: 60, end_s: 62.4 });
    expect(overlayWindow(62.45, review, analysed)).toBe("outside");
    // The last analysed frame stays valid for the maximum age, so its span is still requested.
    expect(overlayWindow(62.6, review, analysed, 0.25)).toBe("chunk:3");
    expect(overlayWindow(62.7, review, analysed, 0.25)).toBe("outside");
    expect(chunkSpan("outside", analysed)).toBeNull();
  });
});

describe("geometry", () => {
  it("places times proportionally and clamps them", () => {
    expect(percent(14.5, 62.5)).toBeCloseTo(23.2, 9);
    expect(percent(-1, 62.5)).toBe(0);
    expect(percent(80, 62.5)).toBe(100);
    expect(percent(5, 0)).toBe(0);
  });

  it("accepts the displayed video only at the recorded aspect ratio", () => {
    expect(geometryMatches(1920, 1080, 640, 360)).toBe(true);
    expect(geometryMatches(1080, 1920, 640, 360)).toBe(false);
    expect(geometryMatches(0, 0, 640, 360)).toBe(true);
  });
});
