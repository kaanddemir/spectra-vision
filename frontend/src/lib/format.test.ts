import { describe, expect, it } from "vitest";

import { formatBytes, formatCodec, formatContainer, formatDate, formatDuration, formatFrameRate, formatLength, formatTimecode, MISSING, sentence } from "./format";

describe("format", () => {
  it("formats sizes in decimal units", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(48_000_001)).toBe("48.0 MB");
    expect(formatBytes(120_500_000)).toBe("121 MB");
    expect(formatBytes(null)).toBe("—");
  });

  it("formats durations and positions", () => {
    expect(formatDuration(62.5)).toBe("1:03");
    expect(formatDuration(3729)).toBe("1:02:09");
    expect(formatDuration(null)).toBe("—");
    expect(formatTimecode(14.5)).toBe("0:14.5");
    expect(formatTimecode(59.96)).toBe("1:00.0");
  });

  it("names codecs and containers", () => {
    expect(formatCodec("h264")).toBe("H.264");
    expect(formatCodec("prores")).toBe("PRORES");
    expect(formatContainer("mov,mp4,m4a,3gp,3g2,mj2")).toBe("MP4 / QuickTime");
  });

  it("turns server messages into sentences without altering file names", () => {
    expect(sentence("the video file is unavailable")).toBe("The video file is unavailable.");
    expect(sentence("deleted.mp4 was moved or deleted")).toBe("deleted.mp4 was moved or deleted.");
    expect(sentence("Already a sentence.")).toBe("Already a sentence.");
  });
});

describe("formatLength", () => {
  it("reads as words, never as a timecode", () => {
    expect(formatLength(6.6)).toBe("7 sec");
    expect(formatLength(0.2)).toBe("1 sec");
    expect(formatLength(60)).toBe("1 min");
    expect(formatLength(62.5)).toBe("1 min 3 sec");
    expect(formatLength(3600)).toBe("1 h");
    expect(formatLength(3729)).toBe("1 h 2 min");
    expect(formatLength(null)).toBe(MISSING);
    expect(formatLength(Number.NaN)).toBe(MISSING);
  });
});

describe("formatFrameRate", () => {
  it("formats the reported rate naturally", () => {
    expect(formatFrameRate(30)).toBe("30 FPS");
    expect(formatFrameRate(29.97)).toBe("29.97 FPS");
    expect(formatFrameRate(30000 / 1001)).toBe("29.97 FPS");
    expect(formatFrameRate(60)).toBe("60 FPS");
    expect(formatFrameRate(23.976)).toBe("23.98 FPS");
    expect(formatFrameRate(25.0)).toBe("25 FPS");
  });

  it("says Unknown for missing or invalid rates", () => {
    for (const value of [null, undefined, 0, -30, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(formatFrameRate(value)).toBe("Unknown");
    }
  });
});

describe("formatDate", () => {
  it("shows the calendar date only, and a dash for missing or invalid dates", () => {
    expect(formatDate("2026-10-20T13:31:00+00:00")).toBe("20 Oct 2026");
    expect(formatDate(null)).toBe(MISSING);
    expect(formatDate("not a date")).toBe(MISSING);
  });
});
