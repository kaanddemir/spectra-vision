import { describe, expect, it } from "vitest";

import { placeTooltip } from "./tooltipPosition";

const VIEWPORT = { width: 1000, height: 800 };
const TIP = { width: 200, height: 30 };
const GAP = 6;
const MARGIN = 8;
const at = (left: number, top: number, width = 28, height = 28) => ({ left, top, width, height });
const place = (anchor: ReturnType<typeof at>, preferred: "top" | "bottom" = "bottom", tip = TIP) =>
  placeTooltip(anchor, tip, VIEWPORT, preferred, GAP, MARGIN);

function inside(position: { top: number; left: number }, tip = TIP) {
  expect(position.left).toBeGreaterThanOrEqual(MARGIN);
  expect(position.top).toBeGreaterThanOrEqual(MARGIN);
  expect(position.left + tip.width).toBeLessThanOrEqual(VIEWPORT.width - MARGIN);
  expect(position.top + tip.height).toBeLessThanOrEqual(VIEWPORT.height - MARGIN);
}

describe("tooltip placement", () => {
  it("centres below the trigger when there is room", () => {
    expect(place(at(486, 400))).toEqual({ top: 434, left: 400, placement: "bottom" });
  });

  it("shifts left at the right edge, keeping the margin", () => {
    const position = place(at(960, 400));
    expect(position.left).toBe(VIEWPORT.width - MARGIN - TIP.width);
    inside(position);
  });

  it("shifts right at the left edge", () => {
    const position = place(at(0, 400));
    expect(position.left).toBe(MARGIN);
    inside(position);
  });

  it("flips above at the bottom edge and below at the top edge", () => {
    const low = place(at(486, 770));
    expect(low).toMatchObject({ placement: "top", top: 770 - GAP - TIP.height });
    inside(low);
    const high = place(at(486, 0), "top");
    expect(high).toMatchObject({ placement: "bottom", top: 28 + GAP });
    inside(high);
  });

  it("stays in each corner", () => {
    for (const anchor of [at(0, 0), at(972, 0), at(0, 772), at(972, 772)]) {
      for (const preferred of ["top", "bottom"] as const) inside(place(anchor, preferred));
    }
  });

  it("keeps the preferred side when neither side fits, clamped into the viewport", () => {
    const tall = { width: 200, height: 790 };
    const position = place(at(486, 400), "bottom", tall);
    expect(position.placement).toBe("bottom");
    expect(position.top).toBe(MARGIN);
  });

  it("pins a tip wider than the viewport to the left margin", () => {
    expect(place(at(486, 400), "bottom", { width: 1200, height: 30 }).left).toBe(MARGIN);
  });
});
