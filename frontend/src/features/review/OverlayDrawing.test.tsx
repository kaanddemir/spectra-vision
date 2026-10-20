import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { LaneBoundary, OverlayFrame } from "../../api/types";
import { fixture } from "../../test/fixtures";
import { OverlayDrawing, boundaryPoints } from "./OverlayDrawing";

const recorded = fixture("overlay");
const frame: OverlayFrame = recorded.overlay.frames[0]!;
const all = { boxes: true, lane: true };

function draw(changes: Partial<OverlayFrame> = {}, layers = all, size: [number, number] = [640, 360]) {
  const { container } = render(<OverlayDrawing frame={{ ...frame, ...changes }} width={size[0]} height={size[1]} layers={layers} />);
  return container.querySelector("svg")!;
}

describe("overlay drawing", () => {
  it("keeps recorded source-pixel boxes and lets the view box scale them with the video", () => {
    const svg = draw({}, all, [1280, 720]);
    expect(svg).toHaveAttribute("viewBox", "0 0 1280 720");
    expect(svg).toHaveAttribute("preserveAspectRatio", "xMidYMid meet");
    const rect = svg.querySelector('[data-track="1"] rect')!;
    expect(rect).toHaveAttribute("x", String(frame.tracks![0]!.box.x1));
    expect(rect).toHaveAttribute("width", "100");
    expect(svg.querySelector("text")).toHaveTextContent("#1 Car");
  });

  it("places normalised lane points in the recorded frame size", () => {
    const boundary: LaneBoundary = { near: [0.3, 0.9], far: [0.45, 0.5], points: [], measured: true, evidence_y: null };
    expect(boundaryPoints(boundary, 640, 360)).toBe("192,324 288,180");
    const curved = { ...boundary, points: [[0.3, 0.9], [0.35, 0.7], [0.45, 0.5]] as [number, number][] };
    expect(boundaryPoints(curved, 1000, 500)).toBe("300,450 350,350 450,250");
  });

  it("draws the extrapolated part of a lane beyond its evidence dashed, never as observed", () => {
    const svg = draw({}, { boxes: false, lane: true });
    const observed = svg.querySelectorAll("polyline:not(.overlay-drawing__lane--extrapolated)");
    const beyond = svg.querySelectorAll("polyline.overlay-drawing__lane--extrapolated");
    expect(observed).toHaveLength(2);
    expect(beyond).toHaveLength(2);
    // evidence_y 0.6 of 360 px: observed below row 216, extrapolated above it.
    const clip = svg.querySelector(`[id="${observed[0]!.getAttribute("clip-path")!.slice(5, -1)}"] rect`)!;
    expect(clip).toHaveAttribute("y", "216");
  });

  it("draws carried-over lane boundaries dashed throughout", () => {
    const lane = { ...frame.lane!, left: { ...frame.lane!.left!, measured: false }, right: null };
    const svg = draw({ lane }, { boxes: false, lane: true });
    expect(svg.querySelectorAll("polyline")).toHaveLength(1);
    expect(svg.querySelector("polyline")).toHaveClass("overlay-drawing__lane--extrapolated");
  });

  it("marks predicted boxes and unrelated tracks", () => {
    const tracks = [
      { ...frame.tracks![0]!, predicted: true },
      { ...frame.tracks![0]!, track_id: 7, related: false },
    ];
    const svg = draw({ tracks });
    expect(svg.querySelector('[data-track="1"]')).toHaveClass("overlay-drawing__track--predicted", "overlay-drawing__track--related");
    expect(svg.querySelector('[data-track="1"] text')).toHaveTextContent("#1 Car (predicted)");
    expect(svg.querySelector('[data-track="7"]')).not.toHaveClass("overlay-drawing__track--related");
  });

  it("draws only the layers that are switched on", () => {
    expect(draw({}, { boxes: false, lane: true }).querySelectorAll("rect:not(clipPath rect)")).toHaveLength(0);
    expect(draw({}, { boxes: true, lane: false }).querySelectorAll("polyline")).toHaveLength(0);
  });
});
