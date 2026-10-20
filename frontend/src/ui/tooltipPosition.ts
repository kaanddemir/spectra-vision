/**
 * Where a tooltip goes: centred on its trigger on the preferred side, flipped to the other side when it
 * does not fit there, and shifted horizontally (and, if it fits nowhere, vertically) so that it always
 * stays at least `margin` inside the viewport.
 */
export type Placement = "top" | "bottom";

export interface Box {
  top: number;
  left: number;
  width: number;
  height: number;
}

export interface Position {
  top: number;
  left: number;
  placement: Placement;
}

export function placeTooltip(
  anchor: Box,
  tip: { width: number; height: number },
  viewport: { width: number; height: number },
  preferred: Placement,
  gap: number,
  margin: number,
): Position {
  const above = anchor.top - gap - tip.height;
  const below = anchor.top + anchor.height + gap;
  const fitsAbove = above >= margin;
  const fitsBelow = below + tip.height <= viewport.height - margin;
  const placement: Placement =
    preferred === "top" ? (fitsAbove || !fitsBelow ? "top" : "bottom") : fitsBelow || !fitsAbove ? "bottom" : "top";
  const top = clamp(placement === "top" ? above : below, margin, viewport.height - margin - tip.height);
  const left = clamp(anchor.left + anchor.width / 2 - tip.width / 2, margin, viewport.width - margin - tip.width);
  return { top, left, placement };
}

/** Like Math.min(Math.max(value, low), high), but prefers `low` when the range is empty (tip wider than the viewport). */
function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(value, high));
}
