export function timelineKey(row) {
  return `${row?.frameIndex ?? ""}:${row?.timeSec ?? ""}`;
}
