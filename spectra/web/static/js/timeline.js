export function timelineKey(row) {
  const t = Number(row?.timeSec);
  return Number.isFinite(t) ? (Math.round(t * 100) / 100).toFixed(2) : "";
}
