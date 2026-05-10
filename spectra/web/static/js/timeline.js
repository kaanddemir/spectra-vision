export function timelineKey(row) {
  const fi = Number(row?.frameIndex);
  if (Number.isFinite(fi)) return `f${fi}`;
  const t = Number(row?.timestampSec);
  return Number.isFinite(t) ? (Math.round(t * 100) / 100).toFixed(2) : "";
}
