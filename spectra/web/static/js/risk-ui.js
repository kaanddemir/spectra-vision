export function stateClass(stateOrBand) {
  const value = String(stateOrBand || "").toLowerCase();
  if (value === "danger" || value === "high" || value === "critical") return "danger";
  if (value === "caution" || value === "medium") return "caution";
  if (value === "safe" || value === "low") return "safe";
  return "none";
}
