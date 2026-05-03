export const MISSING = "-";

export const isReal = (value) => {
  if (value === undefined || value === null) return false;
  if (typeof value === "string") {
    const text = value.trim();
    return text !== "" && text !== "-" && !/^n\/a$/i.test(text);
  }
  return true;
};

export const num = (value, defaultValue = null) => {
  if (value === null || value === undefined || value === "") return defaultValue;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : defaultValue;
};

export const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
