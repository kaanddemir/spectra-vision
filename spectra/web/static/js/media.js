export function mediaSrc(value) {
  if (!value) return "";
  if (/^(data:|blob:|https?:)/i.test(value)) return value;
  return `data:image/png;base64,${value}`;
}
