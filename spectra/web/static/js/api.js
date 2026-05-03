export async function parseJsonResponse(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

export async function postAnalysis(formData) {
  const response = await fetch("/api/analyze", { method: "POST", body: formData });
  const payload = await parseJsonResponse(response);
  if (!response.ok) {
    throw new Error(payload.detail || `Analysis failed (HTTP ${response.status}).`);
  }
  return payload;
}
