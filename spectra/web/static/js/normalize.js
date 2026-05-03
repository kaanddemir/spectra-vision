export function cleanResponsePayload(response) {
  return response?.payload && typeof response.payload === "object" ? response.payload : (response || {});
}
