import { afterEach, describe, expect, it, vi } from "vitest";

import { errorReply, jsonReply, mockApi } from "../test/api";
import { ApiError, ServerUnavailableError, api, errorMessage } from "./client";

afterEach(() => vi.restoreAllMocks());

describe("api client", () => {
  it("returns parsed JSON for successful requests and sends JSON bodies", async () => {
    const server = mockApi({ "POST /api/videos": jsonReply({ id: "x" }, 201) });
    await expect(api.registerVideo("0123456789abcdef", "a/b.mp4")).resolves.toEqual({ id: "x" });
    expect(server.calls("POST /api/videos")[0]!.body).toEqual({ folder_id: "0123456789abcdef", path: "a/b.mp4" });
    const init = server.fetchMock.mock.calls[0]![1]!;
    expect(new Headers(init.headers).get("content-type")).toBe("application/json");
  });

  it("encodes query parameters", async () => {
    const server = mockApi({ "GET /api/library/abc/entries": jsonReply({}) });
    await api.libraryEntries("abc", "trips/2026 & more");
    expect(server.requests[0]!.path).toBe("/api/library/abc/entries?path=trips%2F2026+%26+more");
  });

  it("turns string details into sentences", async () => {
    mockApi({ "GET /api/videos/abc": errorReply(404, "video not registered") });
    const error = await api.video("abc").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 404, message: "Video not registered." });
  });

  it("keeps structured refusal reasons", async () => {
    mockApi({ "POST /api/videos": errorReply(409, { message: "video not eligible", reasons: ["Play it first."] }) });
    const error = (await api.registerVideo("a", "b").catch((e: unknown) => e)) as ApiError;
    expect(error.message).toBe("Video not eligible.");
    expect(error.reasons).toEqual(["Play it first."]);
  });

  it("describes validation errors without exposing their structure", async () => {
    mockApi({ "POST /api/videos": errorReply(422, [{ loc: ["body", "path"], msg: "too long" }]) });
    await expect(api.registerVideo("a", "b")).rejects.toThrow("The request was not accepted.");
  });

  it("reports an unreachable server for network failures and non-JSON proxy errors", async () => {
    const server = mockApi({ "GET /api/health": "network-error" });
    await expect(api.health()).rejects.toBeInstanceOf(ServerUnavailableError);
    server.on("GET /api/health", { status: 502, body: "Bad gateway", json: false });
    await expect(api.health()).rejects.toBeInstanceOf(ServerUnavailableError);
    expect(errorMessage(new ServerUnavailableError())).toBe("Spectra is not responding.");
  });

  it("accepts empty 204 responses", async () => {
    mockApi({ "DELETE /api/videos/abc": { status: 204 } });
    await expect(api.forgetVideo("abc")).resolves.toBeUndefined();
  });
});
