import { act, cleanup, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Analysis } from "../../api/types";
import { errorReply, jsonReply, standardApi } from "../../test/api";
import { fixture, video } from "../../test/fixtures";
import { renderApp } from "../../test/render";
import { installFakeVideo } from "../../test/video";

const latest: Analysis = fixture("analyses")[0]!;
const base = `/api/analyses/${latest.id}`;
const exportPath = `GET ${base}/export`;
const document_ = { format: "spectra.analysis-export", format_version: "1.0", events: [] };
const disposition = { "content-disposition": `attachment; filename="drive_${latest.id.slice(0, 8)}_spectra.json"` };

let fake: ReturnType<typeof installFakeVideo> | null = null;
let saved: { name: string; type: string }[] = [];
beforeEach(() => {
  saved = [];
  URL.createObjectURL = vi.fn(() => "blob:export");
  URL.revokeObjectURL = vi.fn();
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
    saved.push({ name: this.download, type: this.href });
  });
});
afterEach(() => {
  cleanup();
  fake?.restore();
  fake = null;
  vi.restoreAllMocks();
});

async function open(overrides: Parameters<typeof standardApi>[0] = {}, analysis: Analysis = latest) {
  fake = installFakeVideo();
  const server = await standardApi({
    [`GET ${base}`]: jsonReply(analysis),
    [`GET ${base}/timeline`]: jsonReply(fixture("timeline")),
    [`GET /api/videos/${latest.video_id}`]: jsonReply(video("ready")),
    [`GET ${base}/reviews`]: jsonReply({ analysis_id: latest.id, decisions: [] }),
    [`GET ${base}/edits`]: jsonReply({ analysis_id: latest.id, edits: [] }),
    ...overrides,
  });
  const app = renderApp(`/analyses/${analysis.id}`);
  return { server, ...app };
}

describe("JSON export", () => {
  it("downloads one file with the server's name, without requiring any review", async () => {
    const { user, server } = await open({ [exportPath]: { body: document_, headers: disposition } });
    await user.click(await screen.findByRole("button", { name: "Export JSON" }));
    await waitFor(() => expect(saved).toEqual([{ name: `drive_${latest.id.slice(0, 8)}_spectra.json`, type: "blob:export" }]));
    expect(server.calls(exportPath)).toHaveLength(1);
    expect(server.requests.filter((r) => r.method !== "GET")).toHaveLength(0); // exporting changes nothing
    expect(screen.queryByRole("status", { name: /export/i })).not.toBeInTheDocument();
    expect(screen.queryByText(/exported/i)).not.toBeInTheDocument(); // the browser's download is the confirmation
  });

  it("shows that it is working and makes one download for repeated clicks", async () => {
    let answer: (() => void) | null = null;
    const { user, server } = await open({
      [exportPath]: () => new Promise((resolve) => (answer = () => resolve({ body: document_, headers: disposition }))),
    });
    const button = await screen.findByRole("button", { name: "Export JSON" });
    await user.click(button);
    await user.click(button);
    expect(button).toHaveAttribute("aria-busy", "true");
    expect(button).toHaveTextContent("Exporting…");
    expect(saved).toHaveLength(0); // nothing is announced before the download starts
    await act(async () => answer!());
    await waitFor(() => expect(saved).toHaveLength(1));
    expect(server.calls(exportPath)).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Export JSON" })).toBeEnabled();
  });

  it("explains a refusal once and allows another attempt", async () => {
    const { user, server } = await open({
      [exportPath]: errorReply(409, "the evidence for this analysis is not ready; open its review to prepare it, then export again"),
    });
    await user.click(await screen.findByRole("button", { name: "Export JSON" }));
    expect(await screen.findByText("Could not export")).toBeInTheDocument();
    expect(screen.getByText(/The evidence for this analysis is not ready/)).toBeInTheDocument();
    expect(saved).toHaveLength(0);
    server.on(exportPath, { body: document_, headers: disposition });
    await user.click(screen.getByRole("button", { name: "Export JSON" }));
    await waitFor(() => expect(saved).toHaveLength(1));
  });

  it("leaves an unreachable server to the global banner", async () => {
    const { user, server } = await open({ [exportPath]: "network-error" });
    server.on("GET /api/health", "network-error");
    await user.click(await screen.findByRole("button", { name: "Export JSON" }));
    expect(await screen.findByText("Server unavailable")).toBeInTheDocument();
    expect(screen.queryByText("Could not export")).not.toBeInTheDocument();
    expect(saved).toHaveLength(0);
  });

  it("uses a safe fallback name when the server suggests none", async () => {
    const { user } = await open({ [exportPath]: jsonReply(document_) });
    await user.click(await screen.findByRole("button", { name: "Export JSON" }));
    await waitFor(() => expect(saved).toEqual([{ name: `spectra_${latest.id.slice(0, 8)}.json`, type: "blob:export" }]));
  });

  it("is offered only when the analysis has reviewable results", async () => {
    const running = { ...latest, state: "running" as const, result: null, evidence: { state: "not_applicable" as const, build_key: null, message: null } };
    await open({}, running);
    expect(await screen.findByRole("button", { name: "Cancel" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Export JSON" })).not.toBeInTheDocument();
  });
});
