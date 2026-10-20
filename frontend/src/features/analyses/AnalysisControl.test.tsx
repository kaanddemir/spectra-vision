import { act, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Analysis, Video } from "../../api/types";
import { errorReply, jsonReply, standardApi } from "../../test/api";
import { installEventSource } from "../../test/eventSource";
import { fixture, video } from "../../test/fixtures";
import { renderApp } from "../../test/render";

let sources: ReturnType<typeof installEventSource>;
beforeEach(() => {
  sources = installEventSource();
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const ready = video("ready");
const base = fixture("analyses")[0]!;

function job(changes: Partial<Analysis> & { revision: number }): Analysis {
  return {
    ...structuredClone(base),
    id: "a".repeat(32),
    result: null,
    evidence: { state: "not_applicable", build_key: null, message: null },
    state: "queued",
    queue_position: 1,
    stop_reason: null,
    message: null,
    progress: { stage: null, frames_analysed: 0, media_time_s: null, fraction: null },
    ...changes,
  };
}

const queued = job({ revision: 1 });
const running = (revision: number, progress: Partial<Analysis["progress"]> = {}) =>
  job({ revision, state: "running", queue_position: null, progress: { ...queued.progress, stage: "analysing frames", ...progress } });
const finished = (revision: number, state: Analysis["state"], changes: Partial<Analysis> = {}) =>
  job({ revision, state, queue_position: null, ...changes });
const completedResult = {
  result: { cache_hit: false, perception_key: "b".repeat(64), analysis_key: "c".repeat(64), proposals: 3, notes: [] },
};

async function open(v: Video = ready, history: Analysis[] = [], overrides = {}) {
  const server = await standardApi({
    [`GET /api/videos/${v.id}`]: jsonReply(v),
    [`GET /api/analyses?video_id=${v.id}`]: jsonReply(history),
    ...overrides,
  });
  const view = renderApp(`/videos/${v.id}`);
  await screen.findByRole("heading", { level: 1, name: v.file_name });
  return { server, ...view };
}

const section = () => screen.getByRole("region", { name: "Analysis" });
const push = (analysis: Analysis) =>
  act(async () => {
    sources.latest().emit("analysis", analysis);
    await new Promise((resolve) => setTimeout(resolve, 0));
  });

describe("starting an analysis", () => {
  it("is offered only for eligible videos", async () => {
    const pending = video("needs_check");
    await open(pending, [], { [`HEAD /api/videos/${pending.id}/media`]: { status: 200 } });
    expect(await within(section()).findByRole("button", { name: "Analyse" })).toBeDisabled();
    // Reasons about the video itself are shown by Analysis readiness, not repeated here.
    expect(await screen.findByText("Checking", { selector: ".readiness__status" })).toBeInTheDocument();
    expect(within(section()).queryAllByText(/./, { selector: "p" })).toHaveLength(0);
    expect(within(section()).queryByText(/readiness/i)).not.toBeInTheDocument();
    expect(sources.all()).toHaveLength(0);
  });

  it("is unavailable while the server cannot analyse", async () => {
    await open(ready, [], { "GET /api/health": jsonReply(fixture("healthDegraded")) });
    expect(await within(section()).findByText("Memory pressure is critical.")).toBeInTheDocument();
    expect(within(section()).getByRole("button", { name: "Analyse" })).toBeDisabled();
  });

  it("queues once, follows the job live and never starts by itself", async () => {
    let release!: () => void;
    const { server, user } = await open(ready, [], {
      "POST /api/analyses": () => new Promise((resolve) => (release = () => resolve(jsonReply(queued, 201)))),
    });
    expect(server.calls("POST /api/analyses")).toHaveLength(0);
    const button = await within(section()).findByRole("button", { name: "Analyse" });
    await user.click(button);
    expect(button).toBeDisabled();
    await user.click(button);
    release();
    // Progress appears in place; no toast repeats it.
    expect(await within(section()).findByText("Next in queue")).toBeInTheDocument();
    expect(screen.queryByText("Analysis queued")).not.toBeInTheDocument();
    expect(server.calls("POST /api/analyses").map((c) => c.body)).toEqual([{ video_id: ready.id }]);
    expect(within(section()).getByRole("button", { name: "Cancel" })).toBeEnabled();
    expect(sources.latest().url).toBe(`/api/analyses/${queued.id}/events`);
  });

  it("sends one request for clicks that arrive before the button is disabled", async () => {
    const { server } = await open(ready, [], { "POST /api/analyses": jsonReply(queued, 201) });
    const button = await within(section()).findByRole("button", { name: "Analyse" });
    act(() => {
      button.click();
      button.click();
      button.click();
    });
    await waitFor(() => expect(server.calls("POST /api/analyses")).toHaveLength(1));
    expect(await screen.findByText("Next in queue")).toBeInTheDocument();
    expect(server.calls("POST /api/analyses")).toHaveLength(1);
    expect(screen.queryByText("Analysis already in progress")).not.toBeInTheDocument();
  });

  it("reports an analysis that was already in progress", async () => {
    const { user } = await open(ready, [], { "POST /api/analyses": jsonReply(running(3), 200) });
    await user.click(await within(section()).findByRole("button", { name: "Analyse" }));
    expect(await screen.findByText("Analysis already in progress")).toBeInTheDocument();
    expect(within(section()).getByText("Analysing video…")).toBeInTheDocument();
  });

  it("explains a refusal with the server's reasons", async () => {
    const { user } = await open(ready, [], {
      "POST /api/analyses": errorReply(503, { message: "analysis is unavailable", reasons: ["The models are missing."] }),
    });
    await user.click(await within(section()).findByRole("button", { name: "Analyse" }));
    const alert = await within(section()).findByRole("alert");
    expect(alert).toHaveTextContent("Analysis is unavailable.");
    expect(alert).toHaveTextContent("The models are missing.");
  });
});

describe("completed result", () => {
  it.each([
    [0, "· 0 proposals"],
    [1, "· 1 proposal"],
    [2, "· 2 proposals"],
  ])("treats %i proposals as a completed result", async (proposals, text) => {
    const done = finished(3, "completed", { ...completedResult, result: { ...completedResult.result!, proposals } });
    await open(ready, [done]);
    expect(await within(section()).findByText(text)).toBeInTheDocument();
    expect(within(section()).getByText("Completed")).toBeInTheDocument();
    expect(within(section()).getByRole("link", { name: "Open review" })).toHaveAttribute("href", `/analyses/${done.id}`);
    expect(within(section()).queryByRole("alert")).not.toBeInTheDocument();
  });
});

describe("live progress", () => {
  it("shows reported progress only and ignores stale or repeated states", async () => {
    await open(ready, [running(2)]);
    const bar = await within(section()).findByRole("progressbar", { name: "Analysis progress" });
    expect(bar).not.toHaveAttribute("aria-valuenow");
    expect(within(section()).queryByText(/%$/)).not.toBeInTheDocument();
    await push(running(5, { fraction: 0.42, frames_analysed: 120, media_time_s: 12 }));
    expect(within(section()).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "42");
    expect(within(section()).getByText("42%")).toBeInTheDocument();
    expect(within(section()).getByText("Analysing video…")).toBeInTheDocument();
    expect(within(section()).queryByText(/frames|0:12/)).not.toBeInTheDocument();
    expect(within(section()).queryByText("Running")).not.toBeInTheDocument();
    await push(running(4, { fraction: 0.1, frames_analysed: 10 }));
    await push(job({ revision: 3 }));
    expect(within(section()).getByText("42%")).toBeInTheDocument();
  });

  it("waits for evidence, announces the outcome once and closes the stream", async () => {
    await open(ready, [running(2)]);
    await within(section()).findByRole("progressbar");
    const source = sources.latest();
    await push(finished(6, "completed", { ...completedResult, evidence: { state: "pending", build_key: null, message: null } }));
    expect(within(section()).getByText("Preparing evidence…")).toBeInTheDocument();
    await push(finished(7, "completed", { ...completedResult, evidence: { state: "ready", build_key: "d".repeat(64), message: null } }));
    act(() => source.emit("end", {}));
    expect(await screen.findByText("Analysis completed")).toBeInTheDocument();
    expect(screen.getAllByText("Analysis completed")).toHaveLength(1);
    expect(within(section()).getByText("· 3 proposals")).toBeInTheDocument();
    const status = within(section()).getByText("Completed").closest(".status-text");
    expect(status).toHaveClass("status-text--success");
    expect(within(section()).queryByText("Completed", { selector: ".badge *" })).not.toBeInTheDocument();
    const review = within(section()).getByRole("link", { name: "Open review" });
    expect(review).toHaveAttribute("href", `/analyses/${queued.id}`);
    expect(review).toHaveClass("button--primary", "button--md", "analysis-latest__review");
    const again = within(section()).getByRole("button", { name: "Analyse again" });
    expect(again).toBeEnabled();
    expect(again).toHaveClass("button--secondary", "button--sm");
    expect(source.closed).toBe(true);
    expect(sources.all()).toHaveLength(1);
  });

  it("lets the browser reconnect, then falls back to polling", async () => {
    const { server } = await open(ready, [running(2)], { [`GET /api/analyses/${queued.id}`]: jsonReply(running(9, { fraction: 0.9 })) });
    await within(section()).findByRole("progressbar");
    const source = sources.latest();
    act(() => source.fail());
    expect(source.closed).toBe(false);
    await push(running(3, { fraction: 0.3 }));
    act(() => source.fail());
    act(() => source.fail());
    act(() => source.fail());
    expect(source.closed).toBe(true);
    await waitFor(() => expect(within(section()).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "90"), { timeout: 4000 });
    expect(server.calls(`GET /api/analyses/${queued.id}`).length).toBeGreaterThanOrEqual(1);
    expect(sources.all()).toHaveLength(1);
  });

  it("polls when the browser has no event streams", async () => {
    vi.stubGlobal("EventSource", undefined);
    await open(ready, [running(2)], { [`GET /api/analyses/${queued.id}`]: jsonReply(finished(8, "cancelled", { stop_reason: "cancelled" })) });
    expect(await screen.findByText("Analysis cancelled", {}, { timeout: 4000 })).toBeInTheDocument();
  });

  it("recovers a running analysis after a reload without submitting again", async () => {
    const { server } = await open(ready, [running(4, { fraction: 0.5 })]);
    expect(await within(section()).findByRole("progressbar")).toHaveAttribute("aria-valuenow", "50");
    expect(server.calls("POST /api/analyses")).toHaveLength(0);
    expect(sources.all()).toHaveLength(1);
  });
});

describe("cancelling", () => {
  it("shows a request until the server confirms the cancellation", async () => {
    const { server, user } = await open(ready, [running(2)], {
      [`POST /api/analyses/${queued.id}/cancel`]: jsonReply({ ...running(3), cancel_requested: true }),
    });
    await user.click(await within(section()).findByRole("button", { name: "Cancel" }));
    const pending = await within(section()).findByRole("button", { name: "Cancelling…" });
    expect(pending).toBeDisabled();
    expect(screen.queryByText("Analysis cancelled")).not.toBeInTheDocument();
    await push(finished(5, "cancelled", { stop_reason: "cancelled" }));
    expect(await screen.findByText("Analysis cancelled")).toBeInTheDocument();
    expect(server.calls(`POST /api/analyses/${queued.id}/cancel`)).toHaveLength(1);
  });

  it("says when the analysis had already finished", async () => {
    const { user } = await open(ready, [running(2)], {
      [`POST /api/analyses/${queued.id}/cancel`]: errorReply(409, "analysis already finished"),
    });
    await user.click(await within(section()).findByRole("button", { name: "Cancel" }));
    expect(await screen.findByText("Already finished")).toBeInTheDocument();
  });
});

describe("outcomes", () => {
  it.each([
    ["incomplete", "Incomplete", "Decoding ended early; the end of the video was not analysed.", completedResult],
    ["resource_stopped", "Stopped for memory", "Stopped to protect system memory.", { stop_reason: "resources" }],
    ["failed", "Failed", "The analysis stopped unexpectedly.", { stop_reason: "error" }],
    ["cancelled", "Cancelled", "Cancelled on request.", { stop_reason: "cancelled" }],
  ] as const)("shows a %s analysis with its reason, never as completed", async (state, label, reason, changes) => {
    await open(ready, [finished(9, state, changes)]);
    expect(await within(section()).findByText(label)).toBeInTheDocument();
    expect(within(section()).getByText(reason)).toBeInTheDocument();
    expect(within(section()).queryByText("Completed")).not.toBeInTheDocument();
    expect(sources.all()).toHaveLength(0);
  });

  it("summarises an analysis with its evidence state", async () => {
    const failedEvidence = finished(9, "completed", {
      ...completedResult,
      evidence: { state: "failed", build_key: null, message: "interrupted; rebuilt when the timeline is requested" },
    });
    await standardApi({ [`GET /api/analyses/${failedEvidence.id}`]: jsonReply(failedEvidence) });
    renderApp(`/analyses/${failedEvidence.id}`);
    const title = await screen.findByText("Evidence not prepared");
    expect(title.closest(".alert")).toHaveTextContent("Interrupted; rebuilt when the timeline is requested.");
    expect(screen.queryByText("Analysis details")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();
  });

  it("follows and cancels from the analysis page", async () => {
    const { user } = (await standardApi({
      [`GET /api/analyses/${queued.id}`]: jsonReply(running(2, { fraction: 0.2 })),
      [`POST /api/analyses/${queued.id}/cancel`]: jsonReply({ ...running(3), cancel_requested: true }),
    }), renderApp(`/analyses/${queued.id}`));
    expect(await screen.findByRole("progressbar")).toHaveAttribute("aria-valuenow", "20");
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(await screen.findByRole("button", { name: "Cancelling…" })).toBeDisabled();
    await push(finished(4, "cancelled", { stop_reason: "cancelled" }));
    expect(await screen.findByText("Analysis cancelled")).toBeInTheDocument();
  });
});
