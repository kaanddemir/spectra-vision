import { act, cleanup, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { keys, mergeDecisions } from "../../api/queries";
import type { Analysis, ReviewDecision, ReviewDecisions, Timeline } from "../../api/types";
import { errorReply, jsonReply, standardApi, type Recorded } from "../../test/api";
import { fixture, video } from "../../test/fixtures";
import { renderApp } from "../../test/render";
import { installFakeVideo } from "../../test/video";
import decisionCss from "./ReviewDecision.css?raw";

const latest: Analysis = fixture("analyses")[0]!;
const base = `/api/analyses/${latest.id}`;
const timeline = fixture("timeline");
const [approach, cutIn] = timeline.entries as [Timeline["entries"][0], Timeline["entries"][0]];
const reviewsPath = `GET ${base}/reviews`;
const putPath = (eventId: string) => `PUT ${base}/reviews/${eventId}`;

let fake: ReturnType<typeof installFakeVideo> | null = null;
afterEach(() => {
  cleanup();
  fake?.restore();
  fake = null;
});

/** A server that stores decisions like the API: revisions, compare-and-set and no record for "not reviewed". */
function reviewServer(initial: ReviewDecision[] = []) {
  const stored = new Map(initial.map((d) => [d.event_id, d]));
  const list = (): ReviewDecisions => ({ analysis_id: latest.id, decisions: [...stored.values()] });
  const handlers = Object.fromEntries(
    timeline.entries.map((entry) => [
      putPath(entry.event_id),
      ({ body }: Recorded) => {
        const { decision, expected_revision } = body as { decision: ReviewDecision["decision"]; expected_revision: number };
        const current = stored.get(entry.event_id) ?? { event_id: entry.event_id, decision: "not_reviewed", revision: 0, updated_at: null };
        if (decision === current.decision) return jsonReply(current);
        if (expected_revision !== current.revision) {
          return errorReply(409, { message: "this decision was changed elsewhere; the latest decision is shown", current });
        }
        const saved = { event_id: entry.event_id, decision, revision: current.revision + 1, updated_at: "2026-10-20T12:00:00+00:00" };
        stored.set(entry.event_id, saved);
        return jsonReply(saved);
      },
    ]),
  );
  return { stored, handlers: { [reviewsPath]: () => jsonReply(list()), ...handlers } };
}

async function open(path = `/analyses/${latest.id}?event=${approach.event_id}`, overrides: Parameters<typeof standardApi>[0] = {}) {
  fake = installFakeVideo();
  const packages = fixture("packages");
  const server = await standardApi({
    [`GET ${base}`]: jsonReply(latest),
    [`GET ${base}/timeline`]: jsonReply(timeline),
    [`GET /api/videos/${latest.video_id}`]: jsonReply(video("ready")),
    ...Object.fromEntries(Object.entries(packages).map(([id, pk]) => [`GET ${base}/packages/${id}`, jsonReply(pk)])),
    [`GET ${base}/edits`]: jsonReply({ analysis_id: latest.id, edits: [] }),
    ...overrides,
  });
  const app = renderApp(path);
  await screen.findByLabelText("Source video");
  return { server, ...app };
}

const review = () => screen.getByRole("region", { name: "Your review" });
const button = (name: string) => within(review()).getByRole("button", { name });
const proposals = () => screen.getByRole("list", { name: "Proposals in time order" });
const item = (title: string) => within(proposals()).getByRole("button", { name: new RegExp(title) });

describe("review decisions", () => {
  it("starts not reviewed, separate from the analyser's status, and says what a decision does not mean", async () => {
    const { handlers } = reviewServer();
    await open(undefined, handlers);
    expect(await within(review()).findByText("Not reviewed")).toBeInTheDocument();
    expect(button("Accept")).toHaveAttribute("aria-pressed", "false");
    expect(button("Reject")).toHaveAttribute("aria-pressed", "false");
    expect(within(review()).queryByRole("button", { name: "Clear" })).not.toBeInTheDocument();
    // One header row: analyser status, the decision, then the event's actions; no visible "Your review" heading.
    const header = review().closest<HTMLElement>(".section__header")!;
    expect(header).toContainElement(within(header).getByRole("button", { name: "Edit" }));
    expect(within(review()).queryByRole("heading")).not.toBeInTheDocument();
    expect(review()).not.toHaveTextContent("Does not confirm an incident");
    expect(screen.getByRole("region", { name: "Event proposals" })).toHaveTextContent("0 of 3 reviewed");
    // Unreviewed proposals carry no extra status in the list; the analyser's status stays.
    expect(within(proposals()).queryByText("Not reviewed")).not.toBeInTheDocument();
    expect(item("Rapid approach")).toHaveTextContent(/Conditions held|Uncertain/);
  });

  it("shows the decision as a status apart from the Accept and Reject actions", async () => {
    const { handlers } = reviewServer();
    const { user } = await open(undefined, handlers);
    const status = () => review().querySelector(".review-decision__status")!;
    await within(review()).findByText("Not reviewed");
    // Not reviewed is a status, never a button; the actions form their own labelled group.
    expect(within(review()).queryByRole("button", { name: /Not reviewed/ })).not.toBeInTheDocument();
    const actions = within(review()).getByRole("group", { name: "Decide" });
    expect(within(actions).getAllByRole("button").map((b) => b.textContent)).toEqual(["Accept", "Reject"]);
    expect(status()).not.toContainElement(button("Accept"));
    expect(status()).toHaveClass("review-decision__status--not_reviewed");
    await user.click(button("Accept"));
    await within(review()).findByText("Accepted");
    expect(status()).toHaveClass("review-decision__status--accepted");
    expect(button("Accept")).toHaveClass("review-decision__action--accept");
    await user.click(button("Reject"));
    await within(review()).findByText("Rejected");
    expect(status()).toHaveClass("review-decision__status--rejected");
    expect(button("Reject")).toHaveAttribute("aria-pressed", "true");
    expect(button("Accept")).toHaveAttribute("aria-pressed", "false");
  });

  it("marks a selected decision by border, background, colour and weight, from theme tokens", () => {
    for (const [kind, tone] of [["accept", "success"], ["reject", "danger"]]) {
      const rule = new RegExp(`\\.review-decision__action--${kind}\\.button\\[aria-pressed="true"\\]\\s*{([^}]*)}`).exec(decisionCss)![1]!;
      expect(rule).toContain(`border-color: var(--color-${tone}-border)`);
      expect(rule).toContain(`background: var(--color-${tone}-bg)`);
      expect(rule).toContain(`color: var(--color-${tone}-text)`);
      expect(rule).toContain("font-weight: var(--weight-semibold)");
    }
    expect(decisionCss).not.toMatch(/#[0-9a-f]{3,8}\b|rgb\(/i);
  });

  it("accepts, rejects, clears and accepts again, sending the revision on screen", async () => {
    const { handlers } = reviewServer();
    const { user, server } = await open(undefined, handlers);
    const analyserStatus = item("Rapid approach").querySelector(".badge")!.textContent;
    await user.click(await within(review()).findByRole("button", { name: "Accept" }));
    expect(await within(review()).findByText("Accepted")).toBeInTheDocument();
    expect(button("Accept")).toHaveAttribute("aria-pressed", "true");
    expect(within(item("Rapid approach")).getByText("Accepted")).toBeInTheDocument();
    expect(item("Rapid approach").querySelector(".badge")!.textContent).toBe(analyserStatus);
    expect(screen.getByRole("region", { name: "Event proposals" })).toHaveTextContent("1 of 3 reviewed");

    await user.click(button("Reject"));
    expect(await within(review()).findByText("Rejected")).toBeInTheDocument();
    expect(within(item("Rapid approach")).getByText("Rejected")).toBeInTheDocument();

    await user.click(button("Clear"));
    expect(await within(review()).findByText("Not reviewed")).toBeInTheDocument();
    expect(within(item("Rapid approach")).queryByText("Rejected")).not.toBeInTheDocument();

    await user.click(button("Accept"));
    await within(review()).findByText("Accepted");
    // A pressed toggle returns the proposal to not reviewed.
    await user.click(button("Accept"));
    await within(review()).findByText("Not reviewed");

    expect(server.calls(putPath(approach.event_id)).map((r) => r.body)).toEqual([
      { decision: "accepted", expected_revision: 0 },
      { decision: "rejected", expected_revision: 1 },
      { decision: "not_reviewed", expected_revision: 2 },
      { decision: "accepted", expected_revision: 3 },
      { decision: "not_reviewed", expected_revision: 4 },
    ]);
  });

  it("shows stored decisions after a reload and keeps each proposal's own decision", async () => {
    const { handlers, stored } = reviewServer();
    const first = await open(undefined, handlers);
    await first.user.click(await within(review()).findByRole("button", { name: "Reject" }));
    await within(review()).findByText("Rejected");
    cleanup();
    fake?.restore();

    const { user, server } = await open(`/analyses/${latest.id}?event=${cutIn.event_id}`, handlers);
    expect(stored.get(approach.event_id)?.decision).toBe("rejected");
    expect(await within(review()).findByText("Not reviewed")).toBeInTheDocument();
    expect(within(item("Rapid approach")).getByText("Rejected")).toBeInTheDocument();
    await user.click(button("Accept"));
    await within(review()).findByText("Accepted");
    expect(server.calls(putPath(cutIn.event_id))).toHaveLength(1);
    expect(server.calls(putPath(approach.event_id))).toHaveLength(0);
    await user.click(item("Rapid approach"));
    expect(await within(review()).findByText("Rejected")).toBeInTheDocument();
  });

  it("reads the generated API fixture", async () => {
    const reviews: ReviewDecisions = fixture("reviews");
    await open(`/analyses/${latest.id}?event=${cutIn.event_id}`, { [reviewsPath]: jsonReply(reviews) });
    expect(reviews.decisions[0]!.event_id).toBe(cutIn.event_id);
    expect(await within(review()).findByText("Accepted")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Event proposals" })).toHaveTextContent("1 of 3 reviewed");
  });

  it("shows the latest decision when it was changed elsewhere", async () => {
    const { handlers, stored } = reviewServer();
    const { user, server } = await open(undefined, handlers);
    await within(review()).findByText("Not reviewed");
    // Another window rejects the proposal after this page loaded.
    stored.set(approach.event_id, { event_id: approach.event_id, decision: "rejected", revision: 1, updated_at: null });
    await user.click(button("Accept"));
    expect(await screen.findByText("Not saved")).toBeInTheDocument();
    expect(review()).toHaveTextContent("This decision was changed elsewhere; the latest decision is shown.");
    expect(await within(review()).findByText("Rejected")).toBeInTheDocument();
    expect(server.calls(reviewsPath).length).toBeGreaterThanOrEqual(2);
    expect(server.calls(putPath(approach.event_id))).toHaveLength(1); // never retried automatically
  });

  it("never lets an overtaken refresh replace a newer decision", async () => {
    const { handlers } = reviewServer();
    let release: (() => void) | null = null;
    const { user, client, server } = await open(undefined, handlers);
    await within(review()).findByText("Not reviewed");
    // A refresh starts before the decision is saved and answers after it, with the older state.
    server.on(reviewsPath, () => new Promise((resolve) => (release = () => resolve(jsonReply({ analysis_id: latest.id, decisions: [] })))));
    void client.invalidateQueries({ queryKey: keys.reviews(latest.id) });
    await waitFor(() => expect(release).not.toBeNull());
    await user.click(button("Accept"));
    await within(review()).findByText("Accepted");
    await act(async () => release!());
    await waitFor(() => expect(client.isFetching({ queryKey: keys.reviews(latest.id) })).toBe(0));
    expect(within(review()).getByText("Accepted")).toBeInTheDocument();
  });

  it("sends one request for repeated clicks and shows that it is saving", async () => {
    let answer: (() => void) | null = null;
    const saved = { event_id: approach.event_id, decision: "accepted" as const, revision: 1, updated_at: null };
    const { user, server } = await open(undefined, {
      [reviewsPath]: jsonReply({ analysis_id: latest.id, decisions: [] }),
      [putPath(approach.event_id)]: () => new Promise((resolve) => (answer = () => resolve(jsonReply(saved)))),
    });
    const accept = await within(review()).findByRole("button", { name: "Accept" });
    await user.click(accept);
    await user.click(accept);
    expect(accept).toHaveAttribute("aria-busy", "true");
    expect(button("Reject")).toBeDisabled();
    expect(within(review()).getByText("Not reviewed")).toBeInTheDocument(); // nothing is assumed before the answer
    await act(async () => answer!());
    expect(await within(review()).findByText("Accepted")).toBeInTheDocument();
    expect(server.calls(putPath(approach.event_id))).toHaveLength(1);
  });

  it("keeps the stored decision when saving fails", async () => {
    const { user } = await open(undefined, {
      [reviewsPath]: jsonReply({ analysis_id: latest.id, decisions: [] }),
      [putPath(approach.event_id)]: errorReply(500, "database is locked"),
    });
    await user.click(await within(review()).findByRole("button", { name: "Reject" }));
    expect(await within(review()).findByText("Not saved")).toBeInTheDocument();
    expect(review()).toHaveTextContent("Database is locked.");
    expect(within(review()).getByText("Not reviewed")).toBeInTheDocument();
    expect(button("Reject")).toBeEnabled();
  });

  it("explains a proposal the server does not know", async () => {
    const { user } = await open(undefined, {
      [reviewsPath]: jsonReply({ analysis_id: latest.id, decisions: [] }),
      [putPath(approach.event_id)]: errorReply(404, "this analysis has no such event proposal"),
    });
    await user.click(await within(review()).findByRole("button", { name: "Accept" }));
    expect(await within(review()).findByText("This analysis has no such event proposal.")).toBeInTheDocument();
    expect(within(review()).getByText("Not reviewed")).toBeInTheDocument();
  });

  it("leaves an unreachable server to the global banner and says the decision was not saved", async () => {
    const { user, server } = await open(undefined, {
      [reviewsPath]: jsonReply({ analysis_id: latest.id, decisions: [] }),
      [putPath(approach.event_id)]: "network-error",
    });
    server.on("GET /api/health", "network-error");
    await user.click(await within(review()).findByRole("button", { name: "Accept" }));
    expect(await screen.findByText("Server unavailable")).toBeInTheDocument();
    expect(within(review()).getByText("Not saved. Try again when Spectra responds.")).toBeInTheDocument();
    expect(within(review()).queryByRole("alert")).not.toBeInTheDocument();
    expect(within(review()).getByText("Not reviewed")).toBeInTheDocument();
  });

  it("shows loading, then a retry when decisions cannot be loaded, while the evidence stays available", async () => {
    let fail = true;
    const { user } = await open(undefined, {
      [reviewsPath]: () => (fail ? errorReply(500, "database is locked") : jsonReply(fixture("reviews"))),
    });
    expect(await within(review()).findByText("Decisions could not be loaded")).toBeInTheDocument();
    expect(within(review()).queryByRole("button", { name: "Accept" })).not.toBeInTheDocument();
    expect(await screen.findByRole("region", { name: "Why this event?" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Event proposals" })).not.toHaveTextContent("reviewed.");
    fail = false;
    await user.click(within(review()).getByRole("button", { name: "Try again" }));
    expect(await within(review()).findByRole("button", { name: "Accept" })).toBeInTheDocument();
    expect(within(item("Cut-in")).getByText("Accepted")).toBeInTheDocument();
  });

  it("says decisions are loading", async () => {
    await open(undefined, { [reviewsPath]: () => new Promise(() => undefined) });
    expect(within(review()).getByText("Loading decision…")).toBeInTheDocument();
    expect(within(review()).queryByRole("button")).not.toBeInTheDocument();
  });

  it("asks for no decision when nothing was proposed", async () => {
    const { server } = await open(`/analyses/${latest.id}`, {
      [`GET ${base}/timeline`]: jsonReply({ ...timeline, entries: [], segments: [] }),
      [reviewsPath]: jsonReply({ analysis_id: latest.id, decisions: [] }),
    });
    expect(await screen.findByText("No events proposed")).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Your review" })).not.toBeInTheDocument();
    expect(screen.queryByText(/reviewed\./)).not.toBeInTheDocument();
    expect(server.requests.filter((r) => r.method === "PUT")).toHaveLength(0);
  });
});

describe("mergeDecisions", () => {
  const d = (event_id: string, decision: ReviewDecision["decision"], revision: number): ReviewDecision => ({
    event_id,
    decision,
    revision,
    updated_at: null,
  });

  it("keeps the newest revision of each proposal", () => {
    const current = { analysis_id: "a", decisions: [d("e1", "accepted", 2), d("e2", "rejected", 1)] };
    const older = { analysis_id: "a", decisions: [d("e1", "rejected", 1)] };
    expect(mergeDecisions(current, older).decisions).toEqual([d("e1", "accepted", 2), d("e2", "rejected", 1)]);
    expect(mergeDecisions(current, d("e1", "not_reviewed", 3)).decisions[0]).toEqual(d("e1", "not_reviewed", 3));
    expect(mergeDecisions(undefined, { analysis_id: "a", decisions: [] })).toEqual({ analysis_id: "a", decisions: [] });
  });

  it("stores no record for an answer about a proposal that was never reviewed", () => {
    expect(mergeDecisions({ analysis_id: "a", decisions: [] }, d("e1", "not_reviewed", 0)).decisions).toEqual([]);
  });
});
