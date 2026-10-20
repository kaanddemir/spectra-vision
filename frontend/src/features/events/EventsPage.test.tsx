import { act, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { errorReply, jsonReply, standardApi } from "../../test/api";
import { fixture } from "../../test/fixtures";
import { renderApp } from "../../test/render";
import { formatTimecode } from "../../lib/format";
import { latestResults } from "./EventsPage";

afterEach(() => vi.restoreAllMocks());

const analyses = fixture("analyses");
const [latest, superseded, failed] = analyses as [typeof analyses[0], typeof analyses[0], typeof analyses[0]];

describe("event library", () => {
  it("uses the newest analysis with a result for each video", () => {
    expect(latestResults(analyses).map((a) => a.id)).toEqual([latest.id]);
    expect(latestResults([superseded]).map((a) => a.id)).toEqual([superseded.id]);
    expect(latestResults([failed])).toEqual([]);
  });

  it("explains that nothing has been analysed yet", async () => {
    await standardApi();
    renderApp("/events");
    expect(await screen.findByRole("heading", { name: "No analysed videos yet" })).toBeInTheDocument();
  });

  it("lists proposals from stored timelines without fabricating any", async () => {
    const server = await standardApi({
      "GET /api/analyses": jsonReply(analyses),
      [`GET /api/analyses/${latest.id}/timeline`]: jsonReply(fixture("timeline")),
      [`GET /api/analyses/${latest.id}/reviews`]: jsonReply({
        analysis_id: latest.id,
        decisions: [{ event_id: fixture("timeline").entries[1]!.event_id, decision: "rejected", revision: 1, updated_at: "2026-10-20T10:00:00Z" }],
      }),
    });
    const { user } = renderApp("/events");
    const table = await screen.findByRole("table", { name: "Event proposals" });
    expect(within(table).getAllByRole("row")).toHaveLength(4);
    const approach = within(table).getByRole("link", { name: "Rapid approach" }).closest("tr")!;
    const entry = fixture("timeline").entries[0]!;
    expect(within(approach).getByRole("link", { name: "Rapid approach" })).toHaveAttribute(
      "href",
      `/analyses/${latest.id}?event=${entry.event_id}`,
    );
    expect(within(approach).getByRole("link", { name: "drive.mp4" })).toHaveAttribute("href", `/videos/${latest.video_id}`);
    // The detected interval, not the peak or the wider review span.
    expect(within(approach).getByText(`${formatTimecode(entry.t_start)} – ${formatTimecode(entry.t_end)}`)).toBeInTheDocument();
    expect(within(approach).queryByText("0:14.5")).not.toBeInTheDocument();
    // The video name reads like a file name elsewhere, not as a red link.
    expect(within(approach).getByRole("link", { name: "drive.mp4" })).toHaveClass("cell-primary__title");
    // Proposals without a stored decision are not reviewed; stored decisions are shown as saved.
    expect(await within(approach).findByText("Not reviewed")).toBeInTheDocument();
    const cutIn = within(table).getByRole("link", { name: "Cut-in" }).closest("tr")!;
    expect(within(cutIn).getByText("Rejected")).toBeInTheDocument();
    expect(within(approach).getByText("Conditions held")).toBeInTheDocument();
    const contact = within(table).getByText("Possible contact between other road users").closest("tr")!;
    expect(within(contact).getByText("Experimental check")).toBeInTheDocument();
    expect(within(contact).getByText("Uncertain")).toBeInTheDocument();
    expect(screen.queryByText(/Not confirmed incidents/)).not.toBeInTheDocument();
    expect(screen.queryByText(/is not a verdict/)).not.toBeInTheDocument();
    // Internal analyser identifiers are not shown.
    expect(within(table).queryByText(/apparent-approach|road-user-contact|lane-entry-cut-in/)).not.toBeInTheDocument();
    expect(within(table).getAllByRole("columnheader").map((h) => h.textContent)).toEqual(["Event", "Video", "Time", "Status", "Your review"]);
    expect(server.calls(`GET /api/analyses/${latest.id}/timeline`)).toHaveLength(1);
    expect(server.calls(`GET /api/analyses/${superseded.id}/timeline`)).toHaveLength(0);
    expect(server.calls(`GET /api/analyses/${failed.id}/timeline`)).toHaveLength(0);
    await user.click(screen.getByRole("tab", { name: "Impact 1" }));
    expect(within(screen.getByRole("table", { name: "Event proposals" })).getAllByRole("row")).toHaveLength(2);
  });

  it("does not read an empty timeline as 'nothing happened'", async () => {
    await standardApi({
      "GET /api/analyses": jsonReply([latest]),
      [`GET /api/analyses/${latest.id}/timeline`]: jsonReply({ ...fixture("timeline"), entries: [] }),
    });
    renderApp("/events");
    const heading = await screen.findByRole("heading", { name: "No events proposed" });
    expect(heading.closest(".empty-state")).toHaveClass("empty-state--page");
    expect(screen.getByText("No events were suggested for review. You can still check the videos yourself.")).toBeInTheDocument();
    expect(screen.queryByText(/analysed video/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Clear filters" })).not.toBeInTheDocument();
  });

  it("never rebuilds evidence that is not ready, and says so", async () => {
    const notReady = { ...latest, evidence: { state: "failed" as const, build_key: null, message: "interrupted" } };
    const server = await standardApi({ "GET /api/analyses": jsonReply([notReady]) });
    renderApp("/events");
    expect(await screen.findByText("Evidence for 1 analysis is not available")).toBeInTheDocument();
    expect(server.calls(`GET /api/analyses/${latest.id}/timeline`)).toHaveLength(0);
  });
});

describe("analysis summary", () => {
  it("opens the review of a proposal linked from the library", async () => {
    const entry = fixture("timeline").entries[1]!;
    await standardApi({
      [`GET /api/analyses/${latest.id}`]: jsonReply(latest),
      [`GET /api/analyses/${latest.id}/timeline`]: jsonReply(fixture("timeline")),
      [`GET /api/videos/${latest.video_id}`]: jsonReply(fixture("videos").ready),
      [`GET /api/analyses/${latest.id}/packages/${entry.event_id}`]: jsonReply(fixture("packages")[entry.event_id]),
    });
    renderApp(`/analyses/${latest.id}?event=${entry.event_id}`);
    expect(await screen.findByRole("heading", { level: 1, name: "drive.mp4" })).toBeInTheDocument();
    expect(await screen.findByRole("heading", { level: 2, name: "Cut-in" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Back to video" })).toHaveAttribute("href", `/videos/${latest.video_id}`);
  });

  it("explains why a failed analysis stopped", async () => {
    await standardApi({ [`GET /api/analyses/${failed.id}`]: jsonReply(failed) });
    renderApp(`/analyses/${failed.id}`);
    expect(await screen.findByRole("alert")).toHaveTextContent("The video changed after it was registered.");
  });
});

describe("event table layout", () => {
  it("gives the proposal the most space and lets the video name truncate with its full name kept", async () => {
    await standardApi({
      "GET /api/analyses": jsonReply(analyses),
      [`GET /api/analyses/${latest.id}/timeline`]: jsonReply(fixture("timeline")),
    });
    renderApp("/events");
    const table = await screen.findByRole("table", { name: "Event proposals" });
    expect(table).toHaveClass("table--fit");
    expect(within(table).getByRole("columnheader", { name: "Event" })).toHaveClass("cell-primary");
    expect(within(table).getByRole("columnheader", { name: "Video" })).toHaveClass("cell-flex");
    const video = within(table).getAllByRole("link", { name: "drive.mp4" })[0]!;
    expect(video).toHaveClass("truncate");
    expect(video).toHaveAttribute("title", "drive.mp4");
    expect(video.closest("td")).toHaveClass("cell-flex");
  });
});

describe("event library reviews", () => {
  it("shows a decision made in the review when the list is opened again", async () => {
    const entry = fixture("timeline").entries[0]!;
    let decision = "not_reviewed";
    const server = await standardApi({
      "GET /api/analyses": jsonReply([latest]),
      [`GET /api/analyses/${latest.id}/timeline`]: jsonReply(fixture("timeline")),
      [`GET /api/analyses/${latest.id}/reviews`]: () =>
        jsonReply({
          analysis_id: latest.id,
          decisions: decision === "not_reviewed" ? [] : [{ event_id: entry.event_id, decision, revision: 1, updated_at: "2026-10-20T10:00:00Z" }],
        }),
    });
    const { router } = renderApp("/events");
    const row = () => screen.getByRole("link", { name: "Rapid approach" }).closest("tr")!;
    expect(await within(await screen.findByRole("table")).findAllByText("Not reviewed")).toHaveLength(3);
    decision = "accepted";
    await act(() => router.navigate("/videos"));
    await screen.findByRole("heading", { level: 1, name: "Videos" });
    await act(() => router.navigate("/events"));
    await waitFor(() => expect(server.calls(`GET /api/analyses/${latest.id}/reviews`)).toHaveLength(2));
    expect(await within(row()).findByText("Accepted")).toBeInTheDocument();
  });

  it("says when decisions could not be read instead of showing them as not reviewed", async () => {
    await standardApi({
      "GET /api/analyses": jsonReply([latest]),
      [`GET /api/analyses/${latest.id}/timeline`]: jsonReply(fixture("timeline")),
      [`GET /api/analyses/${latest.id}/reviews`]: errorReply(500, "broken"),
    });
    renderApp("/events");
    const table = await screen.findByRole("table", { name: "Event proposals" });
    expect(await within(table).findAllByText("Unavailable")).toHaveLength(3);
    expect(within(table).queryByText("Not reviewed")).not.toBeInTheDocument();
  });
});
