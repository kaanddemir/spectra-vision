import { act, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { jsonReply, standardApi } from "../../test/api";
import { allVideos, video } from "../../test/fixtures";
import { renderApp } from "../../test/render";

afterEach(() => vi.restoreAllMocks());

const row = (name: string) => screen.getByRole("link", { name }).closest("tr")!;

describe("video library", () => {
  it("lists registered videos with their metadata and one status each", async () => {
    await standardApi();
    renderApp("/videos");
    const table = await screen.findByRole("table", { name: "Registered videos" });
    expect(within(table).getAllByRole("row")).toHaveLength(8);
    expect(within(table).getAllByRole("columnheader").map((th) => th.textContent!.trim())).toEqual([
      "Name",
      "Status",
      "Duration",
      "Size",
      "Registered",
    ]);
    expect(within(row("drive.mp4")).getByText("1:03")).toBeInTheDocument();
    expect(within(row("drive.mp4")).getByText("48.0 MB")).toBeInTheDocument();
    expect(within(row("drive.mp4")).queryByText("1920 × 1080")).not.toBeInTheDocument();
    expect(within(row("drive.mp4")).queryByText(/H\.264|MP4 \/ QuickTime/)).not.toBeInTheDocument();
    // Every status category, as a short inline label with its full description kept for assistive technology.
    const statuses: [string, string, string][] = [
      ["drive.mp4", "Ready", "Ready for analysis"],
      ["rain.mp4", "Reading", "Reading details"],
      ["old.avi", "Unsupported", "Unsupported format"],
      ["deleted.mp4", "Missing", "Source missing"],
      ["broken.mp4", "Unreadable", "Could not read"],
      ["city.mp4", "Needs attention", "Needs playback check"],
      ["tunnel.mov", "Can't play", "Not playable here"],
    ];
    for (const [name, short, full] of statuses) {
      const status = row(name).querySelector(".status-text")!;
      expect(status.querySelector(".status-text__icon svg")).toBeInTheDocument();
      expect(status).toHaveTextContent(short);
      expect(status).toHaveAttribute("title", full);
      expect(status).toHaveTextContent(full);
    }
    expect(table.querySelector(".badge")).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "drive.mp4" })).toHaveAttribute("href", "/videos/00000000000000000000000000000001");
  });

  it("shows the registration date without the time, keeping the full timestamp", async () => {
    await standardApi();
    renderApp("/videos");
    await screen.findByRole("table", { name: "Registered videos" });
    const drive = video("ready");
    const time = row("drive.mp4").querySelector("time")!;
    expect(time).toHaveAttribute("dateTime", drive.registered_at);
    expect(time.textContent).toMatch(/^\d{1,2} [A-Z][a-z]{2} \d{4}$/);
  });

  it("keeps its search behaviour under a shorter placeholder", async () => {
    await standardApi();
    const { user } = renderApp("/videos");
    await screen.findByRole("table", { name: "Registered videos" });
    const search = screen.getByRole("searchbox", { name: "Search videos" });
    expect(search).toHaveAttribute("placeholder", "Search videos…");
    await user.type(search, "hevc");
    expect(screen.getByRole("link", { name: "tunnel.mov" })).toBeInTheDocument();
    expect(screen.getAllByRole("row")).toHaveLength(2);
  });

  it("filters by status with counted tabs", async () => {
    await standardApi();
    const { user, router } = renderApp("/videos");
    await screen.findByRole("table", { name: "Registered videos" });
    expect(screen.getByRole("tab", { name: "All 7" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Ready for analysis 1" })).toBeInTheDocument();
    // Videos still being read have no tab of their own; they are listed under All with their status.
    expect(screen.getAllByRole("tab").map((t) => t.textContent)).toEqual(["All 7", "Ready for analysis 1", "Needs attention 5"]);
    expect(screen.getByRole("link", { name: "rain.mp4" })).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Needs attention 5" }));
    expect(router.state.location.search).toBe("?status=attention");
    expect(screen.queryByRole("link", { name: "drive.mp4" })).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "old.avi" })).toBeInTheDocument();
    // An old link to the removed filter shows everything.
    await act(() => router.navigate("/videos?status=processing"));
    await waitFor(() => expect(screen.getByRole("tab", { name: "All 7" })).toHaveAttribute("aria-selected", "true"));
    expect(screen.getByRole("link", { name: "drive.mp4" })).toBeInTheDocument();
  });

  it("searches by name, folder or codec and offers to clear an empty result", async () => {
    await standardApi();
    const { user } = renderApp("/videos?status=ready");
    await screen.findByRole("table", { name: "Registered videos" });
    await user.type(screen.getByRole("searchbox", { name: "Search videos" }), "hevc");
    expect(await screen.findByText("No results found")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(screen.getByRole("searchbox", { name: "Search videos" })).toHaveValue("");
    expect(screen.getByRole("tab", { name: "All 7" })).toHaveAttribute("aria-selected", "true");
    await user.type(screen.getByRole("searchbox", { name: "Search videos" }), "trips city");
    expect(screen.getByRole("link", { name: "city.mp4" })).toBeInTheDocument();
    expect(screen.getAllByRole("row")).toHaveLength(2);
  });

  it("explains how to add the first video", async () => {
    await standardApi({ "GET /api/videos": jsonReply([]) });
    renderApp("/videos");
    expect(await screen.findByRole("heading", { name: "No videos registered yet" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Browse library" })).toHaveAttribute("href", "/library");
  });

  it("keeps polling while details are being read", async () => {
    const server = await standardApi({ "GET /api/videos": jsonReply([video("probing")]) });
    renderApp("/videos");
    expect(await screen.findByText("Reading", { selector: ".status-text span" })).toBeInTheDocument();
    server.on("GET /api/videos", jsonReply([video("ready", { id: video("probing").id, file_name: "rain.mp4" })]));
    expect(await screen.findByText("Ready", { selector: ".status-text span" }, { timeout: 3000 })).toBeInTheDocument();
    expect(server.calls("GET /api/videos").length).toBeGreaterThanOrEqual(2);
  });

  it("marks data that could not be refreshed as possibly out of date", async () => {
    const server = await standardApi({ "GET /api/videos": jsonReply(allVideos()) });
    const { client } = renderApp("/videos");
    await screen.findByRole("table", { name: "Registered videos" });
    server.on("GET /api/videos", "network-error");
    await client.refetchQueries({ queryKey: ["videos"] });
    expect(await screen.findByText(/^Last updated/)).toBeInTheDocument();
    expect(screen.getByText(/Data below may be out of date/)).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Registered videos" })).toBeInTheDocument();
    server.on("GET /api/videos", jsonReply(allVideos()));
    await client.refetchQueries({ queryKey: ["videos"] });
    await waitFor(() => expect(screen.queryByText(/^Last updated/)).not.toBeInTheDocument());
  });
});

describe("video table layout", () => {
  it("sizes columns to content with the name taking the rest and its full name kept", async () => {
    await standardApi();
    renderApp("/videos");
    const table = await screen.findByRole("table", { name: "Registered videos" });
    expect(table).toHaveClass("table--fit");
    expect(within(table).getByRole("columnheader", { name: "Name" })).toHaveClass("cell-primary");
    const name = within(table).getAllByRole("link").find((link) => link.classList.contains("cell-primary__title"))!;
    expect(name).toHaveClass("truncate");
    expect(name).toHaveAttribute("title", name.textContent);
  });
});
