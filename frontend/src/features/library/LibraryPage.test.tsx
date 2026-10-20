import { screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { errorReply, jsonReply, standardApi } from "../../test/api";
import { fixture, video } from "../../test/fixtures";
import { renderApp } from "../../test/render";

afterEach(() => vi.restoreAllMocks());

const FOLDER = fixture("folders")[0]!;
const ROOT = `GET /api/library/${FOLDER.id}/entries?path=`;
const cell = (name: string) => screen.getByText(name, { selector: ".cell-primary__title" }).closest("tr")!;

describe("library browsing", () => {
  it("offers to add a folder when there are none, centred as a page-level empty state", async () => {
    await standardApi({ "GET /api/library": jsonReply([]) });
    renderApp("/library");
    const heading = await screen.findByRole("heading", { name: "No library folders yet" });
    expect(heading.closest(".empty-state")).toHaveClass("empty-state--page");
    expect(screen.getAllByRole("button", { name: "Add folder" })).toHaveLength(1);
    expect(screen.queryByText(/--library/)).not.toBeInTheDocument();
    expect(screen.getByText("Browse folders and add videos to Spectra.")).toBeInTheDocument();
  });

  it("lists folders and videos with their registration status", async () => {
    await standardApi();
    renderApp("/library");
    await screen.findByRole("table", { name: "Contents of Dashcam" });
    expect(within(cell("2026-10 trips")).getByText("Folder")).toBeInTheDocument();
    const ready = cell("drive.mp4").querySelector(".status-text")!;
    expect(ready).toHaveTextContent("Ready");
    expect(ready).toHaveAttribute("title", "Ready for analysis");
    expect(ready.querySelector(".status-text__icon svg")).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Contents of Dashcam" }).querySelector(".badge")).not.toBeInTheDocument();
    expect(within(cell("drive.mp4")).getByRole("link", { name: "Open drive.mp4" })).toHaveAttribute(
      "href",
      `/videos/${video("ready").id}`,
    );
    expect(within(cell("drive.mp4")).queryByRole("checkbox")).not.toBeInTheDocument();
    expect(within(cell("highway.mp4")).getByText("Not registered")).toBeInTheDocument();
    expect(within(cell("highway.mp4")).getByText("121 MB")).toBeInTheDocument();
    // The absolute path is not shown; the location is the folder's name, with the path on hover.
    expect(screen.queryByText(FOLDER.path)).not.toBeInTheDocument();
    const crumbs = screen.getByRole("navigation", { name: "Folder path" });
    expect(within(crumbs).getByText("Dashcam")).toHaveAttribute("title", FOLDER.path);
  });

  it("uses one quiet style for Open and Register, right-aligned", async () => {
    await standardApi();
    renderApp("/library");
    await screen.findByRole("table", { name: "Contents of Dashcam" });
    const open = within(cell("drive.mp4")).getByRole("link", { name: "Open drive.mp4" });
    const register = within(cell("highway.mp4")).getByRole("button", { name: "Register highway.mp4" });
    for (const action of [open, register]) {
      expect(action).toHaveClass("button", "button--ghost", "button--sm");
      expect(action.closest("td")).toHaveClass("cell-actions");
    }
  });

  it("shows modification dates without the time and searches files by name", async () => {
    await standardApi();
    const { user } = renderApp("/library");
    await screen.findByRole("table", { name: "Contents of Dashcam" });
    const time = cell("highway.mp4").querySelector("time")!;
    expect(time.textContent).toMatch(/^\d{1,2} [A-Z][a-z]{2} \d{4}$/);
    expect(time.getAttribute("dateTime")).toMatch(/T\d{2}:\d{2}/);
    const search = screen.getByRole("searchbox", { name: "Filter this folder" });
    expect(search).toHaveAttribute("placeholder", "Search files…");
    await user.type(search, "high");
    expect(screen.getAllByRole("row")).toHaveLength(2);
    expect(cell("highway.mp4")).toBeInTheDocument();
  });

  it("tells library folders with the same name apart", async () => {
    const { folderLabels } = await import("./LibraryPage");
    const base = fixture("folders")[0]!;
    const a = { ...base, id: "a".repeat(16), name: "datasets", path: "/Users/me/Spectra/datasets" };
    const b = { ...base, id: "b".repeat(16), name: "datasets", path: "/Volumes/Archive/datasets" };
    const c = { ...base, id: "c".repeat(16), name: "Dashcam", path: "/Users/me/Dashcam" };
    expect([...folderLabels([a, b, c]).values()]).toEqual(["datasets (Spectra)", "datasets (Archive)", "Dashcam"]);
    const twin = { ...a, id: "d".repeat(16), path: "/Other/Spectra/datasets" };
    expect(folderLabels([a, twin]).get(a.id)).toBe(a.path);
  });

  it("opens subfolders and returns through the breadcrumbs", async () => {
    const server = await standardApi();
    const { user, router } = renderApp("/library");
    await user.click(await screen.findByRole("button", { name: "2026-10 trips" }));
    expect(await screen.findByRole("table", { name: "Contents of 2026-10 trips" })).toBeInTheDocument();
    expect(router.state.location.search).toBe(`?folder=${FOLDER.id}&path=2026-10+trips`);
    const crumbs = screen.getByRole("navigation", { name: "Folder path" });
    expect(within(crumbs).getByText("2026-10 trips")).toHaveAttribute("aria-current", "page");
    expect(cell("city.mp4").querySelector(".status-text")).toHaveTextContent("Needs attention");
    await user.click(within(crumbs).getByRole("button", { name: "Dashcam" }));
    expect(await screen.findByRole("table", { name: "Contents of Dashcam" })).toBeInTheDocument();
    expect(server.calls(`GET /api/library/${FOLDER.id}/entries`).map((c) => c.path)).toContain(
      `/api/library/${FOLDER.id}/entries?path=2026-10+trips`,
    );
  });

  it("registers one video in place and refreshes the listing", async () => {
    const server = await standardApi({ "POST /api/videos": jsonReply(video("probing", { file_name: "highway.mp4" }), 201) });
    const { user } = renderApp("/library");
    await user.click(await screen.findByRole("button", { name: "Register highway.mp4" }));
    expect(await screen.findByText("Registered 1 video")).toBeInTheDocument();
    expect(server.calls("POST /api/videos").map((c) => c.body)).toEqual([{ folder_id: FOLDER.id, path: "highway.mp4" }]);
    await waitFor(() => expect(server.calls(ROOT).length).toBeGreaterThanOrEqual(2));
  });

  it("registers a selection of videos", async () => {
    const server = await standardApi({ "POST /api/videos": jsonReply(video("probing"), 201) });
    const { user } = renderApp("/library");
    await user.click(await screen.findByRole("checkbox", { name: "Select highway.mp4" }));
    expect(screen.getByRole("checkbox", { name: "Select all unregistered videos" })).toBePartiallyChecked();
    await user.click(screen.getByRole("checkbox", { name: "Select all unregistered videos" }));
    expect(screen.getByText("2 videos selected")).toBeInTheDocument();
    expect(cell("night.webm")).toHaveClass("is-selected");
    await user.click(screen.getByRole("button", { name: "Register selected" }));
    expect(await screen.findByText("Registered 2 videos")).toBeInTheDocument();
    expect(server.calls("POST /api/videos").map((c) => (c.body as { path: string }).path)).toEqual(["highway.mp4", "night.webm"]);
    expect(screen.queryByText("2 videos selected")).not.toBeInTheDocument();
  });

  it("reports videos that could not be registered", async () => {
    await standardApi({ "POST /api/videos": errorReply(404, "file not found in the library folder") });
    const { user } = renderApp("/library");
    await user.click(await screen.findByRole("button", { name: "Register night.webm" }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("1 video not registered");
    expect(alert).toHaveTextContent("night.webm: File not found in the library folder.");
  });

  it("stops registering and adds no toast of its own when the server goes away", async () => {
    const server = await standardApi({ "POST /api/videos": "network-error" });
    const { user } = renderApp("/library");
    await screen.findByRole("table", { name: "Contents of Dashcam" });
    server.on("GET /api/health", "network-error");
    await user.click(screen.getByRole("checkbox", { name: "Select all unregistered videos" }));
    await user.click(screen.getByRole("button", { name: "Register selected" }));
    expect(await screen.findByText("Server unavailable")).toBeInTheDocument();
    expect(server.calls("POST /api/videos")).toHaveLength(1);
    expect(screen.queryByText(/not registered$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^Registered/)).not.toBeInTheDocument();
  });

  it("switches between several library folders", async () => {
    const second = fixture("twoFolders")[1]!;
    const listing = { ...fixture("listing"), folder: second, entries: [] };
    const server = await standardApi({
      "GET /api/library": jsonReply(fixture("twoFolders")),
      [`GET /api/library/${second.id}/entries?path=`]: jsonReply(listing),
    });
    const { user } = renderApp("/library");
    await screen.findByRole("table", { name: "Contents of Dashcam" });
    await user.selectOptions(screen.getByRole("combobox", { name: "Library folder" }), "Archive");
    expect(await screen.findByRole("heading", { name: "No videos or folders here" })).toBeInTheDocument();
    expect(server.calls(`GET /api/library/${second.id}/entries`)).toHaveLength(1);
  });

  it("filters the folder by name and warns about truncated listings", async () => {
    await standardApi({ [ROOT]: jsonReply({ ...fixture("listing"), truncated: true }) });
    const { user } = renderApp("/library");
    expect(await screen.findByText("Folder partly listed")).toBeInTheDocument();
    await user.type(screen.getByRole("searchbox", { name: "Filter this folder" }), "night");
    expect(screen.getAllByRole("row")).toHaveLength(2);
  });

  it("clears a search with no results without leaving the folder", async () => {
    await standardApi();
    const { user, router } = renderApp(`/library?folder=${FOLDER.id}`);
    await screen.findByRole("table", { name: "Contents of Dashcam" });
    await user.type(screen.getByRole("searchbox", { name: "Filter this folder" }), "zzz");
    expect(screen.getByRole("heading", { name: "No results found" })).toBeInTheDocument();
    expect(screen.getByText("Try a different search.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(screen.getByRole("searchbox", { name: "Filter this folder" })).toHaveValue("");
    expect(screen.getByRole("table", { name: "Contents of Dashcam" })).toBeInTheDocument();
    expect(router.state.location.search).toBe(`?folder=${FOLDER.id}`);
  });

  it("reports a folder that no longer exists", async () => {
    await standardApi({ [`GET /api/library/${FOLDER.id}/entries?path=gone`]: errorReply(404, "folder not found") });
    renderApp(`/library?folder=${FOLDER.id}&path=gone`);
    expect(await screen.findByRole("heading", { name: "Folder not found" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Go to Dashcam" })).toBeInTheDocument();
  });
});

describe("library folders", () => {
  const SECOND = fixture("twoFolders")[1]!;
  const added = { ...FOLDER, id: "aaaaaaaaaaaaaaaa", name: "Trips", path: "/Users/reviewer/Trips", source: "added" as const, registered_videos: 0 };
  const emptyListing = (folder: typeof FOLDER) => jsonReply({ folder, path: "", parent: null, entries: [], truncated: false });

  it("adds a folder through the native dialog and opens it at once", async () => {
    const server = await standardApi({
      "POST /api/library/folders": jsonReply({ folder: added, created: true }, 201),
      [`GET /api/library/${added.id}/entries?path=`]: emptyListing(added),
    });
    const { user, router } = renderApp("/library");
    await screen.findByRole("table", { name: "Contents of Dashcam" });
    server.on("GET /api/library", jsonReply([...fixture("folders"), added]));
    await user.click(screen.getByRole("button", { name: "Add folder" }));
    expect(await screen.findByText("Folder added")).toBeInTheDocument();
    expect(server.calls("POST /api/library/folders")[0]!.body).toBeUndefined();
    await waitFor(() => expect(router.state.location.search).toBe(`?folder=${added.id}`));
    expect(await screen.findByRole("heading", { name: "No videos or folders here" })).toBeInTheDocument();
  });

  it("does nothing when the dialog is cancelled and reports failures", async () => {
    const server = await standardApi({ "POST /api/library/folders": jsonReply({ folder: null, created: false }) });
    const { user } = renderApp("/library");
    await screen.findByRole("table", { name: "Contents of Dashcam" });
    await user.click(screen.getByRole("button", { name: "Add folder" }));
    await waitFor(() => expect(server.calls("POST /api/library/folders")).toHaveLength(1));
    expect(screen.queryByText("Folder added")).not.toBeInTheDocument();
    server.on("POST /api/library/folders", errorReply(400, "Spectra's own data folder cannot be a library folder"));
    await user.click(screen.getByRole("button", { name: "Add folder" }));
    expect(await screen.findByText("Folder not added")).toBeInTheDocument();
  });

  it("removes a folder without registrations at once, and never offers removal of startup folders", async () => {
    const server = await standardApi({
      "GET /api/library": jsonReply(fixture("twoFolders")),
      [`GET /api/library/${SECOND.id}/entries?path=`]: emptyListing(SECOND),
      [`DELETE /api/library/folders/${SECOND.id}`]: { status: 204 },
    });
    const { user } = renderApp(`/library?folder=${FOLDER.id}`);
    await screen.findByRole("table", { name: "Contents of Dashcam" });
    expect(screen.queryByRole("button", { name: "Remove folder" })).not.toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Library folder"), SECOND.id);
    await user.click(await screen.findByRole("button", { name: "Remove folder" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(await screen.findByText("Folder removed")).toBeInTheDocument();
    expect(server.calls(`DELETE /api/library/folders/${SECOND.id}`)).toHaveLength(1);
  });

  it("confirms before removing a folder with registered videos", async () => {
    const used = { ...SECOND, registered_videos: 3 };
    const server = await standardApi({
      "GET /api/library": jsonReply([used]),
      [`GET /api/library/${used.id}/entries?path=`]: emptyListing(used),
      [`DELETE /api/library/folders/${used.id}`]: { status: 204 },
    });
    const { user } = renderApp("/library");
    await user.click(await screen.findByRole("button", { name: "Remove folder" }));
    const dialog = await screen.findByRole("dialog", { name: "Remove this folder?" });
    expect(within(dialog).getByText(/3 registered videos/)).toBeInTheDocument();
    expect(within(dialog).getByText(/No files are deleted/)).toBeInTheDocument();
    expect(server.calls(`DELETE /api/library/folders/${used.id}`)).toHaveLength(0);
    await user.click(within(dialog).getByRole("button", { name: "Remove folder" }));
    await waitFor(() => expect(server.calls(`DELETE /api/library/folders/${used.id}`)).toHaveLength(1));
  });

  it("explains an unavailable folder without listing it", async () => {
    const gone = { ...SECOND, available: false };
    const server = await standardApi({ "GET /api/library": jsonReply([gone]) });
    renderApp("/library");
    expect(await screen.findByRole("heading", { name: "Folder not available" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove folder" })).toBeInTheDocument();
    expect(server.calls(`GET /api/library/${gone.id}/entries`)).toHaveLength(0);
  });
});

describe("library table layout", () => {
  it("sizes columns to content and gives the name the remaining space", async () => {
    await standardApi();
    renderApp("/library");
    const table = await screen.findByRole("table", { name: "Contents of Dashcam" });
    expect(table).toHaveClass("table--fit");
    expect(within(table).getByRole("columnheader", { name: "Name" })).toHaveClass("cell-primary");
    expect(table.closest(".table-region")).toBeInTheDocument();
  });

  it("shows folder rows without placeholder metadata, at the same column positions", async () => {
    await standardApi();
    renderApp("/library");
    await screen.findByRole("table", { name: "Contents of Dashcam" });
    const folder = cell("2026-10 trips");
    const file = cell("highway.mp4");
    expect(folder).toHaveClass("is-folder");
    expect(folder.children).toHaveLength(file.children.length);
    const [, , , size, modified] = [...folder.children];
    expect(size).toBeEmptyDOMElement();
    expect(modified).toBeEmptyDOMElement();
    expect(within(folder).queryByText("—")).not.toBeInTheDocument();
  });

  it("truncates long names but keeps the full name available", async () => {
    const long = "a".repeat(120) + ".mp4";
    const listing = fixture("listing");
    await standardApi({
      [ROOT]: jsonReply({ ...listing, entries: [{ ...listing.entries.find((e) => e.kind === "video" && !e.video_id)!, name: long, path: long }] }),
    });
    renderApp("/library");
    const name = await screen.findByText(long, { selector: ".cell-primary__title" });
    expect(name).toHaveClass("truncate");
    expect(name).toHaveAttribute("title", long);
    expect(screen.getByRole("button", { name: `Register ${long}` })).toBeInTheDocument();
  });
});
