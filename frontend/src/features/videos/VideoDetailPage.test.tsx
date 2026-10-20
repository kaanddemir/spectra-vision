import { act, cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Video } from "../../api/types";
import { errorReply, jsonReply, standardApi } from "../../test/api";
import { fixture, video } from "../../test/fixtures";
import { renderApp } from "../../test/render";
import { CHECK_TIMEOUT_MS } from "./usePlaybackCheck";

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

const url = (v: Video) => `/videos/${v.id}`;
const panel = () => screen.findByLabelText("Analysis", { selector: "aside" });
const checkPath = (v: Video) => `POST /api/videos/${v.id}/playback-check`;
const load = () => HTMLMediaElement.prototype.load as unknown as ReturnType<typeof vi.fn>;

/** Each readiness row as [label, status]. */
async function readiness(): Promise<[string, string][]> {
  const section = await screen.findByRole("region", { name: "Analysis readiness" });
  return [...section.querySelectorAll(".readiness__main")].map((row) => [
    row.querySelector(".readiness__label")!.textContent!,
    row.querySelector(".readiness__status")!.textContent!,
  ]);
}

/** Headings of the side panel's sections, in order. */
async function sections(): Promise<string[]> {
  const aside = await panel();
  return [...aside.querySelectorAll("h2")].map((h) => h.textContent!);
}

/** Video information as [label, value] pairs. */
async function information(): Promise<[string, string][]> {
  const section = await screen.findByRole("region", { name: "Video information" });
  return [...section.querySelectorAll(".description-list__item")].map((item) => [
    item.querySelector("dt")!.textContent!,
    item.querySelector("dd")!.textContent!,
  ]);
}

async function rowNote(label: string): Promise<HTMLElement | null> {
  const section = await screen.findByRole("region", { name: "Analysis readiness" });
  const row = within(section).getByText(label).closest(".readiness__row")!;
  return row.querySelector<HTMLElement>(".readiness__note");
}

async function detailApi(v: Video, overrides = {}) {
  return standardApi({
    [`GET /api/videos/${v.id}`]: jsonReply(v),
    [`GET /api/analyses?video_id=${v.id}`]: jsonReply([]),
    [`HEAD /api/videos/${v.id}/media`]: { status: 200 },
    ...overrides,
  });
}

function element(name: string): HTMLVideoElement {
  return screen.getByLabelText(`Preview of ${name}`) as HTMLVideoElement;
}

/** The browser "answers" the element: a decoded frame of the given size, or a media error. */
function answer(name: string, result: { frame?: [number, number]; errorCode?: number }) {
  const media = element(name);
  if (result.frame) {
    Object.defineProperty(media, "videoWidth", { configurable: true, value: result.frame[0] });
    Object.defineProperty(media, "videoHeight", { configurable: true, value: result.frame[1] });
    fireEvent(media, new Event("loadeddata"));
  } else {
    Object.defineProperty(media, "error", { configurable: true, value: { code: result.errorCode } });
    fireEvent(media, new Event("error"));
  }
}

/** Wait until the automatic check has loaded the element, then let the browser answer. */
async function browserAnswers(name: string, result: { frame?: [number, number]; errorCode?: number }) {
  await waitFor(() => expect(load()).toHaveBeenCalled());
  answer(name, result);
}

describe("video detail", () => {
  it("shows the file name, its location and four facts under the player, and nothing technical", async () => {
    const ready = video("ready");
    await detailApi(ready);
    renderApp(url(ready));
    expect(await screen.findByRole("heading", { level: 1, name: "drive.mp4" })).toBeInTheDocument();
    expect(document.title).toBe("Spectra");
    expect(await screen.findByText("Dashcam / drive.mp4")).toBeInTheDocument();
    expect(within(screen.getByRole("main")).queryByRole("link", { name: "Videos" })).not.toBeInTheDocument();
    // The player, then its four properties, on the left; the three analysis sections on the right.
    expect(await sections()).toEqual(["Analysis", "Analysis readiness", "Previous analyses"]);
    const media = screen.getByLabelText("Preview of drive.mp4").closest(".workspace__media")!;
    expect([...media.children].map((child) => child.className.split(" ")[0])).toEqual(["media-stage", "section"]);
    expect(within(media as HTMLElement).getByRole("region", { name: "Video information" })).toBeInTheDocument();
    expect(await information()).toEqual([
      ["Resolution", "1920 × 1080"],
      ["Duration", "1 min 3 sec"],
      ["Frame rate", "29.97 FPS"],
      ["File size", "48.0 MB"],
    ]);
    expect(screen.getByRole("region", { name: "Video information" }).querySelector(".description-list")).toHaveClass("description-list--4", "description-list--center");
    expect(screen.queryByText("Codec")).not.toBeInTheDocument();
    expect(screen.queryByText("H.264")).not.toBeInTheDocument();
    expect(await readiness()).toEqual([
      ["Source file", "Ready"],
      ["Video format", "Supported"],
      ["Browser playback", "Verified"],
    ]);
    for (const hidden of ["Details", "History"]) {
      expect(screen.queryByRole("heading", { name: hidden })).not.toBeInTheDocument();
    }
    for (const hidden of [ready.media!.sha256, "MP4 / QuickTime", "AAC", "29.97 fps", "Per-frame, from the container", "Eligible for analysis"]) {
      expect(screen.queryByText(hidden)).not.toBeInTheDocument();
    }
    expect(screen.queryByRole("button", { name: /Check playback/ })).not.toBeInTheDocument();
    expect(screen.getByLabelText("Preview of drive.mp4")).toHaveAttribute("src", ready.media_url);
  });

  it("follows probing until the details are ready", async () => {
    const probing = video("probing");
    const server = await detailApi(probing);
    renderApp(url(probing));
    expect(await screen.findByText("Reading the video's details…")).toBeInTheDocument();
    expect(await readiness()).toEqual([
      ["Source file", "Ready"],
      ["Video format", "Checking"],
      ["Browser playback", "Unavailable"],
    ]);
    expect(screen.queryByText("Video details are still being read.")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Preview of rain.mp4")).not.toBeInTheDocument();
    server.on(`GET /api/videos/${probing.id}`, jsonReply(video("needs_check", { id: probing.id, file_name: "rain.mp4" })));
    expect(await screen.findByLabelText("Preview of rain.mp4", {}, { timeout: 3000 })).toBeInTheDocument();
  });

  it("explains an unsupported format once and never checks it", async () => {
    const unsupported = video("unsupported");
    const server = await detailApi(unsupported);
    renderApp(url(unsupported));
    const reason = unsupported.playback.reasons[0]!;
    expect(await screen.findByText("Browsers cannot play this format.", { selector: ".media-stage__message p" })).toBeInTheDocument();
    expect(screen.getAllByText(reason)).toHaveLength(1);
    expect(await rowNote("Video format")).toHaveTextContent(reason);
    expect((await readiness())[1]).toEqual(["Video format", "Unsupported"]);
    expect(within(await panel()).getByRole("button", { name: "Analyse" })).toBeDisabled();
    expect(server.calls(`HEAD /api/videos/${unsupported.id}/media`)).toHaveLength(0);
  });

  it("reports a file that could not be read and a missing source", async () => {
    const unreadable = video("unreadable");
    await detailApi(unreadable);
    renderApp(url(unreadable));
    expect(await screen.findByText("The video could not be read, so it cannot be played.")).toBeInTheDocument();
    expect((await readiness())[1]).toEqual(["Video format", "Unreadable"]);
    expect(screen.queryByText("broken.mp4: no decodable video frame.")).not.toBeInTheDocument();
    cleanup();
    vi.restoreAllMocks();
    const missing = video("missing");
    await detailApi(missing);
    renderApp(url(missing));
    expect(await screen.findByText("deleted.mp4 was moved or deleted.", { selector: ".media-stage__message p" })).toBeInTheDocument();
    expect((await readiness())[0]).toEqual(["Source file", "Missing"]);
  });

  it("shows a clear message for an unknown video", async () => {
    await standardApi({ "GET /api/videos/ffffffffffffffffffffffffffffffff": errorReply(404, "video not registered") });
    renderApp("/videos/ffffffffffffffffffffffffffffffff");
    expect(await screen.findByRole("heading", { name: "Video not found" })).toBeInTheDocument();
  });
});

describe("analysis panel", () => {
  it("offers one enabled Analyse action for an eligible video, with no readiness badge", async () => {
    const ready = video("ready");
    await detailApi(ready);
    renderApp(url(ready));
    expect(await within(await panel()).findByRole("button", { name: "Analyse" })).toBeEnabled();
    expect(within(await panel()).queryByText(/eligible/i)).not.toBeInTheDocument();
    const previous = screen.getByRole("region", { name: "Previous analyses" });
    expect(await within(previous).findByText("No previous analyses.")).toBeInTheDocument();
    expect(within(previous).queryByRole("listitem")).not.toBeInTheDocument();
  });

  it("keeps Previous analyses visible and empty when only the latest analysis exists", async () => {
    const ready = video("ready");
    const latest = fixture("analyses").filter((a) => a.video_id === ready.id).slice(0, 1);
    await detailApi(ready, { [`GET /api/analyses?video_id=${ready.id}`]: jsonReply(latest) });
    renderApp(url(ready));
    expect(await within(await panel()).findByRole("link", { name: "Open review" })).toBeInTheDocument();
    const previous = screen.getByRole("region", { name: "Previous analyses" });
    expect(within(previous).getByText("No previous analyses.")).toBeInTheDocument();
    expect(within(previous).queryByRole("link")).not.toBeInTheDocument();
  });

  it("leaves reasons about the video to readiness and states none of them twice", async () => {
    const missing = video("missing");
    await detailApi(missing);
    renderApp(url(missing));
    const analysis = await screen.findByRole("region", { name: "Analysis" });
    expect(within(analysis).getByRole("button", { name: "Analyse" })).toBeDisabled();
    expect((await readiness())[0]).toEqual(["Source file", "Missing"]);
    expect(within(analysis).queryByText(/moved or deleted/)).not.toBeInTheDocument();
    expect(within(await panel()).getAllByText(/moved or deleted/)).toHaveLength(1);
  });

  it("states a blocking reason that readiness does not cover in the Analysis section", async () => {
    const ready = video("ready");
    await detailApi(ready, { "GET /api/health": jsonReply(fixture("healthDegraded")) });
    renderApp(url(ready));
    const analysis = await screen.findByRole("region", { name: "Analysis" });
    expect(await within(analysis).findByText("Memory pressure is critical.")).toBeInTheDocument();
    expect(within(analysis).getByRole("button", { name: "Analyse" })).toBeDisabled();
    expect(await readiness()).toEqual([
      ["Source file", "Ready"],
      ["Video format", "Supported"],
      ["Browser playback", "Verified"],
    ]);
    expect(within(screen.getByRole("main")).getAllByText("Memory pressure is critical.")).toHaveLength(1);
  });

  it("opens the latest result's review and lists earlier analyses, newest first", async () => {
    const ready = video("ready");
    const analyses = fixture("analyses").filter((a) => a.video_id === ready.id);
    await detailApi(ready, { [`GET /api/analyses?video_id=${ready.id}`]: jsonReply(analyses) });
    renderApp(url(ready));
    expect(await within(await panel()).findByRole("link", { name: "Open review" })).toHaveAttribute("href", `/analyses/${analyses[0]!.id}`);
    const previous = screen.getByRole("region", { name: "Previous analyses" });
    const links = within(previous).getAllByRole("link");
    expect(links.map((link) => link.getAttribute("href"))).toEqual(analyses.slice(1).map((a) => `/analyses/${a.id}`));
    expect(screen.queryByRole("button", { name: "Forget video" })).not.toBeInTheDocument();
    expect(within(previous).queryByText("Completed", { selector: ".badge *" })).not.toBeInTheDocument();
  });
});

describe("previous analyses", () => {
  const ready = video("ready");
  const base = fixture("analyses").find((a) => a.video_id === ready.id && a.result)!;
  /** The latest, then `count` earlier analyses, one minute apart unless `times` is given. */
  const history = (count: number, times?: string[]) => [
    { ...base, id: "latest", created_at: "2026-10-20T17:59:00Z" },
    ...Array.from({ length: count }, (_, i) => ({
      ...base,
      id: `earlier-${i}`,
      created_at: times?.[i] ?? `2026-10-20T17:${String(40 - i).padStart(2, "0")}:00Z`,
      result: { ...base.result!, proposals: i },
    })),
  ];
  async function openWith(list: ReturnType<typeof history>) {
    await detailApi(ready, { [`GET /api/analyses?video_id=${ready.id}`]: jsonReply(list) });
    const app = renderApp(url(ready));
    const region = await screen.findByRole("region", { name: "Previous analyses" });
    await within(region).findAllByRole("link");
    return { ...app, region };
  }

  it("lists up to three without a scrolling region", async () => {
    const { region } = await openWith(history(3));
    const list = within(region).getByRole("list", { name: "Previous analyses, newest first" });
    expect(within(list).getAllByRole("link")).toHaveLength(3);
    expect(list).not.toHaveClass("history--scroll");
    expect(list).not.toHaveAttribute("tabindex");
    const first = within(region).getAllByRole("link")[0]!;
    expect(first).toHaveTextContent("Completed");
    expect(first).toHaveTextContent("· 0 proposals");
    expect(first.querySelector(".status-text--success")).not.toBeNull();
  });

  it("keeps every record in one list that scrolls beyond three rows, with no expand control", async () => {
    const { region } = await openWith(history(6));
    const list = within(region).getByRole("list", { name: "Previous analyses, newest first" });
    expect(within(list).getAllByRole("link").map((l) => l.getAttribute("href"))).toEqual(
      [0, 1, 2, 3, 4, 5].map((i) => `/analyses/earlier-${i}`),
    );
    expect(list).toHaveClass("history--scroll");
    expect(list).toHaveAttribute("tabindex", "0");
    expect(within(region).getByRole("heading", { name: "Previous analyses" }).closest("ul")).toBeNull();
    expect(within(region).queryByRole("button")).not.toBeInTheDocument();
  });

  it("orders by time and tells apart analyses from the same minute", async () => {
    const { region } = await openWith(history(3, ["2026-10-20T17:23:05Z", "2026-10-20T17:23:41Z", "2026-10-20T16:38:00Z"]));
    const links = within(region).getAllByRole("link");
    expect(links.map((l) => l.getAttribute("href"))).toEqual(["/analyses/earlier-1", "/analyses/earlier-0", "/analyses/earlier-2"]);
    const dates = links.map((l) => l.querySelector(".history__date")!.textContent);
    expect(dates[0]).toMatch(/:\d\d:\d\d$/);
    expect(dates[1]).toMatch(/:\d\d:\d\d$/);
    expect(dates[0]).not.toBe(dates[1]);
    expect(dates[2]).not.toMatch(/:\d\d:\d\d$/);
  });

  it("states each analysis's own outcome and opens it", async () => {
    const list = history(2);
    list[2] = { ...list[2]!, state: "failed", result: null, stop_reason: "worker_error" };
    const { user, region, router } = await openWith(list);
    const failed = within(region).getAllByRole("link")[1]!;
    expect(failed).not.toHaveTextContent("Completed");
    expect(failed).not.toHaveTextContent("proposal");
    await user.click(failed);
    expect(router.state.location.pathname).toBe("/analyses/earlier-1");
  });
});

describe("automatic playback check", () => {
  it("checks once on opening and reports a decoded frame bound to the loaded contents, without playing", async () => {
    const pending = video("needs_check");
    const play = vi.spyOn(HTMLMediaElement.prototype, "play");
    const server = await detailApi(pending, { [checkPath(pending)]: jsonReply(video("ready", { id: pending.id, file_name: "city.mp4" })) });
    renderApp(url(pending));
    await waitFor(async () => expect((await readiness())[2]).toEqual(["Browser playback", "Checking"]));
    expect(within(await screen.findByRole("region", { name: "Analysis" })).getByRole("button", { name: "Analyse" })).toBeDisabled();
    await browserAnswers("city.mp4", { frame: [1920, 1080] });
    expect(await within(await panel()).findByRole("button", { name: "Analyse" })).toBeEnabled();
    expect(server.calls(checkPath(pending)).map((c) => c.body)).toEqual([
      { playable: true, detail: "Decoded a 1920×1080 frame.", sha256: pending.media!.sha256 },
    ]);
    expect(server.calls(`HEAD /api/videos/${pending.id}/media`)).toHaveLength(1);
    expect(load()).toHaveBeenCalledTimes(1);
    expect(play).not.toHaveBeenCalled();
    expect(server.calls("POST /api/analyses")).toHaveLength(0);
    expect((await readiness())[2]).toEqual(["Browser playback", "Verified"]);
  });

  it("never re-checks a video that already played here", async () => {
    const ready = video("ready");
    const server = await detailApi(ready);
    renderApp(url(ready));
    await screen.findByLabelText("Analysis", { selector: "aside" });
    await within(await panel()).findByRole("button", { name: "Analyse" });
    expect(server.calls(`HEAD /api/videos/${ready.id}/media`)).toHaveLength(0);
    expect(load()).not.toHaveBeenCalled();
  });

  it("does not repeat the check when the video is reloaded with the same contents", async () => {
    const pending = video("needs_check");
    const server = await detailApi(pending);
    const { client } = renderApp(url(pending));
    await browserAnswers("city.mp4", { errorCode: 2 });
    await client.refetchQueries({ queryKey: ["videos", pending.id] });
    await client.refetchQueries({ queryKey: ["videos", pending.id] });
    expect(server.calls(`HEAD /api/videos/${pending.id}/media`)).toHaveLength(1);
  });

  it("records a format the browser cannot decode and keeps analysis disabled", async () => {
    const pending = video("needs_check");
    const refused = video("not_playable", { id: pending.id, file_name: "city.mp4", media: pending.media, media_url: pending.media_url });
    const server = await detailApi(pending, { [checkPath(pending)]: jsonReply(refused) });
    renderApp(url(pending));
    await browserAnswers("city.mp4", { errorCode: 4 });
    await waitFor(async () => expect((await readiness())[2]).toEqual(["Browser playback", "Failed"]));
    expect(await rowNote("Browser playback")).toHaveTextContent("This browser cannot play the video.");
    expect(screen.getAllByText(/cannot play the video|could not play the video/)).toHaveLength(1);
    expect(within(await panel()).getByRole("button", { name: "Analyse" })).toBeDisabled();
    expect(server.calls(checkPath(pending))[0]!.body).toMatchObject({ playable: false, detail: "This browser does not support the video's format." });
  });

  it("completes through the person's own Play when the browser loads nothing by itself", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const pending = video("needs_check");
    const server = await detailApi(pending, { [checkPath(pending)]: jsonReply(video("ready", { id: pending.id, file_name: "city.mp4" })) });
    renderApp(url(pending));
    await waitFor(() => expect(load()).toHaveBeenCalled());
    await act(async () => {
      vi.advanceTimersByTime(CHECK_TIMEOUT_MS + 10);
    });
    await waitFor(async () => expect((await readiness())[2]).toEqual(["Browser playback", "Press Play"]));
    expect(await rowNote("Browser playback")).toHaveTextContent("Play the video once to verify it.");
    expect(server.calls(checkPath(pending))).toHaveLength(0);
    answer("city.mp4", { frame: [1920, 1080] }); // the frame the person's Play produced
    await waitFor(() => expect(server.calls(checkPath(pending))).toHaveLength(1));
    expect(server.calls(checkPath(pending))[0]!.body).toMatchObject({ playable: true });
    expect(load()).toHaveBeenCalledTimes(1);
  });

  it("does not reload a player the person is already using", async () => {
    const pending = video("needs_check");
    const server = await detailApi(pending, { [checkPath(pending)]: jsonReply(video("ready", { id: pending.id, file_name: "city.mp4" })) });
    vi.spyOn(HTMLMediaElement.prototype, "paused", "get").mockReturnValue(false);
    renderApp(url(pending));
    await waitFor(() => expect(server.calls(`HEAD /api/videos/${pending.id}/media`)).toHaveLength(1));
    expect(load()).not.toHaveBeenCalled();
    answer("city.mp4", { frame: [1920, 1080] });
    await waitFor(() => expect(server.calls(checkPath(pending))).toHaveLength(1));
  });

  it("does not blame the browser when the server refuses to stream", async () => {
    const pending = video("needs_check");
    const server = await detailApi(pending, { [`HEAD /api/videos/${pending.id}/media`]: { status: 409 } });
    renderApp(url(pending));
    await waitFor(async () => expect((await readiness())[2]).toEqual(["Browser playback", "Not checked"]));
    const note = (await rowNote("Browser playback"))!;
    expect(note).toHaveTextContent(/will not stream this video right now/);
    expect(within(note).getByRole("button", { name: "Try again" })).toBeInTheDocument();
    expect(screen.getAllByText(/will not stream this video right now/)).toHaveLength(1);
    expect(server.calls(checkPath(pending))).toHaveLength(0);
    expect(load()).not.toHaveBeenCalled();
  });

  it("does not record an interrupted download, and tries again on request", async () => {
    const pending = video("needs_check");
    const server = await detailApi(pending);
    const { user } = renderApp(url(pending));
    await browserAnswers("city.mp4", { errorCode: 2 });
    expect(await within(await panel()).findByText(/The video could not be downloaded from the server\./)).toBeInTheDocument();
    expect(server.calls(checkPath(pending))).toHaveLength(0);
    await user.click(within(await panel()).getByRole("button", { name: "Try again" }));
    await waitFor(() => expect(server.calls(`HEAD /api/videos/${pending.id}/media`)).toHaveLength(2));
  });

  it("abandons a check when the file changes and checks the new contents instead", async () => {
    const pending = video("needs_check");
    const server = await detailApi(pending, { [checkPath(pending)]: jsonReply(video("ready", { id: pending.id, file_name: "city.mp4" })) });
    const { client } = renderApp(url(pending));
    await waitFor(() => expect(load()).toHaveBeenCalledTimes(1));
    const changed = { ...pending, media: { ...pending.media!, sha256: "f".repeat(64) } };
    act(() => client.setQueryData(["videos", pending.id], changed));
    await waitFor(() => expect(load()).toHaveBeenCalledTimes(2));
    answer("city.mp4", { frame: [1920, 1080] });
    await waitFor(() => expect(server.calls(checkPath(pending))).toHaveLength(1));
    // Only the new contents are reported; nothing is attached to the contents first loaded.
    expect(server.calls(checkPath(pending))[0]!.body).toMatchObject({ sha256: "f".repeat(64) });
  });

  it("shows the server's refusal when the contents changed before the report arrived", async () => {
    const pending = video("needs_check");
    await detailApi(pending, { [checkPath(pending)]: errorReply(409, "the video changed after it was loaded; check it again") });
    renderApp(url(pending));
    await browserAnswers("city.mp4", { frame: [1920, 1080] });
    expect(await within(await panel()).findByText(/The video changed after it was loaded; check it again\./)).toBeInTheDocument();
  });

  it("leaves an unreachable server to the global banner and checks again when it returns", async () => {
    const pending = video("needs_check");
    const server = await detailApi(pending, {
      [`HEAD /api/videos/${pending.id}/media`]: "network-error",
      [checkPath(pending)]: jsonReply(video("ready", { id: pending.id, file_name: "city.mp4" })),
    });
    server.on("GET /api/health", "network-error");
    const { client } = renderApp(url(pending));
    expect(await screen.findByText("Server unavailable")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try again" })).not.toBeInTheDocument();
    expect(load()).not.toHaveBeenCalled();

    server.on("GET /api/health", jsonReply(fixture("health")));
    server.on(`HEAD /api/videos/${pending.id}/media`, { status: 200 });
    await client.refetchQueries({ queryKey: ["health"] });
    await client.refetchQueries({ queryKey: ["videos", pending.id] });
    await browserAnswers("city.mp4", { frame: [1920, 1080] });
    await waitFor(() => expect(server.calls(checkPath(pending))).toHaveLength(1));
  });

  it("checks each video separately and ignores a result from the previous one", async () => {
    const first = video("needs_check");
    const secondId = "b".repeat(32);
    const second = video("needs_check", {
      id: secondId,
      file_name: "night.mp4",
      media_url: `/api/videos/${secondId}/media`,
      media: { ...first.media!, sha256: "e".repeat(64) },
    });
    const server = await detailApi(first, {
      [`GET /api/videos/${second.id}`]: jsonReply(second),
      [`GET /api/analyses?video_id=${second.id}`]: jsonReply([]),
      [`HEAD /api/videos/${second.id}/media`]: { status: 200 },
    });
    const { router } = renderApp(url(first));
    await waitFor(() => expect(load()).toHaveBeenCalledTimes(1));
    await act(() => router.navigate(url(second)));
    await waitFor(() => expect(load()).toHaveBeenCalledTimes(2));
    expect(server.calls(checkPath(first))).toHaveLength(0);
    expect(server.calls(`HEAD /api/videos/${second.id}/media`)).toHaveLength(1);
  });
});

describe("results from another browser", () => {
  it("does not count another browser's success here: checks again and waits before allowing analysis", async () => {
    const elsewhere = video("ready");
    vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue("Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0");
    const server = await detailApi(elsewhere, {
      [checkPath(elsewhere)]: jsonReply({
        ...elsewhere,
        playback: { ...elsewhere.playback, browser_check: { ...elsewhere.playback.browser_check!, user_agent: navigator.userAgent } },
      }),
    });
    renderApp(url(elsewhere));
    const analysis = await screen.findByRole("region", { name: "Analysis" });
    await waitFor(async () => expect((await readiness())[2]).toEqual(["Browser playback", "Checking"]));
    expect(within(analysis).getByRole("button", { name: "Analyse" })).toBeDisabled();
    await browserAnswers("drive.mp4", { frame: [1920, 1080] });
    await waitFor(async () => expect((await readiness())[2]).toEqual(["Browser playback", "Verified"]));
    expect(within(analysis).getByRole("button", { name: "Analyse" })).toBeEnabled();
    expect(server.calls(checkPath(elsewhere))).toHaveLength(1);
  });

  it("compares the user agent as the server stores it, truncated", async () => {
    const { resultHere } = await import("./usePlaybackCheck");
    const long = "x".repeat(400);
    const stored = video("ready");
    const truncated = { ...stored, playback: { ...stored.playback, browser_check: { ...stored.playback.browser_check!, user_agent: long.slice(0, 300) } } };
    expect(resultHere(truncated, long)).not.toBeNull();
    expect(resultHere(truncated, "y".repeat(400))).toBeNull();
    const anonymous = { ...stored, playback: { ...stored.playback, browser_check: { ...stored.playback.browser_check!, user_agent: null } } };
    expect(resultHere(anonymous, long)).toBeNull();
  });
});

describe("registration management", () => {
  it("forgets a video without analyses after confirmation", async () => {
    const pending = video("needs_check");
    const server = await detailApi(pending, { [`DELETE /api/videos/${pending.id}`]: { status: 204 } });
    const { user, router } = renderApp(url(pending));
    await user.click(await screen.findByRole("button", { name: "Forget video" }));
    const dialog = screen.getByRole("dialog", { name: "Forget this video?" });
    expect(dialog).toHaveTextContent("stays where it is, unchanged");
    await user.click(within(dialog).getByRole("button", { name: "Forget video" }));
    await waitFor(() => expect(router.state.location.pathname).toBe("/videos"));
    expect(server.calls(`DELETE /api/videos/${pending.id}`)).toHaveLength(1);
    expect(await screen.findByText("Video forgotten")).toBeInTheDocument();
  });
});

describe("video information", () => {
  it("keeps all four properties and says Unknown for values not known yet", async () => {
    const probing = video("probing");
    await detailApi(probing);
    renderApp(url(probing));
    expect(await information()).toEqual([
      ["Resolution", "Unknown"],
      ["Duration", "Unknown"],
      ["Frame rate", "Unknown"],
      ["File size", "48.0 MB"],
    ]);
  });

  it("shows the reported frame rate and never invents one", async () => {
    const ready = video("ready");
    const without = { ...ready, media: { ...ready.media!, nominal_fps: null } };
    await detailApi(without);
    renderApp(url(without));
    expect((await information())[2]).toEqual(["Frame rate", "Unknown"]);
  });
});
