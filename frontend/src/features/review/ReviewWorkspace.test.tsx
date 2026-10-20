import layoutCss from "../../styles/layout.css?raw";
import timelineCss from "./ReviewTimeline.css?raw";
import playerCss from "./ReviewPlayer.css?raw";
import { formatTimecode } from "../../lib/format";
import { act, cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Analysis, EvidencePackage, OverlayResponse, Timeline } from "../../api/types";
import { errorReply, jsonReply, standardApi } from "../../test/api";
import { fixture, video } from "../../test/fixtures";
import { renderApp } from "../../test/render";
import { type FakeVideoOptions, installFakeVideo } from "../../test/video";
import evidenceCss from "./EvidencePanel.css?raw";

const latest: Analysis = fixture("analyses")[0]!;
const base = `/api/analyses/${latest.id}`;
const timeline = fixture("timeline");
const [approach, cutIn, contact] = timeline.entries as [Timeline["entries"][0], Timeline["entries"][0], Timeline["entries"][0]];
const overlayPath = `GET ${base}/packages/${approach.event_id}/overlay`;

let fake: ReturnType<typeof installFakeVideo> | null = null;
afterEach(() => {
  cleanup();
  fake?.restore();
  fake = null;
});

interface Open {
  path?: string;
  overrides?: Parameters<typeof standardApi>[0];
  media?: FakeVideoOptions;
  timeline?: Timeline;
  overlay?: OverlayResponse;
}

async function open({ path = `/analyses/${latest.id}`, overrides = {}, media, timeline: data = timeline, overlay = fixture("overlay") }: Open = {}) {
  fake = installFakeVideo(media);
  const packages = fixture("packages");
  const server = await standardApi({
    [`GET ${base}`]: jsonReply(latest),
    [`GET ${base}/timeline`]: jsonReply(data),
    [`GET /api/videos/${latest.video_id}`]: jsonReply(video("ready")),
    ...Object.fromEntries(Object.entries(packages).map(([id, pk]) => [`GET ${base}/packages/${id}`, jsonReply(pk)])),
    [overlayPath]: jsonReply(overlay),
    [`GET ${base}/reviews`]: jsonReply({ analysis_id: latest.id, decisions: [] }),
    [`GET ${base}/edits`]: jsonReply({ analysis_id: latest.id, edits: [] }),
    ...overrides,
  });
  const app = renderApp(path);
  await screen.findByLabelText("Source video");
  return { server, ...app, media: fake };
}

const proposals = () => screen.getByRole("list", { name: "Proposals in time order" });
const proposal = (title: string) => within(proposals()).getByRole("button", { name: new RegExp(title) });
const slider = () => screen.getByRole("slider", { name: "Playback position" });
const chip = () => document.querySelector(".review-player__chip");

describe("review workspace", () => {
  it("opens a ready analysis with the player, timeline and proposals in order", async () => {
    const { server } = await open();
    expect(screen.getByRole("heading", { level: 1, name: "drive.mp4" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Back to video" })).toHaveAttribute("href", `/videos/${latest.video_id}`);
    expect(screen.queryByText("Completed")).not.toBeInTheDocument();
    expect(screen.queryByText(/^Requested /)).not.toBeInTheDocument();
    const back = screen.getByRole("link", { name: "Back to video" });
    expect(back).toHaveTextContent("");
    expect(back.closest(".page-header__title-row")).toContainElement(screen.getByRole("heading", { level: 1 }));
    expect(within(proposals()).getAllByRole("button").map((b) => b.textContent)).toEqual([
      expect.stringContaining("Rapid approach"),
      expect.stringContaining("Cut-in"),
      expect.stringContaining("Possible contact between other road users"),
    ]);
    expect(screen.getByRole("region", { name: "Event proposals" })).toHaveTextContent("Not confirmed incidents");
    expect(screen.getByText("Select an event to review")).toBeInTheDocument();
    expect(screen.getByLabelText("Source video")).toHaveAttribute("src", `/api/videos/${latest.video_id}/media`);
    expect(screen.queryByText("Review not available yet")).not.toBeInTheDocument();
    // Overlays are off until asked for.
    expect(server.calls(overlayPath)).toHaveLength(0);
  });

  it("goes back to the video's page, also after a direct load", async () => {
    const { user, router } = await open();
    await user.click(screen.getByRole("link", { name: "Back to video" }));
    expect(router.state.location.pathname).toBe(`/videos/${latest.video_id}`);
  });

  it("keeps technical details, notes and analyser identifiers off the page", async () => {
    const noted = { ...latest, result: { ...latest.result!, notes: ["system memory pressure reached level 2; analysis continued"] } };
    await open({ overrides: { [`GET ${base}`]: jsonReply(noted) } });
    for (const gone of ["Analysis details", "Notes", "Analysers", "Source SHA-256", "Frames analysed"]) {
      expect(screen.queryByText(gone)).not.toBeInTheDocument();
    }
    expect(screen.queryByText(/memory pressure/)).not.toBeInTheDocument();
    expect(screen.queryByText(latest.source_sha256)).not.toBeInTheDocument();
    expect(screen.queryByText("camera-vehicle-impact")).not.toBeInTheDocument();
    expect(screen.queryByText(/did not run/)).not.toBeInTheDocument();
  });

  it("says plainly when a check did not run, since its silence means nothing", async () => {
    const analyzers = timeline.analyzers.map((a) =>
      a.name === "camera-vehicle-impact" ? { ...a, status: "unavailable" as const, coverage: "unavailable" as const, reason: "model missing" } : a,
    );
    await open({ timeline: { ...timeline, analyzers } });
    const alert = screen.getByText("One check did not run").closest(".alert")!;
    expect(alert).toHaveTextContent("Impact: this check did not run, so no such events could be proposed.");
    expect(alert).not.toHaveTextContent("camera-vehicle-impact");
  });

  it("treats zero proposals as a result and still shows coverage and the playhead", async () => {
    await open({ timeline: { ...timeline, entries: [], segments: [] } });
    const region = screen.getByRole("region", { name: "Event proposals" });
    expect(within(region).getByRole("heading", { name: "No events proposed" })).toBeInTheDocument();
    expect(within(region).getByText("Review the video for anything that may have been missed.")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(document.querySelectorAll('[data-coverage="analysed"]').length).toBeGreaterThan(0);
    expect(screen.getByTestId("playhead")).toBeInTheDocument();
    expect(screen.queryByText("Select an event to review")).not.toBeInTheDocument();
    expect(screen.getAllByText(/No events proposed/)).toHaveLength(1);
  });

  it("keeps the reason an analysis stopped early", async () => {
    const stopped = { ...latest, state: "resource_stopped" as const, stop_reason: "memory_limit", message: "the memory limit was reached at 0:41" };
    await standardApi({ [`GET ${base}`]: jsonReply(stopped) });
    renderApp(`/analyses/${latest.id}`);
    expect(await screen.findByText("The memory limit was reached at 0:41.")).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 1, name: "drive.mp4" })).toBeInTheDocument();
  });

  it("selects a proposal, seeks to its review span and keeps the choice in the address", async () => {
    const { user, router, media } = await open();
    media.loadMetadata();
    await user.click(proposal("Cut-in"));
    expect(proposal("Cut-in")).toHaveAttribute("aria-current", "true");
    expect(router.state.location.search).toBe(`?event=${cutIn.event_id}`);
    expect(media.currentTime).toBe(cutIn.review.start_s);
    expect(await screen.findByRole("heading", { level: 2, name: "Cut-in" })).toBeInTheDocument();
    expect(await screen.findByText(/The moment the tracked vehicle entered the lane was not seen\./)).toBeInTheDocument();
  });

  it("opens a direct link to a proposal after a reload and seeks once the video is ready", async () => {
    const { media } = await open({ path: `/analyses/${latest.id}?event=${contact.event_id}` });
    expect(proposal("Possible contact")).toHaveAttribute("aria-current", "true");
    expect(media.currentTime).toBe(0);
    media.loadMetadata();
    expect(media.currentTime).toBe(contact.review.start_s);
    expect(await screen.findByText(/This check is experimental and not yet validated\./)).toBeInTheDocument();
  });

  it("says when a linked proposal is not part of the analysis", async () => {
    await open({ path: `/analyses/${latest.id}?event=0123456789abcdef` });
    expect(screen.getByText("Proposal not found")).toBeInTheDocument();
    expect(within(proposals()).queryByRole("button", { current: true })).not.toBeInTheDocument();
  });

  it("moves between proposals with the arrow keys", async () => {
    const { user } = await open();
    proposal("Rapid approach").focus();
    await user.keyboard("{ArrowDown}");
    expect(proposal("Cut-in")).toHaveFocus();
    await user.keyboard("{ArrowUp}{ArrowUp}");
    expect(proposal("Rapid approach")).toHaveFocus();
  });

  it("cancels the previous evidence request when another proposal is chosen", async () => {
    const { user, server } = await open({
      overrides: { [`GET ${base}/packages/${approach.event_id}`]: () => new Promise(() => undefined) },
    });
    await user.click(proposal("Rapid approach"));
    await waitFor(() => expect(server.calls(`GET ${base}/packages/${approach.event_id}`)).toHaveLength(1));
    const first = server.fetchMock.mock.calls.find(([input]) => String(input).endsWith(`/packages/${approach.event_id}`));
    await user.click(proposal("Cut-in"));
    await waitFor(() => expect(first?.[1]?.signal?.aborted).toBe(true));
  });
});

describe("timeline", () => {
  it("places suggested events and coverage proportionally, with no review window before a selection", async () => {
    const withGap = { ...timeline, gaps: [{ start_s: 20, end_s: 22.5 }] };
    const { media } = await open({ timeline: withGap });
    media.loadMetadata();
    const marker = document.querySelector(`[data-marker="${approach.event_id}"]`)!;
    expect(marker).toHaveAttribute("data-peak", "23.200%"); // 14.5 s of 62.5 s
    const middle = (((approach.t_start + approach.t_end) / 2 / 62.5) * 100).toFixed(3);
    expect(document.querySelector(`[data-event="${approach.event_id}"]`)!.closest("li")).toHaveAttribute("data-left", `${middle}%`);
    expect(document.querySelectorAll("[data-window]")).toHaveLength(0);
    expect(document.querySelectorAll("[data-segment]")).toHaveLength(0);
    const runs = [...document.querySelectorAll("[data-coverage]")].map((r) => [
      r.getAttribute("data-coverage"),
      r.getAttribute("x"),
      r.getAttribute("width"),
    ]);
    expect(runs).toEqual([
      ["analysed", "0.000%", "32.000%"],
      ["unanalysed", "32.000%", "4.000%"],
      ["analysed", "36.000%", "63.840%"],
      ["unanalysed", "99.840%", "0.160%"],
    ]);
    expect(screen.getByText("Not analysed")).toBeInTheDocument();
  });

  it("uses the recorded duration, not the browser's, for an offset stream", async () => {
    // Chrome reports such a file's duration without its start offset while presenting frames at container time.
    const shifted = { ...timeline, video: { ...timeline.video!, start_s: 0.3, duration_s: 62.5 } };
    const { media, user } = await open({ timeline: shifted, media: { duration: 62.5 } });
    media.loadMetadata();
    expect(screen.getByText(/\/ 1:02\.5$/)).toBeInTheDocument();
    slider().focus();
    await user.keyboard("{End}");
    expect(media.currentTime).toBeCloseTo(62.8, 9);
  });

  it("follows presented frames with the playhead", async () => {
    const { media } = await open();
    media.loadMetadata();
    media.presentFrame(31.25);
    expect(screen.getByTestId("playhead")).toHaveAttribute("data-progress", "50.000%");
    expect(slider()).toHaveAttribute("aria-valuenow", "31.3");
    expect(media.pendingFrameCallbacks()).toBe(1); // the next frame is requested again
  });

  it("seeks where the progress bar is clicked, and selects an event clicked on the Event timeline", async () => {
    const { media } = await open();
    media.loadMetadata();
    vi.spyOn(slider(), "getBoundingClientRect").mockReturnValue({ left: 100, width: 1000 } as DOMRect);
    fireEvent.pointerDown(slider(), { clientX: 600, button: 0, pointerId: 1 });
    expect(media.currentTime).toBeCloseTo(31.25, 6);
    fireEvent.pointerUp(slider(), { pointerId: 1 });
    fireEvent.click(document.querySelector(`[data-event="${contact.event_id}"]`)!);
    expect(proposal("Possible contact")).toHaveAttribute("aria-current", "true");
    expect(media.currentTime).toBe(contact.review.start_s);
  });

  it("moves the playback position from the keyboard on the progress bar", async () => {
    const { user, media } = await open();
    media.loadMetadata();
    slider().focus();
    await user.keyboard("{End}");
    expect(media.currentTime).toBe(62.5);
    expect(slider()).toHaveAttribute("aria-valuetext", "1:02.5 of 1:02.5");
    await user.keyboard("{Home}{ArrowRight}{PageUp}");
    expect(media.currentTime).toBe(11);
    expect(slider()).toHaveAttribute("aria-valuetext", "0:11.0 of 1:02.5");
  });
});

describe("player", () => {
  it("plays, pauses and skips from buttons and the keyboard", async () => {
    const { user, media } = await open();
    media.loadMetadata();
    await user.click(screen.getByRole("button", { name: "Play" }));
    expect(media.play).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "Pause" })).toBeInTheDocument();
    screen.getByRole("group", { name: "Video player" }).focus();
    await user.keyboard(" ");
    expect(media.pause).toHaveBeenCalledTimes(1);
    await user.keyboard("{ArrowRight}{ArrowRight}{ArrowLeft}");
    expect(media.currentTime).toBe(5);
    await user.click(screen.getByRole("button", { name: "Back 5 seconds" }));
    expect(media.currentTime).toBe(0);
  });

  it("keeps controls and the progress bar inside the player, starting muted", async () => {
    const { media } = await open();
    const toolbar = screen.getByRole("toolbar", { name: "Playback" });
    expect(screen.getByRole("group", { name: "Video player" })).toContainElement(toolbar);
    expect(screen.getByRole("group", { name: "Video player" })).toContainElement(slider());
    expect(within(toolbar).getByRole("button", { name: "Play" })).toBeDisabled();
    expect(within(toolbar).getByRole("button", { name: "Playback speed: 1×" })).toBeDisabled();
    media.loadMetadata();
    expect(within(toolbar).getByRole("button", { name: "Play" })).toBeEnabled();
    expect(media.element().muted).toBe(true);
    expect(media.play).not.toHaveBeenCalled();
    expect(within(toolbar).queryByRole("button", { name: /frame/i })).not.toBeInTheDocument();
  });

  it("changes playback speed without changing timestamps", async () => {
    const { user, media } = await open();
    media.loadMetadata();
    const toolbar = screen.getByRole("toolbar", { name: "Playback" });
    expect(within(toolbar).getByRole("button", { name: "Playback speed: 1×" })).toBeEnabled();
    media.timeUpdate(12.5, "seeked");
    await user.click(screen.getByRole("button", { name: "Playback speed: 1×" }));
    const speeds = screen.getByRole("group", { name: "Playback speed" });
    expect(within(speeds).getAllByRole("radio").map((r) => r.closest("label")!.textContent)).toEqual(["0.25×", "0.5×", "1×", "1.5×", "2×"]);
    await user.click(within(speeds).getByRole("radio", { name: "0.25×" }));
    expect(media.element().playbackRate).toBe(0.25);
    expect(screen.getByRole("button", { name: "Playback speed: 0.25×" })).toHaveAttribute("aria-expanded", "false");
    expect(slider()).toHaveAttribute("aria-valuetext", "0:12.5 of 1:02.5");
    await user.click(screen.getByRole("button", { name: "Forward 5 seconds" }));
    expect(media.currentTime).toBe(17.5);
  });

  it("offers sound only for a source with audio", async () => {
    const { user, media } = await open();
    media.loadMetadata();
    await user.click(screen.getByRole("button", { name: "Unmute" }));
    expect(media.element().muted).toBe(false);
    expect(screen.getByRole("button", { name: "Mute" })).toHaveAttribute("aria-pressed", "true");
    cleanup();
    fake?.restore();
    const silent = video("ready");
    silent.media!.audio_codecs = [];
    await open({ overrides: { [`GET /api/videos/${latest.video_id}`]: jsonReply(silent) } });
    expect(screen.queryByRole("button", { name: "Unmute" })).not.toBeInTheDocument();
  });

  it("opens the whole player full screen where the browser allows it", async () => {
    const enabled = Object.getOwnPropertyDescriptor(Document.prototype, "fullscreenEnabled");
    Object.defineProperty(document, "fullscreenEnabled", { configurable: true, value: true });
    const request = vi.fn(() => Promise.resolve());
    Object.defineProperty(HTMLElement.prototype, "requestFullscreen", { configurable: true, writable: true, value: request });
    try {
      const { user } = await open();
      await user.click(screen.getByRole("button", { name: "Full screen" }));
      expect(request).toHaveBeenCalledTimes(1);
      expect(request.mock.contexts[0]).toHaveClass("review-player");
    } finally {
      delete (document as unknown as Record<string, unknown>).fullscreenEnabled;
      if (enabled) Object.defineProperty(Document.prototype, "fullscreenEnabled", enabled);
      delete (HTMLElement.prototype as unknown as Record<string, unknown>).requestFullscreen;
    }
  });

  it("offers no full screen where the browser does not allow it", async () => {
    await open();
    expect(screen.queryByRole("button", { name: "Full screen" })).not.toBeInTheDocument();
  });

  it("follows currentTime on animation frames where frame callbacks are missing", async () => {
    const frames: FrameRequestCallback[] = [];
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => frames.push(callback));
    const cancel = vi.spyOn(window, "cancelAnimationFrame").mockImplementation(() => undefined);
    const { user, media } = await open({ media: { frameCallbacks: false } });
    media.loadMetadata();
    media.timeUpdate(12.5, "seeked");
    expect(screen.getByTestId("playhead")).toHaveAttribute("data-progress", "20.000%");
    await user.click(screen.getByRole("button", { name: "Play" }));
    expect(frames).toHaveLength(1);
    media.timeUpdate(25, "timeupdate");
    act(() => frames.shift()!(0));
    expect(screen.getByTestId("playhead")).toHaveAttribute("data-progress", "40.000%");
    await user.click(screen.getByRole("button", { name: "Pause" }));
    expect(cancel).toHaveBeenCalled();
  });

  it("removes its media listeners and frame callbacks when it closes", async () => {
    const { media, unmount } = await open();
    media.loadMetadata();
    expect(media.pendingFrameCallbacks()).toBe(1);
    unmount();
    expect(media.pendingFrameCallbacks()).toBe(0);
  });

  it("explains a format the browser cannot play and keeps the evidence", async () => {
    const { media } = await open({ path: `/analyses/${latest.id}?event=${approach.event_id}` });
    media.fail(4);
    expect(screen.getByRole("alert")).toHaveTextContent("This browser cannot play this video's format.");
    expect(screen.queryByRole("button", { name: "Play" })).not.toBeInTheDocument();
    expect(await screen.findByRole("region", { name: "Why this event?" })).toBeInTheDocument();
  });

  it("does not offer playback when the server will not stream the video", async () => {
    fake = installFakeVideo();
    await standardApi({
      [`GET ${base}`]: jsonReply(latest),
      [`GET ${base}/timeline`]: jsonReply(timeline),
      [`GET /api/videos/${latest.video_id}`]: jsonReply(
        video("missing", { id: latest.video_id, media: { ...video("missing").media!, sha256: latest.source_sha256 } }),
      ),
    });
    renderApp(`/analyses/${latest.id}`);
    expect(await screen.findByText("This video cannot be played here")).toBeInTheDocument();
    expect(screen.getByText("deleted.mp4 was moved or deleted.")).toBeInTheDocument();
    expect(document.querySelector("video")).toBeNull();
  });

  it("refuses to play a source that changed after the analysis", async () => {
    fake = installFakeVideo();
    const changed = video("ready");
    changed.media!.sha256 = "f".repeat(64);
    await standardApi({
      [`GET ${base}`]: jsonReply(latest),
      [`GET ${base}/timeline`]: jsonReply(timeline),
      [`GET /api/videos/${latest.video_id}`]: jsonReply(changed),
    });
    renderApp(`/analyses/${latest.id}`);
    expect(await screen.findByText(/The video file changed after this analysis/)).toBeInTheDocument();
    expect(document.querySelector("video")).toBeNull();
  });
});

describe("overlays", () => {
  async function withOverlays(options: Open = {}) {
    const app = await open({ path: `/analyses/${latest.id}?event=${approach.event_id}`, ...options });
    app.media.loadMetadata();
    await app.user.click(screen.getByRole("button", { name: "Overlays" }));
    return app;
  }

  it("draws the recorded frame at the current time in source pixels", async () => {
    const { media, server } = await withOverlays();
    media.presentFrame(12.04);
    const drawing = await screen.findByTestId("overlay-drawing");
    expect(server.calls(overlayPath)[0]?.path).toBe(`${base}/packages/${approach.event_id}/overlay`);
    expect(drawing).toHaveAttribute("viewBox", "0 0 640 360");
    expect(drawing).toHaveAttribute("preserveAspectRatio", "xMidYMid meet");
    // Frame 12.0 (index 30 of the interval): box x1 = 100 + 120, as recorded.
    const box = drawing.querySelector('[data-track="1"] rect')!;
    expect([box.getAttribute("x"), box.getAttribute("y"), box.getAttribute("width"), box.getAttribute("height")]).toEqual([
      "220",
      "100",
      "100",
      "100",
    ]);
    expect(drawing.querySelector('[data-track="1"]')).toHaveClass("overlay-drawing__track--related");
    expect(chip()).toBeNull();
  });

  it("offers toggles only for layers the analysis produced", async () => {
    const response = fixture("overlay");
    response.layers.lane = false;
    const { user } = await withOverlays({ overlay: response });
    await user.click(await screen.findByRole("button", { name: "Overlay layers" }));
    expect(await screen.findByRole("checkbox", { name: "Boxes" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Boxes" })).toHaveFocus();
    expect(screen.queryByRole("checkbox", { name: "Lanes" })).not.toBeInTheDocument();
  });

  it("switches boxes and lanes independently", async () => {
    const { user, media } = await withOverlays();
    media.presentFrame(12);
    const drawing = await screen.findByTestId("overlay-drawing");
    expect(drawing.querySelectorAll("polyline").length).toBeGreaterThan(0);
    await user.click(await screen.findByRole("button", { name: "Overlay layers" }));
    await user.click(await screen.findByRole("checkbox", { name: "Lanes" }));
    expect(screen.getByRole("checkbox", { name: "Lanes" })).not.toBeChecked();
    expect(screen.getByTestId("overlay-drawing").querySelectorAll("polyline")).toHaveLength(0);
    await user.click(screen.getByRole("checkbox", { name: "Boxes" }));
    expect(screen.getByTestId("overlay-drawing").querySelectorAll("rect")).toHaveLength(0);
    await user.click(screen.getByRole("button", { name: "Overlays" }));
    expect(screen.queryByTestId("overlay-drawing")).not.toBeInTheDocument();
  });

  it("removes a stale overlay and says 'not analysed here' between valid frames", async () => {
    const response = fixture("overlay");
    // Keep only the frames at 10.0 s and 11.0 s, each valid for 0.25 s.
    response.overlay.frames = response.overlay.frames
      .filter((f) => f.t === 10 || f.t === 11)
      .map((f) => ({ ...f, valid_until: f.t + 0.25 }));
    const { media } = await withOverlays({ overlay: response });
    media.presentFrame(10.2);
    expect(await screen.findByTestId("overlay-drawing")).toBeInTheDocument();
    media.presentFrame(10.5);
    expect(screen.queryByTestId("overlay-drawing")).not.toBeInTheDocument();
    expect(chip()).toHaveTextContent("Not analysed here");
    media.presentFrame(11);
    expect(screen.getByTestId("overlay-drawing")).toBeInTheDocument();
  });

  it("draws a lane observed on an earlier frame without saying no lane was observed", async () => {
    const response = fixture("overlay");
    // The server carries the 12.0 s lane to the 12.1 s frame, on which lane estimation was not scheduled.
    const index = response.overlay.frames.findIndex((f) => f.t === 12.1);
    const earlier = response.overlay.frames[index - 1]!;
    response.overlay.frames[index] = { ...response.overlay.frames[index]!, lane: { ...earlier.lane!, observed_t: earlier.t } };
    const { media } = await withOverlays({ overlay: response });
    media.presentFrame(12.12);
    const drawing = await screen.findByTestId("overlay-drawing");
    expect(drawing.querySelectorAll("polyline").length).toBeGreaterThan(0);
    expect(chip()).toBeNull();
  });

  it("tells an empty analysed frame from a frame the tracker skipped", async () => {
    const response = fixture("overlay");
    response.overlay.frames[0] = { ...response.overlay.frames[0]!, tracks: [] };
    response.overlay.frames[1] = { ...response.overlay.frames[1]!, tracks: null, lane: null };
    const { media } = await withOverlays({ overlay: response });
    media.presentFrame(9);
    await waitFor(() => expect(chip()).toHaveTextContent("No tracked objects"));
    media.presentFrame(9.1);
    expect(chip()).toHaveTextContent("Tracker did not run here · No lane observed");
  });

  it("requests a short span outside the review interval and nothing where nothing was analysed", async () => {
    const chunk = `${base}/packages/${approach.event_id}/overlay?start_s=20&end_s=40`;
    const { media, server } = await withOverlays({ overrides: { [`GET ${chunk}`]: jsonReply({ ...fixture("overlay"), overlay: { ...fixture("overlay").overlay, frames: [] } }) } });
    media.presentFrame(30);
    await waitFor(() => expect(server.calls(`GET ${chunk}`)).toHaveLength(1));
    expect(await screen.findByText("Not analysed here")).toBeInTheDocument();
    const before = server.requests.length;
    media.presentFrame(62.7); // past the last analysed frame (62.4 s) and its 0.25 s validity
    expect(chip()).toHaveTextContent("Not analysed here");
    expect(server.requests.filter((r) => r.path.includes("/overlay")).length).toBe(
      server.requests.slice(0, before).filter((r) => r.path.includes("/overlay")).length,
    );
  });

  it("maps container time to analysis time with the stream start, whatever the earliest seekable position", async () => {
    const shifted = { ...timeline, video: { ...timeline.video!, start_s: 1.4 } };
    // Chrome reports 0 as the earliest position of an MP4 whose first frame it presents at its start time.
    const { media } = await withOverlays({ timeline: shifted, media: { earliest: 0 } });
    expect(media.currentTime).toBeCloseTo(approach.review.start_s + 1.4, 9);
    media.presentFrame(15.9); // container time of analysis time 14.5
    const drawing = await screen.findByTestId("overlay-drawing");
    // Frame 14.5 is index 55 of the interval: box x1 = 100 + 145.
    expect(drawing.querySelector('[data-track="1"] rect')).toHaveAttribute("x", "245");
  });

  it("hides overlays whose recorded size does not match the displayed video", async () => {
    const { media } = await withOverlays({ media: { width: 1080, height: 1920 } });
    media.presentFrame(12);
    await waitFor(() => expect(chip()).toHaveTextContent("Overlay size does not match this video"));
    expect(screen.queryByTestId("overlay-drawing")).not.toBeInTheDocument();
  });

  it("says when overlay data is unavailable", async () => {
    const { media } = await withOverlays({
      overrides: { [overlayPath]: errorReply(409, "the analysis records are no longer in the perception cache") },
    });
    media.presentFrame(12);
    await waitFor(() => expect(chip()).toHaveTextContent("Overlays unavailable"));
  });
});

describe("evidence", () => {
  const summaryOf = () => screen.getByRole("region", { name: approach.title });
  const part = (name: string) => screen.findByRole("region", { name });
  const withPackage = (id: string, edit: (pk: EvidencePackage) => void) => {
    const pk = fixture("packages")[id]!;
    edit(pk);
    return { [`GET ${base}/packages/${id}`]: jsonReply(pk) };
  };

  it("asks for a choice in a compact state while nothing is selected", async () => {
    await open();
    expect(screen.getByRole("heading", { name: "Select an event to review" })).toBeInTheDocument();
    expect(screen.getByText("Choose an event from the list or timeline.")).toBeInTheDocument();
    expect(document.querySelector(".empty-state--compact")).not.toBeNull();
  });

  it("summarises a rapid approach in one sentence, keeping estimates out of it", async () => {
    const { user, media, server } = await open({ path: `/analyses/${latest.id}?event=${approach.event_id}` });
    media.loadMetadata();
    const why = await part("Why this event?");
    expect(why).toHaveTextContent(
      /^Why this event\?A car appeared to approach rapidly, growing larger in the footage\.$/,
    );
    expect(screen.queryByText(/1\.2 s|time to contact/)).not.toBeInTheDocument();
    const summary = summaryOf();
    expect(within(summary).getByText("Conditions held")).toBeInTheDocument();
    expect(within(summary).queryByText(/not a confirmed incident/)).not.toBeInTheDocument();
    expect(screen.getAllByText(/Not confirmed incidents/)).toHaveLength(1);
    expect(document.querySelector(".event-review__columns--single")).not.toBeNull();
    expect(screen.queryByRole("region", { name: "Uncertainty" })).not.toBeInTheDocument();
    await user.click(within(summary).getByRole("button", { name: "Play from 0:12.0" }));
    expect(within(summary).getByRole("button", { name: "Play from 0:12.0" })).toHaveTextContent("0:12.0 – 0:14.5");
    expect(media.currentTime).toBe(12);
    // The whole package was loaded; its technical detail is simply not displayed.
    expect(server.calls(`GET ${base}/packages/${approach.event_id}`)).toHaveLength(1);
    for (const gone of ["Supporting evidence", "Tracks and lanes", "Limits", "Rank", "Peak", "Review span", "Analyser"]) {
      expect(screen.queryByText(gone)).not.toBeInTheDocument();
    }
    expect(screen.queryByText(/apparent-approach|Not ranked|30% of frame height|#1/)).not.toBeInTheDocument();
  });

  it("places why and uncertainty in two columns that stack on narrow screens", () => {
    expect(evidenceCss).toMatch(/\.event-review__columns \{[^}]*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\);/);
    expect(evidenceCss).toMatch(/@media \(max-width: 640px\) \{\s*\.event-review__columns \{\s*grid-template-columns: minmax\(0, 1fr\);/);
  });

  it("says the vehicle ahead only when it was mostly in the camera's lane", async () => {
    await open({
      path: `/analyses/${latest.id}?event=${approach.event_id}`,
      overrides: withPackage(approach.event_id, (pk) => {
        pk.proposal.evidence.push({ basis: "observed", name: "ego_lane_fraction", t: 12, t_end: 14.5, track_id: 1, value: 0.8, unit: "fraction", note: "" });
      }),
    });
    expect(await part("Why this event?")).toHaveTextContent(/The car ahead appeared to approach rapidly/);
  });

  it("explains an uncertain lane relation and an identity jump as recorded", async () => {
    await open({
      path: `/analyses/${latest.id}?event=${approach.event_id}`,
      overrides: withPackage(approach.event_id, (pk) => {
        pk.proposal.state = "uncertain";
        pk.proposal.reasons = ["lane_not_established", "identity_jump"];
      }),
    });
    expect(await part("Uncertainty")).toHaveTextContent(
      "The tracked car's lane position could not be verified reliably. The tracked car's box jumped in size, so tracking may have switched to another object.",
    );
    const columns = document.querySelector(".event-review__columns")!;
    expect(columns).not.toHaveClass("event-review__columns--single");
    expect([...columns.children].map((c) => c.getAttribute("aria-label"))).toEqual(["Why this event?", "Uncertainty"]);
  });

  it("summarises a cut-in from its observed lane relations and keeps its uncertainty", async () => {
    await open({ path: `/analyses/${latest.id}?event=${cutIn.event_id}` });
    expect(await part("Why this event?")).toHaveTextContent("A vehicle was seen moving from the left lane into the camera's lane.");
    expect(await part("Uncertainty")).toHaveTextContent(
      "The moment the tracked vehicle entered the lane was not seen. The tracked vehicle's lane position was not observed.",
    );
    expect(within(screen.getByRole("region", { name: "Cut-in" })).getByText("Uncertain")).toBeInTheDocument();
  });

  it("summarises a camera impact without claiming a collision", async () => {
    const impact = { ...approach, type: "impact" as const, subtype: "camera_vehicle" as const, title: "Possible impact" };
    await open({
      path: `/analyses/${latest.id}?event=${approach.event_id}`,
      timeline: { ...timeline, entries: [impact, cutIn, contact] },
      overrides: withPackage(approach.event_id, (pk) => {
        pk.proposal.type = "impact";
        pk.proposal.subtype = "camera_vehicle";
        pk.proposal.evidence = [
          { basis: "observed", name: "close_track_lost", t: 14, t_end: null, track_id: 1, value: 0.6, unit: "fraction_of_height", note: "" },
          { basis: "estimated", name: "camera_jolt_z", t: 14.3, t_end: null, track_id: null, value: 9.1, unit: "z", note: "" },
        ];
        pk.tracks[0]!.ends_early = true;
        pk.notes = ["track_ends_early"];
      }),
    });
    const why = await part("Why this event?");
    expect(why).toHaveTextContent(
      /^Why this event\?A car came very close to the camera, then the footage shook abruptly\.$/,
    );
    expect(await part("Uncertainty")).toHaveTextContent(
      /^UncertaintyThe shake was judged from the footage alone, so a collision is not confirmed\. The tracked car was lost from view before the event ended\.$/,
    );
  });

  it("summarises contact between road users and does not call it contact", async () => {
    await open({
      path: `/analyses/${latest.id}?event=${contact.event_id}`,
      overrides: withPackage(contact.event_id, (pk) => {
        pk.proposal.evidence = [
          { basis: "observed", name: "boxes_meet", t: 44, t_end: 44.8, track_id: null, value: null, unit: null, note: "" },
          { basis: "estimated", name: "velocity_change_z", t: 44, t_end: 45.8, track_id: null, value: 7, unit: "z", note: "" },
        ];
      }),
    });
    expect(await part("Why this event?")).toHaveTextContent(
      /^Why this event\?Two road users appeared to meet, and one of them seemed to change motion abruptly\.$/,
    );
    expect(await part("Uncertainty")).toHaveTextContent(/^UncertaintyContact itself was not observed; they may only have looked close\./);
    // The experimental note and the not-validated reason say the same thing once.
    expect((await part("Uncertainty")).textContent!.match(/This check is experimental and not yet validated\./g)).toHaveLength(1);
  });

  it("falls back honestly when the evidence does not explain the proposal", async () => {
    await open({ path: `/analyses/${latest.id}?event=${contact.event_id}` });
    expect(await part("Why this event?")).toHaveTextContent(
      "Spectra did not record enough detail to explain this event. Please check the video yourself.",
    );
    // The fallback adds no explanation, but the analyser's own caveats remain.
    expect(await part("Uncertainty")).toHaveTextContent("Contact itself was not observed");
    expect(screen.queryByText(/collision|crash|fault|confirmed\b(?! incidents)/i)).not.toBeInTheDocument();
  });

  it("keeps missing lanes, unanalysed stretches, warnings and unjudged patterns as uncertainty", async () => {
    await open({
      path: `/analyses/${latest.id}?event=${cutIn.event_id}`,
      overrides: withPackage(cutIn.event_id, (pk) => {
        pk.lanes = { available: false, ego_lane: [], relations: [] };
        pk.gaps = [{ start_s: 27.1, end_s: 28.4 }];
        pk.warnings = [{ code: "decode_errors", message: "3 packets could not be decoded", t: 27.0 }];
        pk.unresolved = [{ reason: "motion_estimate_missing", t_start: 29, t_end: 30, track_ids: [2] }];
      }),
    });
    const limits = await part("Uncertainty");
    for (const text of [
      "Lane detection did not run in this analysis.",
      "Part of the footage around this event was not analysed (0:27.1 – 0:28.4).",
      "The video could not be read cleanly at 0:27.0: 3 packets could not be decoded.",
      "Spectra could not assess a nearby moment (0:29.0 – 0:30.0). Camera motion could not be measured when it mattered.",
    ]) {
      expect(limits).toHaveTextContent(text);
    }
  });

  it("links related proposals without merging them", async () => {
    const packages = fixture("packages");
    const pk = packages[approach.event_id]!;
    pk.related = [{ kind: "shares_track", event_id: contact.event_id }];
    const { user } = await open({
      path: `/analyses/${latest.id}?event=${approach.event_id}`,
      overrides: { [`GET ${base}/packages/${approach.event_id}`]: jsonReply(pk) },
    });
    const related = await screen.findByRole("region", { name: "Related events" });
    expect(within(related).getByText("Shares a track with possible contact between other road users")).toBeInTheDocument();
    await user.click(within(related).getByRole("button", { name: "Select" }));
    expect(proposal("Possible contact")).toHaveAttribute("aria-current", "true");
  });

  it("offers a retry when the package cannot be loaded", async () => {
    await open({
      path: `/analyses/${latest.id}?event=${approach.event_id}`,
      overrides: { [`GET ${base}/packages/${approach.event_id}`]: errorReply(409, "the stored evidence failed verification") },
    });
    expect(await screen.findByText("Evidence could not be loaded")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });
});

describe("workspace sections", () => {
  it("names the timeline as its own section and starts the event review under a rule", async () => {
    await open({ path: `/analyses/${latest.id}?event=${approach.event_id}` });
    const timelineSection = screen.getByRole("region", { name: "Event timeline" });
    expect(timelineSection).not.toContainElement(screen.getByRole("slider", { name: "Playback position" }));
    expect(timelineSection).not.toContainElement(screen.getByRole("toolbar", { name: "Playback" }));
    expect(timelineSection.querySelector('[data-testid="playhead"]')).toBeNull();
    // Spacing alone separates the timeline from the player; the rule stays only above the event review.
    expect(timelineCss).not.toMatch(/\.review-timeline\s*{[^}]*border-top/);
    // Equal space above and below the event header: the review's top padding matches the section's gap.
    expect(layoutCss).toMatch(/\.workspace__below\s*{[^}]*padding-top: var\(--space-6\);/);
    expect(evidenceCss).toMatch(/\.section:has\(> \.section__header \.event-review__header-actions\)\s*{\s*gap: var\(--space-6\);/);
    expect(layoutCss).toMatch(/@media \(max-width: 640px\) \{\s*\.workspace \.button--sm \{\s*height: var\(--control-h-md\);/);
    expect(layoutCss).toMatch(/\.workspace__below\s*{[^}]*border-top: var\(--border-width\) solid var\(--color-border\);/);
  });
});

describe("player focus", () => {
  it("focuses the player on a click without scrolling the page", async () => {
    await open({ path: `/analyses/${latest.id}?event=${approach.event_id}` });
    const stage = screen.getByRole("group", { name: "Video player" });
    const focus = vi.spyOn(stage, "focus");
    const pressed = fireEvent.mouseDown(screen.getByLabelText("Source video"));
    expect(pressed).toBe(false); // the browser's own focus-and-scroll is replaced
    expect(focus).toHaveBeenCalledWith({ preventScroll: true });
    expect(stage).toHaveFocus();
  });
});

describe("clips", () => {
  const clipPath = `POST ${base}/packages/${approach.event_id}/clip`;

  it("creates a clip only on request and links to it", async () => {
    vi.spyOn(HTMLMediaElement.prototype, "canPlayType").mockReturnValue("maybe");
    const { user, server } = await open({
      path: `/analyses/${latest.id}?event=${approach.event_id}`,
      overrides: { [clipPath]: jsonReply(fixture("clip")) },
    });
    expect(server.calls(clipPath)).toHaveLength(0);
    await user.click(await screen.findByRole("button", { name: "Create clip" }));
    // The clip appears in place; no toast repeats it.
    expect(await screen.findByRole("link", { name: "Open clip" })).toHaveAttribute("href", fixture("clip").media_url);
    expect(screen.queryByText("Clip ready")).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Download" })).toHaveAttribute("download", fixture("clip").clip.file);
    expect(server.calls(clipPath)).toHaveLength(1);
    expect(screen.queryByRole("button", { name: "Create clip" })).not.toBeInTheDocument();
    // The actions replace Create clip in the header; details live in Download's tooltip, not on the page.
    const header = screen.getByRole("heading", { level: 2, name: approach.title }).closest("section")!.querySelector(".section__header")!;
    expect(header).toContainElement(screen.getByRole("link", { name: "Open clip" }));
    const body = header.closest("section")!.querySelector(".section__body")!;
    expect(body).not.toHaveTextContent(/copied without re-encoding|Open clip|Download/);
    expect(screen.getByRole("link", { name: "Download" })).toHaveAccessibleDescription(/from the keyframe before the review span/);
    expect(document.querySelector(".clip-result")).toBeNull();
  });

  it("shows Creating… while the clip is made", async () => {
    let answer: (() => void) | null = null;
    const { user } = await open({
      path: `/analyses/${latest.id}?event=${approach.event_id}`,
      overrides: { [clipPath]: () => new Promise((resolve) => (answer = () => resolve(jsonReply(fixture("clip"))))) },
    });
    await user.click(await screen.findByRole("button", { name: "Create clip" }));
    const busy = screen.getByRole("button", { name: "Creating…" });
    expect(busy).toHaveAttribute("aria-busy", "true");
    expect(busy).toBeDisabled();
    await act(async () => answer!());
    expect(await screen.findByRole("link", { name: "Download" })).toBeInTheDocument();
  });

  it("does not carry a clip's links to another proposal", async () => {
    vi.spyOn(HTMLMediaElement.prototype, "canPlayType").mockReturnValue("maybe");
    const { user } = await open({
      path: `/analyses/${latest.id}?event=${approach.event_id}`,
      overrides: { [clipPath]: jsonReply(fixture("clip")) },
    });
    await user.click(await screen.findByRole("button", { name: "Create clip" }));
    await screen.findByRole("link", { name: "Open clip" });
    const list = screen.getByRole("list", { name: "Proposals in time order" });
    const other = within(list).getAllByRole("button").find((b) => !b.textContent!.includes(approach.title))!;
    await user.click(other);
    expect(await screen.findByRole("button", { name: "Create clip" })).toBeEnabled();
    expect(screen.queryByRole("link", { name: "Open clip" })).not.toBeInTheDocument();
  });

  it("sends one clip request for repeated clicks", async () => {
    const { server } = await open({
      path: `/analyses/${latest.id}?event=${approach.event_id}`,
      overrides: { [clipPath]: jsonReply(fixture("clip")) },
    });
    const button = await screen.findByRole("button", { name: "Create clip" });
    act(() => {
      button.click();
      button.click();
    });
    await waitFor(() => expect(screen.queryByRole("button", { name: "Create clip" })).not.toBeInTheDocument());
    expect(server.calls(clipPath)).toHaveLength(1);
  });

  it("offers a download instead when the browser reports it cannot play the clip", async () => {
    vi.spyOn(HTMLMediaElement.prototype, "canPlayType").mockReturnValue("");
    const { user } = await open({
      path: `/analyses/${latest.id}?event=${approach.event_id}`,
      overrides: { [clipPath]: jsonReply({ ...fixture("clip"), mime_type: "video/x-matroska" }) },
    });
    await user.click(await screen.findByRole("button", { name: "Create clip" }));
    expect(await screen.findByRole("link", { name: "Download" })).toHaveAccessibleDescription(
      /This browser may not play the clip\. Download it to open it elsewhere\./,
    );
    expect(screen.queryByRole("link", { name: "Open clip" })).not.toBeInTheDocument();
  });

  it("explains a source that cannot be clipped, inline", async () => {
    const { user } = await open({
      path: `/analyses/${latest.id}?event=${approach.event_id}`,
      overrides: { [clipPath]: errorReply(422, "a clip cannot be cut from this video: no keyframe before the span") },
    });
    await user.click(await screen.findByRole("button", { name: "Create clip" }));
    const alert = await screen.findByText("No clip");
    expect(alert.closest(".alert")).toHaveTextContent("A clip cannot be cut from this video: no keyframe before the span.");
  });

  it("leaves an unreachable server to the global banner and keeps the clip action available", async () => {
    const { user, server } = await open({
      path: `/analyses/${latest.id}?event=${approach.event_id}`,
      overrides: { [clipPath]: "network-error" },
    });
    server.on("GET /api/health", "network-error");
    await user.click(await screen.findByRole("button", { name: "Create clip" }));
    expect(await screen.findByText("Server unavailable")).toBeInTheDocument();
    expect(screen.queryByText("No clip")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create clip" })).toBeEnabled();
  });
});

describe("evidence states", () => {
  it("waits for evidence that is being prepared without requesting it", async () => {
    const pending = { ...latest, evidence: { state: "pending" as const, build_key: null, message: null } };
    const { installEventSource } = await import("../../test/eventSource");
    installEventSource();
    const server = await standardApi({ [`GET ${base}`]: jsonReply(pending) });
    renderApp(`/analyses/${latest.id}`);
    expect(await screen.findByText("Preparing evidence")).toBeInTheDocument();
    expect(server.calls(`GET ${base}/timeline`)).toHaveLength(0);
  });

  it("prepares missing evidence only when asked, then opens the review", async () => {
    fake = installFakeVideo();
    const failed = { ...latest, evidence: { state: "failed" as const, build_key: null, message: "evidence preparation was interrupted" } };
    let answer: Analysis = failed;
    const server = await standardApi({
      [`GET ${base}`]: () => jsonReply(answer),
      [`GET ${base}/timeline`]: () => {
        answer = latest;
        return jsonReply(timeline);
      },
      [`GET /api/videos/${latest.video_id}`]: jsonReply(video("ready")),
    });
    const { user } = renderApp(`/analyses/${latest.id}`);
    expect(await screen.findByText("Evidence not prepared")).toBeInTheDocument();
    expect(server.calls(`GET ${base}/timeline`)).toHaveLength(0);
    await user.click(screen.getByRole("button", { name: "Prepare evidence" }));
    expect(await screen.findByRole("list", { name: "Proposals in time order" })).toBeInTheDocument();
    expect(server.calls(`GET ${base}/timeline`)).toHaveLength(1);
  });

  it("shows why evidence could not be prepared", async () => {
    const failed = { ...latest, evidence: { state: "failed" as const, build_key: null, message: null } };
    await standardApi({
      [`GET ${base}`]: jsonReply(failed),
      [`GET ${base}/timeline`]: errorReply(409, "the analysis records are no longer in the perception cache; analyse the video again"),
    });
    const { user } = renderApp(`/analyses/${latest.id}`);
    await user.click(await screen.findByRole("button", { name: "Prepare evidence" }));
    const title = await screen.findByText("Evidence could not be prepared");
    expect(title.closest(".alert")).toHaveTextContent("no longer in the perception cache; analyse the video again.");
  });

  it("reviews an incomplete analysis and marks the unanalysed tail", async () => {
    const incomplete = { ...latest, state: "incomplete" as const };
    const partial = { ...timeline, state: "incomplete" as const, analysed_end_s: 50 };
    const { media } = await open({ timeline: partial, overrides: { [`GET ${base}`]: jsonReply(incomplete) } });
    media.loadMetadata();
    expect(screen.getByText("Decoding ended early; the end of the video was not analysed.")).toBeInTheDocument();
    expect([...document.querySelectorAll('[data-coverage="unanalysed"]')].map((r) => r.getAttribute("x"))).toEqual(["80.000%"]);
  });
});

describe("player controls", () => {
  const stage = () => screen.getByRole("group", { name: "Video player" });
  const shown = () => stage().getAttribute("data-controls");

  it("shows controls while paused and fades them after an idle period while playing", async () => {
    const { media } = await open();
    media.loadMetadata();
    expect(shown()).toBe("shown");
    vi.useFakeTimers();
    try {
      fireEvent.click(screen.getByRole("button", { name: "Play" }));
      expect(shown()).toBe("shown");
      act(() => vi.advanceTimersByTime(2600));
      expect(shown()).toBe("hidden");
      fireEvent.pointerMove(stage(), { pointerType: "mouse" });
      expect(shown()).toBe("shown");
      act(() => vi.advanceTimersByTime(2600));
      expect(shown()).toBe("hidden");
      fireEvent.keyDown(stage(), { key: "Shift" });
      expect(shown()).toBe("shown");
      act(() => vi.advanceTimersByTime(2600));
      // Keyboard focus inside the controls keeps them shown.
      act(() => screen.getByRole("button", { name: "Pause" }).focus());
      expect(shown()).toBe("shown");
      act(() => vi.advanceTimersByTime(10000));
      expect(shown()).toBe("shown");
      act(() => screen.getByRole("button", { name: "Pause" }).blur());
      expect(shown()).toBe("hidden");
      fireEvent.click(screen.getByRole("button", { name: "Pause" }));
      expect(shown()).toBe("shown");
    } finally {
      vi.useRealTimers();
    }
  });

  it("keeps controls shown while the pointer rests on them", async () => {
    const { media } = await open();
    media.loadMetadata();
    vi.useFakeTimers();
    try {
      fireEvent.click(screen.getByRole("button", { name: "Play" }));
      fireEvent.pointerEnter(document.querySelector(".review-player__chrome")!, { pointerType: "mouse" });
      act(() => vi.advanceTimersByTime(10000));
      expect(shown()).toBe("shown");
    } finally {
      vi.useRealTimers();
    }
  });

  it("reveals hidden controls on a tap without pausing, and toggles playback on a tap while they are shown", async () => {
    const { media } = await open();
    media.loadMetadata();
    vi.useFakeTimers();
    try {
      fireEvent.click(screen.getByRole("button", { name: "Play" }));
      act(() => vi.advanceTimersByTime(2600));
      const picture = screen.getByLabelText("Source video");
      fireEvent.pointerDown(picture, { pointerType: "touch" });
      fireEvent.click(picture);
      expect(shown()).toBe("shown");
      expect(media.pause).not.toHaveBeenCalled();
      fireEvent.pointerDown(picture, { pointerType: "touch" });
      fireEvent.click(picture);
      expect(media.pause).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not take seek-bar presses as a click on the picture", async () => {
    const { media } = await open();
    media.loadMetadata();
    vi.spyOn(slider(), "getBoundingClientRect").mockReturnValue({ left: 0, width: 1000 } as DOMRect);
    fireEvent.pointerDown(slider(), { clientX: 250, button: 0, pointerId: 1 });
    fireEvent.pointerUp(slider(), { pointerId: 1 });
    expect(media.currentTime).toBeCloseTo(15.625, 6);
    expect(media.play).not.toHaveBeenCalled();
  });

  it("opens menus upwards over the picture and keeps the overlay layers in a compact menu", () => {
    expect(playerCss).toMatch(/\.review-player__chrome \.popover__panel\s*{[^}]*bottom: calc\(100% \+ var\(--popover-gap\)\);/);
    expect(playerCss).toMatch(/\[data-controls="hidden"\] \.review-player__chrome[^{]*{[^}]*opacity: 0;[^}]*pointer-events: none;/);
    expect(playerCss).toMatch(/transition: opacity var\(--duration-base\)/); // zero under reduced motion
  });

  it("replaces the controls with the reason when the video cannot be decoded", async () => {
    const { media } = await open();
    media.loadMetadata();
    media.fail(4);
    expect(await screen.findByText("This browser cannot play this video's format.")).toBeInTheDocument();
    expect(screen.queryByRole("toolbar", { name: "Playback" })).not.toBeInTheDocument();
  });
});

describe("event timeline", () => {
  const section = () => screen.getByRole("region", { name: "Event timeline" });
  const key = () => within(section()).getByRole("list", { name: "Timeline key" });

  const position = () => within(section()).getByRole("slider", { name: "Timeline position" });

  it("follows playback in step with the player's progress bar", async () => {
    const { media } = await open();
    media.loadMetadata();
    media.presentFrame(31.25);
    expect(screen.getByTestId("timeline-playhead")).toHaveAttribute("x1", "50.000%");
    expect(screen.getByTestId("playhead")).toHaveAttribute("data-progress", "50.000%");
    expect(position()).toHaveAttribute("aria-valuenow", "31.3");
  });

  it("seeks where clicked or dragged, clamped to the video, showing the target time", async () => {
    const { media } = await open();
    media.loadMetadata();
    vi.spyOn(position(), "getBoundingClientRect").mockReturnValue({ left: 100, width: 1000 } as DOMRect);
    fireEvent.pointerDown(position(), { clientX: 600, button: 0, pointerId: 1, pointerType: "mouse" });
    expect(media.currentTime).toBeCloseTo(31.25, 6);
    expect(section().querySelector(".review-timeline__tip")).toHaveTextContent("0:31.3");
    fireEvent.pointerMove(position(), { clientX: 1500, pointerId: 1, pointerType: "mouse" });
    expect(media.currentTime).toBe(62.5);
    fireEvent.pointerUp(position(), { pointerId: 1, pointerType: "mouse" });
    fireEvent.pointerMove(position(), { clientX: 0, pointerId: 1, pointerType: "mouse" });
    expect(media.currentTime).toBe(62.5); // hovering after release does not seek
    expect(section().querySelector(".review-timeline__tip")).toHaveTextContent("0:00.0");
    fireEvent.pointerLeave(position());
    expect(section().querySelector(".review-timeline__tip")).toBeNull();
  });

  it("seeks from the keyboard and says when the position is not analysed", async () => {
    const { user, media } = await open();
    media.loadMetadata();
    position().focus();
    await user.keyboard("{End}");
    expect(media.currentTime).toBe(62.5);
    expect(position()).toHaveAttribute("aria-valuetext", "1:02.5 of 1:02.5, not analysed");
    await user.keyboard("{Home}{ArrowRight}{PageUp}");
    expect(media.currentTime).toBe(11);
  });

  it("selects an event without also seeking to the pressed point, and never changes its times", async () => {
    const { media } = await open();
    media.loadMetadata();
    vi.spyOn(position(), "getBoundingClientRect").mockReturnValue({ left: 0, width: 1000 } as DOMRect);
    const hit = document.querySelector<HTMLElement>(`[data-event="${contact.event_id}"]`)!;
    fireEvent.pointerDown(hit, { clientX: 999, button: 0, pointerId: 1 });
    fireEvent.click(hit);
    expect(media.currentTime).toBe(contact.review.start_s);
    expect(hit).toHaveAccessibleName(
      `${contact.title}, ${formatTimecode(contact.t_start)} to ${formatTimecode(contact.t_end)}, peak at ${formatTimecode(contact.t_peak)}`,
    );
  });

  it("keeps short events at a usable target size", () => {
    expect(timelineCss).toMatch(/width: max\(var\(--w\), var\(--control-h-sm\)\);/);
  });

  it("selects an event from its marker by keyboard and moves the video to it", async () => {
    const { user, media } = await open();
    media.loadMetadata();
    const marker = within(section()).getByRole("button", { name: new RegExp(`^${cutIn.title},`) });
    marker.focus();
    await user.keyboard("{Enter}");
    expect(proposal("Cut-in")).toHaveAttribute("aria-current", "true");
    expect(marker).toHaveAttribute("aria-current", "true");
    expect(media.currentTime).toBe(cutIn.review.start_s);
  });

  it("shows the selected event's review window and names only what is drawn", async () => {
    await open({ path: `/analyses/${latest.id}?event=${approach.event_id}` });
    const window = document.querySelector("[data-window]")!;
    expect(window).toHaveAttribute("x", `${((approach.review.start_s / 62.5) * 100).toFixed(3)}%`);
    expect(within(key()).getAllByRole("listitem").map((li) => li.textContent)).toEqual(
      expect.arrayContaining(["Analysed", "Review window", "Suggested event"]),
    );
  });

  it("leaves out legend items with nothing to show", async () => {
    await open({ timeline: { ...timeline, entries: [], segments: [], gaps: [], analysed_start_s: 0, analysed_end_s: 62.5 } });
    expect(within(key()).getAllByRole("listitem").map((li) => li.textContent)).toEqual(["Analysed"]);
  });
});
