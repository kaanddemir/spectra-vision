import { act, cleanup, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { Analysis, EventEdit, ReviewDecision, Timeline } from "../../api/types";
import { formatExactTime, parseTime } from "../../lib/format";
import { errorReply, jsonReply, standardApi, type Recorded } from "../../test/api";
import { fixture, video } from "../../test/fixtures";
import { renderApp } from "../../test/render";
import { installFakeVideo } from "../../test/video";
import dialogCss from "../../ui/Dialog.css?raw";
import editCss from "./EventEdit.css?raw";

const latest: Analysis = fixture("analyses")[0]!;
const base = `/api/analyses/${latest.id}`;
const timeline = fixture("timeline");
const [approach, cutIn] = timeline.entries as [Timeline["entries"][0], Timeline["entries"][0]];
const editsPath = `GET ${base}/edits`;
const putPath = (eventId: string) => `PUT ${base}/edits/${eventId}`;
const DURATION = timeline.video!.duration_s!;

let fake: ReturnType<typeof installFakeVideo> | null = null;
afterEach(() => {
  cleanup();
  fake?.restore();
  fake = null;
});

type Body = Pick<EventEdit, "t_start" | "t_end" | "description"> & { expected_revision: number };

/** Stores edits like the API: revisions, compare-and-set and no record for an unedited proposal. */
function editServer(initial: EventEdit[] = []) {
  const stored = new Map(initial.map((e) => [e.event_id, e]));
  const handlers = Object.fromEntries(
    timeline.entries.map((entry) => [
      putPath(entry.event_id),
      ({ body }: Recorded) => {
        const { expected_revision, ...next } = body as Body;
        const current = stored.get(entry.event_id) ?? { event_id: entry.event_id, t_start: null, t_end: null, description: null, revision: 0, updated_at: null };
        if (current.t_start === next.t_start && current.t_end === next.t_end && current.description === next.description) return jsonReply(current);
        if (expected_revision !== current.revision) {
          return errorReply(409, { message: "this event was edited elsewhere; the latest edit is shown", current });
        }
        const saved = { event_id: entry.event_id, ...next, revision: current.revision + 1, updated_at: "2026-10-20T12:00:00+00:00" };
        stored.set(entry.event_id, saved);
        return jsonReply(saved);
      },
    ]),
  );
  return { stored, handlers: { [editsPath]: () => jsonReply({ analysis_id: latest.id, edits: [...stored.values()] }), ...handlers } };
}

async function open(path = `/analyses/${latest.id}?event=${approach.event_id}`, overrides: Parameters<typeof standardApi>[0] = {}, decisions: ReviewDecision[] = []) {
  fake = installFakeVideo();
  const packages = fixture("packages");
  const server = await standardApi({
    [`GET ${base}`]: jsonReply(latest),
    [`GET ${base}/timeline`]: jsonReply(timeline),
    [`GET /api/videos/${latest.video_id}`]: jsonReply(video("ready")),
    ...Object.fromEntries(Object.entries(packages).map(([id, pk]) => [`GET ${base}/packages/${id}`, jsonReply(pk)])),
    [`GET ${base}/reviews`]: jsonReply({ analysis_id: latest.id, decisions }),
    ...overrides,
  });
  const app = renderApp(path);
  await screen.findByLabelText("Source video");
  return { server, ...app };
}

const panel = () => screen.getByRole("heading", { level: 2, name: approach.title }).closest("section")!;
const form = () => screen.getByRole("form", { name: "Edit event" });
const dialog = () => screen.getByRole("dialog", { name: "Edit event" });
const closed = () => expect(screen.queryByRole("dialog", { name: "Edit event" })).not.toBeInTheDocument();
// jsdom does not turn Escape into the dialog's cancel event, so send the event a browser would.
const escape = () => act(() => void dialog().dispatchEvent(new Event("cancel", { cancelable: true })));
const field = (label: string) => within(form()).getByLabelText(label) as HTMLInputElement;
const proposals = () => screen.getByRole("list", { name: "Proposals in time order" });

async function startEditing(user: ReturnType<typeof renderApp>["user"]) {
  await user.click(await within(panel()).findByRole("button", { name: "Edit" }));
  return form();
}

async function type(user: ReturnType<typeof renderApp>["user"], label: string, value: string) {
  await user.clear(field(label));
  if (value) await user.type(field(label), value);
}

describe("event editing", () => {
  it("keeps the read-only view clean and opens a compact modal on request", async () => {
    const { handlers } = editServer();
    const { user } = await open(undefined, handlers);
    expect(await within(panel()).findByRole("button", { name: "Edit" })).toBeInTheDocument();
    expect(screen.queryByRole("form", { name: "Edit event" })).not.toBeInTheDocument();
    expect(screen.queryByText(/Adjusted by you/)).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Your note" })).not.toBeInTheDocument();
    await startEditing(user);
    expect(field("Start").value).toBe(formatExactTime(approach.t_start));
    expect(field("End").value).toBe(formatExactTime(approach.t_end));
    expect(field("Your note").value).toBe("");
    // The detected interval and the wider playback context are named apart.
    // The detected interval appears once, beside Reset; no longer explanations.
    expect(form()).toHaveTextContent("Detected: 0:12.0 – 0:14.5");
    expect(dialog()).not.toHaveTextContent(/Playback context|Plain text, up to/);
    expect(field("Your note")).toHaveAttribute("placeholder", "Optional");
    expect(field("Your note")).toHaveAttribute("maxLength", "2000");
    // A labelled modal whose subtitle names the event and its detected interval; focus starts at Start.
    expect(dialog()).toHaveAttribute("open");
    expect(dialog()).toHaveAccessibleDescription("Rapid approach");
    expect(field("Start")).toHaveFocus();
    // The form lives only inside the modal dialog.
    expect(form().closest("dialog")).toBe(dialog());
  });

  it("saves adjusted times and a note, shows them as the reviewer's and keeps the detected times", async () => {
    const { handlers } = editServer();
    const { user, server } = await open(undefined, handlers);
    await startEditing(user);
    await type(user, "Start", "0:11.24");
    await type(user, "End", "15");
    await user.type(field("Your note"), "Truck <b>brakes</b>\nwet road");
    await user.click(within(dialog()).getByRole("button", { name: "Save" }));
    await waitFor(() => expect(screen.queryByRole("form", { name: "Edit event" })).not.toBeInTheDocument());
    expect(server.calls(putPath(approach.event_id)).map((r) => r.body)).toEqual([
      { t_start: 11.24, t_end: 15, description: "Truck <b>brakes</b>\nwet road", expected_revision: 0 },
    ]);
    expect(within(panel()).getByRole("button", { name: "Play from 0:11.2" })).toHaveTextContent("0:11.2 – 0:15.0");
    expect(panel()).toHaveTextContent("Adjusted by you · detected 0:12.0 – 0:14.5");
    const note = screen.getByRole("region", { name: "Your note" });
    expect(note.querySelector("p")!.textContent).toBe("Truck <b>brakes</b>\nwet road"); // plain text, not HTML
    expect(note.querySelector("b")).toBeNull();
    // Spectra's explanation stays separate.
    expect(screen.getByRole("region", { name: "Why this event?" })).not.toHaveTextContent("Truck");
    const item = within(proposals()).getByRole("button", { name: /Rapid approach/ });
    expect(item).toHaveTextContent("0:11.2 – 0:15.0 · Adjusted by you");
  });

  it("keeps every decimal of a time and never rounds", async () => {
    const { handlers } = editServer([{ event_id: approach.event_id, t_start: 12.040001, t_end: 14.5, description: null, revision: 1, updated_at: null }]);
    const { user, server } = await open(undefined, handlers);
    await startEditing(user);
    expect(field("Start").value).toBe("0:12.040001");
    await type(user, "End", "0:14.123");
    await user.click(within(dialog()).getByRole("button", { name: "Save" }));
    await waitFor(() => expect(server.calls(putPath(approach.event_id))).toHaveLength(1));
    expect(server.calls(putPath(approach.event_id))[0]!.body).toMatchObject({ t_start: 12.040001, t_end: 14.123 });
  });

  it.each([
    ["Start", "abc", "Enter a time such as 0:12.5 or 1:02.04."],
    ["Start", "0:75", "Enter a time such as 0:12.5 or 1:02.04."],
    ["Start", "0:12.1234567", "Enter a time such as 0:12.5 or 1:02.04."],
    ["Start", "-1", "Enter a time such as 0:12.5 or 1:02.04."],
    ["End", "0:12.0", "The end must be after the start."],
    ["End", "0:11", "The end must be after the start."],
    ["End", "5:00", `The video ends at ${formatExactTime(DURATION)}.`],
  ])("refuses %s %s with a clear message and sends nothing", async (label, value, message) => {
    const { handlers } = editServer();
    const { user, server } = await open(undefined, handlers);
    await startEditing(user);
    await type(user, label, value);
    await user.click(within(dialog()).getByRole("button", { name: "Save" }));
    expect(await within(form()).findByText(message)).toBeInTheDocument();
    expect(field(label)).toHaveAttribute("aria-invalid", "true");
    expect(server.calls(putPath(approach.event_id))).toHaveLength(0);
  });

  it("cancels without saving", async () => {
    const { handlers } = editServer();
    const { user, server } = await open(undefined, handlers);
    await startEditing(user);
    await type(user, "Start", "0:11");
    await user.click(within(dialog()).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("form", { name: "Edit event" })).not.toBeInTheDocument();
    expect(server.calls(putPath(approach.event_id))).toHaveLength(0);
    await startEditing(user);
    expect(field("Start").value).toBe(formatExactTime(approach.t_start));
  });

  it("restores the detected times and clears the note, keeping the review decision", async () => {
    const accepted: ReviewDecision = { event_id: approach.event_id, decision: "accepted", revision: 1, updated_at: null };
    const { handlers } = editServer([{ event_id: approach.event_id, t_start: 11, t_end: 15, description: "old note", revision: 3, updated_at: null }]);
    const { user, server } = await open(undefined, handlers, [accepted]);
    expect(await screen.findByRole("region", { name: "Your note" })).toHaveTextContent("old note");
    await startEditing(user);
    await user.click(within(form()).getByRole("button", { name: "Reset to detected times" }));
    expect(field("Your note").value).toBe("old note"); // Reset leaves the note alone
    await user.click(within(form()).getByRole("button", { name: "Clear note" }));
    expect(field("Start").value).toBe(formatExactTime(approach.t_start));
    await user.click(within(dialog()).getByRole("button", { name: "Save" }));
    await waitFor(() => expect(screen.queryByRole("form", { name: "Edit event" })).not.toBeInTheDocument());
    expect(server.calls(putPath(approach.event_id))[0]!.body).toEqual({ t_start: null, t_end: null, description: null, expected_revision: 3 });
    expect(screen.queryByText(/Adjusted by you/)).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Your note" })).not.toBeInTheDocument();
    expect(within(screen.getByRole("region", { name: "Your review" })).getByText("Accepted")).toBeInTheDocument();
    expect(server.requests.filter((r) => r.path.includes("/reviews/"))).toHaveLength(0);
  });

  it("does not decide when editing, and deciding keeps the edit", async () => {
    const { handlers } = editServer();
    const { user, server } = await open(undefined, {
      ...handlers,
      [`PUT ${base}/reviews/${approach.event_id}`]: jsonReply({ event_id: approach.event_id, decision: "rejected", revision: 1, updated_at: null }),
    });
    await startEditing(user);
    await user.type(field("Your note"), "a note");
    await user.click(within(dialog()).getByRole("button", { name: "Save" }));
    expect(await screen.findByRole("region", { name: "Your note" })).toBeInTheDocument();
    expect(within(screen.getByRole("region", { name: "Your review" })).getByText("Not reviewed")).toBeInTheDocument();
    await user.click(within(screen.getByRole("region", { name: "Your review" })).getByRole("button", { name: "Reject" }));
    await within(screen.getByRole("region", { name: "Your review" })).findByText("Rejected");
    expect(screen.getByRole("region", { name: "Your note" })).toHaveTextContent("a note");
    expect(server.calls(putPath(approach.event_id))).toHaveLength(1);
  });

  it("shows stored edits after a reload, per proposal", async () => {
    const edits = fixture("edits");
    const { user } = await open(`/analyses/${latest.id}?event=${cutIn.event_id}`, { [editsPath]: jsonReply(edits) });
    expect(screen.queryByRole("region", { name: "Your note" })).not.toBeInTheDocument();
    expect(within(proposals()).getByRole("button", { name: /Rapid approach/ })).toHaveTextContent("Adjusted by you");
    await user.click(within(proposals()).getByRole("button", { name: /Rapid approach/ }));
    expect(await screen.findByRole("region", { name: "Your note" })).toHaveTextContent(edits.edits[0]!.description!);
  });

  it("explains an edit changed elsewhere and loads the latest", async () => {
    const { handlers, stored } = editServer();
    const { user, server } = await open(undefined, handlers);
    await startEditing(user);
    stored.set(approach.event_id, { event_id: approach.event_id, t_start: null, t_end: null, description: "from another window", revision: 1, updated_at: null });
    await user.type(field("Your note"), "mine");
    await user.click(within(dialog()).getByRole("button", { name: "Save" }));
    expect(await within(form()).findByText("This event was edited elsewhere; the latest edit is shown.")).toBeInTheDocument();
    expect(field("Your note").value).toBe("mine"); // the typed text is kept
    await waitFor(() => expect(server.calls(editsPath).length).toBeGreaterThanOrEqual(2));
    expect(server.calls(putPath(approach.event_id))).toHaveLength(1);
    await user.click(within(dialog()).getByRole("button", { name: "Cancel" }));
    expect(await screen.findByRole("region", { name: "Your note" })).toHaveTextContent("from another window");
  });

  it("shows the server's refusal and keeps the form", async () => {
    const { user } = await open(undefined, {
      [editsPath]: jsonReply({ analysis_id: latest.id, edits: [] }),
      [putPath(approach.event_id)]: errorReply(422, { message: "the end time cannot be after the end of the video (62.5 s)" }),
    });
    await startEditing(user);
    await type(user, "End", "0:20");
    await user.click(within(dialog()).getByRole("button", { name: "Save" }));
    expect(await within(form()).findByText("The end time cannot be after the end of the video (62.5 s).")).toBeInTheDocument();
  });

  it("sends one request while saving and says when it could not reach the server", async () => {
    let answer: (() => void) | null = null;
    const { user, server } = await open(undefined, {
      [editsPath]: jsonReply({ analysis_id: latest.id, edits: [] }),
      [putPath(approach.event_id)]: () => new Promise((resolve) => (answer = () => resolve("network-error"))),
    });
    server.on("GET /api/health", "network-error");
    await startEditing(user);
    await user.type(field("Your note"), "x");
    const save = within(dialog()).getByRole("button", { name: "Save" });
    await user.click(save);
    await user.click(save);
    expect(save).toHaveAttribute("aria-busy", "true");
    await act(async () => answer!());
    expect(await within(form()).findByText("Not saved. Try again when Spectra responds.")).toBeInTheDocument();
    expect(server.calls(putPath(approach.event_id))).toHaveLength(1);
  });

  it("offers a retry when edits cannot be loaded and hides Edit until they are", async () => {
    let fail = true;
    const { user } = await open(undefined, {
      [editsPath]: () => (fail ? errorReply(500, "database is locked") : jsonReply({ analysis_id: latest.id, edits: [] })),
    });
    expect(await screen.findByText("Your edits could not be loaded")).toBeInTheDocument();
    expect(within(panel()).queryByRole("button", { name: "Edit" })).not.toBeInTheDocument();
    fail = false;
    await user.click(within(panel()).getAllByRole("button", { name: "Try again" })[0]!);
    expect(await within(panel()).findByRole("button", { name: "Edit" })).toBeInTheDocument();
  });

  it("leaves the timeline's proposal markers at the detected times", async () => {
    const svg = () => document.querySelector(".review-timeline")!.innerHTML.replace(/_r_[0-9a-z]+_/g, "id");
    await open(undefined, { [editsPath]: jsonReply({ analysis_id: latest.id, edits: [] }) });
    const plain = svg();
    cleanup();
    fake?.restore();
    await open(undefined, { [editsPath]: jsonReply(fixture("edits")) });
    await screen.findByText(/Adjusted by you · detected/);
    expect(svg()).toBe(plain);
  });

  it("closes on Escape or the backdrop when nothing changed and returns focus to Edit", async () => {
    const { handlers } = editServer();
    const { user, server } = await open(undefined, handlers);
    const edit = await within(panel()).findByRole("button", { name: "Edit" });
    expect(edit).toHaveAttribute("aria-haspopup", "dialog");
    await startEditing(user);
    await escape();
    await waitFor(closed);
    expect(edit).toHaveFocus();
    await startEditing(user);
    await user.click(dialog()); // the dialog element itself is the backdrop area
    await waitFor(closed);
    expect(edit).toHaveFocus();
    expect(server.calls(putPath(approach.event_id))).toHaveLength(0);
  });

  it("keeps unsaved input on Escape and backdrop clicks; Cancel and Close discard it", async () => {
    const { handlers } = editServer();
    const { user, server } = await open(undefined, handlers);
    await startEditing(user);
    await type(user, "Start", "0:11");
    await escape();
    await user.click(dialog());
    expect(dialog()).toHaveAttribute("open");
    expect(field("Start").value).toBe("0:11");
    // Clicking inside the form is not a backdrop click either.
    await user.click(field("End"));
    expect(dialog()).toHaveAttribute("open");
    await user.click(within(dialog()).getByRole("button", { name: "Close" }));
    closed();
    await startEditing(user);
    expect(field("Start").value).toBe(formatExactTime(approach.t_start));
    expect(server.calls(putPath(approach.event_id))).toHaveLength(0);
  });

  it("cannot be closed while saving, then closes and shows the saved times", async () => {
    let answer: (() => void) | null = null;
    let stored = false;
    const saved = { event_id: approach.event_id, t_start: 11, t_end: 14.5, description: null, revision: 1, updated_at: null };
    const { user, server } = await open(undefined, {
      [editsPath]: () => jsonReply({ analysis_id: latest.id, edits: stored ? [saved] : [] }),
      [putPath(approach.event_id)]: () => new Promise((resolve) => (answer = () => resolve(jsonReply(saved)))),
    });
    await startEditing(user);
    await type(user, "Start", "0:11");
    await user.click(within(dialog()).getByRole("button", { name: "Save" }));
    expect(within(dialog()).getByRole("button", { name: "Close" })).toBeDisabled();
    expect(within(dialog()).getByRole("button", { name: "Cancel" })).toBeDisabled();
    await escape();
    await user.click(dialog());
    expect(dialog()).toHaveAttribute("open");
    stored = true;
    await act(async () => answer!());
    await waitFor(closed);
    expect(within(panel()).getByRole("button", { name: "Play from 0:11.0" })).toHaveTextContent("0:11.0 – 0:14.5");
    expect(panel()).toHaveTextContent("Adjusted by you · detected 0:12.0 – 0:14.5");
    expect(server.calls(putPath(approach.event_id))).toHaveLength(1);
  });

  it("fits narrow screens: the dialog keeps margins and scrolls", () => {
    expect(dialogCss).toMatch(/\.dialog\s*{[^}]*width: min\(var\(--dialog-w\), calc\(100vw - 2 \* var\(--space-4\)\)\);/);
    expect(dialogCss).toMatch(/\.dialog\s*{[^}]*max-height: calc\(100dvh - 2 \* var\(--space-4\)\);[^}]*overflow-y: auto;/);
  });

  it("stacks the time fields on narrow screens", () => {
    expect(editCss).toMatch(/@media \(max-width: 640px\)\s*{\s*\.event-edit__times\s*{\s*grid-template-columns: minmax\(0, 1fr\);/);
    expect(editCss).toMatch(/\.event-note__text\s*{[^}]*white-space: pre-wrap;/);
  });
});

describe("exact times", () => {
  it.each([
    [12, "0:12.0"],
    [12.04, "0:12.04"],
    [0.040000001, "0:00.040000001"],
    [62.5, "1:02.5"],
    [3723.25, "1:02:03.25"],
  ])("formats %s as %s and reads it back unchanged", (value, text) => {
    expect(formatExactTime(value)).toBe(text);
    if (text.split(".")[1]!.length <= 6) expect(parseTime(text)).toBe(value);
  });

  it.each([
    ["12.5", 12.5],
    ["1:02.34", 62.34],
    [" 0:00 ", 0],
    ["1:02:03.5", 3723.5],
    ["0:12.000001", 12.000001],
  ])("reads %s", (text, value) => expect(parseTime(text)).toBe(value));

  it.each(["", "1:2:3:4", "0:60", "1:60:00", "12,5", "1e3", "0:12.1234567", "-1", "abc"])("refuses %s", (text) =>
    expect(parseTime(text)).toBeNull(),
  );
});
