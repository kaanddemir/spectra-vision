import { screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import indexHtml from "../../index.html?raw";
import { jsonReply, standardApi } from "../test/api";
import { fixture } from "../test/fixtures";
import { renderApp } from "../test/render";

afterEach(() => vi.restoreAllMocks());

describe("application shell", () => {
  it("opens the video library from the root address", async () => {
    await standardApi();
    const { router } = renderApp("/");
    expect(await screen.findByRole("heading", { level: 1, name: "Videos" })).toBeInTheDocument();
    expect(router.state.location.pathname).toBe("/videos");
    expect(document.title).toBe("Spectra");
  });

  it("marks the current section and navigates between sections", async () => {
    await standardApi();
    const { user } = renderApp("/videos");
    const nav = screen.getByRole("navigation", { name: "Main" });
    expect(within(nav).getByRole("link", { name: "Videos" })).toHaveAttribute("aria-current", "page");
    await user.click(within(nav).getByRole("link", { name: "Library" }));
    expect(await screen.findByRole("heading", { level: 1, name: "Library" })).toBeInTheDocument();
    expect(within(nav).getByRole("link", { name: "Library" })).toHaveAttribute("aria-current", "page");
    await user.click(within(nav).getByRole("link", { name: "Events" }));
    expect(await screen.findByRole("heading", { level: 1, name: "Events" })).toBeInTheDocument();
  });

  it("offers a skip link to the main content", async () => {
    await standardApi();
    renderApp("/videos");
    expect(screen.getByRole("link", { name: "Skip to content" })).toHaveAttribute("href", "#main");
    expect(screen.getByRole("main")).toHaveAttribute("id", "main");
  });

  it("explains unknown addresses", async () => {
    await standardApi();
    renderApp("/nowhere");
    expect(await screen.findByRole("heading", { name: "Page not found" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Go to videos" })).toHaveAttribute("href", "/videos");
  });

  it("is laid out for any viewport width", () => {
    expect(indexHtml).toContain('<meta name="viewport" content="width=device-width, initial-scale=1.0" />');
    expect(indexHtml).toContain('<script src="/theme-init.js"></script>');
    expect(indexHtml.indexOf("theme-init.js")).toBeLessThan(indexHtml.indexOf("/src/main.tsx"));
  });
});

describe("server status", () => {
  it("shows analysis readiness and a short health summary on request", async () => {
    await standardApi();
    const { user } = renderApp("/videos");
    const button = await screen.findByRole("button", { name: /Analysis available/ });
    expect(button).toHaveTextContent("Ready");
    expect(button).toHaveAttribute("aria-expanded", "false");
    await user.click(button);
    expect(button).toHaveAttribute("aria-expanded", "true");
    const panel = document.getElementById(button.getAttribute("aria-controls")!)!;
    const rows = within(panel).getAllByRole("term").map((term) => [term.textContent, term.nextElementSibling?.textContent]);
    expect(rows).toEqual([["Version", "2.0.0"], ["Models", "Ready"], ["Memory", "Normal"]]);
    // Technical details and the readiness label are not repeated in the panel.
    for (const hidden of [/Schema/, /verified/, /Library folders/, /level/, /Analysis available/]) {
      expect(within(panel).queryByText(hidden)).not.toBeInTheDocument();
    }
    await user.keyboard("{Escape}");
    expect(button).toHaveAttribute("aria-expanded", "false");
  });

  it("never reports Ready while analysis is blocked, and explains why", async () => {
    await standardApi({ "GET /api/health": jsonReply(fixture("healthDegraded")) });
    const { user } = renderApp("/videos");
    const button = await screen.findByRole("button", { name: /Analysis unavailable/ });
    expect(button).toHaveTextContent("Unavailable");
    expect(button).not.toHaveTextContent("Ready");
    await user.click(button);
    expect(screen.getByText("Memory pressure is critical.")).toBeVisible();
    expect(screen.getByText("Critical")).toBeVisible();
    expect(screen.queryByText(/level 4/)).not.toBeInTheDocument();
  });

  it.each([
    [{ pressure_level: 2, state: "warning" }, true, "Ready", "Warning"],
    [{ pressure_level: null, state: "unavailable" }, true, "Ready", "Unknown"],
  ])("follows the server's readiness when memory is %o", async (memory, available, label, memoryLabel) => {
    const health = { ...fixture("health"), memory, analysis: { available, reasons: [] } };
    await standardApi({ "GET /api/health": jsonReply(health) });
    const { user } = renderApp("/videos");
    const button = await screen.findByRole("button", { name: /Analysis available/ });
    expect(button).toHaveTextContent(label);
    await user.click(button);
    expect(screen.getByText(memoryLabel)).toBeVisible();
  });

  it("shows unavailable models without a count", async () => {
    const base = fixture("health");
    const health = { ...base, models: { ...base.models, available: false, message: "Missing weights" },
      analysis: { available: false, reasons: ["Models are unavailable."] } };
    await standardApi({ "GET /api/health": jsonReply(health) });
    const { user } = renderApp("/videos");
    await user.click(await screen.findByRole("button", { name: /Analysis unavailable/ }));
    const models = screen.getByText("Models").nextElementSibling;
    expect(models).toHaveTextContent("Unavailable");
    expect(screen.queryByText("Missing weights")).not.toBeInTheDocument();
  });

  it("reports an unreachable server once, in plain words, and recovers on retry", async () => {
    const server = await standardApi({ "GET /api/health": "network-error", "GET /api/videos": "network-error" });
    const { user } = renderApp("/videos");
    expect(await screen.findByText("Server unavailable")).toBeInTheDocument();
    expect(screen.getByText("Check that Spectra is running.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Server offline/ })).toBeInTheDocument();
    await waitFor(() => expect(server.calls("GET /api/videos").length).toBeGreaterThan(0));
    expect(screen.queryByText("Videos could not be loaded")).not.toBeInTheDocument();
    expect(screen.queryByText(/not responding/)).not.toBeInTheDocument();
    expect(screen.queryByText(/python|--library/)).not.toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /Retry|Try again/ })).toHaveLength(1);
    server.on("GET /api/health", jsonReply(fixture("health")));
    server.on("GET /api/videos", jsonReply([]));
    await user.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(screen.queryByText("Server unavailable")).not.toBeInTheDocument());
    expect(await screen.findByText("No videos registered yet")).toBeInTheDocument();
  });
});

describe("one report per problem", () => {
  it.each([
    ["/events", "GET /api/analyses", "Analyses could not be loaded"],
    ["/library", "GET /api/library", "Library folders could not be loaded"],
  ])("%s shows only the global banner when the server is unreachable", async (path, route, pageError) => {
    const server = await standardApi({ "GET /api/health": "network-error", [route]: "network-error" });
    renderApp(path);
    expect(await screen.findByText("Server unavailable")).toBeInTheDocument();
    await waitFor(() => expect(server.calls(route).length).toBeGreaterThan(0));
    expect(screen.queryByText(pageError)).not.toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /Retry|Try again/ })).toHaveLength(1);
  });

  it("checks health at once when a request finds the server gone, and reloads failed data when it returns", async () => {
    const server = await standardApi({ "GET /api/analyses": jsonReply([]) });
    const { client } = renderApp("/events");
    expect(await screen.findByRole("heading", { name: "No analysed videos yet" })).toBeInTheDocument();
    const before = server.calls("GET /api/health").length;
    server.on("GET /api/health", "network-error");
    server.on("GET /api/analyses", "network-error");
    await client.refetchQueries({ queryKey: ["analyses"] });
    expect(await screen.findByText("Server unavailable")).toBeInTheDocument();
    expect(server.calls("GET /api/health").length).toBeGreaterThan(before);
    expect(screen.getByText(/Data shown may be out of date/)).toBeInTheDocument();
    expect(screen.queryByText(/^Last updated/)).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "No analysed videos yet" })).toBeInTheDocument();

    server.on("GET /api/health", jsonReply(fixture("health")));
    server.on("GET /api/analyses", jsonReply([]));
    await client.refetchQueries({ queryKey: ["health"] });
    await waitFor(() => expect(screen.queryByText("Server unavailable")).not.toBeInTheDocument());
    await waitFor(() => expect(client.getQueryState(["analyses"])?.status).toBe("success"));
  });

  it("still reports a page-specific error while the server is online", async () => {
    await standardApi({ "GET /api/analyses": { status: 500, body: { detail: "the analysis index is unreadable" } } });
    renderApp("/events");
    expect(await screen.findByText("Analyses could not be loaded")).toBeInTheDocument();
    expect(screen.getByText("The analysis index is unreadable.")).toBeInTheDocument();
    expect(screen.queryByText("Server unavailable")).not.toBeInTheDocument();
  });
});

describe("page headings", () => {
  it.each([
    ["/videos", "Videos", "View and manage your videos and analyses."],
    ["/library", "Library", "Browse folders and add videos to Spectra."],
    ["/events", "Events", "Browse event proposals across your videos."],
  ])("%s has a short subtitle", async (path, title, subtitle) => {
    await standardApi();
    renderApp(path);
    expect(await screen.findByRole("heading", { level: 1, name: title })).toBeInTheDocument();
    expect(screen.getByText(subtitle)).toHaveClass("page-header__description");
  });

  it.each([
    ["/videos", "No videos registered yet", { "GET /api/videos": jsonReply([]) }],
    ["/events", "No analysed videos yet", {}],
    ["/library", "No library folders yet", { "GET /api/library": jsonReply([]) }],
  ])("%s centres its no-data state in the page content area", async (path, title, overrides) => {
    await standardApi(overrides);
    renderApp(path);
    const state = (await screen.findByRole("heading", { name: title })).closest(".empty-state")!;
    expect(state).toHaveClass("empty-state--page");
    expect(state.parentElement).toHaveClass("app-main");
  });
});
