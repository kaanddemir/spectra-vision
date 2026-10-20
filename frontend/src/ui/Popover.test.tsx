import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { Popover } from "./Popover";

function rect(left: number, top: number, width: number, height: number): DOMRect {
  return { left, top, width, height, right: left + width, bottom: top + height } as DOMRect;
}

/** Supplied geometry tests positioning logic only; jsdom does not render responsive layouts. */
function setup(width: number, height: number, left: number, top: number, align: "left" | "right" = "right") {
  vi.spyOn(window, "innerWidth", "get").mockReturnValue(width);
  vi.spyOn(window, "innerHeight", "get").mockReturnValue(height);
  const boxes = { anchor: rect(left, top, 36, 36) };
  vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(200);
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
    return this.classList.contains("popover__panel")
      ? rect(align === "left" ? boxes.anchor.left : boxes.anchor.right - 240, 0, 240, 200)
      : boxes.anchor;
  });
  const { container } = render(
    <Popover trigger={(props) => <button type="button" {...props}>Open menu</button>}>
      <button type="button" data-autofocus>Choose</button>
    </Popover>,
  );
  const panel = container.querySelector<HTMLElement>(".popover__panel")!;
  // jsdom does not resolve inherited custom properties; supply the token values for this geometry.
  panel.style.setProperty("--popover-gap", "8px");
  panel.style.setProperty("--popover-margin", "16px");
  return { panel, boxes };
}

describe("popover viewport positioning", () => {
  it.each([375, 430, 768, 1024, 1280, 1440, 1920])("keeps panels inside both edges at width %i", (width) => {
    const { panel, boxes } = setup(width, 800, 16, 100);
    fireEvent.click(screen.getByRole("button", { name: "Open menu" }));
    expect(panel.style.getPropertyValue("--popover-shift")).toBe("204px");
    expect(panel).not.toHaveClass("popover__panel--up");

    // Reposition an open menu when its anchor moves beyond the right edge.
    boxes.anchor = rect(width + 100, 100, 36, 36);
    fireEvent(window, new Event("resize"));
    const shiftedLeft = boxes.anchor.right - 240 + parseFloat(panel.style.getPropertyValue("--popover-shift"));
    expect(shiftedLeft + 240).toBe(width - 16);
  });

  it("keeps the left-aligned playback speed menu inside the right edge", () => {
    const { panel } = setup(375, 800, 323, 100, "left");
    fireEvent.click(screen.getByRole("button", { name: "Open menu" }));
    expect(panel.style.getPropertyValue("--popover-shift")).toBe("-204px");
  });

  it("flips and bounds its scrollable height when scrolling and resizing a short viewport", () => {
    const { panel, boxes } = setup(375, 320, 16, 260, "left");
    fireEvent.click(screen.getByRole("button", { name: "Open menu" }));
    expect(panel).toHaveClass("popover__panel--up");
    expect(panel.style.getPropertyValue("--popover-available-height")).toBe("236px");

    boxes.anchor = rect(16, 130, 36, 36);
    fireEvent.scroll(document);
    expect(panel).not.toHaveClass("popover__panel--up");
    expect(panel.style.getPropertyValue("--popover-available-height")).toBe("130px");

    vi.spyOn(window, "innerHeight", "get").mockReturnValue(200);
    fireEvent(window, new Event("resize"));
    expect(panel).toHaveClass("popover__panel--up");
    expect(panel.style.getPropertyValue("--popover-available-height")).toBe("106px");
  });

  it("preserves autofocus, Escape focus return and outside-pointer dismissal", async () => {
    const user = userEvent.setup();
    const { panel } = setup(375, 800, 16, 100);
    const trigger = screen.getByRole("button", { name: "Open menu" });
    await user.click(trigger);
    expect(screen.getByRole("button", { name: "Choose" })).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(panel).not.toBeVisible();
    expect(trigger).toHaveFocus();
    await user.click(trigger);
    fireEvent.pointerDown(document.body);
    expect(panel).not.toBeVisible();
  });
});
