import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import tooltipCss from "./Tooltip.css?raw";
import { IconButton } from "./Button";
import { icons } from "./icons";
import { Tooltip } from "./Tooltip";

function rect(left: number, top: number, width: number, height: number): DOMRect {
  return { left, top, width, height, right: left + width, bottom: top + height, x: left, y: top, toJSON: () => ({}) } as DOMRect;
}

/** jsdom has no layout: give the trigger and the tooltip boxes, and the viewport a size. */
function layout(anchor: DOMRect, tip: DOMRect) {
  vi.spyOn(document.documentElement, "clientWidth", "get").mockReturnValue(1000);
  vi.spyOn(document.documentElement, "clientHeight", "get").mockReturnValue(800);
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
    return this.getAttribute("role") === "tooltip" ? tip : anchor;
  });
}

afterEach(() => vi.restoreAllMocks());

describe("tooltip", () => {
  it("opens on hover and keyboard focus, describes its trigger and closes on Escape", async () => {
    const user = userEvent.setup();
    render(<IconButton icon={icons.removeFolder} label="Remove folder" />);
    const button = screen.getByRole("button", { name: "Remove folder" });
    const tip = screen.getByRole("tooltip", { hidden: true });
    expect(tip).not.toBeVisible();
    await user.hover(button);
    expect(tip).toBeVisible();
    expect(button).toHaveAccessibleDescription("Remove folder");
    await user.unhover(button);
    expect(tip).not.toBeVisible();
    await user.tab();
    expect(button).toHaveFocus();
    expect(tip).toBeVisible();
    await user.keyboard("{Escape}");
    expect(tip).not.toBeVisible();
  });

  it("is placed against the viewport and kept inside it at the right edge", () => {
    layout(rect(960, 300, 28, 28), rect(0, 0, 200, 30));
    render(<IconButton icon={icons.removeFolder} label="Remove folder" />);
    fireEvent.mouseEnter(screen.getByRole("button").parentElement!);
    const tip = screen.getByRole("tooltip");
    expect(tip).toHaveAttribute("data-positioned", "true");
    // jsdom does not resolve custom properties, so gap and margin are 0 here; tooltipPosition.test covers them.
    expect(tip.style.getPropertyValue("--tooltip-x")).toBe("800px");
    expect(tip.style.getPropertyValue("--tooltip-y")).toBe("328px");
    expect(tip).toHaveClass("tooltip--bottom");
  });

  it("flips above a trigger at the bottom of the viewport", () => {
    layout(rect(400, 780, 28, 28), rect(0, 0, 200, 30));
    render(
      <Tooltip content="Remove folder">
        <button type="button">Remove</button>
      </Tooltip>,
    );
    fireEvent.focus(screen.getByRole("button"));
    expect(screen.getByRole("tooltip")).toHaveClass("tooltip--top");
  });

  it("wraps long labels and paths within a bounded width and is never clipped by scroll containers", () => {
    expect(tooltipCss).toMatch(/position: fixed;/);
    expect(tooltipCss).toMatch(/max-width: min\(var\(--tooltip-max-w\), calc\(100vw - 2 \* var\(--tooltip-margin\)\)\);/);
    expect(tooltipCss).toMatch(/overflow-wrap: anywhere;/);
    const path = "/Users/reviewer/Movies/Dashcam/2026/October/a-very-long-folder-name-without-spaces";
    render(
      <Tooltip content={path}>
        <button type="button">Path</button>
      </Tooltip>,
    );
    fireEvent.focus(screen.getByRole("button"));
    expect(screen.getByRole("tooltip")).toHaveTextContent(path);
  });
});
