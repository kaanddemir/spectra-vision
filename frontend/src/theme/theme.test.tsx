import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";

import initScript from "../../public/theme-init.js?raw";
import { fakeColorScheme } from "../test/media";
import { ThemeToggle } from "../ui/ThemeToggle";
import { STORAGE_KEY, resolveTheme } from "./theme";
import { ThemeProvider } from "./ThemeProvider";

const theme = () => document.documentElement.getAttribute("data-theme");
const originalMatchMedia = window.matchMedia;

afterEach(() => {
  window.matchMedia = originalMatchMedia;
});

/** Renders the toggle and opens its menu. */
function renderToggle() {
  const result = render(
    <ThemeProvider>
      <ThemeToggle />
    </ThemeProvider>,
  );
  fireEvent.click(screen.getByRole("button", { name: /Colour theme/ }));
  return result;
}

describe("theme resolution", () => {
  it("follows the system unless a manual choice exists", () => {
    expect(resolveTheme("system", true)).toBe("dark");
    expect(resolveTheme("system", false)).toBe("light");
    expect(resolveTheme("light", true)).toBe("light");
    expect(resolveTheme("dark", false)).toBe("dark");
  });
});

describe("pre-paint script", () => {
  const run = () => new Function(initScript)();

  it("applies the system theme without a saved choice", () => {
    fakeColorScheme(true);
    run();
    expect(theme()).toBe("dark");
  });

  it("applies a saved manual choice over the system theme", () => {
    fakeColorScheme(true);
    window.localStorage.setItem(STORAGE_KEY, "light");
    run();
    expect(theme()).toBe("light");
  });

  it("ignores an unknown saved value", () => {
    fakeColorScheme(false);
    window.localStorage.setItem(STORAGE_KEY, "purple");
    run();
    expect(theme()).toBe("light");
  });
});

describe("theme toggle", () => {
  it("starts from the system theme with 'System' selected", () => {
    fakeColorScheme(true);
    renderToggle();
    expect(theme()).toBe("dark");
    expect(screen.getByRole("radio", { name: "System" })).toBeChecked();
    expect(screen.getByRole("group", { name: "Colour theme" })).toBeInTheDocument();
  });

  it("shows the current choice on one button and closes after a pointer selection", async () => {
    fakeColorScheme(false);
    const user = userEvent.setup();
    renderToggle();
    const button = screen.getByRole("button", { name: "Colour theme: System" });
    expect(button).toHaveAttribute("aria-expanded", "true");
    await user.click(screen.getByRole("radio", { name: "Dark" }));
    expect(button).toHaveAttribute("aria-expanded", "false");
    expect(button).toHaveAccessibleName("Colour theme: Dark");
    await user.click(button);
    await user.keyboard("{Escape}");
    expect(button).toHaveAttribute("aria-expanded", "false");
    expect(button).toHaveFocus();
  });

  it("switches, persists and restores a manual choice", async () => {
    fakeColorScheme(false);
    const user = userEvent.setup();
    const { unmount } = renderToggle();
    await user.click(screen.getByRole("radio", { name: "Dark" }));
    expect(theme()).toBe("dark");
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe("dark");
    unmount();
    document.documentElement.removeAttribute("data-theme");
    renderToggle();
    expect(theme()).toBe("dark");
    expect(screen.getByRole("radio", { name: "Dark" })).toBeChecked();
  });

  it("forgets the manual choice when returning to the system theme", async () => {
    fakeColorScheme(true);
    window.localStorage.setItem(STORAGE_KEY, "light");
    const user = userEvent.setup();
    renderToggle();
    expect(theme()).toBe("light");
    await user.click(screen.getByRole("radio", { name: "System" }));
    expect(theme()).toBe("dark");
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("tracks system changes only while following the system", async () => {
    const scheme = fakeColorScheme(false);
    const user = userEvent.setup();
    renderToggle();
    act(() => scheme.setDark(true));
    expect(theme()).toBe("dark");
    await user.click(screen.getByRole("radio", { name: "Light" }));
    act(() => scheme.setDark(false));
    act(() => scheme.setDark(true));
    expect(theme()).toBe("light");
  });

  it("is operable with the arrow keys", async () => {
    fakeColorScheme(false);
    const user = userEvent.setup();
    renderToggle();
    expect(screen.getByRole("radio", { name: "System" })).toHaveFocus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("radio", { name: "Light" })).toBeChecked();
    expect(theme()).toBe("light");
  });

  it("still works when storage is unavailable", async () => {
    fakeColorScheme(false);
    const original = Object.getOwnPropertyDescriptor(window, "localStorage")!;
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      get() {
        throw new DOMException("denied", "SecurityError");
      },
    });
    try {
      const user = userEvent.setup();
      renderToggle();
      await user.click(screen.getByRole("radio", { name: "Dark" }));
      expect(theme()).toBe("dark");
      expect(() => new Function(initScript)()).not.toThrow();
    } finally {
      Object.defineProperty(window, "localStorage", original);
    }
  });
});
