/**
 * Enforces the design system: readable contrast in both themes, colours and lengths only from
 * tokens, the documented breakpoints, and one icon library used through src/ui.
 */
import { describe, expect, it } from "vitest";

import tokens from "./tokens.css?raw";

const styles = import.meta.glob<string>("../**/*.css", { query: "?raw", import: "default", eager: true });
const sources = import.meta.glob<string>(["../**/*.{ts,tsx}", "!../**/*.test.{ts,tsx}", "!../test/**"], {
  query: "?raw",
  import: "default",
  eager: true,
});

function block(selector: string): Record<string, string> {
  const start = tokens.indexOf(`${selector} {`);
  expect(start, `token block ${selector}`).toBeGreaterThanOrEqual(0);
  const body = tokens.slice(start, tokens.indexOf("}", start));
  return Object.fromEntries([...body.matchAll(/--([\w-]+):\s*([^;]+);/g)].map((m) => [m[1]!, m[2]!.trim()]));
}

const base = block(":root");
const themes = {
  light: { ...base, ...block(':root,\n:root[data-theme="light"]') },
  dark: { ...base, ...block(':root[data-theme="dark"]') },
};

function color(theme: Record<string, string>, name: string): string {
  let value = theme[name];
  for (let depth = 0; value?.startsWith("var("); depth += 1) {
    value = theme[value.slice(6, -1)];
    if (depth > 5) break;
  }
  expect(value, `--${name}`).toMatch(/^#[0-9a-f]{6}$/);
  return value!;
}

function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r! + 0.7152 * g! + 0.0722 * b!;
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi! + 0.05) / (lo! + 0.05);
}

const SURFACES = ["color-bg", "color-surface", "color-surface-subtle"];
const TEXT = ["color-text", "color-text-secondary", "color-accent-text"];
const STATUSES = ["success", "warning", "danger", "info", "neutral"];

describe.each(Object.entries(themes))("%s theme contrast", (_, theme) => {
  it.each(TEXT.flatMap((fg) => SURFACES.map((bg) => [fg, bg])))("%s on %s is at least 4.5:1", (fg, bg) => {
    expect(contrast(color(theme, fg!), color(theme, bg!))).toBeGreaterThanOrEqual(4.5);
  });

  it.each(STATUSES)("%s text is readable on its own background and on surfaces", (status) => {
    const text = color(theme, `color-${status}-text`);
    for (const bg of [`color-${status}-bg`, "color-surface", "color-bg"]) {
      expect(contrast(text, color(theme, bg))).toBeGreaterThanOrEqual(4.5);
    }
  });

  it("keeps text on filled controls, tooltips and media readable", () => {
    expect(contrast(color(theme, "color-on-accent"), color(theme, "color-accent"))).toBeGreaterThanOrEqual(4.5);
    expect(contrast(color(theme, "color-on-accent"), color(theme, "color-accent-hover"))).toBeGreaterThanOrEqual(4.5);
    expect(contrast(color(theme, "color-tooltip-text"), color(theme, "color-tooltip-bg"))).toBeGreaterThanOrEqual(4.5);
    expect(contrast(color(theme, "color-media-text"), color(theme, "color-media-bg"))).toBeGreaterThanOrEqual(4.5);
  });

  it("gives control outlines, focus rings and filled buttons 3:1 against surfaces", () => {
    for (const bg of ["color-bg", "color-surface"]) {
      expect(contrast(color(theme, "color-border-control"), color(theme, bg))).toBeGreaterThanOrEqual(3);
      expect(contrast(color(theme, "color-focus"), color(theme, bg))).toBeGreaterThanOrEqual(3);
      expect(contrast(color(theme, "color-accent"), color(theme, bg))).toBeGreaterThanOrEqual(3);
    }
  });

  it("defines every semantic colour in both themes", () => {
    const names = (t: Record<string, string>) => Object.keys(t).filter((k) => k.startsWith("color-")).sort();
    expect(names(theme)).toEqual(names(themes.light));
  });
});

describe("review overlays", () => {
  it("use one set of colours in both themes, since the media stage is dark in both", () => {
    const overlay = Object.keys(base).filter((name) => name.startsWith("color-overlay-"));
    expect(overlay.length).toBeGreaterThan(0);
    for (const name of overlay) {
      expect(themes.light[name]).toBe(themes.dark[name]);
      expect(block(':root[data-theme="dark"]')[name], name).toBeUndefined();
    }
    for (const name of ["color-overlay-track", "color-overlay-related", "color-overlay-lane", "color-overlay-text"]) {
      expect(contrast(color(base, name), color(themes.dark, "color-media-bg"))).toBeGreaterThanOrEqual(4.5);
      expect(contrast(color(base, name), color(themes.light, "color-media-bg"))).toBeGreaterThanOrEqual(4.5);
    }
  });
});

describe("token discipline", () => {
  const componentStyles = Object.entries(styles).filter(([path]) => !path.endsWith("/tokens.css"));
  const BREAKPOINTS = new Set(["640px", "960px", "1280px"]);

  it("finds the stylesheets and sources it checks", () => {
    expect(componentStyles.length).toBeGreaterThan(10);
    expect(Object.keys(sources).length).toBeGreaterThan(20);
  });

  it.each(componentStyles)("%s uses colour tokens only", (_, css) => {
    expect(css).not.toMatch(/#[0-9a-fA-F]{3,8}\b/);
    expect(css).not.toMatch(/\b(rgb|rgba|hsl|hsla)\(/);
  });

  it.each(componentStyles)("%s uses spacing and size tokens only", (_, css) => {
    // Media queries hold the documented breakpoints; a reasoned `token-exception` comment marks
    // the few lengths that cannot be tokens.
    const outsideMedia = css
      .split("\n")
      .filter((line) => !line.trim().startsWith("@media") && !/token-exception: \S/.test(line))
      .join("\n");
    expect(outsideMedia).not.toMatch(/(?<![\w-])\d*\.?\d+(px|rem)\b/);
  });

  it.each(componentStyles)("%s uses the documented breakpoints", (_, css) => {
    for (const match of css.matchAll(/@media[^{]*\((?:max|min)-width:\s*([\d.]+px)\)/g)) {
      expect(BREAKPOINTS.has(match[1]!), match[0]).toBe(true);
    }
  });

  it.each(Object.entries(sources))("%s draws icons through the icon system", (path, code) => {
    const iconModules = ["/ui/icons.ts", "/ui/Icon.tsx"];
    const valueImport = /import\s+(?!type\b)[^;]*from\s+"lucide-react"/.test(code);
    if (!iconModules.some((m) => path.endsWith(m))) expect(valueImport).toBe(false);
    expect(code).not.toMatch(/from\s+"(react-icons|@heroicons|@mui\/icons|@tabler\/icons|@phosphor-icons)/);
  });

  it.each(Object.entries(sources))("%s has no inline styles, raw colours or emoji", (_, code) => {
    expect(code).not.toMatch(/\sstyle=\{/);
    expect(code).not.toMatch(/#[0-9a-fA-F]{6}\b/);
    expect(code).not.toMatch(/\p{Extended_Pictographic}/u);
  });
});
