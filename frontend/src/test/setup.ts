import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

afterEach(() => {
  cleanup();
  window.localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
});

// jsdom implements neither modal dialogs nor media loading; these stand-ins only toggle state.
if (!HTMLDialogElement.prototype.showModal) {
  HTMLDialogElement.prototype.showModal = function showModal(this: HTMLDialogElement) {
    this.setAttribute("open", "");
  };
  HTMLDialogElement.prototype.close = function close(this: HTMLDialogElement) {
    this.removeAttribute("open");
  };
}
HTMLMediaElement.prototype.load = vi.fn();
// The browser that produced the fixtures' stored playback results, so they count as checked "here".
const FIXTURE_USER_AGENT =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15";
Object.defineProperty(window.navigator, "userAgent", { configurable: true, get: () => FIXTURE_USER_AGENT });
window.scrollTo = vi.fn() as unknown as typeof window.scrollTo;

// The title index.html sets; pages never change it.
document.title = "Spectra";
