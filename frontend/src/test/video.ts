/**
 * A controllable stand-in for loaded <video> elements in jsdom, which neither loads nor plays media.
 * Install it before rendering (the player reads browser capabilities when the element mounts), set
 * what the browser would report (duration, size, earliest position, frame callbacks), then drive
 * playback events by hand. `restore` removes every patch.
 */
import { act, fireEvent } from "@testing-library/react";
import { vi } from "vitest";

export interface FakeVideoOptions {
  duration?: number;
  width?: number;
  height?: number;
  /** The browser's earliest seekable position (non-zero when it exposes container time). */
  earliest?: number;
  /** Whether the browser supports requestVideoFrameCallback. */
  frameCallbacks?: boolean;
}

const PATCHED = [
  "currentTime",
  "duration",
  "paused",
  "ended",
  "readyState",
  "videoWidth",
  "videoHeight",
  "error",
  "seekable",
  "play",
  "pause",
  "requestVideoFrameCallback",
  "cancelVideoFrameCallback",
] as const;

export function installFakeVideo(options: FakeVideoOptions = {}) {
  const { duration = 62.5, width = 1920, height = 1080, earliest = 0, frameCallbacks = true } = options;
  const proto = HTMLVideoElement.prototype;
  const saved = new Map(PATCHED.map((name) => [name, Object.getOwnPropertyDescriptor(proto, name)]));
  const state = { currentTime: earliest, paused: true, readyState: 0, error: null as { code: number } | null };
  const callbacks = new Map<number, VideoFrameRequestCallback>();
  let nextHandle = 1;
  const element = () => {
    const found = document.querySelector("video");
    if (!found) throw new Error("no video element is rendered");
    return found;
  };
  const define = (name: string, get: () => unknown, set?: (value: never) => void) =>
    Object.defineProperty(proto, name, { configurable: true, get, set });

  define("currentTime", () => state.currentTime, (value: number) => {
    state.currentTime = value;
  });
  define("duration", () => (state.readyState > 0 ? duration + earliest : NaN));
  define("paused", () => state.paused);
  define("ended", () => false);
  define("readyState", () => state.readyState);
  define("videoWidth", () => (state.readyState > 0 ? width : 0));
  define("videoHeight", () => (state.readyState > 0 ? height : 0));
  define("error", () => state.error);
  define("seekable", () => ({
    length: state.readyState > 0 ? 1 : 0,
    start: () => earliest,
    end: () => earliest + duration,
  }));
  const play = vi.fn(function (this: HTMLVideoElement) {
    state.paused = false;
    fireEvent(this, new Event("play"));
    return Promise.resolve();
  });
  const pause = vi.fn(function (this: HTMLVideoElement) {
    state.paused = true;
    fireEvent(this, new Event("pause"));
  });
  const requestFrame = vi.fn((callback: VideoFrameRequestCallback) => {
    const handle = nextHandle++;
    callbacks.set(handle, callback);
    return handle;
  });
  const cancelFrame = vi.fn((handle: number) => {
    callbacks.delete(handle);
  });
  Object.defineProperty(proto, "play", { configurable: true, writable: true, value: play });
  Object.defineProperty(proto, "pause", { configurable: true, writable: true, value: pause });
  if (frameCallbacks) {
    Object.defineProperty(proto, "requestVideoFrameCallback", { configurable: true, writable: true, value: requestFrame });
    Object.defineProperty(proto, "cancelVideoFrameCallback", { configurable: true, writable: true, value: cancelFrame });
  }

  return {
    element,
    play,
    pause,
    requestFrame,
    cancelFrame,
    pendingFrameCallbacks: () => callbacks.size,
    get currentTime() {
      return state.currentTime;
    },
    /** The browser has read the file's metadata. */
    loadMetadata() {
      state.readyState = 1;
      act(() => {
        fireEvent(element(), new Event("loadedmetadata"));
      });
    },
    /** The browser presents a frame with this media time (requestVideoFrameCallback). */
    presentFrame(mediaTime: number) {
      state.currentTime = mediaTime;
      const pending = [...callbacks.values()];
      callbacks.clear();
      act(() => {
        for (const callback of pending) callback(performance.now(), { mediaTime } as VideoFrameCallbackMetadata);
      });
    },
    /** A time update or a completed seek, as browsers report them without frame callbacks. */
    timeUpdate(mediaTime: number, type: "timeupdate" | "seeked" = "timeupdate") {
      state.currentTime = mediaTime;
      act(() => {
        fireEvent(element(), new Event(type));
      });
    },
    fail(code: number) {
      state.error = { code };
      act(() => {
        fireEvent(element(), new Event("error"));
      });
    },
    restore() {
      for (const [name, descriptor] of saved) {
        if (descriptor) Object.defineProperty(proto, name, descriptor);
        else delete (proto as unknown as Record<string, unknown>)[name];
      }
    },
  };
}
