/**
 * The player's position in analysis time, outside React state. Components subscribe to a derived value
 * (a rounded time, an overlay frame index, an overlay span) and re-render only when that value changes,
 * so playback never re-renders the whole workspace.
 */
import { useCallback, useSyncExternalStore } from "react";

export class MediaClock {
  private time = 0;
  private readonly listeners = new Set<() => void>();

  get = (): number => this.time;

  set(t: number): void {
    if (t === this.time || !Number.isFinite(t)) return;
    this.time = t;
    this.listeners.forEach((listener) => listener());
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };
}

/** A value derived from the clock; `select` must return a primitive so unchanged values skip rendering. */
export function useClock<T extends string | number | boolean | null>(clock: MediaClock, select: (t: number) => T): T {
  const snapshot = useCallback(() => select(clock.get()), [clock, select]);
  return useSyncExternalStore(clock.subscribe, snapshot, snapshot);
}

export const tenths = (t: number) => Math.round(t * 10) / 10;
