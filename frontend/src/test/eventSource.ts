/** A controlled EventSource: tests push server events and connection failures by hand. */
import { vi } from "vitest";

export class FakeEventSource {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 2;
  static instances: FakeEventSource[] = [];

  readonly url: string;
  readyState = FakeEventSource.OPEN;
  onerror: ((event: Event) => void) | null = null;
  private listeners = new Map<string, Set<(event: MessageEvent) => void>>();

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: (event: MessageEvent) => void) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(listener);
  }

  removeEventListener(type: string, listener: (event: MessageEvent) => void) {
    this.listeners.get(type)?.delete(listener);
  }

  close() {
    this.readyState = FakeEventSource.CLOSED;
  }

  /** Deliver a server event, as the browser would. */
  emit(type: string, data: unknown) {
    if (this.readyState === FakeEventSource.CLOSED) return;
    const event = new MessageEvent(type, { data: JSON.stringify(data) });
    this.listeners.get(type)?.forEach((listener) => listener(event));
  }

  /** A dropped connection; `permanent` mimics the browser giving up. */
  fail(permanent = false) {
    this.readyState = permanent ? FakeEventSource.CLOSED : FakeEventSource.CONNECTING;
    this.onerror?.(new Event("error"));
  }

  get closed() {
    return this.readyState === FakeEventSource.CLOSED;
  }
}

export function installEventSource() {
  FakeEventSource.instances = [];
  vi.stubGlobal("EventSource", FakeEventSource);
  return {
    latest: () => FakeEventSource.instances.at(-1)!,
    all: () => FakeEventSource.instances,
  };
}
