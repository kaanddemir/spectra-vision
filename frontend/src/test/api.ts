/**
 * A controlled stand-in for the local API: tests declare responses per "METHOD /path" and inspect
 * the requests the application made. Unhandled requests fail loudly.
 */
import { vi } from "vitest";

export interface Recorded {
  method: string;
  path: string;
  body: unknown;
}

type Reply = { status?: number; body?: unknown; json?: boolean; headers?: Record<string, string> } | "network-error";
type Handler = Reply | ((request: Recorded) => Reply | Promise<Reply>);

export function jsonReply(body: unknown, status = 200): Reply {
  return { status, body };
}

export function errorReply(status: number, detail: unknown): Reply {
  return { status, body: { detail } };
}

export function mockApi(initial: Record<string, Handler> = {}) {
  const handlers = new Map<string, Handler>(Object.entries(initial));
  const requests: Recorded[] = [];

  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = new URL(typeof input === "string" ? input : input instanceof URL ? input.href : input.url, "http://127.0.0.1:5173");
    const method = (init?.method ?? "GET").toUpperCase();
    const path = url.pathname + url.search;
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : undefined;
    const request = { method, path, body };
    requests.push(request);
    const handler = handlers.get(`${method} ${path}`) ?? handlers.get(`${method} ${url.pathname}`);
    if (handler === undefined) {
      return new Response(JSON.stringify({ detail: `unhandled test request ${method} ${path}` }), {
        status: 599,
        headers: { "content-type": "application/json" },
      });
    }
    const reply = typeof handler === "function" ? await handler(request) : handler;
    if (reply === "network-error") throw new TypeError("Failed to fetch");
    const status = reply.status ?? 200;
    if (reply.json === false) return new Response(String(reply.body ?? ""), { status, headers: { "content-type": "text/plain" } });
    return new Response(status === 204 ? null : JSON.stringify(reply.body ?? null), {
      status,
      headers: { "content-type": "application/json", ...reply.headers },
    });
  });

  return {
    requests,
    fetchMock,
    on(route: string, handler: Handler) {
      handlers.set(route, handler);
    },
    calls(route: string): Recorded[] {
      const [method, path] = route.split(" ");
      return requests.filter((r) => r.method === method && (r.path === path || r.path.split("?")[0] === path));
    },
  };
}

/** A healthy server with one library folder, every fixture video and no analyses. */
export async function standardApi(overrides: Record<string, Handler> = {}) {
  const { fixture, allVideos } = await import("./fixtures");
  const listing = fixture("listing");
  return mockApi({
    "GET /api/health": jsonReply(fixture("health")),
    "GET /api/library": jsonReply(fixture("folders")),
    [`GET /api/library/${listing.folder.id}/entries?path=`]: jsonReply(listing),
    [`GET /api/library/${listing.folder.id}/entries?path=2026-10+trips`]: jsonReply(fixture("sublisting")),
    "GET /api/videos": jsonReply(allVideos()),
    "GET /api/analyses": jsonReply([]),
    ...overrides,
  });
}
