/**
 * Live analysis state. The server's event stream is the primary source; each event carries the
 * analysis revision, so older or repeated states are ignored. If the stream cannot be used, the
 * analysis is polled until it settles. Nothing is ever retried or restarted automatically.
 */
import { useQueryClient, type QueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { api } from "./client";
import { keys } from "./queries";
import type { Analysis } from "./types";

export const FALLBACK_POLL_MS = 2000;
/** Consecutive stream errors before switching to polling; the browser reconnects in between. */
export const MAX_STREAM_ERRORS = 3;

export const isActive = (a: Pick<Analysis, "state">) => a.state === "queued" || a.state === "running";

/** Final state reached and evidence no longer being prepared: nothing more will change. */
export const isSettled = (a: Analysis) => !isActive(a) && a.evidence.state !== "pending";

/** Store a newer analysis state in every cache that shows it. Returns false for stale states. */
export function storeAnalysis(client: QueryClient, next: Analysis): boolean {
  const current = client.getQueryData<Analysis>(keys.analysis(next.id));
  if (current && current.revision >= next.revision) return false;
  client.setQueryData(keys.analysis(next.id), next);
  for (const [key, list] of client.getQueriesData<Analysis[]>({ queryKey: ["analyses"] })) {
    if (!list) continue;
    const filter = key[1] as { videoId?: string } | undefined;
    const found = list.find((a) => a.id === next.id);
    if (found && found.revision >= next.revision) continue;
    if (!found && filter?.videoId && filter.videoId !== next.video_id) continue;
    client.setQueryData(key, found ? list.map((a) => (a.id === next.id ? next : a)) : [next, ...list]);
  }
  return true;
}

export type LiveMode = "idle" | "stream" | "polling";

/** Follow one analysis until it settles. */
export function useAnalysisLive(analysis: Analysis | undefined): LiveMode {
  const client = useQueryClient();
  const [mode, setMode] = useState<LiveMode>("idle");
  const id = analysis?.id;
  const follow = analysis ? !isSettled(analysis) : false;

  useEffect(() => {
    if (!id || !follow) {
      setMode("idle");
      return;
    }
    let timer: number | undefined;
    let source: EventSource | null = null;
    let stopped = false;

    const poll = () => {
      setMode("polling");
      const tick = async () => {
        try {
          const next = await api.analysis(id);
          storeAnalysis(client, next);
          if (isSettled(next)) return;
        } catch {
          // The server banner reports unavailability; keep checking at the same slow pace.
        }
        if (!stopped) timer = window.setTimeout(tick, FALLBACK_POLL_MS);
      };
      timer = window.setTimeout(tick, FALLBACK_POLL_MS);
    };

    if (typeof EventSource === "undefined") {
      poll();
    } else {
      let errors = 0;
      source = new EventSource(api.analysisEventsUrl(id));
      setMode("stream");
      source.addEventListener("analysis", (event) => {
        errors = 0;
        storeAnalysis(client, JSON.parse((event as MessageEvent<string>).data) as Analysis);
      });
      source.addEventListener("end", () => source?.close());
      source.onerror = () => {
        errors += 1;
        if (source && (source.readyState === EventSource.CLOSED || errors >= MAX_STREAM_ERRORS)) {
          source.close();
          source = null;
          poll();
        }
      };
    }
    return () => {
      stopped = true;
      source?.close();
      window.clearTimeout(timer);
    };
  }, [client, id, follow]);

  return mode;
}
