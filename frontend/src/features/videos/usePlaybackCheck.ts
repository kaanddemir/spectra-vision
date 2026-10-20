/**
 * The in-browser playback check: load the registered video in the page's <video> element, wait for a
 * decoded frame (a picture with real dimensions, not only metadata), and report the result to the server
 * bound to the file contents (SHA-256) that were loaded. The server refuses a result for other contents.
 *
 * It runs by itself once per file contents when the page opens. Loading and decoding need no user gesture
 * and nothing is played: no sound, no frames advanced, the position stays at the start. A browser that
 * will not load data before the person interacts (some mobile browsers) is not failed: the check then
 * completes when the person presses Play.
 *
 * Only definite answers are reported. A refused or interrupted download, a timeout, or a video that
 * changed while it was being checked is never recorded as "not playable".
 */
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState, type RefObject } from "react";

import { ApiError, errorMessage } from "../../api/client";
import { keys, useReportPlayback } from "../../api/queries";
import type { Video } from "../../api/types";

export const CHECK_TIMEOUT_MS = 20000;

/** "waiting": the browser did not load a frame by itself; pressing Play completes the check. */
export type CheckPhase = "idle" | "loading" | "waiting" | "saving" | "failed";

export interface CheckState {
  phase: CheckPhase;
  /** Why the latest attempt did not produce a result. */
  message: string | null;
}

type LoadOutcome =
  | { kind: "playable"; detail: string }
  | { kind: "unplayable"; detail: string }
  | { kind: "unavailable"; detail: string };

const MEDIA_ERRORS: Record<number, LoadOutcome> = {
  1: { kind: "unavailable", detail: "Loading was interrupted." },
  2: { kind: "unavailable", detail: "The video could not be downloaded from the server." },
  3: { kind: "unplayable", detail: "This browser could not decode the video." },
  4: { kind: "unplayable", detail: "This browser does not support the video's format." },
};

/**
 * Resolves once the element has decoded the first frame, failed, or timed out. With `reload` false the
 * element is not reloaded (so the person's playback is never disturbed) and there is no timeout: it waits
 * for the frame the person's own Play produces.
 */
export function loadFirstFrame(element: HTMLVideoElement, signal: AbortSignal, reload = true): Promise<LoadOutcome> {
  return new Promise((resolve) => {
    const finish = (outcome: LoadOutcome) => {
      element.removeEventListener("loadeddata", onLoaded);
      element.removeEventListener("error", onError);
      signal.removeEventListener("abort", onAbort);
      window.clearTimeout(timer);
      resolve(outcome);
    };
    const onLoaded = () => {
      if (element.videoWidth > 0 && element.videoHeight > 0) {
        finish({ kind: "playable", detail: `Decoded a ${element.videoWidth}×${element.videoHeight} frame.` });
      } else {
        finish({ kind: "unplayable", detail: "The browser loaded the file but decoded no picture." });
      }
    };
    const onError = () =>
      finish(MEDIA_ERRORS[element.error?.code ?? 0] ?? { kind: "unplayable", detail: "The browser could not play the video." });
    const onAbort = () => finish({ kind: "unavailable", detail: "The check was cancelled." });
    const timer = reload
      ? window.setTimeout(() => finish({ kind: "unavailable", detail: TIMEOUT_DETAIL }), CHECK_TIMEOUT_MS)
      : undefined;
    element.addEventListener("loadeddata", onLoaded);
    element.addEventListener("error", onError);
    signal.addEventListener("abort", onAbort);
    if (!reload) {
      if (element.readyState >= HAVE_CURRENT_DATA) onLoaded();
      return;
    }
    // "metadata" may stop before a frame is decoded; the check needs one.
    element.preload = "auto";
    element.load();
  });
}

const HAVE_CURRENT_DATA = 2;
const TIMEOUT_DETAIL = "No frame was decoded within 20 seconds.";

/** Whether the person has started using the player; then it is never reloaded under them. */
function inUse(element: HTMLVideoElement): boolean {
  return !element.paused || element.currentTime > 0;
}

/** The server keeps the first 300 characters of the user agent that reported a result. */
const STORED_USER_AGENT_LENGTH = 300;

/**
 * The stored result if it was produced by this browser, else null. The server keeps one result per file
 * contents, from whichever browser checked last; a decode in one browser says nothing about another.
 */
export function resultHere(video: Video, userAgent = navigator.userAgent): Video["playback"]["browser_check"] {
  const check = video.playback.browser_check;
  return check && check.user_agent === userAgent.slice(0, STORED_USER_AGENT_LENGTH) ? check : null;
}

/** Whether this browser has itself decoded the current file contents. */
export function verifiedHere(video: Video): boolean {
  return Boolean(resultHere(video)?.playable);
}

/** Whether this browser should check the video: it can be streamed and has not decoded it here yet. */
export function needsCheck(video: Video): boolean {
  return Boolean(
    video.media_url &&
      video.media &&
      video.source.status === "available" &&
      video.probe.state === "ready" &&
      video.playback.assessment !== "unsupported" &&
      !verifiedHere(video),
  );
}

export function usePlaybackCheck(video: Video, mediaRef: RefObject<HTMLVideoElement | null>) {
  const [state, setState] = useState<CheckState>({ phase: "idle", message: null });
  const report = useReportPlayback();
  const client = useQueryClient();
  const latest = useRef(video);
  latest.current = video;
  const controller = useRef<AbortController | null>(null);
  // The file contents last checked automatically; a new file (or new contents) is checked again.
  const attempted = useRef<string | null>(null);

  useEffect(() => () => controller.current?.abort(), []);

  const start = useCallback(async () => {
    const identity = latest.current;
    const element = mediaRef.current;
    if (!identity.media_url || !identity.media || !element) return;
    const reload = !inUse(element);
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    const refresh = () => void client.invalidateQueries({ queryKey: keys.video(identity.id) });
    setState({ phase: "loading", message: null });

    // Confirm the server will stream this file before blaming the browser for a failed load.
    try {
      const head = await fetch(identity.media_url, { method: "HEAD", signal: abort.signal });
      if (!head.ok) {
        refresh();
        setState({
          phase: "failed",
          message: "The server will not stream this video right now. It may have been moved or changed.",
        });
        return;
      }
    } catch {
      if (abort.signal.aborted) return;
      // Unreachable server: refreshing the video lets the global banner report it; nothing is shown here,
      // and the automatic check runs again once the video is reloaded from the returning server.
      attempted.current = null;
      refresh();
      setState({ phase: "idle", message: null });
      return;
    }

    let outcome = await loadFirstFrame(element, abort.signal, reload);
    if (abort.signal.aborted) return;
    if (outcome.kind === "unavailable" && outcome.detail === TIMEOUT_DETAIL) {
      // The browser would not load a frame by itself: the person's Play completes the check.
      setState({ phase: "waiting", message: null });
      outcome = await loadFirstFrame(element, abort.signal, false);
      if (abort.signal.aborted) return;
    }
    if (outcome.kind === "unavailable") {
      setState({ phase: "failed", message: outcome.detail });
      return;
    }
    const current = latest.current;
    if (current.id !== identity.id || current.media?.sha256 !== identity.media.sha256) {
      setState({ phase: "failed", message: "The video changed during the check." });
      return;
    }
    setState({ phase: "saving", message: null });
    try {
      await report.mutateAsync({
        videoId: identity.id,
        playable: outcome.kind === "playable",
        detail: outcome.detail,
        sha256: identity.media.sha256,
      });
      if (!abort.signal.aborted) setState({ phase: "idle", message: null });
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) refresh();
      if (!abort.signal.aborted) setState({ phase: "failed", message: errorMessage(error) });
    }
  }, [client, mediaRef, report]);

  // Once per file contents, re-evaluated whenever the video is reloaded; a success here is never re-checked.
  const identityKey = video.media ? `${video.id}:${video.media.sha256}` : null;
  const wanted = needsCheck(video);
  useEffect(() => {
    if (!wanted || !identityKey || attempted.current === identityKey) return;
    attempted.current = identityKey;
    void start();
  }, [wanted, identityKey, start, video]);

  return { state, start };
}
