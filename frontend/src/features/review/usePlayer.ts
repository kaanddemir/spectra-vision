/**
 * Controls one review <video>: playback state, seeking in analysis time, and a clock that follows the
 * displayed frame. Where the browser supports `requestVideoFrameCallback`, the clock takes each presented
 * frame's own media time; otherwise it reads `currentTime` on every animation frame while playing and on
 * seek and time updates while paused. Every listener and callback is removed when the element changes.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { MediaClock } from "./mediaClock";
import { toAnalysisTime, toMediaTime } from "./timing";

export interface PlayerState {
  status: "loading" | "ready" | "error";
  paused: boolean;
  waiting: boolean;
  /** Length of the playable video in analysis time, once known. */
  duration: number | null;
  /** Display size of the decoded video, once known. */
  size: { width: number; height: number } | null;
  error: string | null;
  /** Whether frames are followed with requestVideoFrameCallback. */
  frameCallbacks: boolean;
  /** Playback speed; it changes only how fast media time passes, never the recorded timestamps. */
  rate: number;
  muted: boolean;
}

export interface Player {
  clock: MediaClock;
  state: PlayerState;
  /** Attach to the <video> element. */
  ref: (element: HTMLVideoElement | null) => void;
  seek: (t: number) => void;
  seekBy: (delta: number) => void;
  toggle: () => void;
  setRate: (rate: number) => void;
  setMuted: (muted: boolean) => void;
}

const INITIAL: PlayerState = {
  status: "loading",
  paused: true,
  waiting: false,
  duration: null,
  size: null,
  error: null,
  frameCallbacks: false,
  rate: 1,
  muted: true,
};

/** What a media error means for the person reviewing, or null for an aborted load. */
export function describeMediaError(error: MediaError | null): string | null {
  switch (error?.code) {
    case 4: // MEDIA_ERR_SRC_NOT_SUPPORTED
      return "This browser cannot play this video's format.";
    case 3: // MEDIA_ERR_DECODE
      return "This browser could not decode the video.";
    case 2: // MEDIA_ERR_NETWORK
      return "The video stopped loading. Check that the Spectra server is running.";
    case 1: // MEDIA_ERR_ABORTED
      return null;
    default:
      return "The video could not be played.";
  }
}

/**
 * `streamStart` and `knownDuration` come from the analysed video's recorded information. The recorded
 * duration is preferred: Chrome reports an offset MP4's `duration` as the stream's length while placing its
 * frames at container time, so `duration - streamStart` would cut off the end.
 */
export function usePlayer(streamStart: number, knownDuration: number | null = null): Player {
  const clock = useMemo(() => new MediaClock(), []);
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const [state, setState] = useState<PlayerState>(INITIAL);
  const origin = useRef(streamStart);
  origin.current = streamStart;
  const pendingSeek = useRef<number | null>(null);
  const duration = useRef<number | null>(knownDuration);
  const recorded = useRef(knownDuration);
  recorded.current = knownDuration;
  // Kept across element changes; review starts muted so a video never plays sound unexpectedly.
  const preferences = useRef({ rate: 1, muted: true });

  useEffect(() => {
    if (!video) return;
    const frameCallbacks = typeof video.requestVideoFrameCallback === "function";
    let frameHandle: number | null = null;
    let animationHandle: number | null = null;
    const update = (state: Partial<PlayerState>) => setState((current) => ({ ...current, ...state }));
    const read = () => clock.set(toAnalysisTime(video.currentTime, origin.current));
    video.muted = preferences.current.muted;
    video.playbackRate = preferences.current.rate;

    const onFrame = (_: number, metadata: VideoFrameCallbackMetadata) => {
      clock.set(toAnalysisTime(metadata.mediaTime, origin.current));
      frameHandle = video.requestVideoFrameCallback(onFrame);
    };
    const tick = () => {
      read();
      animationHandle = requestAnimationFrame(tick);
    };
    const stopTicking = () => {
      if (animationHandle !== null) cancelAnimationFrame(animationHandle);
      animationHandle = null;
    };

    const readDuration = () => {
      const media = video.duration;
      duration.current =
        recorded.current ?? (Number.isFinite(media) && media > 0 ? toAnalysisTime(media, origin.current) : null);
      return duration.current;
    };
    const onMetadata = () => {
      update({
        status: "ready",
        error: null,
        duration: readDuration(),
        size: video.videoWidth > 0 ? { width: video.videoWidth, height: video.videoHeight } : null,
      });
      if (pendingSeek.current !== null) {
        video.currentTime = toMediaTime(pendingSeek.current, origin.current);
        clock.set(pendingSeek.current);
        pendingSeek.current = null;
      } else {
        read();
      }
    };
    const onPlay = () => {
      update({ paused: false });
      if (!frameCallbacks && animationHandle === null) animationHandle = requestAnimationFrame(tick);
    };
    const onPause = () => {
      update({ paused: true, waiting: false });
      stopTicking();
      read();
    };
    const onTimeUpdate = () => {
      // Frame callbacks are exact while playing; time updates only fill in while paused.
      if (!frameCallbacks || video.paused) read();
    };
    const onError = () => {
      const message = describeMediaError(video.error);
      if (message) update({ status: "error", error: message, waiting: false });
    };
    const onRate = () => {
      preferences.current.rate = video.playbackRate;
      update({ rate: video.playbackRate });
    };
    const onVolume = () => {
      preferences.current.muted = video.muted;
      update({ muted: video.muted });
    };
    const listeners: [string, () => void][] = [
      ["loadedmetadata", onMetadata],
      ["durationchange", () => update({ duration: readDuration() })],
      ["play", onPlay],
      ["pause", onPause],
      ["ended", onPause],
      ["seeked", read],
      ["timeupdate", onTimeUpdate],
      ["waiting", () => update({ waiting: true })],
      ["playing", () => update({ waiting: false })],
      ["canplay", () => update({ waiting: false })],
      ["error", onError],
      ["ratechange", onRate],
      ["volumechange", onVolume],
    ];
    listeners.forEach(([type, listener]) => video.addEventListener(type, listener));
    if (frameCallbacks) frameHandle = video.requestVideoFrameCallback(onFrame);
    update({ frameCallbacks, rate: video.playbackRate, muted: video.muted });
    if (video.readyState >= 1) onMetadata();

    return () => {
      listeners.forEach(([type, listener]) => video.removeEventListener(type, listener));
      if (frameHandle !== null) video.cancelVideoFrameCallback(frameHandle);
      stopTicking();
    };
  }, [video, clock, streamStart]);

  const seek = useCallback(
    (t: number) => {
      const target = Math.max(0, duration.current === null ? t : Math.min(t, duration.current));
      clock.set(target);
      if (!video || video.readyState < 1) {
        pendingSeek.current = target;
        return;
      }
      video.currentTime = toMediaTime(target, origin.current);
    },
    [video, clock],
  );

  const seekBy = useCallback((delta: number) => seek(clock.get() + delta), [seek, clock]);

  const toggle = useCallback(() => {
    if (!video) return;
    if (video.paused || video.ended) {
      // A refused play() (for example an interrupted load) leaves the video paused; the error event explains it.
      void video.play()?.catch(() => undefined);
    } else {
      video.pause();
    }
  }, [video]);

  const setRate = useCallback(
    (rate: number) => {
      preferences.current.rate = rate;
      if (video) video.playbackRate = rate;
      setState((current) => ({ ...current, rate }));
    },
    [video],
  );

  const setMuted = useCallback(
    (muted: boolean) => {
      preferences.current.muted = muted;
      if (video) video.muted = muted;
      setState((current) => ({ ...current, muted }));
    },
    [video],
  );

  return { clock, state, ref: setVideo, seek, seekBy, toggle, setRate, setMuted };
}
