/**
 * The review player: the registered source video, read-only and streamed as stored, with recorded
 * overlays and playback controls over the bottom of the picture, shown while paused or in use. Overlays are requested
 * only while they are switched on, for the selected proposal's review interval or a short span around
 * the playhead.
 *
 * There is no frame stepping: the browser seeks to a time, not to a decoded frame, and the analysis
 * records only its sampled frames, so a step could not be shown to land on the next real frame.
 */
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type FocusEvent,
  type KeyboardEvent,
  type PointerEvent,
  type RefObject,
} from "react";

import { isOffline } from "../../api/client";
import { useOverlay } from "../../api/queries";
import type { OverlayResponse, Span } from "../../api/types";
import { formatTimecode } from "../../lib/format";
import { buttonClass, IconButton } from "../../ui/Button";
import { Icon } from "../../ui/Icon";
import { icons } from "../../ui/icons";
import { Popover } from "../../ui/Popover";
import { Spinner } from "../../ui/Spinner";
import { tenths, useClock } from "./mediaClock";
import { OverlayDrawing, type OverlayLayers } from "./OverlayDrawing";
import { type AnalysedSpan, chunkSpan, frameIndexAt, geometryMatches, overlayWindow } from "./timing";
import type { Player } from "./usePlayer";
import "./ReviewPlayer.css";

const SKIP_S = 5;
const RATES = [0.25, 0.5, 1, 1.5, 2];
/** How long playing video stays uncovered after the last pointer move or key press. */
const IDLE_MS = 2500;
const pct = (value: number) => `${value.toFixed(3)}%`;

interface ReviewPlayerProps {
  player: Player;
  /** The source's media URL, or null when the server will not stream it. */
  src: string | null;
  /** Why the source cannot be played, when `src` is null. */
  unavailable: string[];
  analysisId: string;
  /** The proposal whose overlay data is requested (the selected one, else the first). */
  overlayEventId: string | null;
  /** The selected proposal's review interval, or null without a selection. */
  review: Span | null;
  analysed: AnalysedSpan;
  /** How long an analysed frame's records may be shown after its own time. */
  maxAge: number;
  /** Highlight the selected proposal's tracks. */
  highlightRelated: boolean;
  /** Whether the source has an audio stream; sound controls are offered only then. */
  hasAudio: boolean;
}

export function ReviewPlayer(props: ReviewPlayerProps) {
  const { player, src, unavailable } = props;
  const [overlays, setOverlays] = useState(false);
  const [layers, setLayers] = useState<OverlayLayers>({ boxes: true, lane: true });
  const [available, setAvailable] = useState<OverlayResponse["layers"] | null>(null);
  const { state } = player;
  const root = useRef<HTMLDivElement>(null);
  const fullscreen = useFullscreen(root);
  const ready = state.status === "ready";
  const chrome = useControlsVisibility(ready && !state.paused);
  const lastPointer = useRef<string>("mouse");

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    chrome.wake();
    if (event.target !== event.currentTarget || event.metaKey || event.ctrlKey || event.altKey) return;
    const actions: Record<string, () => void> = {
      " ": player.toggle,
      k: player.toggle,
      ArrowLeft: () => player.seekBy(-SKIP_S),
      ArrowRight: () => player.seekBy(SKIP_S),
      Home: () => player.seek(0),
      End: () => state.duration !== null && player.seek(state.duration),
    };
    const action = actions[event.key];
    if (action && src && state.status === "ready") {
      event.preventDefault();
      action();
    }
  };

  return (
    <div ref={root} className="review-player">
      <div
        className="media-stage review-player__stage"
        data-controls={chrome.visible ? "shown" : "hidden"}
        role="group"
        aria-label="Video player"
        aria-describedby="review-player-keys"
        tabIndex={src ? 0 : undefined}
        onKeyDown={onKeyDown}
        onPointerMove={(event) => event.pointerType === "mouse" && chrome.wake()}
        onPointerLeave={(event) => event.pointerType === "mouse" && chrome.rest()}
        // A click on the picture focuses the stage for keyboard control without scrolling the page to it.
        onMouseDown={(event) => {
          if (!src || (event.target as Element).closest("button, a, input, select, textarea, [role='slider']")) return;
          event.preventDefault();
          event.currentTarget.focus({ preventScroll: true });
        }}
      >
        {src ? (
          <>
            <video
              ref={player.ref}
              src={src}
              preload="metadata"
              playsInline
              disablePictureInPicture
              onPointerDown={(event) => (lastPointer.current = event.pointerType)}
              onClick={() => {
                // A tap on hidden controls only reveals them, as touch has no hover.
                const reveal = lastPointer.current === "touch" && !chrome.visible;
                lastPointer.current = "mouse";
                chrome.wake();
                if (!reveal) player.toggle();
              }}
              aria-label="Source video"
            />
            {overlays && state.status === "ready" ? (
              <OverlayView {...props} layers={layers} onLayers={setAvailable} />
            ) : null}
            {state.status === "loading" ? (
              <div className="review-player__cover" role="status">
                <Spinner size="md" />
                <span>Loading video…</span>
              </div>
            ) : null}
            {state.status === "ready" && state.waiting ? (
              <div className="review-player__corner" role="status">
                <Spinner />
                <span className="visually-hidden">Buffering…</span>
              </div>
            ) : null}
            {state.status === "error" ? (
              <div className="review-player__cover media-stage__message" role="alert">
                <Icon icon={icons.unsupported} size="lg" />
                <p className="type-subheading">{state.error}</p>
                <p>Evidence below is still available.</p>
              </div>
            ) : (
              <div className="review-player__chrome" {...chrome.holdProps}>
                <SeekBar player={player} disabled={!ready} />
                <div className="review-player__controls" role="toolbar" aria-label="Playback">
                  <div className="review-player__group">
                    <IconButton
                      icon={state.paused ? icons.play : icons.pause}
                      label={state.paused ? "Play" : "Pause"}
                      size="sm"
                      onClick={player.toggle}
                      disabled={!ready}
                    />
                    <IconButton icon={icons.back5} label="Back 5 seconds" size="sm" onClick={() => player.seekBy(-SKIP_S)} disabled={!ready} />
                    <IconButton icon={icons.forward5} label="Forward 5 seconds" size="sm" onClick={() => player.seekBy(SKIP_S)} disabled={!ready} />
                    {props.hasAudio ? (
                      <IconButton
                        icon={state.muted ? icons.soundOff : icons.soundOn}
                        label={state.muted ? "Unmute" : "Mute"}
                        size="sm"
                        aria-pressed={!state.muted}
                        onClick={() => player.setMuted(!state.muted)}
                        disabled={!ready}
                      />
                    ) : null}
                    <TimeReadout player={player} />
                  </div>
                  <div className="review-player__group">
                    <SpeedMenu rate={state.rate} disabled={!ready} onChange={player.setRate} />
                    <div className="review-player__group" role="group" aria-label="Overlays">
                      <IconButton
                        icon={icons.overlays}
                        label="Overlays"
                        size="sm"
                        aria-pressed={overlays}
                        onClick={() => setOverlays((on) => !on)}
                        disabled={!ready || props.overlayEventId === null}
                      />
                      {overlays && (available?.tracks || available?.lane) ? (
                        <LayerMenu available={available} layers={layers} onChange={setLayers} />
                      ) : null}
                    </div>
                    {fullscreen.supported ? (
                      <IconButton
                        icon={fullscreen.active ? icons.exitFullscreen : icons.fullscreen}
                        label={fullscreen.active ? "Exit full screen" : "Full screen"}
                        size="sm"
                        onClick={fullscreen.toggle}
                      />
                    ) : null}
                  </div>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="media-stage__message" role="status">
            <Icon icon={icons.unsupported} size="lg" />
            <p className="type-subheading">This video cannot be played here</p>
            {unavailable.map((reason) => (
              <p key={reason}>{reason}</p>
            ))}
            <p>Evidence below is still available.</p>
          </div>
        )}
      </div>
      <p id="review-player-keys" className="visually-hidden">
        Space plays or pauses. Left and right arrows move five seconds.
      </p>
    </div>
  );
}

/**
 * Whether the controls over the picture are shown: always while paused; while playing, after the pointer
 * moves or a key is pressed, until a short idle period passes. Pointer or keyboard focus inside the controls
 * (an open menu included) keeps them shown.
 */
function useControlsVisibility(playing: boolean) {
  const [awake, setAwake] = useState(true);
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  const wake = useCallback(() => {
    setAwake(true);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setAwake(false), IDLE_MS);
  }, []);
  const rest = useCallback(() => {
    window.clearTimeout(timer.current);
    setAwake(false);
  }, []);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  useEffect(() => {
    if (playing) wake();
  }, [playing, wake]);
  const holdProps = {
    onPointerEnter: (event: PointerEvent<HTMLDivElement>) => event.pointerType === "mouse" && setHovered(true),
    onPointerLeave: (event: PointerEvent<HTMLDivElement>) => event.pointerType === "mouse" && setHovered(false),
    onFocus: () => setFocused(true),
    onBlur: (event: FocusEvent<HTMLDivElement>) => {
      if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setFocused(false);
    },
  };
  return { visible: !playing || awake || hovered || focused, wake, rest, holdProps };
}

/**
 * The video's own progress: a thin seekable bar over the whole duration. Pointer and touch drag seek;
 * arrows move one second (five with Shift), Page keys ten, Home and End to either end.
 */
function SeekBar({ player, disabled }: { player: Player; disabled: boolean }) {
  const track = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const time = useClock(player.clock, tenths);
  const duration = player.state.duration ?? 0;
  const progress = useClock(
    player.clock,
    useCallback((t: number) => (duration > 0 ? Math.min(Math.max(t / duration, 0), 1) : 0), [duration]),
  );

  useLayoutEffect(() => {
    track.current?.style.setProperty("--seek-progress", String(progress));
  }, [progress]);

  const timeAt = (clientX: number) => {
    const rect = track.current!.getBoundingClientRect();
    return rect.width > 0 ? Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1) * duration : 0;
  };
  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (disabled || event.button !== 0) return;
    dragging.current = true;
    event.currentTarget.setPointerCapture?.(event.pointerId);
    player.seek(timeAt(event.clientX));
  };
  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (dragging.current) player.seek(timeAt(event.clientX));
  };
  const onPointerUp = (event: PointerEvent<HTMLDivElement>) => {
    dragging.current = false;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
  };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (disabled) return;
    const step = event.shiftKey ? 5 : 1;
    const targets: Record<string, number> = {
      ArrowLeft: time - step,
      ArrowDown: time - step,
      ArrowRight: time + step,
      ArrowUp: time + step,
      PageDown: time - 10,
      PageUp: time + 10,
      Home: 0,
      End: duration,
    };
    const target = targets[event.key];
    if (target === undefined) return;
    event.preventDefault();
    player.seek(Math.min(Math.max(target, 0), duration));
  };

  return (
    <div
      ref={track}
      className="review-player__seek"
      role="slider"
      tabIndex={disabled ? -1 : 0}
      aria-label="Playback position"
      aria-valuemin={0}
      aria-valuemax={Number(duration.toFixed(1))}
      aria-valuenow={time}
      aria-valuetext={`${formatTimecode(time)} of ${formatTimecode(duration)}`}
      aria-disabled={disabled || undefined}
      data-testid="playhead"
      data-progress={pct(progress * 100)}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onKeyDown={onKeyDown}
    >
      <span className="review-player__seek-rail" aria-hidden="true">
        <span className="review-player__seek-fill" />
        <span className="review-player__seek-thumb" />
      </span>
    </div>
  );
}

/** One layer row: its name, with a check while it is drawn. */
function LayerOption({ label, checked, autofocus, onChange }: { label: string; checked: boolean; autofocus: boolean; onChange: () => void }) {
  return (
    <label className="review-player__speed-option review-player__layer-option">
      <input type="checkbox" checked={checked} data-autofocus={autofocus || undefined} onChange={onChange} className="visually-hidden" />
      <span>{label}</span>
      <Icon icon={icons.check} className="review-player__speed-check review-player__layer-check" />
    </label>
  );
}

/** Which recorded layers to draw, offered only for layers the analysis has. */
function LayerMenu({
  available,
  layers,
  onChange,
}: {
  available: OverlayResponse["layers"];
  layers: OverlayLayers;
  onChange: (update: (layers: OverlayLayers) => OverlayLayers) => void;
}) {
  return (
    <Popover
      panelClassName="review-player__speed-panel review-player__layer-panel"
      trigger={(props) => (
        <button
          type="button"
          className={buttonClass({ variant: "ghost", size: "sm" }, "button--icon")}
          aria-label="Overlay layers"
          {...props}
        >
          <Icon icon={icons.chevronDown} />
        </button>
      )}
    >
      <fieldset className="review-player__speeds review-player__layers">
        <legend className="visually-hidden">Overlay layers</legend>
        {available.tracks ? (
          <LayerOption label="Boxes" checked={layers.boxes} autofocus onChange={() => onChange((l) => ({ ...l, boxes: !l.boxes }))} />
        ) : null}
        {available.lane ? (
          <LayerOption label="Lanes" checked={layers.lane} autofocus={!available.tracks} onChange={() => onChange((l) => ({ ...l, lane: !l.lane }))} />
        ) : null}
      </fieldset>
    </Popover>
  );
}

/** Playback speed as a borderless button opening native radio buttons, like the theme menu. */
function SpeedMenu({ rate, disabled, onChange }: { rate: number; disabled: boolean; onChange: (rate: number) => void }) {
  return (
    <Popover
      panelClassName="review-player__speed-panel"
      trigger={(props) => (
        <button
          type="button"
          className={buttonClass({ variant: "ghost", size: "sm" }, "review-player__speed type-numeric")}
          aria-label={`Playback speed: ${rate}×`}
          disabled={disabled}
          {...props}
        >
          {rate}×
        </button>
      )}
    >
      {(close) => (
        <fieldset className="review-player__speeds">
          <legend className="visually-hidden">Playback speed</legend>
          {RATES.map((option) => (
            <label key={option} className="review-player__speed-option type-numeric">
              <input
                type="radio"
                name="playback-speed"
                value={option}
                checked={rate === option}
                data-autofocus={rate === option || undefined}
                onChange={() => onChange(option)}
                onClick={(event) => event.detail > 0 && close()}
                className="visually-hidden"
              />
              <span>{option}×</span>
              {rate === option ? <Icon icon={icons.check} className="review-player__speed-check" /> : null}
            </label>
          ))}
        </fieldset>
      )}
    </Popover>
  );
}

/** Full screen for the whole player (image, overlays and controls), where the browser allows it. */
function useFullscreen(root: RefObject<HTMLDivElement | null>) {
  const supported = typeof document !== "undefined" && document.fullscreenEnabled === true;
  const [active, setActive] = useState(false);
  useEffect(() => {
    if (!supported) return;
    const onChange = () => setActive(document.fullscreenElement !== null && document.fullscreenElement === root.current);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, [supported, root]);
  const toggle = () => {
    if (document.fullscreenElement) void document.exitFullscreen().catch(() => undefined);
    else void root.current?.requestFullscreen().catch(() => undefined);
  };
  return { supported, active, toggle };
}

function TimeReadout({ player }: { player: Player }) {
  const time = useClock(player.clock, tenths);
  return (
    <span className="review-player__time type-numeric" aria-label={`Position ${formatTimecode(time)}`}>
      {formatTimecode(time)}
      <span className="review-player__duration"> / {formatTimecode(player.state.duration)}</span>
    </span>
  );
}

interface OverlayViewProps extends ReviewPlayerProps {
  layers: OverlayLayers;
  onLayers: (layers: OverlayResponse["layers"]) => void;
}

/** Requests overlay data for the span around the playhead and draws the frame valid at the current time. */
function OverlayView({ player, analysisId, overlayEventId, review, analysed, maxAge, highlightRelated, layers, onLayers }: OverlayViewProps) {
  const area = useClock(
    player.clock,
    useCallback((t: number) => overlayWindow(t, review, analysed, maxAge), [review, analysed, maxAge]),
  );
  const span = area === "review" ? null : chunkSpan(area, analysed);
  const requestable = area === "review" || span !== null;
  const query = useOverlay(analysisId, overlayEventId, span, requestable);
  const data = requestable ? query.data : undefined;
  const frames = data?.overlay.frames;
  const index = useClock(
    player.clock,
    useCallback((t: number) => (frames ? frameIndexAt(frames, t) : null), [frames]),
  );

  useEffect(() => {
    if (data) onLayers(data.layers);
  }, [data, onLayers]);

  let message: string | null = null;
  let frame = null;
  const size = player.state.size;
  if (!requestable) message = "Not analysed here";
  else if (query.isError) message = isOffline(query.error) ? null : "Overlays unavailable";
  else if (!data) message = "Loading overlays…";
  else if (!data.layers.tracks && !data.layers.lane) message = "No overlay layers in this analysis";
  else if (size && !geometryMatches(size.width, size.height, data.overlay.width, data.overlay.height)) {
    message = "Overlay size does not match this video";
  } else if (index === null) message = "Not analysed here";
  else {
    frame = data.overlay.frames[index]!;
    const notes: string[] = [];
    if (layers.boxes && data.layers.tracks) {
      if (frame.tracks === null) notes.push("Tracker did not run here");
      else if (frame.tracks.length === 0) notes.push("No tracked objects");
    }
    if (layers.lane && data.layers.lane && (frame.lane === null || (!frame.lane.left && !frame.lane.right))) {
      notes.push("No lane observed");
    }
    message = notes.length > 0 ? notes.join(" · ") : null;
  }

  return (
    <>
      {frame && data ? (
        <OverlayDrawing
          frame={highlightRelated ? frame : { ...frame, tracks: frame.tracks?.map((t) => ({ ...t, related: false })) ?? null }}
          width={data.overlay.width}
          height={data.overlay.height}
          layers={layers}
        />
      ) : null}
      {message ? <p className="review-player__chip">{message}</p> : null}
    </>
  );
}
