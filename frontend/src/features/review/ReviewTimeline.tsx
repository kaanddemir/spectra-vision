/**
 * The Event timeline: the selected event's review window and the suggested events on an upper lane, what was
 * analysed on a thin lower lane, and the playback position across both, over the video's full duration.
 * Clicking or dragging the timeline seeks the video; it never changes an event's times. Each suggested event
 * is a button that selects it, which also moves the video to it.
 */
import { useCallback, useId, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";

import type { Span, TimelineEntry } from "../../api/types";
import { formatTimecode } from "../../lib/format";
import { tenths, useClock } from "./mediaClock";
import { type AnalysedSpan, coverageRuns, isAnalysedAt, percent } from "./timing";
import type { Player } from "./usePlayer";
import "./ReviewTimeline.css";

interface ReviewTimelineProps {
  player: Player;
  duration: number;
  analysed: AnalysedSpan;
  entries: TimelineEntry[];
  selectedId: string | null;
  /** The selected event's review window, or null without a selection. */
  review: Span | null;
  onSelect: (entry: TimelineEntry) => void;
}

const pct = (value: number) => `${value.toFixed(3)}%`;
const clamp = (value: number, max: number) => Math.min(Math.max(value, 0), max);

export function ReviewTimeline({ player, duration, analysed, entries, selectedId, review, onSelect }: ReviewTimelineProps) {
  const id = useId();
  const track = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const [hover, setHover] = useState<number | null>(null);
  const runs = useMemo(() => coverageRuns(analysed, duration), [analysed, duration]);
  const time = useClock(player.clock, tenths);
  const disabled = player.state.status !== "ready";
  const hasAnalysed = runs.some((run) => run.analysed);
  const hasUnanalysed = runs.some((run) => !run.analysed);

  const timeAt = (clientX: number) => {
    const rect = track.current!.getBoundingClientRect();
    return rect.width > 0 ? clamp((clientX - rect.left) / rect.width, 1) * duration : 0;
  };

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (disabled || event.button !== 0 || (event.target as Element).closest("[data-event]")) return;
    dragging.current = true;
    event.currentTarget.setPointerCapture?.(event.pointerId);
    const t = timeAt(event.clientX);
    setHover(t);
    player.seek(t);
  };
  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (disabled) return;
    const t = timeAt(event.clientX);
    if (dragging.current) player.seek(t);
    if (dragging.current || event.pointerType === "mouse") setHover(t);
  };
  const onPointerUp = (event: PointerEvent<HTMLDivElement>) => {
    dragging.current = false;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
    if (event.pointerType !== "mouse") setHover(null);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (disabled || event.target !== event.currentTarget) return;
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
    player.seek(clamp(target, duration));
  };

  const valueText = `${formatTimecode(time)} of ${formatTimecode(duration)}${isAnalysedAt(analysed, time) ? "" : ", not analysed"}`;

  return (
    <section className="review-timeline" aria-labelledby="review-timeline-heading">
      <h2 id="review-timeline-heading" className="review-timeline__heading type-caption">
        Event timeline
      </h2>
      <div
        ref={track}
        className="review-timeline__track"
        role="slider"
        tabIndex={disabled ? -1 : 0}
        aria-label="Timeline position"
        aria-valuemin={0}
        aria-valuemax={Number(duration.toFixed(1))}
        aria-valuenow={time}
        aria-valuetext={valueText}
        aria-disabled={disabled || undefined}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={() => !dragging.current && setHover(null)}
        onKeyDown={onKeyDown}
      >
        <svg className="review-timeline__svg" width="100%" height="100%" aria-hidden="true">
          <defs>
            <pattern id={`${id}-hatch`} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line className="review-timeline__hatch" x1="0" y1="0" x2="0" y2="6" />
            </pattern>
          </defs>
          <rect className="review-timeline__lane" x="0" width="100%" y="0" height="64%" rx="4" />
          {review ? (
            <rect
              className="review-timeline__window"
              data-window=""
              x={pct(percent(review.start_s, duration))}
              width={pct(Math.max(percent(review.end_s, duration) - percent(review.start_s, duration), 0))}
              y="0"
              height="64%"
              rx="4"
            />
          ) : null}
          {entries.map((entry) => (
            <Marker key={entry.event_id} entry={entry} duration={duration} selected={entry.event_id === selectedId} />
          ))}
          {runs.map((run) => (
            <rect
              key={`${run.start_s}-${run.analysed}`}
              className={run.analysed ? "review-timeline__analysed" : "review-timeline__unanalysed"}
              fill={run.analysed ? undefined : `url(#${id}-hatch)`}
              data-coverage={run.analysed ? "analysed" : "unanalysed"}
              x={pct(percent(run.start_s, duration))}
              width={pct(percent(run.end_s, duration) - percent(run.start_s, duration))}
              y="78%"
              height="16%"
            />
          ))}
          <Playhead player={player} duration={duration} />
        </svg>
        <ul className="review-timeline__events" aria-label="Suggested events on the timeline">
          {entries.map((entry) => {
            const start = percent(entry.t_start, duration);
            const end = percent(entry.t_end, duration);
            return (
              <Positioned key={entry.event_id} as="li" left={pct((start + end) / 2)} width={pct(Math.max(end - start, 0))}>
                <button
                  type="button"
                  className="review-timeline__hit"
                  data-event={entry.event_id}
                  aria-current={entry.event_id === selectedId || undefined}
                  aria-label={`${entry.title}, ${formatTimecode(entry.t_start)} to ${formatTimecode(entry.t_end)}, peak at ${formatTimecode(entry.t_peak)}`}
                  title={`${entry.title} · ${formatTimecode(entry.t_start)}–${formatTimecode(entry.t_end)}`}
                  onClick={() => onSelect(entry)}
                />
              </Positioned>
            );
          })}
        </ul>
        {hover !== null ? (
          <Positioned as="span" left={pct(percent(hover, duration))} className="review-timeline__tip type-numeric">
            {formatTimecode(hover)}
          </Positioned>
        ) : null}
      </div>
      <div className="review-timeline__scale type-caption type-numeric" aria-hidden="true">
        <span>{formatTimecode(0)}</span>
        <span>{formatTimecode(duration)}</span>
      </div>
      <ul className="review-timeline__legend type-caption" aria-label="Timeline key">
        {entries.length > 0 ? (
          <li>
            <span className="review-timeline__swatch review-timeline__swatch--proposal" aria-hidden="true" />
            Suggested event
          </li>
        ) : null}
        {review ? (
          <li>
            <span className="review-timeline__swatch review-timeline__swatch--window" aria-hidden="true" />
            Review window
          </li>
        ) : null}
        {hasAnalysed ? (
          <li>
            <span className="review-timeline__swatch review-timeline__swatch--analysed" aria-hidden="true" />
            Analysed
          </li>
        ) : null}
        {hasUnanalysed ? (
          <li>
            <span className="review-timeline__swatch review-timeline__swatch--unanalysed" aria-hidden="true" />
            Not analysed
          </li>
        ) : null}
      </ul>
    </section>
  );
}

/** The detected interval as a red bar, with a thin line at its peak. */
function Marker({ entry, duration, selected }: { entry: TimelineEntry; duration: number; selected: boolean }) {
  const start = percent(entry.t_start, duration);
  const end = percent(entry.t_end, duration);
  const peak = pct(percent(entry.t_peak, duration));
  return (
    <g
      className={`review-timeline__marker${selected ? " review-timeline__marker--selected" : ""}`}
      data-marker={entry.event_id}
      data-peak={peak}
    >
      <rect className="review-timeline__span" x={pct(start)} width={pct(Math.max(end - start, 0))} y="20%" height="24%" rx="2" />
      <line className="review-timeline__peak" x1={peak} x2={peak} y1="10%" y2="54%" />
    </g>
  );
}

/** The playback position: a thin line across both lanes. */
function Playhead({ player, duration }: { player: Player; duration: number }) {
  const x = useClock(
    player.clock,
    useCallback((t: number) => pct(percent(t, duration)), [duration]),
  );
  return <line className="review-timeline__playhead" data-testid="timeline-playhead" x1={x} x2={x} y1="0" y2="100%" />;
}

/** An element placed along the track; measured geometry is set on the element, not as an inline style. */
function Positioned({
  as: Tag,
  left,
  width,
  className,
  children,
}: {
  as: "li" | "span";
  left: string;
  width?: string;
  className?: string;
  children: ReactNode;
}) {
  const ref = useRef<HTMLElement>(null);
  useLayoutEffect(() => {
    ref.current?.style.setProperty("--at", left);
    if (width !== undefined) ref.current?.style.setProperty("--w", width);
  }, [left, width]);
  return (
    <Tag ref={ref as never} className={className} data-left={left}>
      {children}
    </Tag>
  );
}
