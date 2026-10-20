/** The chronological list of event proposals beside the player, and checks that could not run. */
import { useRef, type KeyboardEvent } from "react";

import type { AnalyzerCoverage, EventEdits, ReviewDecisions, TimelineEntry } from "../../api/types";
import { formatTimecode } from "../../lib/format";
import { Alert } from "../../ui/Alert";
import { Badge } from "../../ui/Badge";
import { EmptyState } from "../../ui/EmptyState";
import { icons } from "../../ui/icons";
import { Section } from "../../ui/Section";
import { editFor, effectiveTimes } from "./EventEdit";
import { analyzerLabel } from "./labels";
import { ReviewStatus, decisionFor } from "./ReviewDecision";
import "./ProposalList.css";

export function EvidenceBadge({ state }: { state: TimelineEntry["state"] }) {
  return state === "supported" ? (
    <Badge tone="neutral" icon={icons.check}>
      Conditions held
    </Badge>
  ) : (
    <Badge tone="warning" icon={icons.attention}>
      Uncertain
    </Badge>
  );
}

interface ProposalListProps {
  entries: TimelineEntry[];
  selectedId: string | null;
  onSelect: (entry: TimelineEntry) => void;
  /** Recorded review decisions; undefined while they load or when they could not be loaded. */
  reviews?: ReviewDecisions;
  /** The reviewer's edits; adjusted times are listed as the reviewer's. */
  edits?: EventEdits;
}

export function ProposalList({ entries, selectedId, onSelect, reviews, edits }: ProposalListProps) {
  const list = useRef<HTMLOListElement>(null);

  // Up and down arrows move between proposals; Enter or Space selects.
  const onKeyDown = (event: KeyboardEvent<HTMLOListElement>) => {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    const buttons = [...(list.current?.querySelectorAll<HTMLButtonElement>("button") ?? [])];
    const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
    if (index < 0) return;
    event.preventDefault();
    buttons[Math.min(Math.max(index + (event.key === "ArrowDown" ? 1 : -1), 0), buttons.length - 1)]?.focus();
  };

  const reviewed = reviews ? entries.filter((e) => decisionFor(reviews, e.event_id).decision !== "not_reviewed").length : null;
  return (
    <Section
      title="Event proposals"
      description={
        entries.length > 0
          ? `Not confirmed incidents${reviewed === null ? "" : ` · ${reviewed} of ${entries.length} reviewed`}`
          : undefined
      }
    >
      {entries.length === 0 ? (
        <EmptyState icon={icons.events} title="No events proposed">
          Review the video for anything that may have been missed.
        </EmptyState>
      ) : (
        <ol ref={list} className="proposal-list" aria-label="Proposals in time order" onKeyDown={onKeyDown}>
          {entries.map((entry) => {
            const selected = entry.event_id === selectedId;
            const decision = decisionFor(reviews, entry.event_id).decision;
            const times = effectiveTimes(entry, editFor(edits, entry.event_id));
            return (
              <li key={entry.event_id}>
                <button
                  type="button"
                  className="proposal-list__item"
                  aria-current={selected || undefined}
                  onClick={() => onSelect(entry)}
                >
                  <span className="proposal-list__time type-numeric">{formatTimecode(entry.t_peak)}</span>
                  <span className="proposal-list__text">
                    <span className="proposal-list__title">{entry.title}</span>
                    <span className="type-caption">
                      {formatTimecode(times.start)} – {formatTimecode(times.end)}
                      {times.adjusted ? " · Adjusted by you" : ""}
                      {entry.experimental ? " · Experimental" : ""}
                    </span>
                    {decision !== "not_reviewed" ? (
                      <span className="proposal-list__review">
                        <ReviewStatus decision={decision} />
                      </span>
                    ) : null}
                  </span>
                  <EvidenceBadge state={entry.state} />
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </Section>
  );
}

/**
 * Checks that did not run or failed. Their silence is not evidence of absence, so a reviewer is told;
 * checks that ran are not listed.
 */
export function CoverageWarning({ analyzers }: { analyzers: AnalyzerCoverage[] }) {
  const missing = analyzers.filter((a) => a.coverage === "unavailable" || a.coverage === "failed");
  if (missing.length === 0) return null;
  return (
    <Alert tone="warning" title={missing.length === 1 ? "One check did not run" : "Some checks did not run"}>
      <ul className="reason-list">
        {missing.map((a) => (
          <li key={a.name}>
            {analyzerLabel(a.name)}: this check {a.coverage === "failed" ? "failed" : "did not run"}, so no such events could be proposed.
          </li>
        ))}
      </ul>
    </Alert>
  );
}
