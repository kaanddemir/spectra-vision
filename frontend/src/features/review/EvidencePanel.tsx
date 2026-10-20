/**
 * The selected proposal as a short review summary: its type and time, why the analyser proposed it and
 * what remains uncertain, in plain words mapped from the recorded evidence package (see `summary.ts`).
 * The package is loaded and kept whole; its tables, scores and identifiers are simply not displayed.
 * Nothing here is a verdict; the proposal list states once that proposals are not confirmed incidents. The
 * reviewer's own decision sits in the header, between the analyser's status and the event's actions.
 */
import { useState, type ReactNode } from "react";

import { errorMessage, isOffline } from "../../api/client";
import { useEdits, useEvidencePackage } from "../../api/queries";
import type { EvidencePackage, TimelineEntry } from "../../api/types";
import { QueryStatus } from "../../app/QueryStatus";
import { formatTimecode } from "../../lib/format";
import { Alert } from "../../ui/Alert";
import { Button } from "../../ui/Button";
import { Icon } from "../../ui/Icon";
import { icons } from "../../ui/icons";
import { Section } from "../../ui/Section";
import { ClipButton, ClipResult, useClipRequest } from "./ClipControl";
import { EventEditDialog, EventNote, editFor, effectiveTimes, span } from "./EventEdit";
import { RELATED_LABELS } from "./labels";
import { EvidenceBadge } from "./ProposalList";
import { ReviewDecisionControl } from "./ReviewDecision";
import { explanation, uncertainties } from "./summary";
import "./EvidencePanel.css";

interface EvidencePanelProps {
  analysisId: string;
  entry: TimelineEntry;
  entries: TimelineEntry[];
  onSeek: (t: number) => void;
  onSelect: (entry: TimelineEntry) => void;
  /** End of the video in analysis time, for checking edited times. */
  duration: number | null;
}

export function EvidencePanel({ analysisId, entry, entries, onSeek, onSelect, duration }: EvidencePanelProps) {
  const query = useEvidencePackage(analysisId, entry.event_id);
  const clip = useClipRequest({ analysisId, eventId: entry.event_id });
  const edits = useEdits(analysisId);
  const [editing, setEditing] = useState(false);
  const [session, setSession] = useState(0);
  const edit = editFor(edits.data, entry.event_id);
  const times = effectiveTimes(entry, edit);
  return (
    <Section
      title={entry.title}
      inlineDescription
      description={
        <>
          <Button
            size="sm"
            variant="ghost"
            className="event-review__time"
            onClick={() => onSeek(times.start)}
            aria-label={`Play from ${formatTimecode(times.start)}`}
          >
            {span(times.start, times.end)}
          </Button>
          {times.adjusted ? (
            <span className="event-review__adjusted type-caption">
              <Icon icon={icons.edit} />
              Adjusted by you · detected {span(entry.t_start, entry.t_end)}
            </span>
          ) : null}
        </>
      }
      actions={
        <span className="event-review__header-actions">
          <EvidenceBadge state={entry.state} />
          <ReviewDecisionControl analysisId={analysisId} eventId={entry.event_id} />
          <span className="event-review__tools">
            {edits.data ? (
              <Button
                size="sm"
                variant="ghost"
                icon={icons.edit}
                aria-haspopup="dialog"
                onClick={() => {
                  setSession((n) => n + 1);
                  setEditing(true);
                }}
              >
                Edit
              </Button>
            ) : null}
            <ClipButton clip={clip} />
          </span>
        </span>
      }
    >
      <div className="event-review">
        {edits.isError && edits.data === undefined && !isOffline(edits.error) ? (
          <Alert
            tone="danger"
            title="Your edits could not be loaded"
            actions={
              <Button size="sm" icon={icons.refresh} loading={edits.isFetching} onClick={() => void edits.refetch()}>
                Try again
              </Button>
            }
          >
            {errorMessage(edits.error)}
          </Alert>
        ) : null}
        {edit.description ? <EventNote description={edit.description} /> : null}
        <QueryStatus query={query} loadingLabel="Loading evidence…" errorTitle="Evidence could not be loaded">
          {(pk) => <Summary entry={entry} pk={pk} entries={entries} onSelect={onSelect} />}
        </QueryStatus>
        <ClipResult clip={clip} />
      </div>
      {edits.data ? (
        <EventEditDialog
          key={`${entry.event_id}:${session}`}
          open={editing}
          analysisId={analysisId}
          entry={entry}
          edit={edit}
          duration={duration}
          onClose={() => setEditing(false)}
        />
      ) : null}
    </Section>
  );
}

interface SummaryProps {
  entry: TimelineEntry;
  pk: EvidencePackage;
  entries: TimelineEntry[];
  onSelect: (entry: TimelineEntry) => void;
}

function Summary({ entry, pk, entries, onSelect }: SummaryProps) {
  const why = explanation(entry, pk);
  const limits = uncertainties(entry, pk);
  const related = pk.related
    .map((relation) => ({
      relation,
      entry: entries.find((e) => e.event_id === relation.event_id),
    }))
    .filter((r): r is { relation: typeof r.relation; entry: TimelineEntry } => r.entry !== undefined);

  return (
    <>
      <div className={`event-review__columns${limits.length > 0 ? "" : " event-review__columns--single"}`}>
        <Part title="Why this event?">
          <p>{why.join(" ")}</p>
        </Part>
        {limits.length > 0 ? (
          <Part title="Uncertainty">
            <p>{limits.join(" ")}</p>
          </Part>
        ) : null}
      </div>
      {related.length > 0 ? (
        <Part title="Related events">
          <ul className="row-list">
            {related.map(({ relation, entry: other }) => (
              <li key={`${relation.kind}-${other.event_id}`} className="row-list__item">
                <span className="row-list__text">
                  <span className="type-body">
                    {RELATED_LABELS[relation.kind]} {other.title.toLowerCase()}
                  </span>
                  <span className="type-caption">{formatTimecode(other.t_peak)}</span>
                </span>
                <Button size="sm" variant="ghost" onClick={() => onSelect(other)}>
                  Select
                </Button>
              </li>
            ))}
          </ul>
        </Part>
      ) : null}
    </>
  );
}

function Part({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="event-review__part" aria-label={title}>
      <h3 className="type-subheading">{title}</h3>
      <div className="type-secondary">{children}</div>
    </section>
  );
}
