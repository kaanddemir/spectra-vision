import { useQueries, useQueryClient } from "@tanstack/react-query";
import { useMemo } from "react";
import { Link, useSearchParams } from "react-router";

import { api, isOffline } from "../../api/client";
import { keys, mergeDecisions, useAnalyses } from "../../api/queries";
import type { Analysis, EventType, ReviewDecisions, Timeline, TimelineEntry } from "../../api/types";
import { QueryStatus } from "../../app/QueryStatus";
import { formatTimecode, pluralize } from "../../lib/format";
import { Alert } from "../../ui/Alert";
import { Badge } from "../../ui/Badge";
import { ButtonLink } from "../../ui/Button";
import { EmptyState, NoResults } from "../../ui/EmptyState";
import { Icon } from "../../ui/Icon";
import { icons } from "../../ui/icons";
import { PageHeader } from "../../ui/PageHeader";
import { LoadingState, Spinner } from "../../ui/Spinner";
import { Table } from "../../ui/Table";
import { Tabs } from "../../ui/Tabs";
import { hasResult } from "../analyses/analysisStatus";
import { decisionFor, ReviewStatus } from "../review/ReviewDecision";
import { proposalPath } from "../review/ReviewWorkspace";

type Filter = "all" | EventType;
const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "rapid_approach", label: "Rapid approach" },
  { id: "cut_in", label: "Cut-in" },
  { id: "impact", label: "Impact" },
];
const TABLE_ID = "events-table";
const isFilter = (value: string | null): value is Filter => FILTERS.some((f) => f.id === value);

/** The newest analysis with a result for each video; older results of the same video are superseded. */
export function latestResults(analyses: Analysis[]): Analysis[] {
  const latest = new Map<string, Analysis>();
  for (const analysis of analyses) {
    if (!hasResult(analysis)) continue;
    const current = latest.get(analysis.video_id);
    if (!current || analysis.created_at > current.created_at) latest.set(analysis.video_id, analysis);
  }
  return [...latest.values()];
}

interface EventRow {
  entry: TimelineEntry;
  analysis: Analysis;
  /** The analysis's stored review decisions; undefined while loading, null when they could not be read. */
  reviews: ReviewDecisions | null | undefined;
}

export function EventsPage() {
  const analyses = useAnalyses();
  return (
    <>
      <PageHeader
        title="Events"
        description="Browse event proposals across your videos."
      />
      <QueryStatus query={analyses} loadingLabel="Loading analyses…" errorTitle="Analyses could not be loaded">
        {(list) => <EventLibrary analyses={list} />}
      </QueryStatus>
    </>
  );
}

function EventLibrary({ analyses }: { analyses: Analysis[] }) {
  const results = useMemo(() => latestResults(analyses), [analyses]);
  // Only stored timelines are read here; a failed or pending build is never rebuilt from this list.
  const ready = results.filter((a) => a.evidence.state === "ready");
  const unavailable = results.length - ready.length;
  const timelines = useQueries({
    queries: ready.map((analysis) => ({
      queryKey: keys.timeline(analysis.id),
      queryFn: ({ signal }: { signal: AbortSignal }) => api.timeline(analysis.id, signal),
      staleTime: Infinity,
    })),
  });

  // The reviewer's decisions, read through the same cache as the Review workspace, so a decision made there
  // shows here on return; they are fetched again whenever this page opens.
  const client = useQueryClient();
  const reviews = useQueries({
    queries: ready.map((analysis) => ({
      queryKey: keys.reviews(analysis.id),
      queryFn: async ({ signal }: { signal: AbortSignal }) =>
        mergeDecisions(client.getQueryData<ReviewDecisions>(keys.reviews(analysis.id)), await api.reviews(analysis.id, signal)),
      staleTime: 0,
    })),
  });

  if (results.length === 0) {
    return (
      <EmptyState
        layout="page"
        icon={icons.events}
        title="No analysed videos yet"
        action={
          <ButtonLink to="/videos" icon={icons.videos}>
            Go to videos
          </ButtonLink>
        }
      >
        Proposals appear here after analysis.
      </EmptyState>
    );
  }
  if (timelines.some((t) => t.isPending)) return <LoadingState label="Loading event proposals…" />;

  const rows: EventRow[] = [];
  let failed = 0;
  let offline = 0; // reported once by the global banner, never as missing evidence
  timelines.forEach((query, index) => {
    const timeline: Timeline | undefined = query.data;
    if (!timeline) {
      if (isOffline(query.error)) offline += 1;
      else failed += 1;
      return;
    }
    const review = reviews[index]!;
    const decisions = review.data ?? (review.isError ? null : undefined);
    for (const entry of timeline.entries) rows.push({ entry, analysis: ready[index]!, reviews: decisions });
  });

  // Unread timelines are not evidence of no proposals: never say "No proposals" while some could not be read.
  if (offline > 0 && rows.length === 0) return null;

  // With nothing else to report, the empty state fills the page and is centred in it.
  if (rows.length === 0 && unavailable + failed === 0) return <NoProposals layout="page" />;

  return (
    <div className="stack">
      {unavailable + failed > 0 ? (
        <Alert tone="warning" title={`Evidence for ${pluralize(unavailable + failed, "analysis", "analyses")} is not available`}>
          Its proposals are not listed.
        </Alert>
      ) : null}
      <EventTable rows={rows} />
    </div>
  );
}

function EventTable({ rows }: { rows: EventRow[] }) {
  const [params, setParams] = useSearchParams();
  const raw = params.get("type");
  const filter: Filter = isFilter(raw) ? raw : "all";
  const visible = rows
    .filter((row) => filter === "all" || row.entry.type === filter)
    .sort((a, b) => a.analysis.file_name.localeCompare(b.analysis.file_name) || a.entry.t_peak - b.entry.t_peak);

  if (rows.length === 0) {
    return (
      <NoProposals layout="inline" />
    );
  }

  return (
    <div className="stack">
      <Tabs
        label="Filter events by type"
        controls={TABLE_ID}
        value={filter}
        onChange={(type) => setParams(type === "all" ? {} : { type }, { replace: true })}
        items={FILTERS.map((f) => ({ ...f, count: rows.filter((r) => f.id === "all" || r.entry.type === f.id).length }))}
      />
      <div id={TABLE_ID} role="tabpanel" aria-labelledby={`${TABLE_ID}-tab-${filter}`} className="stack">
        {visible.length === 0 ? (
          <NoResults onClear={filter !== "all" ? () => setParams({}, { replace: true }) : undefined} />
        ) : (
          <Table label="Event proposals" minWidth="lg" columns="fit">
            <thead>
              <tr>
                <th scope="col" className="cell-primary">
                  Event
                </th>
                <th scope="col" className="cell-flex">
                  Video
                </th>
                <th scope="col" className="cell-numeric">
                  Time
                </th>
                <th scope="col">Status</th>
                <th scope="col">Your review</th>
              </tr>
            </thead>
            <tbody>
              {visible.map(({ entry, analysis, reviews }) => (
                <tr key={`${analysis.id}-${entry.event_id}`}>
                  <td className="cell-primary">
                    <div className="cell-primary__content">
                      <Icon icon={icons.events} size="md" className="cell-primary__icon" />
                      <div className="cell-primary__text">
                        <Link className="cell-primary__title truncate" to={proposalPath(analysis.id, entry.event_id)} title={entry.title}>
                          {entry.title}
                        </Link>
                        {entry.experimental ? <span className="cell-primary__subtitle">Experimental check</span> : null}
                      </div>
                    </div>
                  </td>
                  <td className="cell-flex">
                    <Link className="cell-primary__title truncate" to={`/videos/${analysis.video_id}`} title={analysis.file_name}>
                      {analysis.file_name}
                    </Link>
                  </td>
                  {/* The detected interval, as proposed; a reviewer's adjusted times are shown in the review. */}
                  <td className="cell-numeric">
                    {formatTimecode(entry.t_start)} – {formatTimecode(entry.t_end)}
                  </td>
                  <td>
                    {entry.state === "supported" ? (
                      <Badge tone="neutral" icon={icons.check}>
                        Conditions held
                      </Badge>
                    ) : (
                      <Badge tone="warning" icon={icons.attention}>
                        Uncertain
                      </Badge>
                    )}
                  </td>
                  <td>
                    {reviews === undefined ? (
                      <span className="cell-secondary">
                        <Spinner />
                        <span className="visually-hidden">Loading your review…</span>
                      </span>
                    ) : reviews === null ? (
                      <span className="cell-secondary">Unavailable</span>
                    ) : (
                      <ReviewStatus decision={decisionFor(reviews, entry.event_id).decision} />
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </div>
    </div>
  );
}

/** Analyses ran but proposed nothing; that is no evidence that nothing happened. */
function NoProposals({ layout }: { layout: "inline" | "page" }) {
  return (
    <EmptyState layout={layout} icon={icons.events} title="No events proposed">
      No events were suggested for review. You can still check the videos yourself.
    </EmptyState>
  );
}
