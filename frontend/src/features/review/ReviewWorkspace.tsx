/**
 * The review workspace of an analysis with ready evidence: the source video with its toolbar and the
 * analysis timeline, the event proposals with the reviewer's decisions beside it, and the selected proposal's
 * evidence and decision below. The selected proposal lives in the URL (`?event=`), so links to a proposal
 * survive a reload.
 */
import { useCallback, useEffect, useMemo, useRef } from "react";
import { useSearchParams } from "react-router";

import { errorMessage, isNotFound } from "../../api/client";
import { useEdits, useReviews, useTimeline, useVideo } from "../../api/queries";
import type { Analysis, Timeline, TimelineEntry, Video } from "../../api/types";
import { QueryStatus } from "../../app/QueryStatus";
import { formatTimecode, sentence } from "../../lib/format";
import { Alert } from "../../ui/Alert";
import { EmptyState } from "../../ui/EmptyState";
import { icons } from "../../ui/icons";
import { LoadingState } from "../../ui/Spinner";
import { editFor, effectiveTimes } from "./EventEdit";
import { EvidencePanel } from "./EvidencePanel";
import { CoverageWarning, ProposalList } from "./ProposalList";
import { ReviewPlayer } from "./ReviewPlayer";
import { ReviewTimeline } from "./ReviewTimeline";
import type { AnalysedSpan } from "./timing";
import { usePlayer } from "./usePlayer";

export const EVENT_PARAM = "event";

/** A link to one proposal in its analysis. */
export const proposalPath = (analysisId: string, eventId: string) =>
  `/analyses/${analysisId}?${new URLSearchParams({ [EVENT_PARAM]: eventId })}`;

export function ReviewWorkspace({ analysis }: { analysis: Analysis }) {
  const timeline = useTimeline(analysis.id, true);
  return (
    <QueryStatus query={timeline} loadingLabel="Loading review timeline…" errorTitle="The review timeline could not be loaded">
      {(data) => <Workspace analysis={analysis} timeline={data} />}
    </QueryStatus>
  );
}

/** Why the source cannot be played for this analysis, or an empty list when it can. */
export function sourceProblems(analysis: Analysis, video: Video | undefined, error: unknown): string[] {
  if (error) return [isNotFound(error) ? "The video is no longer registered." : errorMessage(error)];
  if (!video) return [];
  if (video.media && video.media.sha256 !== analysis.source_sha256) {
    return ["The video file changed after this analysis, so its evidence no longer matches it."];
  }
  if (video.media_url) return [];
  const reasons = [video.source.message, ...video.playback.reasons].filter((r): r is string => Boolean(r)).map(sentence);
  return reasons.length > 0 ? reasons : ["The server does not stream this video to browsers."];
}

function Workspace({ analysis, timeline }: { analysis: Analysis; timeline: Timeline }) {
  const video = useVideo(analysis.video_id);
  const reviews = useReviews(analysis.id);
  const edits = useEdits(analysis.id);
  const editsData = edits.data;
  const [params, setParams] = useSearchParams();
  const requested = params.get(EVENT_PARAM);
  const entries = timeline.entries;
  const selected = entries.find((e) => e.event_id === requested) ?? null;
  const player = usePlayer(timeline.video?.start_s ?? 0, timeline.video?.duration_s ?? null);
  const { seek } = player;

  const problems = sourceProblems(analysis, video.data, video.error);
  const src = problems.length === 0 && video.data ? video.data.media_url : null;

  const analysed: AnalysedSpan = useMemo(
    () => ({ analysed_start_s: timeline.analysed_start_s, analysed_end_s: timeline.analysed_end_s, gaps: timeline.gaps }),
    [timeline],
  );
  const duration =
    player.state.duration ??
    timeline.video?.duration_s ??
    Math.max(timeline.analysed_end_s ?? 0, ...entries.map((e) => e.review.end_s), 1);

  const select = useCallback(
    (entry: TimelineEntry) => {
      setParams(
        (current) => {
          const next = new URLSearchParams(current);
          next.set(EVENT_PARAM, entry.event_id);
          return next;
        },
        { replace: true },
      );
      // From the playback context, or earlier when the reviewer's adjusted start lies before it.
      seek(Math.min(entry.review.start_s, effectiveTimes(entry, editFor(editsData, entry.event_id)).start));
    },
    [setParams, seek, editsData],
  );

  // A proposal named in the address is shown from its review span once, when the page opens.
  const opened = useRef(false);
  useEffect(() => {
    if (opened.current || !selected) return;
    opened.current = true;
    seek(selected.review.start_s);
  }, [selected, seek]);

  const review = useMemo(
    () => (selected ? { start_s: selected.review.start_s, end_s: selected.review.end_s } : null),
    [selected],
  );

  return (
    <div className="stack stack--lg">
      {timeline.warnings.length > 0 ? (
        <Alert tone="warning" title="Decoding warnings">
          <ul className="reason-list">
            {timeline.warnings.map((w) => (
              <li key={`${w.code}-${w.t}`}>{w.t === null ? sentence(w.message) : `${formatTimecode(w.t)}: ${sentence(w.message)}`}</li>
            ))}
          </ul>
        </Alert>
      ) : null}
      <div className="workspace">
        <div className="workspace__media">
          {video.isPending ? (
            <div className="media-stage">
              <LoadingState label="Loading video…" />
            </div>
          ) : (
            <ReviewPlayer
              player={player}
              src={src}
              unavailable={problems}
              analysisId={analysis.id}
              overlayEventId={selected?.event_id ?? entries[0]?.event_id ?? null}
              review={review}
              analysed={analysed}
              maxAge={timeline.config.overlay_max_age_s}
              highlightRelated={selected !== null}
              hasAudio={(video.data?.media?.audio_codecs.length ?? 0) > 0}
            />
          )}
          <ReviewTimeline
            player={player}
            duration={duration}
            analysed={analysed}
            entries={entries}
            selectedId={selected?.event_id ?? null}
            review={review}
            onSelect={select}
          />
        </div>
        <div className="workspace__aside">
          <ProposalList entries={entries} selectedId={selected?.event_id ?? null} onSelect={select} reviews={reviews.data} edits={edits.data} />
          <CoverageWarning analyzers={timeline.analyzers} />
        </div>
        <div className="workspace__below stack stack--lg">
          {requested && !selected ? (
            <Alert tone="warning" title="Proposal not found">
              This analysis has no proposal with that link. Choose one from the list.
            </Alert>
          ) : null}
          {selected ? (
            <EvidencePanel
              key={selected.event_id}
              analysisId={analysis.id}
              entry={selected}
              entries={entries}
              onSeek={seek}
              onSelect={select}
              duration={timeline.video?.duration_s ?? timeline.analysed_end_s}
            />
          ) : entries.length > 0 && !requested ? (
            <EmptyState icon={icons.events} title="Select an event to review" layout="compact">
              Choose an event from the list or timeline.
            </EmptyState>
          ) : null}
        </div>
      </div>
    </div>
  );
}
