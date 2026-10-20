import { useEffect, useRef } from "react";

import { isActive, isSettled, useAnalysisLive } from "../../api/analysisLive";
import { ApiError, errorMessage, isOffline } from "../../api/client";
import { useCancelAnalysis, useHealth, useStartAnalysis } from "../../api/queries";
import type { Analysis, Video } from "../../api/types";
import { pluralize, sentence } from "../../lib/format";
import { Alert } from "../../ui/Alert";
import { Badge } from "../../ui/Badge";
import { Button, ButtonLink } from "../../ui/Button";
import { icons } from "../../ui/icons";
import { ProgressBar } from "../../ui/ProgressBar";
import { Section } from "../../ui/Section";
import { StatusText } from "../../ui/StatusText";
import { useToast, type ToastInput } from "../../ui/Toast";
import { analysisStatus, hasResult, stopReason } from "./analysisStatus";

/** The one-time toast for an analysis that settled while this page was open. */
export function outcomeToast(analysis: Analysis): ToastInput {
  const key = `analysis-outcome-${analysis.id}`;
  const proposals = analysis.result ? pluralize(analysis.result.proposals, "proposal") : undefined;
  switch (analysis.state) {
    case "completed":
      return { key, tone: "success", title: "Analysis completed", description: proposals };
    case "incomplete":
      return { key, tone: "warning", title: "Analysis incomplete", description: proposals };
    case "cancelled":
      return { key, tone: "info", title: "Analysis cancelled" };
    case "resource_stopped":
      return { key, tone: "warning", title: "Analysis stopped for memory" };
    default:
      return { key, tone: "danger", title: "Analysis failed", description: stopReason(analysis) ?? undefined };
  }
}

/** Announce outcomes once, and only for analyses seen running on this page. */
export function useOutcomeToast(analysis: Analysis | undefined) {
  const toast = useToast();
  const followed = useRef(new Set<string>());
  useEffect(() => {
    if (!analysis) return;
    if (!isSettled(analysis)) followed.current.add(analysis.id);
    else if (followed.current.delete(analysis.id)) toast(outcomeToast(analysis));
  }, [analysis, toast]);
}

/**
 * Progress of a queued or running analysis: one short line, the reported percentage and a thin bar.
 * Only values the server reports are shown; without a fraction the bar is indeterminate and no number appears.
 */
export function AnalysisProgress({ analysis }: { analysis: Analysis }) {
  const { fraction } = analysis.progress;
  const running = analysis.state === "running";
  const label = analysis.cancel_requested
    ? "Cancelling…"
    : running
      ? "Analysing video…"
      : analysis.queue_position && analysis.queue_position > 1
        ? `Position ${analysis.queue_position} in queue`
        : "Next in queue";
  return (
    <div className="analysis-progress">
      <div className="analysis-progress__row">
        <span className="type-secondary" role="status">
          {label}
        </span>
        {running && fraction !== null ? (
          <span className="analysis-progress__percent type-numeric" aria-hidden="true">
            {Math.round(Math.min(Math.max(fraction, 0), 1) * 100)}%
          </span>
        ) : null}
      </div>
      {running ? <ProgressBar label="Analysis progress" value={fraction} size="sm" /> : null}
    </div>
  );
}

interface AnalysisControlProps {
  video: Video;
  analyses: Analysis[] | undefined;
  /**
   * Whether this browser has itself verified playback. The server's eligibility may rest on a result
   * from another browser; analysis also waits for this one.
   */
  verified: boolean;
}

/**
 * Starts, follows and cancels the video's analysis, and opens the latest result. Reasons about the video
 * itself (source, format, playback) are shown by Analysis readiness, never here; this states only what
 * readiness does not: the server's own reasons (models, memory) and a refused request.
 */
export function AnalysisControl({ video, analyses, verified }: AnalysisControlProps) {
  const health = useHealth();
  const start = useStartAnalysis();
  const cancel = useCancelAnalysis();
  const toast = useToast();
  const latest = analyses?.[0];
  const current = latest && !isSettled(latest) ? latest : undefined;
  useAnalysisLive(current);
  useOutcomeToast(latest);

  const serverReasons = health.data && !health.data.analysis.available ? health.data.analysis.reasons : [];
  const videoReady = video.analysis.eligible && verified;
  const blocked = !videoReady || serverReasons.length > 0;

  // Clicks within the same moment reach the handler before the button is disabled; send one request.
  const sending = useRef(false);

  function onStart() {
    if (sending.current) return;
    sending.current = true;
    start.mutate(video.id, {
      // A new analysis shows its progress in place; only a request that changed nothing is explained.
      onSuccess: ({ created }) => {
        if (!created) toast({ tone: "info", title: "Analysis already in progress" });
      },
      onSettled: () => {
        sending.current = false;
      },
    });
  }

  function onCancel(analysis: Analysis) {
    if (sending.current) return;
    sending.current = true;
    cancel.mutate(analysis.id, {
      onSettled: () => {
        sending.current = false;
      },
      onError: (error) => {
        if (isOffline(error)) return;
        toast(
          error instanceof ApiError && error.status === 409
            ? { tone: "info", key: "cancel-too-late", title: "Already finished" }
            : { tone: "danger", key: "cancel-failed", title: "Could not cancel", description: errorMessage(error) },
        );
      },
    });
  }

  const action = current ? (
    isActive(current) ? (
      <Button
        variant="secondary"
        size="sm"
        icon={icons.close}
        disabled={current.cancel_requested}
        loading={cancel.isPending}
        onClick={() => onCancel(current)}
      >
        {current.cancel_requested ? "Cancelling…" : "Cancel"}
      </Button>
    ) : null
  ) : (
    // After a result, review is the next step and analysing again a secondary one (it adds a new record).
    <Button
      variant={latest && hasResult(latest) ? "secondary" : "primary"}
      size={latest && hasResult(latest) ? "sm" : undefined}
      icon={icons.analyse}
      disabled={blocked || !analyses}
      loading={start.isPending}
      onClick={onStart}
    >
      {latest && hasResult(latest) ? "Analyse again" : "Analyse"}
    </Button>
  );

  return (
    <Section title="Analysis" actions={action}>
      <div className="stack stack--sm">
        {current ? (
          isActive(current) ? (
            <AnalysisProgress analysis={current} />
          ) : (
            <p className="type-secondary" role="status">
              Preparing evidence…
            </p>
          )
        ) : start.isError && !isOffline(start.error) ? (
          <Alert tone="danger" title={errorMessage(start.error)}>
            {start.error instanceof ApiError && start.error.reasons.length > 0 ? (
              <ul>
                {start.error.reasons.map((reason) => (
                  <li key={reason}>{sentence(reason)}</li>
                ))}
              </ul>
            ) : null}
          </Alert>
        ) : serverReasons.length > 0 ? (
          <p className="type-secondary">{sentence(serverReasons[0]!)}</p>
        ) : latest ? (
          <LatestOutcome analysis={latest} />
        ) : null}
      </div>
    </Section>
  );
}

/** The latest analysis: its outcome in a few words and the way into its results. */
function LatestOutcome({ analysis }: { analysis: Analysis }) {
  const status = analysisStatus(analysis);
  const reason = stopReason(analysis);
  const result = hasResult(analysis);
  return (
    <div className="analysis-latest">
      <div className="analysis-latest__text">
        {result ? (
          <StatusText tone={status.tone} icon={status.icon}>
            {status.label}
          </StatusText>
        ) : (
          <Badge tone={status.tone} icon={status.icon}>
            {status.label}
          </Badge>
        )}
        <span className="type-secondary">
          {analysis.result ? `${result ? "· " : ""}${pluralize(analysis.result.proposals, "proposal")}` : reason ?? null}
        </span>
      </div>
      {result && reason ? <p className="type-caption">{reason}</p> : null}
      <ButtonLink
        to={`/analyses/${analysis.id}`}
        variant={result ? "primary" : undefined}
        size={result ? undefined : "sm"}
        className={result ? "analysis-latest__review" : undefined}
        icon={result ? icons.events : icons.open}
      >
        {result ? "Open review" : "View details"}
      </ButtonLink>
    </div>
  );
}
