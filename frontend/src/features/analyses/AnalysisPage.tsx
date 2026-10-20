import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useParams } from "react-router";

import { isActive, useAnalysisLive } from "../../api/analysisLive";
import { ApiError, api, errorMessage, isNotFound, isOffline } from "../../api/client";
import { keys, useAnalysis, useCancelAnalysis } from "../../api/queries";
import type { Analysis } from "../../api/types";
import { QueryStatus } from "../../app/QueryStatus";
import { humanize, sentence } from "../../lib/format";
import { Alert } from "../../ui/Alert";
import { Button, ButtonLink } from "../../ui/Button";
import { EmptyState } from "../../ui/EmptyState";
import { icons } from "../../ui/icons";
import { PageHeader } from "../../ui/PageHeader";
import { StatusText } from "../../ui/StatusText";
import { useToast } from "../../ui/Toast";
import { ExportButton } from "../review/ExportButton";
import { ReviewWorkspace } from "../review/ReviewWorkspace";
import { AnalysisProgress, useOutcomeToast } from "./AnalysisControl";
import { analysisStatus, hasResult, stopReason } from "./analysisStatus";

/** One analysis: live progress while it runs, then its review workspace once evidence is ready. */
export function AnalysisPage() {
  const { analysisId = "" } = useParams();
  const analysis = useAnalysis(analysisId);

  if (analysis.isError && isNotFound(analysis.error)) {
    return (
      <EmptyState
        layout="page"
        icon={icons.attention}
        title="Analysis not found"
        action={
          <ButtonLink to="/videos" icon={icons.back}>
            Back to videos
          </ButtonLink>
        }
      />
    );
  }
  return (
    <QueryStatus query={analysis} loadingLabel="Loading analysis…" errorTitle="The analysis could not be loaded">
      {(data) => <AnalysisView analysis={data} />}
    </QueryStatus>
  );
}

function AnalysisView({ analysis }: { analysis: Analysis }) {
  const status = analysisStatus(analysis);
  const reason = stopReason(analysis);
  const cancel = useCancelAnalysis();
  const toast = useToast();
  useAnalysisLive(analysis);
  useOutcomeToast(analysis);
  const reviewable = hasResult(analysis) && analysis.evidence.state === "ready";
  return (
    <>
      <PageHeader
        back={{ to: `/videos/${analysis.video_id}`, label: "Back to video" }}
        title={analysis.file_name}
        meta={
          // A completed analysis needs no status beside its name; any other state is worth saying.
          analysis.state === "completed" ? null : (
            <StatusText tone={status.tone} icon={status.icon} spin={status.spin}>
              {status.label}
            </StatusText>
          )
        }
        actions={
          isActive(analysis) ? (
            <Button
              icon={icons.close}
              disabled={analysis.cancel_requested}
              loading={cancel.isPending}
              onClick={() =>
                cancel.mutate(analysis.id, {
                  onError: (error) => {
                    if (isOffline(error)) return;
                    toast(
                      error instanceof ApiError && error.status === 409
                        ? { tone: "info", key: "cancel-too-late", title: "Already finished" }
                        : { tone: "danger", key: "cancel-failed", title: "Could not cancel", description: errorMessage(error) },
                    );
                  },
                })
              }
            >
              {analysis.cancel_requested ? "Cancelling…" : "Cancel"}
            </Button>
          ) : reviewable ? (
            <ExportButton analysisId={analysis.id} />
          ) : null
        }
      />
      <div className="stack stack--lg">
        {isActive(analysis) ? <AnalysisProgress analysis={analysis} /> : null}
        {reason || analysis.message ? (
          <Alert
            tone={analysis.state === "failed" ? "danger" : analysis.state === "cancelled" ? "info" : "warning"}
            title={reason ?? humanize(analysis.stop_reason ?? analysis.state)}
          >
            {analysis.message ? sentence(analysis.message) : null}
          </Alert>
        ) : null}
        {reviewable ? (
          <ReviewWorkspace analysis={analysis} />
        ) : hasResult(analysis) ? (
          <EvidenceState analysis={analysis} />
        ) : null}
      </div>
    </>
  );
}

/** Evidence of a finished analysis that is not ready: being prepared, or to be prepared on request. */
function EvidenceState({ analysis }: { analysis: Analysis }) {
  const client = useQueryClient();
  // Requesting the timeline rebuilds missing evidence from the stored result and perception cache.
  const prepare = useMutation({
    mutationFn: () => api.timeline(analysis.id),
    onSuccess: (timeline) => {
      client.setQueryData(keys.timeline(analysis.id), timeline);
      void client.invalidateQueries({ queryKey: keys.analysis(analysis.id) });
    },
  });
  if (analysis.evidence.state === "pending") {
    return (
      <Alert tone="info" title="Preparing evidence">
        Review opens when it is ready.
      </Alert>
    );
  }
  const failed = prepare.isError && !isOffline(prepare.error);
  return (
    <Alert
      tone={failed ? "danger" : "warning"}
      title={failed ? "Evidence could not be prepared" : "Evidence not prepared"}
      actions={
        <Button size="sm" icon={icons.refresh} loading={prepare.isPending} onClick={() => prepare.mutate()}>
          Prepare evidence
        </Button>
      }
    >
      {failed
        ? errorMessage(prepare.error)
        : analysis.evidence.message
          ? sentence(analysis.evidence.message)
          : "Prepare the evidence to review this analysis."}
    </Alert>
  );
}
