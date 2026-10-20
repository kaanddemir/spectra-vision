/**
 * The reviewer's own decision on a proposal: Accepted, Rejected or Not reviewed. It is kept apart from the
 * analyser's status ("Conditions held", "Uncertain"), drawn with a person icon, and never states that an
 * incident occurred or who was at fault. Changes are sent with the revision on screen; the display shows
 * only what the server stored, never an assumed outcome.
 */
import { useRef } from "react";

import { ApiError, errorMessage, isOffline } from "../../api/client";
import { useReviews, useSetReview } from "../../api/queries";
import type { ReviewDecision, ReviewDecisionName, ReviewDecisions } from "../../api/types";
import { Alert } from "../../ui/Alert";
import type { Tone } from "../../ui/Badge";
import { Button } from "../../ui/Button";
import { icons } from "../../ui/icons";
import { Spinner } from "../../ui/Spinner";
import { StatusText } from "../../ui/StatusText";
import "./ReviewDecision.css";

export const DECISION_LABELS: Record<ReviewDecisionName, { label: string; tone: Tone; icon: (typeof icons)[keyof typeof icons] }> = {
  not_reviewed: { label: "Not reviewed", tone: "neutral", icon: icons.notReviewed },
  accepted: { label: "Accepted", tone: "success", icon: icons.accepted },
  rejected: { label: "Rejected", tone: "neutral", icon: icons.rejected },
};

/** The stored decision for a proposal; a proposal without a record is not reviewed. */
export function decisionFor(reviews: ReviewDecisions | undefined, eventId: string): ReviewDecision {
  return (
    reviews?.decisions.find((d) => d.event_id === eventId) ?? { event_id: eventId, decision: "not_reviewed", revision: 0, updated_at: null }
  );
}

/** A decision as icon and text, for the proposal list. */
export function ReviewStatus({ decision }: { decision: ReviewDecisionName }) {
  const { label, tone, icon } = DECISION_LABELS[decision];
  return (
    <StatusText tone={tone} icon={icon} description="your review">
      {label}
    </StatusText>
  );
}

interface ReviewDecisionControlProps {
  analysisId: string;
  eventId: string;
}

export function ReviewDecisionControl({ analysisId, eventId }: ReviewDecisionControlProps) {
  const reviews = useReviews(analysisId);
  const mutation = useSetReview(analysisId);
  const sending = useRef(false);
  const current = decisionFor(reviews.data, eventId);
  const { label, tone, icon } = DECISION_LABELS[current.decision];

  const decide = (decision: ReviewDecisionName) => {
    if (sending.current) return; // repeated clicks before the buttons update send one request
    sending.current = true;
    mutation.mutate(
      { eventId, decision, expectedRevision: current.revision },
      {
        onSettled: () => {
          sending.current = false;
        },
      },
    );
  };
  // A pressed toggle returns the proposal to not reviewed.
  const toggle = (decision: ReviewDecisionName) => decide(current.decision === decision ? "not_reviewed" : decision);
  const pending = mutation.isPending ? mutation.variables.decision : null;

  let body;
  if (reviews.isPending) {
    body = (
      <span className="review-decision__state type-secondary" role="status">
        <Spinner size="sm" />
        Loading decision…
      </span>
    );
  } else if (reviews.data === undefined) {
    body = isOffline(reviews.error) ? (
      <p className="type-secondary">Decisions are unavailable while Spectra is not responding.</p>
    ) : (
      <Alert
        tone="danger"
        title="Decisions could not be loaded"
        actions={
          <Button size="sm" icon={icons.refresh} loading={reviews.isFetching} onClick={() => void reviews.refetch()}>
            Try again
          </Button>
        }
      >
        {errorMessage(reviews.error)}
      </Alert>
    );
  } else {
    body = (
      <>
        <span className={`review-decision__state review-decision__status review-decision__status--${current.decision}`} aria-live="polite">
          <StatusText tone={tone} icon={icon}>
            {label}
          </StatusText>
        </span>
        <span className="review-decision__actions" role="group" aria-label="Decide">
          <Button
            size="sm"
            variant="ghost"
            className="review-decision__action review-decision__action--accept"
            icon={icons.accepted}
            aria-pressed={current.decision === "accepted"}
            loading={pending === "accepted"}
            disabled={mutation.isPending}
            onClick={() => toggle("accepted")}
          >
            Accept
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="review-decision__action review-decision__action--reject"
            icon={icons.rejected}
            aria-pressed={current.decision === "rejected"}
            loading={pending === "rejected"}
            disabled={mutation.isPending}
            onClick={() => toggle("rejected")}
          >
            Reject
          </Button>
          {current.decision !== "not_reviewed" ? (
            <Button
              size="sm"
              variant="ghost"
              icon={icons.clearDecision}
              loading={pending === "not_reviewed"}
              disabled={mutation.isPending}
              onClick={() => decide("not_reviewed")}
            >
              Clear
            </Button>
          ) : null}
        </span>
      </>
    );
  }

  return (
    <section className="review-decision" aria-label="Your review">
      <div className="review-decision__row">{body}</div>
      <SaveError error={mutation.isError ? mutation.error : null} />
    </section>
  );
}

function SaveError({ error }: { error: unknown }) {
  if (error === null) return null;
  if (isOffline(error)) return <p className="type-secondary">Not saved. Try again when Spectra responds.</p>;
  if (error instanceof ApiError && error.status === 409) {
    return (
      <Alert tone="warning" title="Not saved">
        {errorMessage(error)}
      </Alert>
    );
  }
  return (
    <Alert tone="danger" title="Not saved">
      {errorMessage(error)}
    </Alert>
  );
}
