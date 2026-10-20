import type { UseQueryResult } from "@tanstack/react-query";
import type { ReactNode } from "react";

import { errorMessage, isOffline } from "../api/client";
import { useHealth } from "../api/queries";
import { formatClock } from "../lib/format";
import { Alert } from "../ui/Alert";
import { Button } from "../ui/Button";
import { icons } from "../ui/icons";
import { LoadingState } from "../ui/Spinner";

interface QueryStatusProps<T> {
  query: UseQueryResult<T>;
  loadingLabel: string;
  /** Shown when the request fails before any data arrived. */
  errorTitle: string;
  children: (data: T) => ReactNode;
}

/**
 * The one pattern for server data: a loading message, an error with a retry action, or the data.
 * When a refresh fails, earlier data stays visible under a warning that says how old it is, so
 * it is never presented as current. An unreachable server is reported once, by the global banner
 * (which also says shown data may be out of date): while the health check confirms it, or is checking,
 * this shows nothing more for it.
 */
export function QueryStatus<T>({ query, loadingLabel, errorTitle, children }: QueryStatusProps<T>) {
  const health = useHealth();
  if (query.isPending) return <LoadingState label={loadingLabel} />;
  const retry = (
    <Button size="sm" icon={icons.refresh} loading={query.isFetching} onClick={() => void query.refetch()}>
      Try again
    </Button>
  );
  const offline = query.isError && isOffline(query.error) && (health.isError || health.isFetching);
  if (offline && query.data === undefined) return null;
  if (query.isError && query.data === undefined) {
    return (
      <Alert tone="danger" title={errorTitle} actions={retry}>
        {errorMessage(query.error)}
      </Alert>
    );
  }
  return (
    <>
      {query.isError && !offline ? <StaleNotice error={query.error} updatedAt={query.dataUpdatedAt} action={retry} /> : null}
      {children(query.data as T)}
    </>
  );
}

function StaleNotice({ error, updatedAt, action }: { error: unknown; updatedAt: number; action: ReactNode }) {
  const reason = errorMessage(error);
  return (
    <Alert tone="warning" title={`Last updated ${formatClock(updatedAt)}`} actions={action}>
      {reason} Data below may be out of date.
    </Alert>
  );
}
