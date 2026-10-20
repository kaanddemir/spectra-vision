/** Server state: query keys, polling rules and mutations shared by every screen. */
import {
  MutationCache,
  QueryCache,
  QueryClient,
  useMutation,
  useQuery,
  useQueryClient,
  type DefaultOptions,
} from "@tanstack/react-query";

import { storeAnalysis } from "./analysisLive";
import { ApiError, api, isOffline } from "./client";
import type { Analysis, EventEdit, EventEdits, ReviewDecision, ReviewDecisionName, ReviewDecisions, Video } from "./types";

export const keys = {
  health: ["health"] as const,
  folders: ["library"] as const,
  entries: (folderId: string, path: string) => ["library", folderId, path] as const,
  videos: ["videos"] as const,
  video: (videoId: string) => ["videos", videoId] as const,
  analyses: (videoId?: string) => (videoId ? (["analyses", { videoId }] as const) : (["analyses"] as const)),
  analysis: (analysisId: string) => ["analysis", analysisId] as const,
  timeline: (analysisId: string) => ["timeline", analysisId] as const,
  evidencePackage: (analysisId: string, eventId: string) => ["package", analysisId, eventId] as const,
  overlay: (analysisId: string, eventId: string, span: { start_s: number; end_s: number } | null) =>
    ["overlay", analysisId, eventId, span?.start_s ?? null, span?.end_s ?? null] as const,
  reviews: (analysisId: string) => ["reviews", analysisId] as const,
  edits: (analysisId: string) => ["edits", analysisId] as const,
};

/** Polling intervals. Probing is short-lived; health changes slowly. */
export const POLL_MS = { probe: 1000, health: 15000, analyses: 5000 } as const;

/** The application's client. `overrides` (tests: no retries) replace individual defaults. */
export function createQueryClient(overrides: DefaultOptions = {}): QueryClient {
  // Any request that finds the server unreachable checks health at once, so the global banner (the one
  // place that reports it) appears without waiting for the next health poll.
  const checkHealth = (error: unknown) => {
    if (isOffline(error) && !client.isFetching({ queryKey: keys.health })) {
      void client.invalidateQueries({ queryKey: keys.health });
    }
  };
  const client: QueryClient = new QueryClient({
    queryCache: new QueryCache({ onError: (error, query) => query.queryKey[0] !== keys.health[0] && checkHealth(error) }),
    mutationCache: new MutationCache({ onError: checkHealth }),
    defaultOptions: {
      queries: {
        staleTime: 2000,
        // A 4xx answer will not change by asking again; an unreachable server might.
        retry: (count, error) => !(error instanceof ApiError) && count < 1,
        refetchOnWindowFocus: true,
        ...overrides.queries,
      },
      mutations: { retry: false, ...overrides.mutations },
    },
  });
  return client;
}

const probing = (video: Video) => video.probe.state === "pending";

export function useHealth() {
  return useQuery({
    queryKey: keys.health,
    queryFn: ({ signal }) => api.health(signal),
    refetchInterval: POLL_MS.health,
  });
}

export function useLibraryFolders() {
  return useQuery({ queryKey: keys.folders, queryFn: ({ signal }) => api.libraryFolders(signal) });
}

export function useAddLibraryFolder() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => api.addLibraryFolder(),
    onSettled: () => void client.invalidateQueries({ queryKey: keys.folders }),
  });
}

/** Removes a folder from the library only: files, registrations and analyses are kept. */
export function useRemoveLibraryFolder() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (folderId: string) => api.removeLibraryFolder(folderId),
    onSettled: () => {
      void client.invalidateQueries({ queryKey: keys.folders });
      void client.invalidateQueries({ queryKey: keys.videos });
    },
  });
}

export function useLibraryEntries(folderId: string | undefined, path: string) {
  return useQuery({
    queryKey: keys.entries(folderId ?? "", path),
    queryFn: ({ signal }) => api.libraryEntries(folderId!, path, signal),
    enabled: Boolean(folderId),
  });
}

export function useVideos() {
  return useQuery({
    queryKey: keys.videos,
    queryFn: ({ signal }) => api.videos(signal),
    refetchInterval: (query) => (query.state.data?.some(probing) ? POLL_MS.probe : false),
  });
}

export function useVideo(videoId: string) {
  return useQuery({
    queryKey: keys.video(videoId),
    queryFn: ({ signal }) => api.video(videoId, signal),
    refetchInterval: (query) => (query.state.data && probing(query.state.data) ? POLL_MS.probe : false),
  });
}

const active = (analysis: Analysis) => analysis.state === "queued" || analysis.state === "running";

/**
 * Analyses, newest first. Screens that follow an analysis live (`useAnalysisLive`) pass
 * `poll: false`; the overall list polls slowly while anything is active.
 */
export function useAnalyses(videoId?: string, { poll = true }: { poll?: boolean } = {}) {
  return useQuery({
    queryKey: keys.analyses(videoId),
    queryFn: ({ signal }) => api.analyses(videoId, signal),
    refetchInterval: (query) => (poll && query.state.data?.some(active) ? POLL_MS.analyses : false),
  });
}

/** One analysis; live updates come from `useAnalysisLive`. */
export function useAnalysis(analysisId: string) {
  return useQuery({
    queryKey: keys.analysis(analysisId),
    queryFn: ({ signal }) => api.analysis(analysisId, signal),
  });
}

/**
 * Stored evidence never changes for an analysis, so it is fetched once. `enabled` is false until the
 * analysis reports its evidence as ready: a timeline request for any other state may rebuild evidence,
 * which happens only on an explicit request.
 */
export function useTimeline(analysisId: string, enabled: boolean) {
  return useQuery({
    queryKey: keys.timeline(analysisId),
    queryFn: ({ signal }) => api.timeline(analysisId, signal),
    staleTime: Infinity,
    enabled,
  });
}

/** One proposal's evidence package. Switching proposals cancels the previous request. */
export function useEvidencePackage(analysisId: string, eventId: string | null) {
  return useQuery({
    queryKey: keys.evidencePackage(analysisId, eventId ?? ""),
    queryFn: ({ signal }) => api.evidencePackage(analysisId, eventId!, signal),
    staleTime: Infinity,
    enabled: eventId !== null,
  });
}

/** Overlay frames for one span; only requested while overlays are shown. */
export function useOverlay(
  analysisId: string,
  eventId: string | null,
  span: { start_s: number; end_s: number } | null,
  enabled: boolean,
) {
  return useQuery({
    queryKey: keys.overlay(analysisId, eventId ?? "", span),
    queryFn: ({ signal }) => api.overlay(analysisId, eventId!, span, signal),
    staleTime: Infinity,
    gcTime: 60_000,
    enabled: enabled && eventId !== null,
  });
}

export function useCreateClip() {
  return useMutation({
    mutationFn: ({ analysisId, eventId }: { analysisId: string; eventId: string }) => api.createClip(analysisId, eventId),
  });
}

/**
 * Decisions only move forward: each proposal keeps the record with the highest revision, so a response that
 * was overtaken (a refresh started before a change was saved) never replaces a newer decision.
 */
export function mergeDecisions(current: ReviewDecisions | undefined, incoming: ReviewDecisions | ReviewDecision): ReviewDecisions {
  const records = "decisions" in incoming ? incoming.decisions : [incoming];
  const analysisId = "decisions" in incoming ? incoming.analysis_id : (current?.analysis_id ?? "");
  return { analysis_id: analysisId, decisions: newest(current?.decisions ?? [], records) };
}

/** Edits move forward by revision in the same way as decisions. */
export function mergeEdits(current: EventEdits | undefined, incoming: EventEdits | EventEdit): EventEdits {
  const records = "edits" in incoming ? incoming.edits : [incoming];
  const analysisId = "edits" in incoming ? incoming.analysis_id : (current?.analysis_id ?? "");
  return { analysis_id: analysisId, edits: newest(current?.edits ?? [], records) };
}

function newest<T extends { event_id: string; revision: number }>(current: T[], incoming: T[]): T[] {
  const byEvent = new Map(current.map((r) => [r.event_id, r]));
  for (const record of incoming) {
    const known = byEvent.get(record.event_id);
    if (!known || record.revision > known.revision) byEvent.set(record.event_id, record);
  }
  return [...byEvent.values()].filter((r) => r.revision > 0);
}

/** A reviewer's edits of one analysis, refreshed when the window regains focus. */
export function useEdits(analysisId: string) {
  const client = useQueryClient();
  return useQuery({
    queryKey: keys.edits(analysisId),
    queryFn: async ({ signal }) => {
      const fresh = await api.edits(analysisId, signal);
      return mergeEdits(client.getQueryData<EventEdits>(keys.edits(analysisId)), fresh);
    },
  });
}

/** Replace a proposal's edit, based on the revision on screen; a 409 or 404 reloads the edits. */
export function useSetEdit(analysisId: string) {
  const client = useQueryClient();
  const key = keys.edits(analysisId);
  return useMutation({
    mutationFn: ({ eventId, edit, expectedRevision }: { eventId: string; edit: Pick<EventEdit, "t_start" | "t_end" | "description">; expectedRevision: number }) =>
      api.setEdit(analysisId, eventId, edit, expectedRevision),
    onSuccess: (saved) => client.setQueryData<EventEdits>(key, (current) => mergeEdits(current, saved)),
    onError: (error) => {
      if (error instanceof ApiError && (error.status === 409 || error.status === 404)) void client.invalidateQueries({ queryKey: key });
    },
  });
}

/** Human review decisions of one analysis, refreshed when the window regains focus. */
export function useReviews(analysisId: string) {
  const client = useQueryClient();
  return useQuery({
    queryKey: keys.reviews(analysisId),
    queryFn: async ({ signal }) => {
      const fresh = await api.reviews(analysisId, signal);
      // Merged with what is cached when the answer arrives, which may include a change saved meanwhile.
      return mergeDecisions(client.getQueryData<ReviewDecisions>(keys.reviews(analysisId)), fresh);
    },
  });
}

/**
 * Record a decision based on the revision on screen. A refusal because it changed elsewhere (409) loads the
 * latest decisions; nothing is retried or decided automatically.
 */
export function useSetReview(analysisId: string) {
  const client = useQueryClient();
  const key = keys.reviews(analysisId);
  return useMutation({
    mutationFn: ({ eventId, decision, expectedRevision }: { eventId: string; decision: ReviewDecisionName; expectedRevision: number }) =>
      api.setReview(analysisId, eventId, decision, expectedRevision),
    onSuccess: (saved) => client.setQueryData<ReviewDecisions>(key, (current) => mergeDecisions(current, saved)),
    onError: (error) => {
      if (error instanceof ApiError && (error.status === 409 || error.status === 404)) void client.invalidateQueries({ queryKey: key });
    },
  });
}

export function useStartAnalysis() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (videoId: string) => api.startAnalysis(videoId),
    onSuccess: ({ analysis }) => storeAnalysis(client, analysis),
  });
}

export function useCancelAnalysis() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (analysisId: string) => api.cancelAnalysis(analysisId),
    onSuccess: (analysis) => storeAnalysis(client, analysis),
  });
}

/** Store a changed video everywhere it is displayed. */
function useStoreVideo() {
  const client = useQueryClient();
  return (video: Video) => {
    client.setQueryData(keys.video(video.id), video);
    client.setQueryData<Video[]>(keys.videos, (list) =>
      list ? (list.some((v) => v.id === video.id) ? list.map((v) => (v.id === video.id ? video : v)) : [...list, video]) : list,
    );
  };
}

export function useRegisterVideo() {
  const client = useQueryClient();
  const store = useStoreVideo();
  return useMutation({
    mutationFn: ({ folderId, path }: { folderId: string; path: string }) => api.registerVideo(folderId, path),
    onSuccess: store,
    onSettled: () => {
      void client.invalidateQueries({ queryKey: keys.folders });
      void client.invalidateQueries({ queryKey: keys.videos });
    },
  });
}

export function useForgetVideo() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (videoId: string) => api.forgetVideo(videoId),
    onSuccess: (_, videoId) => {
      client.removeQueries({ queryKey: keys.video(videoId) });
      client.setQueryData<Video[]>(keys.videos, (list) => list?.filter((v) => v.id !== videoId));
    },
    onSettled: () => {
      void client.invalidateQueries({ queryKey: keys.folders });
      void client.invalidateQueries({ queryKey: keys.videos });
    },
  });
}

export function useReportPlayback() {
  const store = useStoreVideo();
  return useMutation({
    mutationFn: ({ videoId, ...report }: { videoId: string; playable: boolean; detail: string | null; sha256: string }) =>
      api.reportPlayback(videoId, report),
    onSuccess: store,
  });
}
