/**
 * Requests to the local API. Pages are served from the same origin as the API (directly, or through
 * the development proxy), so every path is relative and no cross-origin configuration exists.
 */
import { sentence } from "../lib/format";
import type {
  Analysis,
  ClipResponse,
  EventEdit,
  EventEdits,
  EvidencePackage,
  FolderSelection,
  Health,
  LibraryFolder,
  LibraryListing,
  OverlayResponse,
  PlaybackReport,
  ReviewDecision,
  ReviewDecisionName,
  ReviewDecisions,
  Timeline,
  Video,
} from "./types";

/** The server answered with an error. `message` is written for people. */
export class ApiError extends Error {
  readonly status: number;
  readonly reasons: string[];

  constructor(status: number, message: string, reasons: string[] = []) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.reasons = reasons;
  }
}

/** The server could not be reached, or something other than the server answered. */
export class ServerUnavailableError extends Error {
  constructor(message = "Spectra is not responding.") {
    super(message);
    this.name = "ServerUnavailableError";
  }
}

function describeDetail(detail: unknown, status: number): { message: string; reasons: string[] } {
  if (typeof detail === "string") {
    return { message: sentence(detail), reasons: [] };
  }
  if (detail && typeof detail === "object" && !Array.isArray(detail)) {
    const record = detail as { message?: unknown; reasons?: unknown };
    const reasons = Array.isArray(record.reasons) ? record.reasons.filter((r) => typeof r === "string") : [];
    if (typeof record.message === "string") {
      return { message: sentence(record.message), reasons };
    }
  }
  if (Array.isArray(detail)) {
    return { message: "The request was not accepted.", reasons: [] };
  }
  return { message: status >= 500 ? "Something went wrong on the server. Try again." : "The request was not accepted.", reasons: [] };
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  return (await send<T>(path, init)).body;
}

async function send<T>(path: string, init: RequestInit = {}): Promise<{ body: T; status: number }> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: { Accept: "application/json", ...(init.body ? { "Content-Type": "application/json" } : {}) },
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ServerUnavailableError();
  }
  const isJson = (response.headers.get("content-type") ?? "").includes("application/json");
  if (!response.ok) throw await failure(response, isJson);
  if (response.status === 204) return { body: undefined as T, status: 204 };
  if (!isJson) throw new ServerUnavailableError("The server sent an unexpected response.");
  return { body: (await response.json()) as T, status: response.status };
}

/** The error for a failed response. Every error from the API has a JSON body; anything else came from a
 * proxy or another program. */
async function failure(response: Response, isJson: boolean): Promise<Error> {
  if (!isJson) return new ServerUnavailableError();
  const body = (await response.json().catch(() => ({}))) as { detail?: unknown };
  const { message, reasons } = describeDetail(body.detail, response.status);
  return new ApiError(response.status, message, reasons);
}

/** A file the server sends as a download, with the name it suggests. */
async function download(path: string, fallbackName: string): Promise<{ blob: Blob; fileName: string }> {
  let response: Response;
  try {
    response = await fetch(path, { headers: { Accept: "application/json" } });
  } catch {
    throw new ServerUnavailableError();
  }
  const isJson = (response.headers.get("content-type") ?? "").includes("application/json");
  if (!response.ok) throw await failure(response, isJson);
  if (!isJson) throw new ServerUnavailableError("The server sent an unexpected response.");
  const named = /filename="([^"/\\]+)"/.exec(response.headers.get("content-disposition") ?? "")?.[1];
  return { blob: await response.blob(), fileName: named ?? fallbackName };
}

const json = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const api = {
  health: (signal?: AbortSignal) => request<Health>("/api/health", { signal }),
  libraryFolders: (signal?: AbortSignal) => request<LibraryFolder[]>("/api/library", { signal }),
  /** Opens the folder dialog on the computer running Spectra; the request carries no path. */
  addLibraryFolder: () => request<FolderSelection>("/api/library/folders", { method: "POST" }),
  removeLibraryFolder: (folderId: string) =>
    request<void>(`/api/library/folders/${encodeURIComponent(folderId)}`, { method: "DELETE" }),
  libraryEntries: (folderId: string, path: string, signal?: AbortSignal) =>
    request<LibraryListing>(
      `/api/library/${encodeURIComponent(folderId)}/entries?${new URLSearchParams({ path })}`,
      { signal },
    ),
  videos: (signal?: AbortSignal) => request<Video[]>("/api/videos", { signal }),
  video: (videoId: string, signal?: AbortSignal) =>
    request<Video>(`/api/videos/${encodeURIComponent(videoId)}`, { signal }),
  registerVideo: (folderId: string, path: string) =>
    request<Video>("/api/videos", json({ folder_id: folderId, path })),
  forgetVideo: (videoId: string) =>
    request<void>(`/api/videos/${encodeURIComponent(videoId)}`, { method: "DELETE" }),
  reportPlayback: (videoId: string, report: PlaybackReport) =>
    request<Video>(`/api/videos/${encodeURIComponent(videoId)}/playback-check`, json(report)),
  analyses: (videoId?: string, signal?: AbortSignal) =>
    request<Analysis[]>(`/api/analyses${videoId ? `?${new URLSearchParams({ video_id: videoId })}` : ""}`, {
      signal,
    }),
  analysis: (analysisId: string, signal?: AbortSignal) =>
    request<Analysis>(`/api/analyses/${encodeURIComponent(analysisId)}`, { signal }),
  /** Queue an analysis. `created` is false when the video's queued or running analysis was returned. */
  startAnalysis: async (videoId: string) => {
    const { body, status } = await send<Analysis>("/api/analyses", json({ video_id: videoId }));
    return { analysis: body, created: status === 201 };
  },
  cancelAnalysis: (analysisId: string) =>
    request<Analysis>(`/api/analyses/${encodeURIComponent(analysisId)}/cancel`, { method: "POST" }),
  analysisEventsUrl: (analysisId: string) => `/api/analyses/${encodeURIComponent(analysisId)}/events`,
  timeline: (analysisId: string, signal?: AbortSignal) =>
    request<Timeline>(`/api/analyses/${encodeURIComponent(analysisId)}/timeline`, { signal }),
  evidencePackage: (analysisId: string, eventId: string, signal?: AbortSignal) =>
    request<EvidencePackage>(packagePath(analysisId, eventId), { signal }),
  /** Recorded overlay frames for the proposal's review interval, or a span of at most 120 s. */
  overlay: (analysisId: string, eventId: string, span: { start_s: number; end_s: number } | null, signal?: AbortSignal) =>
    request<OverlayResponse>(
      `${packagePath(analysisId, eventId)}/overlay${
        span ? `?${new URLSearchParams({ start_s: String(span.start_s), end_s: String(span.end_s) })}` : ""
      }`,
      { signal },
    ),
  /** Stream-copy the review interval from the source, or reuse a clip made earlier. */
  createClip: (analysisId: string, eventId: string) =>
    request<ClipResponse>(`${packagePath(analysisId, eventId)}/clip`, { method: "POST" }),
  /** The analysis, its evidence and the reviewer's work as one JSON file; reading it changes nothing. */
  exportAnalysis: (analysisId: string) =>
    download(`/api/analyses/${encodeURIComponent(analysisId)}/export`, `spectra_${analysisId.slice(0, 8)}.json`),
  reviews: (analysisId: string, signal?: AbortSignal) =>
    request<ReviewDecisions>(`/api/analyses/${encodeURIComponent(analysisId)}/reviews`, { signal }),
  edits: (analysisId: string, signal?: AbortSignal) =>
    request<EventEdits>(`/api/analyses/${encodeURIComponent(analysisId)}/edits`, { signal }),
  /** Replace a proposal's edit; refused with 409 when it changed since `expectedRevision`. */
  setEdit: (analysisId: string, eventId: string, edit: Pick<EventEdit, "t_start" | "t_end" | "description">, expectedRevision: number) =>
    request<EventEdit>(`/api/analyses/${encodeURIComponent(analysisId)}/edits/${encodeURIComponent(eventId)}`, {
      method: "PUT",
      body: JSON.stringify({ ...edit, expected_revision: expectedRevision }),
    }),
  /** Refused with 409 when the decision changed since `expectedRevision`. */
  setReview: (analysisId: string, eventId: string, decision: ReviewDecisionName, expectedRevision: number) =>
    request<ReviewDecision>(`/api/analyses/${encodeURIComponent(analysisId)}/reviews/${encodeURIComponent(eventId)}`, {
      method: "PUT",
      body: JSON.stringify({ decision, expected_revision: expectedRevision }),
    }),
};

function packagePath(analysisId: string, eventId: string): string {
  return `/api/analyses/${encodeURIComponent(analysisId)}/packages/${encodeURIComponent(eventId)}`;
}

/** A short, human-readable description of any error thrown by `api`. */
export function errorMessage(error: unknown): string {
  if (error instanceof ApiError || error instanceof ServerUnavailableError) return error.message;
  return "Something went wrong. Try again.";
}

/**
 * Whether a failure is the server being unreachable. The global banner reports that once for the whole
 * application, so pages, alerts and toasts do not repeat it.
 */
export function isOffline(error: unknown): boolean {
  return error instanceof ServerUnavailableError;
}

export function isNotFound(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404;
}
