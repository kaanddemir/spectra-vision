/**
 * Real API responses (see tests/api/frontend_fixtures.py), typed with the application's contracts.
 * Tests copy them before changing anything.
 */
import raw from "./fixtures/api.json";
import type {
  Analysis,
  ClipResponse,
  EventEdits,
  EvidencePackage,
  Health,
  LibraryFolder,
  LibraryListing,
  OverlayResponse,
  ReviewDecisions,
  Timeline,
  Video,
} from "../api/types";

type VideoKey = "ready" | "needs_check" | "probing" | "unsupported" | "missing" | "unreadable" | "not_playable";

interface Fixtures {
  health: Health;
  healthDegraded: Health;
  folders: LibraryFolder[];
  twoFolders: LibraryFolder[];
  listing: LibraryListing;
  sublisting: LibraryListing;
  videos: Record<VideoKey, Video>;
  analyses: Analysis[];
  timeline: Timeline;
  packages: Record<string, EvidencePackage>;
  overlay: OverlayResponse;
  clip: ClipResponse;
  reviews: ReviewDecisions;
  edits: EventEdits;
}

const fixtures = raw as unknown as Fixtures;

/** A deep copy, so a test can change a response without affecting other tests. */
export function fixture<K extends keyof Fixtures>(key: K): Fixtures[K] {
  return structuredClone(fixtures[key]);
}

export function video(key: VideoKey, changes: Partial<Video> = {}): Video {
  return { ...structuredClone(fixtures.videos[key]), ...changes };
}

export const allVideos = (): Video[] => Object.values(structuredClone(fixtures.videos));
