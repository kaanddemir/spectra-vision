/**
 * Response contracts of the local API (spectra/api/schemas.py and the evidence contracts it returns).
 * Field names and literal values are kept identical to the server. Only the fields the application
 * reads are declared for the large evidence contracts.
 */

export interface ModelFile {
  role: "detector" | "segmenter";
  file: string;
  sha256: string;
}

export interface Health {
  status: "ok" | "degraded";
  version: string;
  database: { available: boolean; schema_version: number | null; message: string | null };
  models: { available: boolean; message: string | null; models: ModelFile[] };
  memory: { pressure_level: number | null; state: "normal" | "warning" | "critical" | "unavailable" };
  analysis: { available: boolean; reasons: string[] };
  library_folders: number;
}

export interface LibraryFolder {
  id: string;
  name: string;
  path: string;
  /** "startup": given with --library and not removable here; "added": chosen in the application. */
  source: "startup" | "added";
  /** False when the folder was moved, deleted or cannot be read. */
  available: boolean;
  registered_videos: number;
}

/** The outcome of the native folder dialog; `folder` is null when it was cancelled. */
export interface FolderSelection {
  folder: LibraryFolder | null;
  created: boolean;
}

export interface LibraryEntry {
  name: string;
  path: string;
  kind: "directory" | "video";
  size_bytes: number | null;
  modified_at: string | null;
  video_id: string | null;
}

export interface LibraryListing {
  folder: LibraryFolder;
  path: string;
  parent: string | null;
  entries: LibraryEntry[];
  truncated: boolean;
}

export interface ProbeStatus {
  state: "pending" | "ready" | "failed";
  error: string | null;
  message: string | null;
  probed_at: string | null;
}

export interface MediaInfo {
  sha256: string;
  container: string;
  codec: string;
  audio_codecs: string[];
  width: number;
  height: number;
  rotation: number;
  duration_s: number | null;
  nominal_fps: number | null;
  start_s: number;
  timestamp_source: string;
}

export type SourceStatus = "available" | "missing" | "changed" | "outside_library";

export interface BrowserCheck {
  playable: boolean;
  checked_at: string;
  user_agent: string | null;
  detail: string | null;
}

export type PlaybackAssessment = "likely" | "uncertain" | "unsupported" | "unknown";

export interface Video {
  id: string;
  file_name: string;
  folder_id: string | null;
  path: string | null;
  size_bytes: number;
  modified_at: string;
  registered_at: string;
  probe: ProbeStatus;
  media: MediaInfo | null;
  source: { status: SourceStatus; message: string | null };
  playback: {
    assessment: PlaybackAssessment;
    reasons: string[];
    mime_type: string | null;
    browser_check: BrowserCheck | null;
  };
  analysis: { eligible: boolean; reasons: string[] };
  media_url: string | null;
}

export interface PlaybackReport {
  playable: boolean;
  detail?: string | null;
  sha256?: string | null;
}

export type AnalysisStateName =
  | "queued"
  | "running"
  | "completed"
  | "incomplete"
  | "cancelled"
  | "resource_stopped"
  | "failed";

export interface Analysis {
  id: string;
  video_id: string;
  file_name: string;
  source_sha256: string;
  state: AnalysisStateName;
  revision: number;
  queue_position: number | null;
  cancel_requested: boolean;
  progress: {
    stage: string | null;
    frames_analysed: number;
    media_time_s: number | null;
    fraction: number | null;
  };
  result: {
    cache_hit: boolean;
    perception_key: string | null;
    analysis_key: string | null;
    proposals: number;
    notes: string[];
  } | null;
  evidence: {
    state: "not_applicable" | "pending" | "ready" | "failed";
    build_key: string | null;
    message: string | null;
  };
  stop_reason: string | null;
  message: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export type EventType = "rapid_approach" | "cut_in" | "impact";
export type EvidenceState = "supported" | "uncertain";
export type Coverage = "proposed" | "no_proposal" | "unavailable" | "failed";

export interface TimelineEntry {
  event_id: string;
  type: EventType;
  subtype: "camera_vehicle" | "road_users" | null;
  title_key: string;
  title: string;
  t_start: number;
  t_peak: number;
  t_end: number;
  review: { start_s: number; end_s: number; evidence_start_s: number; evidence_end_s: number };
  state: EvidenceState;
  reasons: string[];
  experimental: boolean;
  analyzer: string;
  track_ids: number[];
  rank: number | null;
  package: string;
  segment: number;
  related: { kind: string; event_id: string }[];
  notes: string[];
}

export interface AnalyzerCoverage {
  name: string;
  version: string;
  experimental: boolean;
  status: "ok" | "unavailable" | "failed";
  coverage: Coverage;
  reason: string | null;
  proposals: number;
  considered: number;
}

export interface ReviewInterval {
  start_s: number;
  end_s: number;
  evidence_start_s: number;
  evidence_end_s: number;
}

export interface Span {
  start_s: number;
  end_s: number;
}

export interface RunWarning {
  code: string;
  message: string;
  t: number | null;
}

export interface UnresolvedPattern {
  reason: string;
  t_start: number;
  t_end: number;
  track_ids: number[];
}

export interface Relation {
  kind: "based_on" | "basis_of" | "shares_track";
  event_id: string;
}

export interface SourceVideo {
  file_name: string;
  sha256: string;
  width: number;
  height: number;
  rotation: number;
  start_s: number;
  duration_s: number | null;
  timestamp_source: string;
}

export interface Timeline {
  evidence_version: string;
  state: "complete" | "incomplete" | "cancelled" | "resource_stopped" | "failed";
  message: string | null;
  analysis_key: string | null;
  config: { overlay_max_age_s: number; gap_s: number };
  video: SourceVideo | null;
  analysed_start_s: number | null;
  analysed_end_s: number | null;
  analyzers: AnalyzerCoverage[];
  entries: TimelineEntry[];
  segments: (Span & { event_ids: string[] })[];
  gaps: Span[];
  warnings: RunWarning[];
}

export type ObjectCategory = "person" | "bicycle" | "motorcycle" | "car" | "bus" | "truck";
export type LaneRelation = "ego_lane" | "left_adjacent" | "right_adjacent" | "beyond_adjacent" | "off_road" | "ambiguous";
export type EgoLaneState = "estimated" | "ambiguous" | "unavailable";

export interface EvidenceItem {
  basis: "observed" | "estimated" | "context";
  name: string;
  t: number;
  t_end: number | null;
  track_id: number | null;
  value: number | string | null;
  unit: string | null;
  note: string;
}

export interface EventProposal {
  event_id: string;
  type: EventType;
  subtype: TimelineEntry["subtype"];
  state: EvidenceState;
  experimental: boolean;
  t_start: number;
  t_peak: number;
  t_end: number;
  track_ids: number[];
  evidence: EvidenceItem[];
  reasons: string[];
  rank: number | null;
  analyzer: string;
}

export interface TrackSummary {
  track_id: number;
  category: ObjectCategory | null;
  first_s: number | null;
  last_s: number | null;
  observed: number;
  predicted: number;
  ends_early: boolean;
}

export interface EvidencePackage {
  evidence_version: string;
  proposal: EventProposal;
  analyzer: { name: string; version: string };
  video: SourceVideo;
  analysed_start_s: number;
  analysed_end_s: number;
  review: ReviewInterval;
  tracks: TrackSummary[];
  lanes: {
    available: boolean;
    ego_lane: (Span & { state: EgoLaneState; frames: number })[];
    relations: (Span & { track_id: number; relation: LaneRelation | null; beyond_evidence: boolean; frames: number })[];
  };
  gaps: Span[];
  warnings: RunWarning[];
  unresolved: UnresolvedPattern[];
  related: Relation[];
  notes: string[];
}

export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** Normalised display coordinates, near end first. */
export interface LaneBoundary {
  near: [number, number];
  far: [number, number];
  points: [number, number][];
  measured: boolean;
  evidence_y: number | null;
}

export interface TrackOverlay {
  track_id: number;
  box: Box;
  category: ObjectCategory;
  score: number | null;
  predicted: boolean;
  related: boolean;
  relation: LaneRelation | null;
}

export interface OverlayFrame {
  t: number;
  valid_until: number;
  /** null: the tracker did not run on this frame. */
  tracks: TrackOverlay[] | null;
  /** null: no ego-lane observation for this frame. */
  lane: { state: EgoLaneState; left: LaneBoundary | null; right: LaneBoundary | null; observed_t: number } | null;
}

export interface OverlayResponse {
  overlay: { event_id: string; width: number; height: number; start_s: number; end_s: number; frames: OverlayFrame[] };
  layers: { tracks: boolean; lane: boolean };
  timing: {
    stream_start_s: number;
    max_age_s: number;
    analysed_start_s: number;
    analysed_end_s: number;
    timestamp_source: string;
  };
  gaps: Span[];
}

export interface ClipResponse {
  clip: {
    file: string;
    size_bytes: number;
    format: string;
    codec: string;
    start_s: number;
    end_s: number;
    origin_s: number;
    first_s: number;
    last_s: number;
    frames: number;
  };
  media_url: string;
  mime_type: string;
  browser_playback: "unverified";
}

/** A person's decision on a proposal; not the analyser's status, and never a finding of an incident or fault. */
export type ReviewDecisionName = "not_reviewed" | "accepted" | "rejected";

export interface ReviewDecision {
  event_id: string;
  decision: ReviewDecisionName;
  /** Increases with every change; 0 when no decision was ever recorded. */
  revision: number;
  updated_at: string | null;
}

export interface ReviewDecisions {
  analysis_id: string;
  /** Proposals with a recorded decision; every other proposal is not reviewed. */
  decisions: ReviewDecision[];
}

/** A reviewer's corrections to a proposal. Null times mean the proposed times apply; the proposal is unchanged. */
export interface EventEdit {
  event_id: string;
  t_start: number | null;
  t_end: number | null;
  /** Plain text written by the reviewer. */
  description: string | null;
  /** Increases with every change; 0 when never edited. */
  revision: number;
  updated_at: string | null;
}

export interface EventEdits {
  analysis_id: string;
  edits: EventEdit[];
}
