/**
 * Short English labels for evidence codes. Unknown codes fall back to a readable form of the code, so a
 * newer server never shows blank text.
 */
import type { Relation } from "../../api/types";
import { humanize } from "../../lib/format";

const label = (table: Record<string, string>) => (code: string) => table[code] ?? humanize(code);

export const reasonLabel = label({
  lane_not_established: "No observed lane relation for most of the pattern",
  boundaries_reidentified: "The camera's lane boundaries changed identity",
  entry_not_observed: "The lane entry itself was not observed",
  identity_jump: "The box size jumped as if the track switched objects",
  not_validated: "Experimental analyser, not validated",
  motion_estimate_missing: "Camera motion was missing when it mattered",
  motion_baseline_insufficient: "Too little earlier camera motion to compare",
});

export const noteLabel = label({
  experimental: "Proposed by an experimental analyser.",
  clipped_at_video_start: "Less context before: the video starts.",
  clipped_at_video_end: "Less context after: the video ends.",
  context_capped: "Context around the proposal was shortened.",
  outside_analysis: "Part of the review span was not analysed.",
  frame_gap: "The review span has a stretch without analysed frames.",
  track_ends_early: "A related track is lost before the span ends.",
  track_predicted: "Some related boxes were predicted by the tracker, not observed.",
  lane_unobserved: "No observed lane relation for a related track.",
  ego_lane_unavailable: "The camera's lane was not estimated here.",
  motion_missing: "Some frames have no camera-motion estimate.",
  run_warning: "A decoding or timing warning applies here.",
  unresolved_nearby: "An analyser could not judge a nearby pattern.",
});

/** What each analyser checks, in words a reviewer knows; internal identifiers are never shown. */
export const analyzerLabel = label({
  "apparent-approach": "Rapid approach",
  "lane-entry-cut-in": "Cut-in",
  "camera-vehicle-impact": "Impact",
  "road-user-contact": "Contact between other road users",
});

export const RELATED_LABELS: Record<Relation["kind"], string> = {
  based_on: "Built on",
  basis_of: "Basis of",
  shares_track: "Shares a track with",
};

export const CATEGORY_LABELS: Record<string, string> = {
  person: "Person",
  bicycle: "Bicycle",
  motorcycle: "Motorcycle",
  car: "Car",
  bus: "Bus",
  truck: "Truck",
};
