/**
 * Plain-English review summary of an evidence package: why the analyser proposed the event and what
 * remains uncertain. Every sentence is mapped deterministically from recorded evidence items, reasons and
 * limits; nothing is inferred beyond them. Observations are stated as seen, estimates are named as
 * estimates, and no sentence says that an incident or a collision happened. The package itself is not
 * changed or filtered.
 */
import type { EvidenceItem, EvidencePackage, TimelineEntry } from "../../api/types";
import { formatTimecode, sentence } from "../../lib/format";
import { CATEGORY_LABELS, noteLabel, reasonLabel } from "./labels";

/** Shown when the recorded evidence does not support any specific explanation. */
export const NO_EXPLANATION = "Spectra did not record enough detail to explain this event. Please check the video yourself.";

const SIDES: Record<string, string> = { left_adjacent: "left", right_adjacent: "right" };

const span = (start: number, end: number) =>
  Math.abs(end - start) < 0.05 ? formatTimecode(start) : `${formatTimecode(start)} – ${formatTimecode(end)}`;


function find(items: EvidenceItem[], name: string): EvidenceItem | undefined {
  return items.find((item) => item.name === name);
}

function numeric(item: EvidenceItem | undefined): number | null {
  return item && typeof item.value === "number" && Number.isFinite(item.value) ? item.value : null;
}

/** The tracked object's category in lower case, or a neutral noun when it was not recorded. */
function objectName(pk: EvidencePackage, trackId: number | null, fallback: string): string {
  const track = pk.tracks.find((t) => t.track_id === trackId) ?? (trackId === null ? pk.tracks[0] : undefined);
  const category = track?.category;
  return category ? (CATEGORY_LABELS[category] ?? category).toLowerCase() : fallback;
}

function rapidApproach(pk: EvidencePackage, items: EvidenceItem[]): string[] {
  const episode = find(items, "approach_episode");
  if (!episode) return [];
  const name = objectName(pk, episode.track_id, "vehicle");
  const inLane = numeric(find(items, "ego_lane_fraction"));
  const subject = inLane !== null && inLane >= 0.5 ? `The ${name} ahead` : `A ${name}`;
  return [`${subject} appeared to approach rapidly, growing larger in the footage.`];
}

function cutIn(pk: EvidencePackage, items: EvidenceItem[]): string[] {
  const relations = items.filter((item) => item.name === "lane_relation");
  const before = relations.find((item) => typeof item.value === "string" && item.value in SIDES);
  const after = relations.find((item) => item.value === "ego_lane");
  if (!before || !after) return [];
  const name = objectName(pk, after.track_id, "vehicle");
  return [`A ${name} was seen moving from the ${SIDES[before.value as string]} lane into the camera's lane.`];
}

function cameraImpact(pk: EvidencePackage, items: EvidenceItem[]): string[] {
  const jolt = find(items, "camera_jolt_z");
  const close = items.find((item) => item.name.startsWith("close_"));
  if (!jolt || !close) return [];
  const name = objectName(pk, close.track_id, "vehicle");
  return [`A ${name} came very close to the camera, then the footage shook abruptly.`];
}

function roadUserContact(items: EvidenceItem[]): string[] {
  if (!find(items, "boxes_meet") || !find(items, "velocity_change_z")) return [];
  return ["Two road users appeared to meet, and one of them seemed to change motion abruptly."];
}

/** One short sentence on what the analyser saw, or `NO_EXPLANATION` when its evidence is incomplete. */
export function explanation(entry: TimelineEntry, pk: EvidencePackage): string[] {
  const items = pk.proposal.evidence;
  const out =
    entry.type === "rapid_approach"
      ? rapidApproach(pk, items)
      : entry.type === "cut_in"
        ? cutIn(pk, items)
        : entry.type === "impact" && entry.subtype === "camera_vehicle"
          ? cameraImpact(pk, items)
          : entry.type === "impact" && entry.subtype === "road_users"
            ? roadUserContact(items)
            : [];
  return out.length > 0 ? out : [NO_EXPLANATION];
}

const capital = (text: string) => text[0]!.toUpperCase() + text.slice(1);

/** Plain sentences for reason and note codes about the proposal's object, named as `it` (e.g. "the truck"). */
function plainTexts(it: string): Record<string, string> {
  const its = `${it}'s`;
  const experimental = "This check is experimental and not yet validated.";
  return {
    lane_not_established: `${capital(its)} lane position could not be verified reliably.`,
    boundaries_reidentified: "The lane markings changed partway, so the lane change may only be apparent.",
    entry_not_observed: `The moment ${it} entered the lane was not seen.`,
    identity_jump: `${capital(its)} box jumped in size, so tracking may have switched to another object.`,
    not_validated: experimental,
    motion_estimate_missing: "Camera motion could not be measured when it mattered.",
    motion_baseline_insufficient: "There was too little earlier camera motion to compare against.",
    experimental,
    track_ends_early: `${capital(it)} was lost from view before the event ended.`,
    track_predicted: `Some of ${its} positions were estimated, not seen.`,
    lane_unobserved: `${capital(its)} lane position was not observed.`,
    ego_lane_unavailable: "The camera's own lane could not be identified here.",
  };
}

/**
 * Everything recorded that limits this proposal, in plain words and without duplicates: the analyser's
 * reasons, the package's notes, lost tracks, missing lane estimation, unanalysed stretches, decoding
 * warnings and nearby patterns that could not be judged. Empty only when nothing was recorded and the
 * analyser's conditions held.
 */
export function uncertainties(entry: TimelineEntry, pk: EvidencePackage): string[] {
  const it = entry.type === "impact" && entry.subtype === "road_users"
    ? "a tracked road user"
    : `the tracked ${objectName(pk, entry.track_ids[0] ?? null, "vehicle")}`;
  const plain = plainTexts(it);
  const reasonText = (code: string) => plain[code] ?? sentence(reasonLabel(code));
  const noteText = (code: string) => plain[code] ?? noteLabel(code);
  // What an analyser cannot see by design stays visible for every proposal of its kind.
  const inherent =
    entry.type === "impact" && entry.subtype === "camera_vehicle"
      ? ["The shake was judged from the footage alone, so a collision is not confirmed."]
      : entry.type === "impact" && entry.subtype === "road_users"
        ? ["Contact itself was not observed; they may only have looked close."]
        : [];
  const out = [
    ...inherent,
    ...pk.proposal.reasons.map(reasonText),
    ...entry.reasons.map(reasonText),
    ...pk.notes.map(noteText),
    ...(pk.tracks.some((t) => t.ends_early) ? [plain.track_ends_early!] : []),
    ...(pk.lanes.available ? [] : ["Lane detection did not run in this analysis."]),
    ...pk.gaps.map((gap) => `Part of the footage around this event was not analysed (${span(gap.start_s, gap.end_s)}).`),
    ...pk.warnings.map((w) => (w.t === null ? sentence(w.message) : `The video could not be read cleanly at ${formatTimecode(w.t)}: ${sentence(w.message)}`)),
    ...pk.unresolved.map((u) => `Spectra could not assess a nearby moment (${span(u.t_start, u.t_end)}). ${reasonText(u.reason)}`),
  ];
  if (entry.state === "uncertain" && out.length === 0) out.push("Some of Spectra's checks for this event could not be completed.");
  return [...new Set(out)];
}
