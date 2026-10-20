/**
 * One status per registered video, derived only from server fields. The same mapping is used by
 * every screen so a video never shows two different states.
 */
import type { LucideIcon } from "lucide-react";

import type { Video } from "../../api/types";
import type { Tone } from "../../ui/Badge";
import { icons } from "../../ui/icons";

export type VideoStatusKind =
  | "probing"
  | "source_missing"
  | "source_changed"
  | "outside_library"
  | "unreadable"
  | "unsupported"
  | "needs_check"
  | "not_playable"
  | "ready";

export interface VideoStatus {
  kind: VideoStatusKind;
  label: string;
  /** One or two words for dense tables; `label` stays the full description. */
  short: string;
  tone: Tone;
  icon: LucideIcon;
  /** Grouping used by the library filters. */
  group: "ready" | "attention" | "processing";
}

const STATUS: Record<VideoStatusKind, Omit<VideoStatus, "kind">> = {
  probing: { label: "Reading details", short: "Reading", tone: "info", icon: icons.pending, group: "processing" },
  source_missing: { label: "Source missing", short: "Missing", tone: "danger", icon: icons.danger, group: "attention" },
  source_changed: { label: "Source changed", short: "Changed", tone: "warning", icon: icons.warning, group: "attention" },
  outside_library: { label: "Outside library", short: "Outside library", tone: "warning", icon: icons.warning, group: "attention" },
  unreadable: { label: "Could not read", short: "Unreadable", tone: "danger", icon: icons.danger, group: "attention" },
  unsupported: { label: "Unsupported format", short: "Unsupported", tone: "danger", icon: icons.unsupported, group: "attention" },
  needs_check: { label: "Needs playback check", short: "Needs attention", tone: "warning", icon: icons.attention, group: "attention" },
  not_playable: { label: "Not playable here", short: "Can't play", tone: "danger", icon: icons.danger, group: "attention" },
  ready: { label: "Ready for analysis", short: "Ready", tone: "success", icon: icons.success, group: "ready" },
};

export function videoStatus(video: Video): VideoStatus {
  const kind = statusKind(video);
  return { kind, ...STATUS[kind] };
}

function statusKind(video: Video): VideoStatusKind {
  switch (video.source.status) {
    case "missing":
      return "source_missing";
    case "changed":
      return "source_changed";
    case "outside_library":
      return "outside_library";
    case "available":
      break;
  }
  if (video.probe.state === "pending") return "probing";
  if (video.probe.state === "failed") return "unreadable";
  if (video.playback.assessment === "unsupported") return "unsupported";
  if (video.analysis.eligible) return "ready";
  const check = video.playback.browser_check;
  if (check && !check.playable) return "not_playable";
  return "needs_check";
}
