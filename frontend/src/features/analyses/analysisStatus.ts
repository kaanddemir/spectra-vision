import type { LucideIcon } from "lucide-react";

import type { Analysis, AnalysisStateName } from "../../api/types";
import type { Tone } from "../../ui/Badge";
import { icons } from "../../ui/icons";

interface AnalysisStatus {
  label: string;
  tone: Tone;
  icon: LucideIcon;
  spin?: boolean;
}

const STATES: Record<AnalysisStateName, AnalysisStatus> = {
  queued: { label: "Queued", tone: "neutral", icon: icons.notStarted },
  running: { label: "Running", tone: "info", icon: icons.pending, spin: true },
  completed: { label: "Completed", tone: "success", icon: icons.success },
  incomplete: { label: "Incomplete", tone: "warning", icon: icons.warning },
  cancelled: { label: "Cancelled", tone: "neutral", icon: icons.unsupported },
  resource_stopped: { label: "Stopped for memory", tone: "warning", icon: icons.warning },
  failed: { label: "Failed", tone: "danger", icon: icons.danger },
};

export function analysisStatus(analysis: Pick<Analysis, "state">): AnalysisStatus {
  return STATES[analysis.state];
}

/** Whether a finished analysis produced reviewable results. */
export function hasResult(analysis: Analysis): boolean {
  return analysis.state === "completed" || analysis.state === "incomplete";
}

const STOP_REASONS: Record<string, string> = {
  cancelled: "Cancelled on request.",
  interrupted: "Interrupted when the server stopped.",
  shutdown: "Stopped because the server shut down.",
  video: "The video changed or could not be read.",
  models: "The analysis models were unavailable.",
  resources: "Stopped to protect system memory.",
  error: "The analysis stopped unexpectedly.",
};

/** Why a finished analysis stopped, in one short sentence; null for complete runs. */
export function stopReason(analysis: Analysis): string | null {
  if (analysis.state === "incomplete") return "Decoding ended early; the end of the video was not analysed.";
  if (!analysis.stop_reason) return null;
  return STOP_REASONS[analysis.stop_reason] ?? null;
}
