import { Icon, type IconSize } from "./Icon";
import { icons } from "./icons";
import "./Spinner.css";

/** Decorative activity indicator; pair it with text or use LoadingState. */
export function Spinner({ size = "sm" }: { size?: IconSize }) {
  return (
    <span className="spinner" aria-hidden="true">
      <Icon icon={icons.pending} size={size} />
    </span>
  );
}

/** A region-level loading message, announced politely. */
export function LoadingState({ label }: { label: string }) {
  return (
    <div className="loading-state" role="status">
      <Spinner size="md" />
      <span>{label}</span>
    </div>
  );
}
