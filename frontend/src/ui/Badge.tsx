import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { Icon } from "./Icon";
import "./Badge.css";

export type Tone = "neutral" | "info" | "success" | "warning" | "danger";

interface BadgeProps {
  tone?: Tone;
  /** Statuses always carry an icon as well as text, so colour is never the only signal. */
  icon?: LucideIcon;
  spin?: boolean;
  children: ReactNode;
}

export function Badge({ tone = "neutral", icon, spin = false, children }: BadgeProps) {
  return (
    <span className={`badge badge--${tone}`}>
      {icon ? (
        <span className={spin ? "badge__icon spinner" : "badge__icon"} aria-hidden="true">
          <Icon icon={icon} />
        </span>
      ) : null}
      <span>{children}</span>
    </span>
  );
}
