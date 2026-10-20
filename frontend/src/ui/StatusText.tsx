import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import type { Tone } from "./Badge";
import { Icon } from "./Icon";
import "./StatusText.css";

interface StatusTextProps {
  tone: Tone;
  /** Statuses always carry an icon as well as text, so colour is never the only signal. */
  icon: LucideIcon;
  spin?: boolean;
  /** A fuller description for assistive technology and the pointer tooltip, when the text is short. */
  description?: string;
  children: ReactNode;
}

/** A status as a coloured icon and plain text, without a pill: for dense lists and tables. */
export function StatusText({ tone, icon, spin = false, description, children }: StatusTextProps) {
  return (
    <span className={`status-text status-text--${tone}`} title={description}>
      <span className={spin ? "spinner status-text__icon" : "status-text__icon"} aria-hidden="true">
        <Icon icon={icon} />
      </span>
      <span>{children}</span>
      {description ? <span className="visually-hidden">: {description}</span> : null}
    </span>
  );
}
