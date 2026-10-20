import type { LucideIcon } from "lucide-react";

import "./Icon.css";

export type IconSize = "sm" | "md" | "lg";

/** Pixel sizes mirror --icon-sm/md/lg; Lucide needs them as attributes. */
export const ICON_PX: Record<IconSize, number> = { sm: 16, md: 20, lg: 24 };
export const ICON_STROKE = 1.75;

interface IconProps {
  icon: LucideIcon;
  size?: IconSize;
  /** Accessible name for a meaningful standalone icon. Omit for decorative icons next to text. */
  label?: string;
  className?: string;
}

/** The only way the application renders a Lucide icon. */
export function Icon({ icon: Glyph, size = "sm", label, className }: IconProps) {
  return (
    <Glyph
      className={["icon", className].filter(Boolean).join(" ")}
      size={ICON_PX[size]}
      strokeWidth={ICON_STROKE}
      aria-hidden={label ? undefined : true}
      aria-label={label}
      role={label ? "img" : undefined}
      focusable="false"
    />
  );
}
