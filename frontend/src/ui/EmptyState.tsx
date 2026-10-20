import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { Button } from "./Button";
import { Icon } from "./Icon";
import { icons } from "./icons";
import "./EmptyState.css";

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
  /**
   * "page" fills the content area below the page header and centres the group in it; use it when the
   * state replaces a page's whole content. "inline" (default) sits within a section, list or panel;
   * "compact" is the same with less vertical padding, for a hint beneath other content.
   */
  layout?: "inline" | "page" | "compact";
}

/** Explains why a region has nothing to show and, where possible, what to do next. */
export function EmptyState({ icon, title, children, action, layout = "inline" }: EmptyStateProps) {
  return (
    <div className={`empty-state empty-state--${layout}`}>
      <span className="empty-state__icon">
        <Icon icon={icon} size="lg" />
      </span>
      <h2 className="type-subheading">{title}</h2>
      {children ? <div className="empty-state__text type-secondary">{children}</div> : null}
      {action ? <div className="empty-state__action">{action}</div> : null}
    </div>
  );
}

/**
 * The one state for a search or filter that hides every item. `onClear` resets the page's search and
 * filters; leave it out when none is active, so the state never suggests clearing nothing.
 */
export function NoResults({ onClear }: { onClear?: () => void }) {
  return (
    <EmptyState
      icon={icons.search}
      title="No results found"
      action={onClear ? <Button onClick={onClear}>Clear filters</Button> : undefined}
    >
      Try a different search.
    </EmptyState>
  );
}
