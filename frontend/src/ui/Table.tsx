import type { ReactNode } from "react";

import "./Table.css";

interface TableProps {
  /** Accessible name of the table (also its scroll region). */
  label: string;
  children: ReactNode;
  /** Minimum width before the table scrolls horizontally inside its region. */
  minWidth?: "md" | "lg";
  /**
   * "fit": columns are as narrow as their content and the `cell-primary` (and `cell-flex`) columns take the
   * rest and truncate. "auto" leaves sizing to the browser, for tables of wrapping text.
   */
  columns?: "fit" | "auto";
}

/**
 * A data table in a horizontally scrollable, keyboard-focusable region. Cells use the shared
 * classes: `cell-numeric`, `cell-actions`, `cell-select`, `cell-primary`, `cell-flex`.
 */
export function Table({ label, children, minWidth = "md", columns = "auto" }: TableProps) {
  return (
    <div className="table-region" role="region" aria-label={label} tabIndex={0}>
      <table className={`table table--min-${minWidth} table--${columns}`} aria-label={label}>
        {children}
      </table>
    </div>
  );
}
