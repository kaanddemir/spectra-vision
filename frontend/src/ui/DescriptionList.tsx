import type { ReactNode } from "react";

import "./DescriptionList.css";

export interface Description {
  term: string;
  value: ReactNode;
  /** Hashes, keys and paths. */
  mono?: boolean;
}

/** Label/value pairs in an aligned grid. Missing values are shown as an em dash, never left blank. */
interface DescriptionListProps {
  items: Description[];
  columns?: 1 | 2 | 3 | 4;
  /** "center" centres each label and value in its column, for a short row of facts. */
  align?: "start" | "center";
}

export function DescriptionList({ items, columns = 2, align = "start" }: DescriptionListProps) {
  const className = ["description-list", `description-list--${columns}`, align === "center" ? "description-list--center" : null];
  return (
    <dl className={className.filter(Boolean).join(" ")}>
      {items.map((item) => (
        <div key={item.term} className="description-list__item">
          <dt className="type-caption">{item.term}</dt>
          <dd className={item.mono ? "type-mono" : "type-numeric"}>{item.value ?? "—"}</dd>
        </div>
      ))}
    </dl>
  );
}
