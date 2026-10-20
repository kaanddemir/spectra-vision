import { useId, type ReactNode } from "react";

import "./Section.css";

interface SectionProps {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  /** "divided" separates the section from the one above with a rule instead of a box. */
  variant?: "plain" | "divided";
  /** Places the description on the title's line instead of under it; it wraps below when space runs out. */
  inlineDescription?: boolean;
}

/** A titled region of a page. */
export function Section({ title, description, actions, children, variant = "plain", inlineDescription = false }: SectionProps) {
  const id = useId();
  return (
    <section aria-labelledby={id} className={`section section--${variant}`}>
      <div className="section__header">
        <div className={inlineDescription ? "section__title section__title--inline" : "section__title"}>
          <h2 id={id} className="type-heading">
            {title}
          </h2>
          {description ? <p className="type-secondary section__description">{description}</p> : null}
        </div>
        {actions ? <div className="cluster">{actions}</div> : null}
      </div>
      <div className="section__body">{children}</div>
    </section>
  );
}
