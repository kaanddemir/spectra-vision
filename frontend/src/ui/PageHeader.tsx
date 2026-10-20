import type { ReactNode } from "react";
import { Link } from "react-router";

import { Icon } from "./Icon";
import { icons } from "./icons";

import "./PageHeader.css";

interface PageHeaderProps {
  title: ReactNode;
  description?: ReactNode;
  /** The parent page, shown as an arrow before the title; `label` is its accessible name. */
  back?: { to: string; label: string };
  /** Status badges next to the title. */
  meta?: ReactNode;
  actions?: ReactNode;
}

/** The single heading block at the top of every page; it owns the page's h1. */
export function PageHeader({ title, description, back, meta, actions }: PageHeaderProps) {
  return (
    <header className="page-header">
      <div className="page-header__row">
        <div className="page-header__text">
          <div className="page-header__title-row">
            {back ? (
              <Link to={back.to} className="page-header__back" aria-label={back.label}>
                <Icon icon={icons.back} />
              </Link>
            ) : null}
            <h1 className="type-title page-header__title">{title}</h1>
            {meta ? <div className="cluster">{meta}</div> : null}
          </div>
          {description ? <p className="page-header__description">{description}</p> : null}
        </div>
        {actions ? <div className="page-header__actions">{actions}</div> : null}
      </div>
    </header>
  );
}
