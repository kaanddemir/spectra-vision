import { Fragment, type ReactNode } from "react";
import { Link } from "react-router";

import { Icon } from "./Icon";
import { icons } from "./icons";
import "./Breadcrumbs.css";

export interface Crumb {
  label: ReactNode;
  /** A link target, or a handler for in-page navigation. The last crumb is the current location. */
  to?: string;
  onSelect?: () => void;
}

export function Breadcrumbs({ items, label = "Breadcrumb" }: { items: Crumb[]; label?: string }) {
  return (
    <nav aria-label={label} className="breadcrumbs">
      <ol>
        {items.map((item, index) => {
          const current = index === items.length - 1;
          return (
            <Fragment key={index}>
              {index > 0 ? (
                <li aria-hidden="true" className="breadcrumbs__separator">
                  <Icon icon={icons.chevronRight} />
                </li>
              ) : null}
              <li>
                {current ? (
                  <span aria-current="page" className="breadcrumbs__current">
                    {item.label}
                  </span>
                ) : item.to !== undefined ? (
                  <Link to={item.to}>{item.label}</Link>
                ) : (
                  <button type="button" className="breadcrumbs__button" onClick={item.onSelect}>
                    {item.label}
                  </button>
                )}
              </li>
            </Fragment>
          );
        })}
      </ol>
    </nav>
  );
}
