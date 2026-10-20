import type { ReactNode } from "react";

import { Icon } from "./Icon";
import { icons } from "./icons";
import type { Tone } from "./Badge";
import "./Alert.css";

type AlertTone = Exclude<Tone, "neutral">;

const ALERT_ICONS = { info: icons.info, success: icons.success, warning: icons.warning, danger: icons.danger };

interface AlertProps {
  tone?: AlertTone;
  title: string;
  children?: ReactNode;
  actions?: ReactNode;
}

/** An inline message about the current screen. Errors are announced immediately, others politely. */
export function Alert({ tone = "info", title, children, actions }: AlertProps) {
  return (
    <div className={`alert alert--${tone}`} role={tone === "danger" ? "alert" : "status"}>
      <Icon icon={ALERT_ICONS[tone]} size="md" className="alert__icon" />
      <div className="alert__body">
        <p className="alert__title">{title}</p>
        {children ? <div className="alert__content">{children}</div> : null}
      </div>
      {actions ? <div className="alert__actions">{actions}</div> : null}
    </div>
  );
}
