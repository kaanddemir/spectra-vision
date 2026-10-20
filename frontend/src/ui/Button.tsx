import type { LucideIcon } from "lucide-react";
import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import { Link, type LinkProps } from "react-router";

import { Icon } from "./Icon";
import { Spinner } from "./Spinner";
import { Tooltip } from "./Tooltip";
import "./Button.css";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
export type ButtonSize = "sm" | "md";

interface Look {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

export function buttonClass({ variant = "secondary", size = "md" }: Look, extra?: string): string {
  return ["button", `button--${variant}`, `button--${size}`, extra].filter(Boolean).join(" ");
}

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement>, Look {
  icon?: LucideIcon;
  /** Shows a spinner, disables the button and keeps its label for assistive technology. */
  loading?: boolean;
  children: ReactNode;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant, size, icon, loading = false, disabled, children, className, type = "button", ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={buttonClass({ variant, size }, className)}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...rest}
    >
      {loading ? <Spinner size="sm" /> : icon ? <Icon icon={icon} /> : null}
      <span>{children}</span>
    </button>
  );
});

interface ButtonLinkProps extends LinkProps, Look {
  icon?: LucideIcon;
  children: ReactNode;
}

/** Navigation that looks like a button. */
export function ButtonLink({ variant, size, icon, children, className, ...rest }: ButtonLinkProps) {
  return (
    <Link className={buttonClass({ variant, size }, className)} {...rest}>
      {icon ? <Icon icon={icon} /> : null}
      <span>{children}</span>
    </Link>
  );
}

interface IconButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> {
  icon: LucideIcon;
  /** Accessible name, also shown as a tooltip. */
  label: string;
  variant?: Exclude<ButtonVariant, "primary">;
  size?: ButtonSize;
}

export function IconButton({ icon, label, variant = "ghost", size = "md", className, type = "button", ...rest }: IconButtonProps) {
  return (
    <Tooltip content={label}>
      <button type={type} aria-label={label} className={buttonClass({ variant, size }, ["button--icon", className].filter(Boolean).join(" "))} {...rest}>
        <Icon icon={icon} />
      </button>
    </Tooltip>
  );
}
