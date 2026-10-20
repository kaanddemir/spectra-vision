import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";

import "./Popover.css";

interface PopoverProps {
  /** Renders the trigger; spread `props` onto one native button. */
  trigger: (props: {
    "aria-expanded": boolean;
    "aria-controls": string;
    onClick: () => void;
  }) => ReactNode;
  children: ReactNode | ((close: () => void) => ReactNode);
  className?: string;
  panelClassName?: string;
}

/** A disclosure anchored below its trigger, or above it when the viewport has no room below. Opening focuses any [data-autofocus] element; Escape or a pointer outside closes it. */
export function Popover({ trigger, children, className, panelClassName }: PopoverProps) {
  const [open, setOpen] = useState(false);
  const [up, setUp] = useState(false);
  const panelId = useId();
  const root = useRef<HTMLDivElement>(null);
  const panel = useRef<HTMLDivElement>(null);

  const update = useCallback(() => {
    if (!root.current || !panel.current) return;
    const element = panel.current;
    const style = getComputedStyle(element);
    const gap = parseFloat(style.getPropertyValue("--popover-gap")) || 0;
    const margin = parseFloat(style.getPropertyValue("--popover-margin")) || 0;
    element.style.removeProperty("--popover-shift");
    element.style.removeProperty("--popover-available-height");
    const trigger = root.current.getBoundingClientRect();
    const height = element.offsetHeight;
    const below = window.innerHeight - trigger.bottom;
    const above = below < height + gap + margin && trigger.top > below;
    setUp(above);
    const rect = element.getBoundingClientRect();
    const left = Math.max(margin, Math.min(rect.left, window.innerWidth - margin - rect.width));
    // Measured geometry only; spacing and appearance remain in the stylesheet and tokens.
    element.style.setProperty("--popover-shift", `${left - rect.left}px`);
    element.style.setProperty("--popover-available-height", `${Math.max(0, (above ? trigger.top : below) - gap - margin)}px`);
  }, []);

  // Measured before paint, so the panel never flashes on the wrong side.
  useLayoutEffect(() => {
    if (open) update();
  }, [open, update]);

  useEffect(() => {
    if (!open) return;
    root.current?.querySelector<HTMLElement>("[data-autofocus]")?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setOpen(false);
      root.current?.querySelector<HTMLButtonElement>("button[aria-controls]")?.focus();
    };
    const onPointer = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onPointer);
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onPointer);
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [open, update]);

  const close = () => setOpen(false);
  return (
    <div className={["popover", className].filter(Boolean).join(" ")} ref={root}>
      {trigger({ "aria-expanded": open, "aria-controls": panelId, onClick: () => setOpen((value) => !value) })}
      <div
        ref={panel}
        id={panelId}
        className={["popover__panel", up ? "popover__panel--up" : null, panelClassName].filter(Boolean).join(" ")}
        hidden={!open}
      >
        {typeof children === "function" ? children(close) : children}
      </div>
    </div>
  );
}
