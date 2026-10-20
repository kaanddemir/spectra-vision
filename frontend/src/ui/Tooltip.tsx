import {
  cloneElement,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type ReactElement,
  type ReactNode,
} from "react";

import { placeTooltip, type Placement } from "./tooltipPosition";
import "./Tooltip.css";

interface TooltipProps {
  content: ReactNode;
  /** One focusable element; it gains aria-describedby. */
  children: ReactElement<{ "aria-describedby"?: string }>;
  /** Preferred side; the tooltip flips when there is no room there. */
  placement?: Placement;
}

/** Read from --tooltip-gap and --tooltip-margin, so positioning follows the design tokens. */
function lengths(element: Element): { gap: number; margin: number } {
  const style = getComputedStyle(element);
  return {
    gap: parseFloat(style.getPropertyValue("--tooltip-gap")) || 0,
    margin: parseFloat(style.getPropertyValue("--tooltip-margin")) || 0,
  };
}

/**
 * A short supplementary label shown on hover and keyboard focus; Escape dismisses it. It is positioned
 * against the viewport (fixed), so scroll containers such as tables never clip it, and it is kept inside
 * the viewport near every edge. Essential information must not live only here: touch users get no hover.
 */
export function Tooltip({ content, children, placement = "bottom" }: TooltipProps) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const [side, setSide] = useState<Placement | null>(null);
  const anchor = useRef<HTMLSpanElement>(null);
  const tip = useRef<HTMLSpanElement>(null);

  const update = useCallback(() => {
    if (!anchor.current || !tip.current) return;
    const { gap, margin } = lengths(tip.current);
    const rect = anchor.current.getBoundingClientRect();
    const size = tip.current.getBoundingClientRect();
    const position = placeTooltip(
      { top: rect.top, left: rect.left, width: rect.width, height: rect.height },
      { width: size.width, height: size.height },
      { width: document.documentElement.clientWidth, height: document.documentElement.clientHeight },
      placement,
      gap,
      margin,
    );
    // Measured coordinates only; every visual property stays in Tooltip.css.
    tip.current.style.setProperty("--tooltip-x", `${position.left}px`);
    tip.current.style.setProperty("--tooltip-y", `${position.top}px`);
    setSide(position.placement);
  }, [placement]);

  useLayoutEffect(() => {
    if (open) update();
    else setSide(null);
  }, [open, update, content]);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && setOpen(false);
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [open, update]);

  const describedBy = [children.props["aria-describedby"], id].filter(Boolean).join(" ");
  return (
    <span
      ref={anchor}
      className="tooltip-anchor"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      {cloneElement(children, { "aria-describedby": describedBy })}
      <span
        ref={tip}
        id={id}
        role="tooltip"
        className={`tooltip tooltip--${side ?? placement}`}
        data-positioned={side ? "true" : undefined}
        hidden={!open}
      >
        {content}
      </span>
    </span>
  );
}
