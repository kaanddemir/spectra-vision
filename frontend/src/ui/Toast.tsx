import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { IconButton } from "./Button";
import { Icon } from "./Icon";
import { icons } from "./icons";
import "./Toast.css";

export type ToastTone = "success" | "info" | "warning" | "danger";

export interface ToastInput {
  tone: ToastTone;
  title: string;
  description?: string;
  /** Toasts with the same key replace each other instead of stacking; defaults to tone and title. */
  key?: string;
}

interface ToastItem extends ToastInput {
  id: string;
}

/** Routine results go quickly; problems stay long enough to read. */
export const TOAST_DURATION_MS: Record<ToastTone, number> = { success: 4000, info: 5000, warning: 8000, danger: 10000 };
const MAX_TOASTS = 3;
const TONE_ICONS = { success: icons.success, info: icons.info, warning: icons.warning, danger: icons.danger };

const ToastContext = createContext<((toast: ToastInput) => void) | null>(null);

/**
 * Temporary feedback for completed actions. Ongoing conditions use banners, page-specific
 * information uses inline alerts, and consequential actions use dialogs.
 */
export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const timers = useRef(new Map<string, number>());

  const dismiss = useCallback((id: string) => {
    window.clearTimeout(timers.current.get(id));
    timers.current.delete(id);
    setToasts((list) => list.filter((t) => t.id !== id));
  }, []);

  const show = useCallback(
    (input: ToastInput) => {
      const id = input.key ?? `${input.tone}:${input.title}`;
      window.clearTimeout(timers.current.get(id));
      timers.current.set(id, window.setTimeout(() => dismiss(id), TOAST_DURATION_MS[input.tone]));
      setToasts((list) => [...list.filter((t) => t.id !== id), { ...input, id }].slice(-MAX_TOASTS));
    },
    [dismiss],
  );

  useEffect(() => {
    const pending = timers.current;
    return () => pending.forEach((timer) => window.clearTimeout(timer));
  }, []);

  const value = useMemo(() => show, [show]);
  return (
    <ToastContext.Provider value={value}>
      {children}
      <section className="toasts" aria-label="Notifications">
        <ol className="toasts__list" aria-live="polite">
          {toasts.map((toast) => (
            <li key={toast.id} className={`toast toast--${toast.tone}`} role={toast.tone === "danger" ? "alert" : "status"}>
              <Icon icon={TONE_ICONS[toast.tone]} size="md" className="toast__icon" />
              <div className="toast__body">
                <p className="toast__title">{toast.title}</p>
                {toast.description ? <p className="toast__description">{toast.description}</p> : null}
              </div>
              <IconButton icon={icons.close} label="Dismiss" size="sm" onClick={() => dismiss(toast.id)} />
            </li>
          ))}
        </ol>
      </section>
    </ToastContext.Provider>
  );
}

export function useToast(): (toast: ToastInput) => void {
  const show = useContext(ToastContext);
  if (!show) throw new Error("useToast must be used inside ToastProvider");
  return show;
}
