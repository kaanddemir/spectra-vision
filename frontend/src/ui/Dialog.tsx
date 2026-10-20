import { useEffect, useId, useRef, type ReactNode } from "react";

import { IconButton } from "./Button";
import { icons } from "./icons";
import "./Dialog.css";

interface DialogProps {
  open: boolean;
  title: string;
  /** A short line under the title, announced as the dialog's description. */
  subtitle?: ReactNode;
  children: ReactNode;
  /** Buttons; the first focusable control receives focus when the dialog opens. */
  actions: ReactNode;
  onClose: () => void;
  /**
   * False while closing would lose something (unsaved input, a request in flight): Escape and the
   * backdrop then do nothing. Explicit buttons, including Close, still call `onClose`.
   */
  dismissible?: boolean;
  /** Close when the backdrop is clicked, if `dismissible`. */
  closeOnBackdrop?: boolean;
  /** Disables the Close button, for example while saving. */
  closeDisabled?: boolean;
}

/**
 * A modal dialog built on the native <dialog> element: the browser traps focus, makes the page
 * inert and closes on Escape. Focus moves to an element marked `data-autofocus`, if any, and
 * returns to the control that opened the dialog when it closes.
 */
export function Dialog({
  open,
  title,
  subtitle,
  children,
  actions,
  onClose,
  dismissible = true,
  closeOnBackdrop = false,
  closeDisabled = false,
}: DialogProps) {
  const ref = useRef<HTMLDialogElement>(null);
  const opener = useRef<HTMLElement | null>(null);
  const pressedBackdrop = useRef(false);
  const titleId = useId();
  const subtitleId = useId();

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) {
      opener.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
      dialog.showModal();
      dialog.querySelector<HTMLElement>("[data-autofocus]")?.focus();
    }
    if (!open && dialog.open) {
      dialog.close();
      if (opener.current?.isConnected) opener.current.focus();
      opener.current = null;
    }
  }, [open]);

  return (
    <dialog
      ref={ref}
      className="dialog"
      aria-labelledby={titleId}
      aria-describedby={open && subtitle ? subtitleId : undefined}
      onCancel={(event) => {
        event.preventDefault();
        if (dismissible) onClose();
      }}
      // Only a press that starts and ends on the backdrop closes, so selecting text cannot.
      onMouseDown={(event) => (pressedBackdrop.current = event.target === event.currentTarget)}
      onClick={(event) => {
        const backdrop = pressedBackdrop.current && event.target === event.currentTarget;
        pressedBackdrop.current = false;
        if (backdrop && closeOnBackdrop && dismissible) onClose();
      }}
    >
      {open ? (
        <div className="dialog__frame">
          <div className="dialog__header">
            <div className="dialog__heading">
              <h2 id={titleId} className="type-heading">
                {title}
              </h2>
              {subtitle ? (
                <p id={subtitleId} className="type-caption">
                  {subtitle}
                </p>
              ) : null}
            </div>
            <IconButton icon={icons.close} label="Close" size="sm" onClick={onClose} disabled={closeDisabled} />
          </div>
          <div className="dialog__body">{children}</div>
          <div className="dialog__actions">{actions}</div>
        </div>
      ) : null}
    </dialog>
  );
}
