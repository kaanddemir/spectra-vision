/**
 * A reviewer's corrections to the selected proposal: adjusted start and end times and a plain-text note.
 * Spectra's proposal, its detected times and its evidence never change; edited times are always labelled as
 * the reviewer's, with the detected times beside them. Edits are independent of the review decision. Times are
 * typed and shown with every recorded decimal (`formatExactTime`, `parseTime`), so nothing is rounded.
 */
import { useId, useState, type FormEvent } from "react";

import { ApiError, errorMessage, isOffline } from "../../api/client";
import { useSetEdit } from "../../api/queries";
import type { EventEdit, EventEdits, TimelineEntry } from "../../api/types";
import { formatExactTime, formatTimecode, parseTime } from "../../lib/format";
import { Alert } from "../../ui/Alert";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { TextArea, TextField } from "../../ui/Field";
import { icons } from "../../ui/icons";
import "./EventEdit.css";

export const DESCRIPTION_MAX = 2000;

/** The recorded edit for a proposal; an unedited proposal has revision 0. */
export function editFor(
  edits: EventEdits | undefined,
  eventId: string,
): EventEdit {
  return (
    edits?.edits.find((e) => e.event_id === eventId) ?? {
      event_id: eventId,
      t_start: null,
      t_end: null,
      description: null,
      revision: 0,
      updated_at: null,
    }
  );
}

/** The times to show and seek to: the reviewer's when adjusted, otherwise the detected ones. */
export function effectiveTimes(
  entry: TimelineEntry,
  edit: EventEdit,
): { start: number; end: number; adjusted: boolean } {
  return edit.t_start !== null && edit.t_end !== null
    ? { start: edit.t_start, end: edit.t_end, adjusted: true }
    : { start: entry.t_start, end: entry.t_end, adjusted: false };
}

export const span = (start: number, end: number) =>
  Math.abs(end - start) < 0.05
    ? formatTimecode(start)
    : `${formatTimecode(start)} – ${formatTimecode(end)}`;

/** The reviewer's note, plain text kept as written. */
export function EventNote({ description }: { description: string }) {
  return (
    <section className="event-note" aria-label="Your note">
      <h3 className="type-subheading">Your note</h3>
      <p className="type-secondary event-note__text">{description}</p>
    </section>
  );
}

interface EventEditDialogProps {
  open: boolean;
  analysisId: string;
  entry: TimelineEntry;
  edit: EventEdit;
  /** End of the video in analysis time; edited times must lie inside it. */
  duration: number | null;
  onClose: () => void;
}

interface Errors {
  start?: string;
  end?: string;
}

/**
 * Editing in a compact modal. Mount it with a fresh `key` per opening so the fields start from the
 * stored edit. With unsaved input, Escape and the backdrop do nothing; Cancel and Close discard it.
 * While saving, nothing closes it.
 */
export function EventEditDialog({
  open,
  analysisId,
  entry,
  edit,
  duration,
  onClose,
}: EventEditDialogProps) {
  const current = effectiveTimes(entry, edit);
  const initial = {
    start: formatExactTime(current.start),
    end: formatExactTime(current.end),
    description: edit.description ?? "",
  };
  const [start, setStart] = useState(initial.start);
  const [end, setEnd] = useState(initial.end);
  const [description, setDescription] = useState(initial.description);
  const [errors, setErrors] = useState<Errors>({});
  const mutation = useSetEdit(analysisId);
  const formId = useId();
  const saving = mutation.isPending;
  const dirty =
    start !== initial.start ||
    end !== initial.end ||
    description !== initial.description;

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (saving) return;
    const t_start = parseTime(start);
    const t_end = parseTime(end);
    const found: Errors = {};
    const format = "Enter a time such as 0:12.5 or 1:02.04.";
    if (t_start === null) found.start = format;
    if (t_end === null) found.end = format;
    if (t_start !== null && t_end !== null) {
      if (t_start >= t_end) found.end = "The end must be after the start.";
      else if (duration !== null && t_end > duration)
        found.end = `The video ends at ${formatExactTime(duration)}.`;
      if (duration !== null && t_start >= duration)
        found.start = `The video ends at ${formatExactTime(duration)}.`;
    }
    setErrors(found);
    if (found.start || found.end || t_start === null || t_end === null) return;
    const detected = t_start === entry.t_start && t_end === entry.t_end;
    const next = {
      t_start: detected ? null : t_start,
      t_end: detected ? null : t_end,
      description: description === "" ? null : description,
    };
    if (
      next.t_start === edit.t_start &&
      next.t_end === edit.t_end &&
      next.description === edit.description
    ) {
      onClose();
      return;
    }
    mutation.mutate(
      { eventId: entry.event_id, edit: next, expectedRevision: edit.revision },
      { onSuccess: onClose },
    );
  };

  const restoreDetected = () => {
    setStart(formatExactTime(entry.t_start));
    setEnd(formatExactTime(entry.t_end));
    setErrors({});
  };

  return (
    <Dialog
      open={open}
      title="Edit event"
      subtitle={entry.title}
      onClose={() => {
        if (!saving) onClose();
      }}
      dismissible={!dirty && !saving}
      closeOnBackdrop
      closeDisabled={saving}
      actions={
        <>
          <Button variant="ghost" onClick={onClose} disabled={saving}>
            Cancel
          </Button>
          <Button type="submit" form={formId} variant="primary" loading={saving}>
            Save
          </Button>
        </>
      }
    >
      <form
        id={formId}
        className="event-edit"
        aria-label="Edit event"
        onSubmit={submit}
        noValidate
      >
        <div className="event-edit__times">
          <TextField
            label="Start"
            value={start}
            inputMode="decimal"
            autoComplete="off"
            spellCheck={false}
            error={errors.start}
            data-autofocus
            onChange={(e) => setStart(e.target.value)}
          />
          <TextField
            label="End"
            value={end}
            inputMode="decimal"
            autoComplete="off"
            spellCheck={false}
            error={errors.end}
            onChange={(e) => setEnd(e.target.value)}
          />
        </div>
        <div className="event-edit__detected">
          <span className="type-caption">
            Detected: {span(entry.t_start, entry.t_end)}
          </span>
          <Button
            size="sm"
            variant="ghost"
            icon={icons.clearDecision}
            onClick={restoreDetected}
            disabled={saving}
            aria-label="Reset to detected times"
          >
            Reset
          </Button>
        </div>
        <TextArea
          label="Your note"
          placeholder="Optional"
          value={description}
          maxLength={DESCRIPTION_MAX}
          rows={3}
          onChange={(e) => setDescription(e.target.value)}
        />
        {description !== "" ? (
          <div className="event-edit__actions">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setDescription("")}
              disabled={saving}
            >
              Clear note
            </Button>
          </div>
        ) : null}
        <SaveError error={mutation.isError ? mutation.error : null} />
      </form>
    </Dialog>
  );
}

function SaveError({ error }: { error: unknown }) {
  if (error === null) return null;
  if (isOffline(error))
    return (
      <p className="type-secondary">
        Not saved. Try again when Spectra responds.
      </p>
    );
  const refused =
    error instanceof ApiError && (error.status === 409 || error.status === 422);
  return (
    <Alert tone={refused ? "warning" : "danger"} title="Not saved">
      {errorMessage(error)}
    </Alert>
  );
}
