/**
 * Optional: a stream copy of one proposal's review interval, made only on request. A clip is a
 * convenience copy of the source; the review itself always plays the original.
 */
import { useRef } from "react";

import { ApiError, errorMessage, isOffline } from "../../api/client";
import { useCreateClip } from "../../api/queries";
import { formatBytes, formatTimecode } from "../../lib/format";
import { Alert } from "../../ui/Alert";
import { Button, buttonClass } from "../../ui/Button";
import { Icon } from "../../ui/Icon";
import { icons } from "../../ui/icons";
import { Tooltip } from "../../ui/Tooltip";

/** Whether this browser reports any chance of playing the type; "" means it cannot. */
export function canPlayType(mimeType: string): boolean {
  try {
    return document.createElement("video").canPlayType(mimeType) !== "";
  } catch {
    return false;
  }
}

interface ClipControlProps {
  analysisId: string;
  eventId: string;
}

/**
 * One place in the event header: Create clip, then Creating…, then Open clip and Download for the
 * clip just made. Details are in Download's tooltip, not on the page. The panel is keyed by
 * proposal, so a clip's links never appear under another proposal.
 */
export function ClipButton({ clip }: { clip: ReturnType<typeof useClipRequest> }) {
  const { mutation } = clip;
  if (mutation.data) {
    const { clip: file, media_url: url, mime_type: mimeType } = mutation.data;
    const playable = canPlayType(mimeType);
    const details =
      `Clip ${formatTimecode(file.first_s)} – ${formatTimecode(file.last_s)} · ${formatBytes(file.size_bytes)}, copied without re-encoding` +
      (file.first_s < file.start_s ? ", from the keyframe before the review span" : "") +
      "." +
      (playable ? "" : " This browser may not play the clip. Download it to open it elsewhere.");
    return (
      <span className="clip-actions">
        {playable ? (
          <a className={buttonClass({ variant: "secondary", size: "sm" })} href={url} target="_blank" rel="noopener">
            <Icon icon={icons.open} />
            <span>Open clip</span>
          </a>
        ) : null}
        <Tooltip content={details}>
          <a className={buttonClass({ variant: "secondary", size: "sm" })} href={url} download={file.file}>
            <Icon icon={icons.download} />
            <span>Download</span>
          </a>
        </Tooltip>
      </span>
    );
  }
  return (
    <Button size="sm" icon={icons.clip} loading={mutation.isPending} onClick={clip.request}>
      {mutation.isPending ? "Creating…" : "Create clip"}
    </Button>
  );
}

export function useClipRequest({ analysisId, eventId }: ClipControlProps) {
  const mutation = useCreateClip();
  const sending = useRef(false);
  const request = () => {
    if (sending.current) return; // repeated clicks before the button updates send one request
    sending.current = true;
    mutation.mutate(
      { analysisId, eventId },
      {
        onSettled: () => {
          sending.current = false;
        },
      },
    );
  };
  return { mutation, request };
}

/** A refused or failed clip request, explained under the summary; Create clip stays in the header. */
export function ClipResult({ clip }: { clip: ReturnType<typeof useClipRequest> }) {
  const { mutation } = clip;
  if (!mutation.isError || isOffline(mutation.error)) return null;
  const refused = mutation.error instanceof ApiError && (mutation.error.status === 422 || mutation.error.status === 409);
  return (
    <Alert tone={refused ? "warning" : "danger"} title="No clip">
      {errorMessage(mutation.error)} The review player is not affected.
    </Alert>
  );
}
