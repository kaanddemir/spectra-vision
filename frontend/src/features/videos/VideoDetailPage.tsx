import { useRef, useState, type ReactNode, type RefObject } from "react";
import { Link, useNavigate, useParams } from "react-router";

import { errorMessage, isNotFound } from "../../api/client";
import { useAnalyses, useForgetVideo, useLibraryFolders, useVideo } from "../../api/queries";
import type { Analysis, Video } from "../../api/types";
import { QueryStatus } from "../../app/QueryStatus";
import {
  formatBytes,
  formatDateTime,
  formatDateTimeSeconds,
  formatFrameRate,
  formatLength,
  formatResolution,
  pluralize,
  sentence,
} from "../../lib/format";
import { Alert } from "../../ui/Alert";
import { Button, ButtonLink } from "../../ui/Button";
import { DescriptionList } from "../../ui/DescriptionList";
import { Dialog } from "../../ui/Dialog";
import { EmptyState } from "../../ui/EmptyState";
import { Icon } from "../../ui/Icon";
import { icons } from "../../ui/icons";
import { PageHeader } from "../../ui/PageHeader";
import { Section } from "../../ui/Section";
import { Spinner } from "../../ui/Spinner";
import { StatusText } from "../../ui/StatusText";
import { useToast } from "../../ui/Toast";
import { AnalysisControl } from "../analyses/AnalysisControl";
import { analysisStatus } from "../analyses/analysisStatus";
import { resultHere, usePlaybackCheck, verifiedHere } from "./usePlaybackCheck";

export function VideoDetailPage() {
  const { videoId = "" } = useParams();
  const video = useVideo(videoId);

  if (video.isError && isNotFound(video.error)) {
    return (
      <EmptyState
        layout="page"
        icon={icons.video}
        title="Video not found"
        action={
          <ButtonLink to="/videos" icon={icons.back}>
            Back to videos
          </ButtonLink>
        }
      />
    );
  }
  return (
    <QueryStatus query={video} loadingLabel="Loading video…" errorTitle="The video could not be loaded">
      {(data) => <VideoDetail key={data.id} video={data} />}
    </QueryStatus>
  );
}

function VideoDetail({ video }: { video: Video }) {
  const analyses = useAnalyses(video.id, { poll: false });
  const folder = useLibraryFolders().data?.find((f) => f.id === video.folder_id);
  const [forgetting, setForgetting] = useState(false);
  const hasAnalyses = (analyses.data?.length ?? 0) > 0;
  const mediaRef = useRef<HTMLVideoElement>(null);
  const check = usePlaybackCheck(video, mediaRef);

  return (
    <>
      <PageHeader
        title={video.file_name}
        description={
          video.path
            ? folder
              ? `${folder.name} / ${video.path}`
              : video.path
            : "Outside the library folders"
        }
        actions={
          // Videos with analyses keep their registration: the server refuses to forget them.
          analyses.isSuccess && !hasAnalyses ? (
            <Button variant="danger" icon={icons.forget} onClick={() => setForgetting(true)}>
              Forget video
            </Button>
          ) : null
        }
      />
      <div className="workspace">
        <div className="workspace__media">
          <Player video={video} mediaRef={mediaRef} />
          <VideoInformation video={video} />
        </div>
        <aside className="workspace__aside" aria-label="Analysis">
          <AnalysisControl video={video} analyses={analyses.data} verified={verifiedHere(video)} />
          <Readiness video={video} check={check} />
          <PreviousAnalyses analyses={analyses} />
        </aside>
      </div>
      <ForgetDialog video={video} open={forgetting} onClose={() => setForgetting(false)} />
    </>
  );
}

function stageMessage(video: Video): string {
  if (video.source.status !== "available") return sentence(video.source.message ?? "The video file is unavailable.");
  if (video.probe.state === "pending") return "Reading the video's details…";
  if (video.probe.state === "failed") return "The video could not be read, so it cannot be played.";
  if (video.playback.assessment === "unsupported") return "Browsers cannot play this format.";
  return "The video cannot be played right now.";
}

function Player({ video, mediaRef }: { video: Video; mediaRef: RefObject<HTMLVideoElement | null> }) {
  return (
    <div className="media-stage">
      {video.media_url ? (
        <video
          ref={mediaRef}
          src={video.media_url}
          controls
          preload="metadata"
          playsInline
          aria-label={`Preview of ${video.file_name}`}
        />
      ) : (
        <div className="media-stage__message">
          {video.probe.state === "pending" ? <Spinner size="lg" /> : <Icon icon={icons.unsupported} size="lg" />}
          <p>{stageMessage(video)}</p>
        </div>
      )}
    </div>
  );
}

/** Four properties a person recognises, under the player. A value that is not known says so. */
function VideoInformation({ video }: { video: Video }) {
  const media = video.media;
  return (
    <Section title="Video information">
      <DescriptionList
        columns={4}
        align="center"
        items={[
          { term: "Resolution", value: media ? formatResolution(media.width, media.height) : UNKNOWN },
          { term: "Duration", value: media ? formatLength(media.duration_s) : UNKNOWN },
          { term: "Frame rate", value: formatFrameRate(media?.nominal_fps) },
          { term: "File size", value: formatBytes(video.size_bytes) },
        ]}
      />
    </Section>
  );
}

const UNKNOWN = "Unknown";

type RowTone = "success" | "warning" | "danger" | "pending" | "neutral";

interface ReadinessRow {
  label: string;
  status: string;
  tone: RowTone;
  /** One short line under the row, only where it tells the person something the status cannot. */
  note?: string | null;
  action?: ReactNode;
}

const ROW_ICONS: Record<RowTone, typeof icons.success> = {
  success: icons.success,
  warning: icons.warning,
  danger: icons.danger,
  pending: icons.pending,
  neutral: icons.notStarted,
};

function sourceRow(video: Video): ReadinessRow {
  switch (video.source.status) {
    case "available":
      return { label: "Source file", status: "Ready", tone: "success" };
    case "missing":
      return { label: "Source file", status: "Missing", tone: "danger", note: "The file was moved or deleted." };
    case "changed":
      return { label: "Source file", status: "Changed", tone: "warning", note: "Register the video again." };
    default:
      return {
        label: "Source file",
        status: "Unavailable",
        tone: "warning",
        note: "Add its folder to the library again.",
      };
  }
}

function formatRow(video: Video): ReadinessRow {
  if (video.probe.state === "pending") return { label: "Video format", status: "Checking", tone: "pending" };
  if (video.probe.state === "failed") return { label: "Video format", status: "Unreadable", tone: "danger" };
  if (video.playback.assessment === "unsupported") {
    const reason = video.playback.reasons[0];
    return { label: "Video format", status: "Unsupported", tone: "danger", note: reason ? sentence(reason) : null };
  }
  return { label: "Video format", status: "Supported", tone: "success" };
}

/** Browser playback, as verified by this browser (a result from another browser does not count here). */
function playbackRow(video: Video, check: ReturnType<typeof usePlaybackCheck>): ReadinessRow {
  const label = "Browser playback";
  const { state, start } = check;
  if (state.phase === "loading" || state.phase === "saving") return { label, status: "Checking", tone: "pending" };
  if (state.phase === "waiting") {
    return { label, status: "Press Play", tone: "neutral", note: "Play the video once to verify it." };
  }
  if (state.phase === "failed") {
    return {
      label,
      status: "Not checked",
      tone: "warning",
      note: state.message,
      action: (
        <Button size="sm" variant="ghost" icon={icons.refresh} onClick={() => void start()}>
          Try again
        </Button>
      ),
    };
  }
  const here = resultHere(video);
  if (here?.playable) return { label, status: "Verified", tone: "success" };
  if (here) return { label, status: "Failed", tone: "danger", note: "This browser cannot play the video." };
  if (!video.media_url) return { label, status: "Unavailable", tone: "neutral" };
  return { label, status: "Not checked", tone: "neutral" };
}

/** Three checks about the video itself, each with a short status from the server's and this browser's state. */
function Readiness({ video, check }: { video: Video; check: ReturnType<typeof usePlaybackCheck> }) {
  const rows = [sourceRow(video), formatRow(video), playbackRow(video, check)];
  return (
    <Section title="Analysis readiness" variant="divided">
      <ul className="readiness">
        {rows.map((row) => (
          <li key={row.label} className="readiness__row">
            <div className="readiness__main">
              <Icon icon={ROW_ICONS[row.tone]} className={`readiness__icon readiness__icon--${row.tone}`} />
              <span className="readiness__label">{row.label}</span>
              <span
                className={`readiness__status readiness__status--${row.tone}`}
                role={row.tone === "pending" ? "status" : undefined}
              >
                {row.status}
              </span>
            </div>
            {row.note || row.action ? (
              <div className="readiness__note">
                {row.note ? <span>{row.note}</span> : null}
                {row.action}
              </div>
            ) : null}
          </li>
        ))}
      </ul>
    </Section>
  );
}

/** Rows shown before the list scrolls; the height itself is three rows in layout.css. */
const HISTORY_ROWS = 3;

/**
 * Earlier analyses of this video, newest first; the latest is shown in the Analysis panel instead. Beyond
 * three, the list keeps the height of three rows and scrolls on its own, so history never lengthens the page.
 */
function PreviousAnalyses({ analyses }: { analyses: ReturnType<typeof useAnalyses> }) {
  const previous = (analyses.data?.slice(1) ?? []).sort((a, b) => b.created_at.localeCompare(a.created_at));
  const scrolls = previous.length > HISTORY_ROWS;
  // Analyses started within the same minute are told apart by their seconds.
  const minutes = previous.map((a) => formatDateTime(a.created_at));
  const label = (analysis: Analysis, index: number) =>
    minutes.indexOf(minutes[index]!) !== minutes.lastIndexOf(minutes[index]!)
      ? formatDateTimeSeconds(analysis.created_at)
      : minutes[index]!;

  return (
    <Section title="Previous analyses" variant="divided">
      <QueryStatus query={analyses} loadingLabel="Loading analyses…" errorTitle="Analyses could not be loaded">
        {() =>
          previous.length === 0 ? (
            <p className="type-secondary">No previous analyses.</p>
          ) : (
            <ul
              className={`history${scrolls ? " history--scroll" : ""}`}
              // A scrolling list takes focus so keyboard users can scroll it.
              tabIndex={scrolls ? 0 : undefined}
              aria-label="Previous analyses, newest first"
            >
              {previous.map((analysis, index) => {
                const status = analysisStatus(analysis);
                return (
                  <li key={analysis.id}>
                    <Link to={`/analyses/${analysis.id}`} className="history__row">
                      <span className="history__text">
                        <span className="history__date type-numeric">{label(analysis, index)}</span>
                        <span className="history__meta">
                          <StatusText tone={status.tone} icon={status.icon} spin={status.spin}>
                            {status.label}
                          </StatusText>
                          {analysis.result ? (
                            <span className="type-caption">· {pluralize(analysis.result.proposals, "proposal")}</span>
                          ) : null}
                        </span>
                      </span>
                      <Icon icon={icons.chevronRight} className="history__chevron" />
                    </Link>
                  </li>
                );
              })}
            </ul>
          )
        }
      </QueryStatus>
    </Section>
  );
}

function ForgetDialog({ video, open, onClose }: { video: Video; open: boolean; onClose: () => void }) {
  const forget = useForgetVideo();
  const navigate = useNavigate();
  const toast = useToast();
  const close = () => {
    forget.reset();
    onClose();
  };
  return (
    <Dialog
      open={open}
      title="Forget this video?"
      onClose={close}
      actions={
        <>
          <Button onClick={close}>Cancel</Button>
          <Button
            variant="danger"
            icon={icons.forget}
            loading={forget.isPending}
            onClick={() =>
              forget.mutate(video.id, {
                onSuccess: () => {
                  toast({ tone: "success", title: "Video forgotten", description: video.file_name });
                  void navigate("/videos", { replace: true });
                },
              })
            }
          >
            Forget video
          </Button>
        </>
      }
    >
      <div className="stack">
        <p>
          The registration is removed. <strong>{video.file_name}</strong> stays where it is, unchanged.
        </p>
        {forget.isError ? (
          <Alert tone="danger" title="The video could not be forgotten">
            {errorMessage(forget.error)}
          </Alert>
        ) : null}
      </div>
    </Dialog>
  );
}
