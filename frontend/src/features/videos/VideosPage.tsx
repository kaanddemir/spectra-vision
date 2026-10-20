import { useMemo } from "react";
import { Link, useSearchParams } from "react-router";

import { useVideos } from "../../api/queries";
import type { Video } from "../../api/types";
import { QueryStatus } from "../../app/QueryStatus";
import { formatBytes, formatDate, formatDuration } from "../../lib/format";
import { ButtonLink } from "../../ui/Button";
import { EmptyState, NoResults } from "../../ui/EmptyState";
import { SearchField } from "../../ui/Field";
import { Icon } from "../../ui/Icon";
import { icons } from "../../ui/icons";
import { PageHeader } from "../../ui/PageHeader";
import { StatusText } from "../../ui/StatusText";
import { Table } from "../../ui/Table";
import { Tabs } from "../../ui/Tabs";
import { videoStatus, type VideoStatus } from "./videoStatus";

type Filter = "all" | VideoStatus["group"];
const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "ready", label: "Ready for analysis" },
  { id: "attention", label: "Needs attention" },
];
const TABLE_ID = "videos-table";

const isFilter = (value: string | null): value is Filter => FILTERS.some((f) => f.id === value);

function matches(video: Video, query: string): boolean {
  if (!query) return true;
  const haystack = [video.file_name, video.path, video.media?.codec, video.media?.container]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return query
    .toLowerCase()
    .split(/\s+/)
    .every((word) => haystack.includes(word));
}

export function VideosPage() {
  const videos = useVideos();
  const [params, setParams] = useSearchParams();
  const rawFilter = params.get("status");
  const filter: Filter = isFilter(rawFilter) ? rawFilter : "all";
  const query = params.get("q") ?? "";

  function update(next: { status?: Filter; q?: string }) {
    const merged = new URLSearchParams(params);
    if (next.status !== undefined) {
      if (next.status === "all") merged.delete("status");
      else merged.set("status", next.status);
    }
    if (next.q !== undefined) {
      if (next.q) merged.set("q", next.q);
      else merged.delete("q");
    }
    setParams(merged, { replace: true });
  }

  return (
    <>
      <PageHeader
        title="Videos"
        description="View and manage your videos and analyses."
        actions={
          <ButtonLink to="/library" variant="primary" icon={icons.register}>
            Add videos
          </ButtonLink>
        }
      />
      <QueryStatus query={videos} loadingLabel="Loading videos…" errorTitle="Videos could not be loaded">
        {(list) =>
          list.length === 0 ? (
            <EmptyState
              layout="page"
              icon={icons.videos}
              title="No videos registered yet"
              action={
                <ButtonLink to="/library" variant="primary" icon={icons.library}>
                  Browse library
                </ButtonLink>
              }
            >
              Add videos from your library folders.
            </EmptyState>
          ) : (
            <VideoList videos={list} filter={filter} query={query} onChange={update} />
          )
        }
      </QueryStatus>
    </>
  );
}

interface VideoListProps {
  videos: Video[];
  filter: Filter;
  query: string;
  onChange: (next: { status?: Filter; q?: string }) => void;
}

function VideoList({ videos, filter, query, onChange }: VideoListProps) {
  const rows = useMemo(
    () =>
      videos
        .map((video) => ({ video, status: videoStatus(video) }))
        .sort((a, b) => b.video.registered_at.localeCompare(a.video.registered_at)),
    [videos],
  );
  const searched = rows.filter((row) => matches(row.video, query));
  const visible = searched.filter((row) => filter === "all" || row.status.group === filter);
  const counts = (id: Filter) => searched.filter((row) => id === "all" || row.status.group === id).length;

  return (
    <div className="stack">
      <Tabs
        label="Filter videos by status"
        controls={TABLE_ID}
        value={filter}
        onChange={(status) => onChange({ status })}
        items={FILTERS.map((f) => ({ ...f, count: counts(f.id) }))}
      />
      <div className="toolbar">
        <SearchField
          label="Search videos"
          placeholder="Search videos…"
          value={query}
          onChange={(event) => onChange({ q: event.target.value })}
        />
      </div>
      <div id={TABLE_ID} role="tabpanel" aria-labelledby={`${TABLE_ID}-tab-${filter}`}>
        {visible.length === 0 ? (
          <NoResults onClear={query || filter !== "all" ? () => onChange({ status: "all", q: "" }) : undefined} />
        ) : (
          <Table label="Registered videos" columns="fit">
            <thead>
              <tr>
                <th scope="col" className="cell-primary">
                  Name
                </th>
                <th scope="col">Status</th>
                <th scope="col" className="cell-numeric">
                  Duration
                </th>
                <th scope="col" className="cell-numeric">
                  Size
                </th>
                <th scope="col">Registered</th>
              </tr>
            </thead>
            <tbody>
              {visible.map(({ video, status }) => (
                <tr key={video.id}>
                  <td className="cell-primary">
                    <div className="cell-primary__content">
                      <Icon icon={icons.video} size="md" className="cell-primary__icon" />
                      <div className="cell-primary__text">
                        <Link to={`/videos/${video.id}`} className="cell-primary__title truncate" title={video.file_name}>
                          {video.file_name}
                        </Link>
                        <span className="cell-primary__subtitle truncate">{video.path ?? "Outside the library folders"}</span>
                      </div>
                    </div>
                  </td>
                  <td>
                    <StatusText
                      tone={status.tone}
                      icon={status.icon}
                      spin={status.kind === "probing"}
                      description={status.short !== status.label ? status.label : undefined}
                    >
                      {status.short}
                    </StatusText>
                  </td>
                  <td className="cell-numeric">{formatDuration(video.media?.duration_s)}</td>
                  <td className="cell-numeric">{formatBytes(video.size_bytes)}</td>
                  <td className="cell-secondary">
                    <time dateTime={video.registered_at}>{formatDate(video.registered_at)}</time>
                  </td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </div>
    </div>
  );
}
