import { useMemo, useState } from "react";
import { useSearchParams } from "react-router";

import { errorMessage, isNotFound, isOffline } from "../../api/client";
import {
  useAddLibraryFolder,
  useLibraryEntries,
  useLibraryFolders,
  useRegisterVideo,
  useRemoveLibraryFolder,
  useVideos,
} from "../../api/queries";
import type { LibraryEntry, LibraryFolder, LibraryListing, Video } from "../../api/types";
import { QueryStatus } from "../../app/QueryStatus";
import { formatBytes, formatDate, pluralize } from "../../lib/format";
import { Alert } from "../../ui/Alert";
import { useToast } from "../../ui/Toast";
import { Breadcrumbs, type Crumb } from "../../ui/Breadcrumbs";
import { Button, ButtonLink, IconButton } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { EmptyState, NoResults } from "../../ui/EmptyState";
import { Checkbox, SearchField, Select } from "../../ui/Field";
import { Icon } from "../../ui/Icon";
import { icons } from "../../ui/icons";
import { PageHeader } from "../../ui/PageHeader";
import { StatusText } from "../../ui/StatusText";
import { Table } from "../../ui/Table";
import { videoStatus } from "../videos/videoStatus";

export function LibraryPage() {
  const folders = useLibraryFolders();
  const [params, setParams] = useSearchParams();
  const addFolder = useAddFolder((folderId) => setParams({ folder: folderId }));
  const hasFolders = (folders.data?.length ?? 0) > 0;
  return (
    <>
      <PageHeader
        title="Library"
        description="Browse folders and add videos to Spectra."
        actions={hasFolders ? <AddFolderButton add={addFolder} /> : undefined}
      />
      <QueryStatus query={folders} loadingLabel="Loading library folders…" errorTitle="Library folders could not be loaded">
        {(list) =>
          list.length === 0 ? (
            <EmptyState
              layout="page"
              icon={icons.library}
              title="No library folders yet"
              action={<AddFolderButton add={addFolder} variant="primary" />}
            >
              Add a folder of videos. Files stay where they are.
            </EmptyState>
          ) : (
            <Browser folders={list} selectedId={params.get("folder")} />
          )
        }
      </QueryStatus>
    </>
  );
}

type AddFolder = ReturnType<typeof useAddFolder>;

/** Adds a folder through the native dialog and opens it; a cancelled dialog changes nothing. */
function useAddFolder(onOpen: (folderId: string) => void) {
  const add = useAddLibraryFolder();
  const toast = useToast();
  return {
    pending: add.isPending,
    run: () =>
      add.mutate(undefined, {
        onSuccess: ({ folder, created }) => {
          if (!folder) return;
          toast(created ? { tone: "success", title: "Folder added", description: folder.name } : { tone: "info", title: "Already in your library", description: folder.name });
          onOpen(folder.id);
        },
        onError: (error) => {
          if (!isOffline(error)) toast({ tone: "danger", key: "add-folder", title: "Folder not added", description: errorMessage(error) });
        },
      }),
  };
}

function AddFolderButton({ add, variant = "secondary" }: { add: AddFolder; variant?: "primary" | "secondary" }) {
  return (
    <Button variant={variant} icon={icons.addFolder} loading={add.pending} onClick={add.run}>
      {add.pending ? "Choose a folder…" : "Add folder"}
    </Button>
  );
}

function RemoveFolder({ folder, onRemoved }: { folder: LibraryFolder; onRemoved: () => void }) {
  const remove = useRemoveLibraryFolder();
  const toast = useToast();
  const [confirming, setConfirming] = useState(false);
  if (folder.source !== "added") return null;
  const run = () =>
    remove.mutate(folder.id, {
      onSuccess: () => {
        setConfirming(false);
        toast({ tone: "success", title: "Folder removed", description: folder.name });
        onRemoved();
      },
      onError: (error) => {
        setConfirming(false);
        if (!isOffline(error)) toast({ tone: "danger", key: "remove-folder", title: "Folder not removed", description: errorMessage(error) });
      },
    });
  return (
    <>
      <IconButton
        icon={icons.removeFolder}
        label="Remove folder"
        disabled={remove.isPending}
        onClick={() => (folder.registered_videos > 0 ? setConfirming(true) : run())}
      />
      <Dialog
        open={confirming}
        title="Remove this folder?"
        onClose={() => setConfirming(false)}
        actions={
          <>
            <Button onClick={() => setConfirming(false)}>Cancel</Button>
            <Button variant="danger" icon={icons.removeFolder} loading={remove.isPending} onClick={run}>
              Remove folder
            </Button>
          </>
        }
      >
        <p>
          {pluralize(folder.registered_videos, "registered video")} in <strong>{folder.name}</strong> stay in Spectra with
          their analyses, but can't be played or analysed until you add the folder again. No files are deleted.
        </p>
      </Dialog>
    </>
  );
}

interface RegistrationResult {
  registered: number;
  failures: { name: string; message: string }[];
}

function Browser({ folders, selectedId }: { folders: LibraryFolder[]; selectedId: string | null }) {
  const [params, setParams] = useSearchParams();
  const folder = folders.find((f) => f.id === selectedId) ?? folders[0]!;
  const path = params.get("path") ?? "";
  const listing = useLibraryEntries(folder.available ? folder.id : undefined, path);
  const videos = useVideos();
  const register = useRegisterVideo();
  const [filter, setFilter] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState<Set<string>>(new Set());
  const toast = useToast();

  function open(nextFolder: string, nextPath: string) {
    const next = new URLSearchParams();
    next.set("folder", nextFolder);
    if (nextPath) next.set("path", nextPath);
    setParams(next);
    setSelected(new Set());
    setFilter("");
  }

  async function registerAll(entries: LibraryEntry[]) {
    setBusy(new Set(entries.map((e) => e.path)));
    const outcome: RegistrationResult = { registered: 0, failures: [] };
    for (const entry of entries) {
      try {
        await register.mutateAsync({ folderId: folder.id, path: entry.path });
        outcome.registered += 1;
      } catch (error) {
        if (isOffline(error)) break; // the global banner reports it; the rest would fail the same way
        outcome.failures.push({ name: entry.name, message: errorMessage(error) });
      }
      setBusy((current) => {
        const next = new Set(current);
        next.delete(entry.path);
        return next;
      });
    }
    setBusy(new Set());
    setSelected(new Set());
    if (outcome.registered > 0) {
      toast({ tone: "success", title: `Registered ${pluralize(outcome.registered, "video")}`, description: "Reading details in the background." });
    }
    if (outcome.failures.length > 0) {
      toast({
        tone: "danger",
        key: "registration-failed",
        title: `${pluralize(outcome.failures.length, "video")} not registered`,
        description: outcome.failures.map((f) => `${f.name}: ${f.message}`).join(" "),
      });
    }
  }

  const labels = folderLabels(folders);
  // The library folder's own name starts the location; its full path stays one hover away.
  const crumbs: Crumb[] = [
    { label: <span title={folder.path}>{labels.get(folder.id)}</span>, onSelect: () => open(folder.id, "") },
  ];
  const parts = path ? path.split("/") : [];
  parts.forEach((part, index) => {
    const target = parts.slice(0, index + 1).join("/");
    crumbs.push({ label: part, onSelect: () => open(folder.id, target) });
  });

  const toolbarFolder = (
    <>
      {folders.length > 1 ? (
        <Select
          label="Library folder"
          value={folder.id}
          onChange={(event) => open(event.target.value, "")}
          options={folders.map((f) => ({ value: f.id, label: f.available ? labels.get(f.id)! : `${labels.get(f.id)} (unavailable)` }))}
        />
      ) : null}
    </>
  );
  // Where you are, as breadcrumbs; the folder's own action sits at the end of the same line.
  const location = (
    <div className="library-location">
      <Breadcrumbs label="Folder path" items={crumbs} />
      <RemoveFolder folder={folder} onRemoved={() => setParams(new URLSearchParams())} />
    </div>
  );

  if (!folder.available) {
    return (
      <div className="stack">
        {folders.length > 1 ? (
          <div className="toolbar">
            <div className="toolbar__group">{toolbarFolder}</div>
          </div>
        ) : null}
        {location}
        <EmptyState icon={icons.folder} title="Folder not available">
          {folder.name} was moved, deleted or can't be read.
          {folder.source === "added" ? " Remove it, or add it again from its new location." : null}
        </EmptyState>
      </div>
    );
  }

  if (listing.isError && isNotFound(listing.error) && path) {
    return (
      <EmptyState
        icon={icons.folder}
        title="Folder not found"
        action={<Button onClick={() => open(folder.id, "")}>Go to {folder.name}</Button>}
      >
        It may have been moved or deleted.
      </EmptyState>
    );
  }

  return (
    <div className="stack">
      <div className="toolbar">
        <div className="toolbar__group">
          {toolbarFolder}
          <SearchField
            label="Filter this folder"
            placeholder="Search files…"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
          />
        </div>
      </div>
      {location}
      <QueryStatus query={listing} loadingLabel="Loading folder…" errorTitle="This folder could not be listed">
        {(data) => (
          <Listing
            listing={data}
            videos={videos.data}
            filter={filter}
            selected={selected}
            busy={busy}
            onSelect={setSelected}
            onOpen={(entryPath) => open(folder.id, entryPath)}
            onRegister={(entries) => void registerAll(entries)}
            onClearFilter={() => setFilter("")}
          />
        )}
      </QueryStatus>
    </div>
  );
}

interface ListingProps {
  listing: LibraryListing;
  videos: Video[] | undefined;
  filter: string;
  selected: Set<string>;
  busy: Set<string>;
  onSelect: (selected: Set<string>) => void;
  onOpen: (path: string) => void;
  onRegister: (entries: LibraryEntry[]) => void;
  onClearFilter: () => void;
}

function Listing({ listing, videos, filter, selected, busy, onSelect, onOpen, onRegister, onClearFilter }: ListingProps) {
  const byId = useMemo(() => new Map((videos ?? []).map((v) => [v.id, v])), [videos]);
  const entries = listing.entries.filter((entry) => entry.name.toLowerCase().includes(filter.trim().toLowerCase()));
  const selectable = entries.filter((entry) => entry.kind === "video" && !entry.video_id);
  const chosen = selectable.filter((entry) => selected.has(entry.path));
  const registering = busy.size > 0;

  function toggle(path: string, on: boolean) {
    const next = new Set(selected);
    if (on) next.add(path);
    else next.delete(path);
    onSelect(next);
  }

  if (listing.entries.length === 0) {
    return (
      <EmptyState icon={icons.folder} title="No videos or folders here">
        Only folders and video files are listed.
      </EmptyState>
    );
  }

  return (
    <div className="stack">
      {listing.truncated ? (
        <Alert tone="warning" title="Folder partly listed">
          Showing the first {listing.entries.length} entries.
        </Alert>
      ) : null}
      <div className="selection-bar" aria-live="polite" hidden={chosen.length === 0}>
        {chosen.length > 0 ? (
          <>
            <span className="type-subheading type-numeric">{pluralize(chosen.length, "video")} selected</span>
            <Button variant="primary" size="sm" icon={icons.register} loading={registering} onClick={() => onRegister(chosen)}>
              Register selected
            </Button>
            <Button variant="ghost" size="sm" disabled={registering} onClick={() => onSelect(new Set())}>
              Clear selection
            </Button>
          </>
        ) : null}
      </div>
      {entries.length === 0 ? (
        <NoResults onClear={filter ? onClearFilter : undefined} />
      ) : (
        <Table label={`Contents of ${listing.path || listing.folder.name}`} columns="fit">
          <thead>
            <tr>
              <th scope="col" className="cell-select">
                <Checkbox
                  label="Select all unregistered videos"
                  hideLabel
                  disabled={selectable.length === 0 || registering}
                  checked={selectable.length > 0 && chosen.length === selectable.length}
                  indeterminate={chosen.length > 0 && chosen.length < selectable.length}
                  onChange={(event) => onSelect(new Set(event.target.checked ? selectable.map((e) => e.path) : []))}
                />
              </th>
              <th scope="col" className="cell-primary">
                Name
              </th>
              <th scope="col">Status</th>
              <th scope="col" className="cell-numeric">
                Size
              </th>
              <th scope="col">Modified</th>
              <th scope="col" className="cell-actions">
                <span className="visually-hidden">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry) => (
              <EntryRow
                key={entry.path}
                entry={entry}
                video={entry.video_id ? byId.get(entry.video_id) : undefined}
                selected={selected.has(entry.path)}
                busy={busy.has(entry.path)}
                disabled={registering}
                onToggle={(on) => toggle(entry.path, on)}
                onOpen={() => onOpen(entry.path)}
                onRegister={() => onRegister([entry])}
              />
            ))}
          </tbody>
        </Table>
      )}
    </div>
  );
}

interface EntryRowProps {
  entry: LibraryEntry;
  video: Video | undefined;
  selected: boolean;
  busy: boolean;
  disabled: boolean;
  onToggle: (on: boolean) => void;
  onOpen: () => void;
  onRegister: () => void;
}

function EntryRow({ entry, video, selected, busy, disabled, onToggle, onOpen, onRegister }: EntryRowProps) {
  if (entry.kind === "directory") {
    return (
      <tr className="is-folder">
        <td className="cell-select" />
        <td className="cell-primary">
          <div className="cell-primary__content">
            <Icon icon={icons.folder} size="md" className="cell-primary__icon" />
            <button type="button" className="cell-primary__title truncate" title={entry.name} onClick={onOpen}>
              {entry.name}
            </button>
          </div>
        </td>
        <td className="cell-secondary">Folder</td>
        <td className="cell-numeric" />
        <td className="cell-secondary" />
        <td className="cell-actions" />
      </tr>
    );
  }
  const status = video ? videoStatus(video) : null;
  return (
    <tr className={selected && !entry.video_id ? "is-selected" : undefined}>
      <td className="cell-select">
        {entry.video_id ? null : (
          <Checkbox label={`Select ${entry.name}`} hideLabel checked={selected} disabled={disabled} onChange={(event) => onToggle(event.target.checked)} />
        )}
      </td>
      <td className="cell-primary">
        <div className="cell-primary__content">
          <Icon icon={icons.video} size="md" className="cell-primary__icon" />
          <span className="cell-primary__title truncate" title={entry.name}>
            {entry.name}
          </span>
        </div>
      </td>
      <td>
        {status ? (
          <StatusText
            tone={status.tone}
            icon={status.icon}
            spin={status.kind === "probing"}
            description={status.short !== status.label ? status.label : undefined}
          >
            {status.short}
          </StatusText>
        ) : entry.video_id ? (
          <StatusText tone="neutral" icon={icons.check}>
            Registered
          </StatusText>
        ) : (
          <span className="type-secondary">Not registered</span>
        )}
      </td>
      <td className="cell-numeric">{formatBytes(entry.size_bytes)}</td>
      <td className="cell-secondary">
        {entry.modified_at ? <time dateTime={entry.modified_at}>{formatDate(entry.modified_at)}</time> : formatDate(null)}
      </td>
      <td className="cell-actions">
        {/* One quiet style for both row actions; the label says whether it navigates or registers. */}
        {entry.video_id ? (
          <ButtonLink to={`/videos/${entry.video_id}`} size="sm" variant="ghost" aria-label={`Open ${entry.name}`}>
            Open
          </ButtonLink>
        ) : (
          <Button size="sm" variant="ghost" loading={busy} disabled={disabled && !busy} onClick={onRegister} aria-label={`Register ${entry.name}`}>
            Register
          </Button>
        )}
      </td>
    </tr>
  );
}

/**
 * Names that tell library folders apart: the folder's own name, with its parent folder's name added
 * when two folders share a name, and the full path when even that is the same.
 */
export function folderLabels(folders: LibraryFolder[]): Map<string, string> {
  const count = (values: string[], value: string) => values.filter((v) => v === value).length;
  const names = folders.map((f) => f.name);
  const withParent = folders.map((f) => {
    if (count(names, f.name) < 2) return f.name;
    const parent = f.path.split("/").filter(Boolean).slice(-2, -1)[0];
    return parent ? `${f.name} (${parent})` : f.path;
  });
  return new Map(folders.map((f, i) => [f.id, count(withParent, withParent[i]!) < 2 ? withParent[i]! : f.path]));
}
