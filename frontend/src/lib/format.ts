/** Display formatting for sizes, durations and times. Missing values become an em dash. */

export const MISSING = "—";

const UNITS = ["B", "KB", "MB", "GB", "TB"];

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return MISSING;
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < UNITS.length - 1) {
    value /= 1000;
    unit += 1;
  }
  const digits = unit === 0 || value >= 100 ? 0 : 1;
  return `${value.toFixed(digits)} ${UNITS[unit]}`;
}

/** Whole-second durations: 0:42, 12:05, 1:02:09. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return MISSING;
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const ss = String(s).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

/** A length in words for reading, not for seeking: 7 sec, 2 min 5 sec, 1 h 3 min. */
export function formatLength(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return MISSING;
  const total = Math.max(1, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return m > 0 ? `${h} h ${m} min` : `${h} h`;
  if (m > 0) return s > 0 ? `${m} min ${s} sec` : `${m} min`;
  return `${s} sec`;
}

/** Most decimal places accepted in an edited time; finer than any recorded timestamp. */
export const TIME_DECIMALS = 6;

/**
 * A position with every recorded decimal and nothing rounded, for editing: 0:07.4, 0:12.04, 1:02:09.0.
 * `parseTime` reads it back to the same number.
 */
export function formatExactTime(seconds: number): string {
  const text = Number.isInteger(seconds) ? `${seconds}.0` : String(seconds);
  if (!/^\d+\.\d+$/.test(text)) return seconds.toFixed(TIME_DECIMALS).replace(/0+$/, "").replace(/\.$/, ".0");
  const [whole, fraction] = text.split(".") as [string, string];
  return `${formatDuration(Number(whole))}.${fraction}`;
}

/**
 * Read a typed position: seconds ("12.5"), m:ss ("1:02.34") or h:mm:ss ("1:02:03.5"), with at most
 * TIME_DECIMALS decimals. The value is built from the digits, so it is exactly the decimal typed.
 * Returns null when the text is not a time.
 */
export function parseTime(text: string): number | null {
  const match = /^\s*(?:(?:(\d+):)?(\d+):)?(\d+)(?:\.(\d{1,6}))?\s*$/.exec(text);
  if (!match) return null;
  const [, h, m, s, fraction] = match;
  if ((m !== undefined && Number(s) >= 60) || (h !== undefined && Number(m) >= 60)) return null;
  const whole = Number(h ?? 0) * 3600 + Number(m ?? 0) * 60 + Number(s);
  return Number(`${whole}.${fraction ?? "0"}`);
}

/** Positions inside a video, to a tenth of a second: 0:07.4, 1:02:09.0. */
export function formatTimecode(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return MISSING;
  const tenths = Math.round(seconds * 10);
  const whole = Math.floor(tenths / 10);
  return `${formatDuration(whole)}.${tenths % 10}`;
}

const dateTime = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

const dateOnly = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric" });

/** A calendar date without the time: 20 Oct 2026. */
export function formatDate(iso: string | null | undefined): string {
  if (!iso) return MISSING;
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? MISSING : dateOnly.format(date);
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return MISSING;
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? MISSING : dateTime.format(date);
}

const dateTimeSeconds = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

/** Date and time with seconds, for telling apart moments within the same minute: 20 Oct 2026, 17:23:05. */
export function formatDateTimeSeconds(iso: string | null | undefined): string {
  if (!iso) return MISSING;
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? MISSING : dateTimeSeconds.format(date);
}

const clock = new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });

export function formatClock(epochMs: number): string {
  return clock.format(new Date(epochMs));
}

export function formatResolution(width: number, height: number): string {
  return `${width} × ${height}`;
}

/**
 * The frame rate the video reports, for reading: 30 FPS, 29.97 FPS, 60 FPS (at most two decimals, no
 * trailing zeros). Missing, zero, negative or non-finite rates are "Unknown"; nothing is estimated.
 */
export function formatFrameRate(fps: number | null | undefined): string {
  if (fps === null || fps === undefined || !Number.isFinite(fps) || fps <= 0) return "Unknown";
  return `${Number(fps.toFixed(2))} FPS`;
}

const CODECS: Record<string, string> = {
  h264: "H.264",
  hevc: "HEVC (H.265)",
  vp8: "VP8",
  vp9: "VP9",
  av1: "AV1",
  mpeg4: "MPEG-4 Part 2",
  theora: "Theora",
  aac: "AAC",
  mp3: "MP3",
  opus: "Opus",
  vorbis: "Vorbis",
  flac: "FLAC",
  alac: "ALAC",
  pcm_s16le: "PCM",
};

export function formatCodec(codec: string): string {
  return CODECS[codec] ?? codec.toUpperCase();
}

const CONTAINERS: Record<string, string> = {
  "mov,mp4,m4a,3gp,3g2,mj2": "MP4 / QuickTime",
  "matroska,webm": "Matroska / WebM",
  ogg: "Ogg",
  avi: "AVI",
  mpegts: "MPEG-TS",
};

export function formatContainer(container: string): string {
  return CONTAINERS[container] ?? container;
}

/** "rapid_approach" → "Rapid approach" for identifiers without an English label from the server. */
export function humanize(identifier: string): string {
  const text = identifier.replace(/_/g, " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

/**
 * Server messages as display sentences, ending with punctuation. A leading plain word is
 * capitalised; a leading file name ("clip.mp4 was moved") is left exactly as it is.
 */
export function sentence(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) return trimmed;
  const capitalised = /^[a-z]+([\s;,:]|$)/.test(trimmed) ? trimmed[0]!.toUpperCase() + trimmed.slice(1) : trimmed;
  return /[.!?]$/.test(capitalised) ? capitalised : `${capitalised}.`;
}
