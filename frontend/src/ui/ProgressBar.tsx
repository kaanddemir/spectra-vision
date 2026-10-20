import "./ProgressBar.css";

interface ProgressBarProps {
  label: string;
  /** 0–1, or null when no reliable fraction exists: the bar is then indeterminate. */
  value: number | null;
  /** "sm" is a thin bar for compact panels. */
  size?: "md" | "sm";
}

export function ProgressBar({ label, value, size = "md" }: ProgressBarProps) {
  const percent = value === null ? null : Math.round(Math.min(Math.max(value, 0), 1) * 100);
  return (
    <div
      className={`progress progress--${size}${percent === null ? " progress--indeterminate" : ""}`}
      role="progressbar"
      aria-label={label}
      aria-valuemin={percent === null ? undefined : 0}
      aria-valuemax={percent === null ? undefined : 100}
      aria-valuenow={percent ?? undefined}
    >
      <progress className="progress__native" max={100} value={percent ?? undefined} aria-hidden="true" />
    </div>
  );
}
