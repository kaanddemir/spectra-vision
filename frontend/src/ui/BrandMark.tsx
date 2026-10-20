/**
 * The Spectra mark, drawn after docs/assets/logo.png: a road under an open arc on the brand red.
 * The only custom graphic in the interface.
 */
export function BrandMark({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 32 32" aria-hidden="true" focusable="false">
      <rect width="32" height="32" rx="7" fill="var(--color-brand)" />
      <g fill="none" stroke="var(--color-on-brand)" strokeLinecap="round">
        <path d="M10.6 22.1A8.6 8.6 0 1 1 21.4 22.1" strokeWidth="1.7" />
        <path d="M15.1 13.8 10.4 25.2M16.9 13.8l4.7 11.4" strokeWidth="1.5" />
        <path d="M16 15.6v.9M16 18v1.3M16 21v1.8" strokeWidth="1.3" />
      </g>
    </svg>
  );
}
