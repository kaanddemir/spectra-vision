# Design system

Spectra's interface is neutral, precise and calm: thin dividers, comfortable spacing, a clear type
hierarchy and the Spectra red used sparingly for brand, primary actions, the current location and focus.
Library screens are structured tables; the review layout is a large media column with a side panel and
detail below. Every rule here is enforced or used by `src/styles/designSystem.test.ts`.

## Rules

1. **Colours exist only in [tokens.css](src/styles/tokens.css).** Components use semantic tokens
   (`--color-text-secondary`, `--color-danger-bg`), never raw values. The test rejects hex, `rgb()` and
   `hsl()` anywhere else, including TypeScript.
2. **Lengths come from tokens.** Spacing, sizes, radii, borders and z-index are tokens. The test rejects
   `px`/`rem` literals outside `tokens.css`, except in media queries and on lines marked
   `/* token-exception: reason */` (currently only the standard visually-hidden pattern).
3. **One pattern, one component.** Screens compose `src/ui` components and the layouts in
   `layout.css`; they do not restyle buttons, tables, badges or alerts locally. No inline `style`.
4. **Status never relies on colour.** Every status has text and an icon.
5. **No boxes inside boxes.** Sections sit on the page surface, separated by spacing or a single rule
   (`Section variant="divided"`). Tables, the media stage, dialogs and toasts are the only bordered surfaces.
   No decorative statistics, gradients or heavy shadows.
6. **Short copy.** Titles state the fact ("Server unavailable", "Not analysed"); a description is added only
   when it changes what the person does next. Safety and eligibility information is never dropped.

## Feedback

| Situation | Mechanism |
|---|---|
| A completed action (registered, forgotten) | `useToast()` toast |
| Information about this page, form or video (eligibility, a failed check, stale data) | inline `Alert` or text |
| An ongoing condition (server unreachable) | the shell's banner |
| A destructive or consequential action | `Dialog` |

Toasts appear bottom-right, at most three. A repeat with the same key (default: tone and title) replaces the
earlier toast instead of stacking. Success closes after 4 s, info 5 s, warning 8 s, errors 10 s; every toast
has a Dismiss button. Errors are announced as alerts, others politely. Loading and background refreshes never
produce toasts.

## Tokens

| Group | Tokens |
|---|---|
| Brand | `--color-brand` `#e40110` (the logo red), `--color-on-brand` |
| Surfaces | `--color-bg`, `--color-surface`, `--color-surface-subtle`, `--color-surface-hover`, `--color-surface-selected` |
| Text | `--color-text`, `--color-text-secondary`, `--color-text-disabled` |
| Lines | `--color-border` (dividers), `--color-border-control` (form-control outlines, ≥ 3:1) |
| Interaction | `--color-accent`, `-hover`, `-pressed`, `--color-on-accent`, `--color-accent-text` (links), `--color-focus` |
| Status | `--color-{success,warning,danger,info,neutral}-{text,bg,border}` |
| Media and overlays | `--color-media-bg`, `--color-media-text`, `--color-backdrop`, `--color-tooltip-{bg,text}`; review overlays `--color-overlay-{track,related,lane,text,halo,chip}`, identical in both themes because the stage is always dark |
| Type | `--font-sans` (system UI), `--font-mono`; `--text-xs…2xl` (12, 13, 14, 16, 20, 26 px) with matching `--leading-*`; `--weight-{regular,medium,semibold}`; `--tracking-{tight,caps}` |
| Space | `--space-0-5…16` on a 4 px grid |
| Shape | `--border-width(-strong)`, `--radius-{sm,md,lg,full}` (4, 6, 8 px; full for dots only) |
| Elevation | `--shadow-sm` (selected toggle), `--shadow-md` (popovers, dialogs, tooltips) |
| Icons | `--icon-{sm,md,lg}` 16/20/24 px, `--icon-stroke` 1.75 |
| Density | `--control-h-{sm,md,lg}` 28/36/40 px, `--table-row-h` 48, `--table-head-h` 40, `--header-h` 56 |
| Layout | `--content-max` 1440, `--page-gutter`, `--aside-w` 360, `--field-max-w`, `--dialog-w`, `--toast-w`, `--table-min-{md,lg}`, `--table-primary-min`, `--timeline-h`, `--proposal-list-max-h` |
| Focus | `--focus-width`, `--focus-offset`, `--underline-offset` |
| Layers | `--z-sticky` 10, `--z-header` 20, `--z-dropdown` 30, `--z-dialog` 50, `--z-toast` 55, `--z-tooltip` 60 |
| Motion | `--duration-fast` 120 ms, `--duration-base` 180 ms, `--duration-spin`, `--ease-standard`; zero under `prefers-reduced-motion` (spinners slow down instead of stopping) |
| Breakpoints | 640 px (compact gutters, single-column details), 960 px (side panel stacks, header wraps), 1280 px (reserved); written literally in media queries |

Both themes define every semantic colour. The test computes WCAG contrast from the file: text and status
text ≥ 4.5:1 on every surface, text on filled controls, tooltips and media ≥ 4.5:1, control outlines,
focus rings and filled buttons ≥ 3:1.

## Typography

System UI fonts (San Francisco on macOS): no web-font download, excellent numerals. Roles in `base.css`:
`type-title` (page h1), `type-heading` (section h2), `type-subheading`, `type-body`, `type-secondary`,
`type-caption`, `type-label` (uppercase table and field labels), `type-numeric` (tabular figures for
times, sizes, counts) and `type-mono` (hashes and keys). Tables use tabular figures in numeric cells and
right-align them.

## Components (`src/ui`)

| Component | Use |
|---|---|
| `Button`, `ButtonLink`, `IconButton` | `primary` (one per area), `secondary`, `ghost`, `danger`; `sm`/`md`; `loading` keeps the label and sets `aria-busy`. Icon buttons require a label, shown as a tooltip. A `ghost` button with `aria-pressed` is a toggle (overlay layers). |
| `TextField`, `TextArea`, `SearchField`, `Select`, `Checkbox` | Native controls with labels, descriptions and errors linked by `aria-describedby`; search labels are visually hidden. |
| `Tabs` | Filters with counts; arrow keys, Home/End, one tab stop, `aria-controls`. |
| `Table` | A named, keyboard-scrollable region; cell classes `cell-primary`, `cell-numeric`, `cell-secondary`, `cell-select`, `cell-actions`. |
| `Badge` | Status with tone, icon and text. |
| `ProgressBar` | Thin accent bar; determinate only with a reported fraction, otherwise indeterminate (no `aria-valuenow`). |
| `Alert` | Inline message; `danger` is announced as an alert, others as status. |
| `EmptyState`, `LoadingState`, `Spinner` | Explain empty regions; announce loading. |
| `PageHeader`, `Breadcrumbs`, `Section`, `DescriptionList` | Page title block with an optional back link to the parent page; breadcrumbs for folder paths only; titled regions (`divided` adds a rule above); label/value grids. |
| `ToastProvider`, `useToast` | Temporary feedback for completed actions (see Feedback). |
| `Dialog` | Native modal `<dialog>`: focus containment, Escape, focus return. |
| `Tooltip` | Supplementary labels on hover and focus; Escape dismisses. |
| `ThemeToggle` | System, Light, Dark as a radio group. |
| `Icon`, `icons`, `BrandMark` | See below. |

Layouts in `layout.css`: `toolbar`, `selection-bar`, `workspace` (media column, side panel, full-width
detail; the review workspace uses the same areas for the player, event timeline and evidence),
`media-stage` (dark 16:9 stage in both themes), `checklist`, `row-list`, `reason-list`.

The review workspace (`src/features/review`) keeps the player as the focus: proposals are a selectable list
with the accent rule on the current one, the timeline is a single SVG positioned in percentages (no inline
styles), and overlay status is one short note in the stage corner. Status shown in the list (the analyser's
checks) is not repeated as a badge in the evidence panel. The reviewer's decision is plain status text with a
person icon (`icons.accepted`, `icons.rejected`, `icons.notReviewed`), never a badge, so it cannot be read as
the analyser's status; it appears in the list only once a proposal is decided. A reviewer's adjusted times are labelled
"Adjusted by you" with `icons.edit` beside the detected times; their note is a separate "Your note" part,
never mixed into Spectra's explanation. `TextArea` shares `TextField`'s frame.

## Icons

Lucide is the only icon library. Screens import meanings from [`icons.ts`](src/ui/icons.ts)
(`icons.register`, `icons.warning`, …) and render them with `Icon`, which applies the size tokens
(16 px default beside text, 20 px in table rows and checklists, 24 px in empty states) and a 1.75 stroke.
Decorative icons are hidden from assistive technology; a standalone meaningful icon takes a `label`.
The test rejects Lucide imports outside the icon modules, other icon libraries and emoji.
`BrandMark` is the only custom graphic, drawn after `docs/assets/logo.png`.

## Themes

The preference is `system` (default), `light` or `dark`, stored as `spectra.theme` in local storage
(`system` removes the key). `public/theme-init.js` runs before the stylesheet and sets
`<html data-theme>` from the saved choice or `prefers-color-scheme`, so the wrong theme never flashes.
`ThemeProvider` follows operating-system changes while the preference is `system`. Themes differ only in
token values; components and layouts are shared. The media stage stays dark in both themes.

## Accessibility

Semantic landmarks and headings, a skip link, visible `:focus-visible` rings, labelled controls,
keyboard-operable tabs, toggles and dialogs, live regions for loading and results, and
`prefers-reduced-motion` support. Interactive controls always perform a real action; features that are
not available are explained in text rather than shown as inactive buttons.
