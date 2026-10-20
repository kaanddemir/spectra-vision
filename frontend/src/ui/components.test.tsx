import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { Alert } from "./Alert";
import { Badge } from "./Badge";
import badgeCss from "./Badge.css?raw";
import statusCss from "./StatusText.css?raw";
import tableCss from "./Table.css?raw";
import { Button, IconButton } from "./Button";
import { DescriptionList } from "./DescriptionList";
import { Dialog } from "./Dialog";
import { EmptyState, NoResults } from "./EmptyState";
import { Checkbox, SearchField, Select, TextField } from "./Field";
import { Icon } from "./Icon";
import { icons } from "./icons";
import { Popover } from "./Popover";
import { LoadingState } from "./Spinner";
import { Table } from "./Table";
import { Tabs } from "./Tabs";

describe("Button", () => {
  it("runs its action and exposes its label", async () => {
    const onClick = vi.fn();
    render(<Button icon={icons.register} onClick={onClick}>Register</Button>);
    await userEvent.click(screen.getByRole("button", { name: "Register" }));
    expect(onClick).toHaveBeenCalledOnce();
  });

  it("is disabled and busy while loading, keeping its name", async () => {
    const onClick = vi.fn();
    render(<Button loading onClick={onClick}>Save</Button>);
    const button = screen.getByRole("button", { name: "Save" });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("aria-busy", "true");
    await userEvent.click(button);
    expect(onClick).not.toHaveBeenCalled();
  });

  it("defaults to type=button so it never submits a form by accident", () => {
    render(<Button>Plain</Button>);
    expect(screen.getByRole("button")).toHaveAttribute("type", "button");
  });
});

describe("IconButton and Tooltip", () => {
  it("has an accessible name and shows it as a tooltip on focus and hover", async () => {
    const user = userEvent.setup();
    render(<IconButton icon={icons.close} label="Close panel" />);
    const button = screen.getByRole("button", { name: "Close panel" });
    const tooltip = screen.getByRole("tooltip", { hidden: true });
    expect(tooltip).not.toBeVisible();
    await user.tab();
    expect(button).toHaveFocus();
    expect(tooltip).toBeVisible();
    expect(button).toHaveAccessibleDescription("Close panel");
    await user.keyboard("{Escape}");
    expect(tooltip).not.toBeVisible();
    await user.hover(button);
    expect(tooltip).toBeVisible();
    await user.unhover(button);
    expect(tooltip).not.toBeVisible();
  });
});

describe("Icon", () => {
  it("is hidden from assistive technology unless it carries a label", () => {
    const { container } = render(
      <>
        <Icon icon={icons.video} />
        <Icon icon={icons.warning} label="Warning" size="lg" />
      </>,
    );
    const [decorative, meaningful] = container.querySelectorAll("svg");
    expect(decorative).toHaveAttribute("aria-hidden", "true");
    expect(decorative).toHaveAttribute("width", "16");
    expect(screen.getByRole("img", { name: "Warning" })).toBe(meaningful);
    expect(meaningful).toHaveAttribute("width", "24");
    expect(meaningful).toHaveAttribute("stroke-width", "1.75");
  });
});

describe("Badge", () => {
  it("centres its icon on the label with the same gap as status text, static or spinning", () => {
    const { container } = render(
      <>
        <Badge tone="warning" icon={icons.warning}>Uncertain</Badge>
        <Badge tone="info" icon={icons.pending} spin>Running</Badge>
      </>,
    );
    const wrappers = container.querySelectorAll(".badge > [aria-hidden]");
    expect([...wrappers].map((w) => w.className)).toEqual(["badge__icon", "badge__icon spinner"]);
    expect(badgeCss).toMatch(/\.badge__icon\s*{\s*display: inline-flex;/);
    const gap = (css: string, cls: string) => new RegExp(`\\.${cls}\\s*{[^}]*gap: ([^;]+);`).exec(css)![1];
    expect(gap(badgeCss, "badge")).toBe(gap(statusCss, "status-text"));
  });

  it("states its status in text as well as colour", () => {
    render(<Badge tone="success" icon={icons.success}>Ready for analysis</Badge>);
    expect(screen.getByText("Ready for analysis")).toBeInTheDocument();
  });
});

describe("Alert", () => {
  it("announces errors assertively and other messages politely", () => {
    render(
      <>
        <Alert tone="danger" title="Failed">Details</Alert>
        <Alert tone="info" title="Note" />
      </>,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("FailedDetails");
    expect(screen.getByRole("status")).toHaveTextContent("Note");
  });
});

describe("form controls", () => {
  it("labels a text field and links its description and error", () => {
    render(<TextField label="Name" description="As shown in lists" error="Required" />);
    const input = screen.getByRole("textbox", { name: "Name" });
    expect(input).toHaveAccessibleDescription("As shown in lists Required");
    expect(input).toBeInvalid();
  });

  it("keeps a visually hidden label on search fields", async () => {
    const onChange = vi.fn();
    render(<SearchField label="Search videos" onChange={onChange} />);
    await userEvent.type(screen.getByRole("searchbox", { name: "Search videos" }), "dash");
    expect(onChange).toHaveBeenCalledTimes(4);
  });

  it("uses a native, labelled select", async () => {
    const onChange = vi.fn();
    render(<Select label="Folder" options={[{ value: "a", label: "Dashcam" }, { value: "b", label: "Archive" }]} onChange={onChange} />);
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Folder" }), "Archive");
    expect(onChange).toHaveBeenCalled();
  });

  it("supports the mixed state on checkboxes", () => {
    render(<Checkbox label="Select all" hideLabel indeterminate checked={false} onChange={() => {}} />);
    expect(screen.getByRole("checkbox", { name: "Select all" })).toBePartiallyChecked();
  });
});

describe("Tabs", () => {
  function Harness() {
    const [value, setValue] = useState<"a" | "b" | "c">("a");
    return (
      <>
        <Tabs
          label="Filter"
          controls="panel"
          value={value}
          onChange={setValue}
          items={[
            { id: "a", label: "All", count: 3 },
            { id: "b", label: "Ready", count: 1 },
            { id: "c", label: "Problems", count: 2 },
          ]}
        />
        <div id="panel">{value}</div>
      </>
    );
  }

  it("selects with clicks and arrow keys and keeps a single tab stop", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const tabs = screen.getAllByRole("tab");
    expect(tabs.map((t) => t.getAttribute("tabindex"))).toEqual(["0", "-1", "-1"]);
    expect(screen.getByRole("tab", { name: "Ready 1" })).toHaveAttribute("aria-controls", "panel");
    await user.click(screen.getByRole("tab", { name: "Problems 2" }));
    expect(screen.getByRole("tab", { name: "Problems 2" })).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "All 3" })).toHaveFocus();
    expect(screen.getByRole("tab", { name: "All 3" })).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{End}");
    expect(screen.getByText("c", { selector: "#panel" })).toBeInTheDocument();
  });
});

describe("Dialog", () => {
  function Harness({ onConfirm }: { onConfirm: () => void }) {
    const [open, setOpen] = useState(false);
    return (
      <>
        <Button onClick={() => setOpen(true)}>Open</Button>
        <Dialog open={open} title="Forget this video?" onClose={() => setOpen(false)} actions={<Button onClick={onConfirm}>Confirm</Button>}>
          The file stays where it is.
        </Dialog>
      </>
    );
  }

  it("opens as a labelled modal and closes on Escape and the close button", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    render(<Harness onConfirm={onConfirm} />);
    await user.click(screen.getByRole("button", { name: "Open" }));
    const dialog = screen.getByRole("dialog", { name: "Forget this video?" });
    expect(dialog).toHaveAttribute("open");
    await user.click(screen.getByRole("button", { name: "Confirm" }));
    expect(onConfirm).toHaveBeenCalledOnce();
    dialog.dispatchEvent(new Event("cancel", { cancelable: true }));
    await vi.waitFor(() => expect(dialog).not.toHaveAttribute("open"));
    await user.click(screen.getByRole("button", { name: "Open" }));
    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(dialog).not.toHaveAttribute("open");
  });
});

describe("Dialog dismissal", () => {
  function Guarded({ dismissible, onClose }: { dismissible: boolean; onClose: () => void }) {
    return (
      <Dialog open title="Edit" subtitle="One line" dismissible={dismissible} closeOnBackdrop onClose={onClose} actions={null}>
        <input aria-label="Value" data-autofocus />
      </Dialog>
    );
  }

  it("focuses the marked field, describes itself and closes on the backdrop only when dismissible", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { rerender } = render(<Guarded dismissible={false} onClose={onClose} />);
    const dialog = screen.getByRole("dialog", { name: "Edit" });
    expect(dialog).toHaveAccessibleDescription("One line");
    expect(screen.getByLabelText("Value")).toHaveFocus();
    await user.click(dialog);
    dialog.dispatchEvent(new Event("cancel", { cancelable: true }));
    await user.click(screen.getByLabelText("Value"));
    expect(onClose).not.toHaveBeenCalled();
    rerender(<Guarded dismissible onClose={onClose} />);
    await user.click(screen.getByLabelText("Value"));
    expect(onClose).not.toHaveBeenCalled();
    await user.click(dialog);
    expect(onClose).toHaveBeenCalledOnce();
  });
});

describe("Table, empty and loading states", () => {
  it("puts a table in a named, focusable scroll region", () => {
    render(
      <Table label="Registered videos">
        <tbody>
          <tr>
            <td>drive.mp4</td>
          </tr>
        </tbody>
      </Table>,
    );
    const region = screen.getByRole("region", { name: "Registered videos" });
    expect(region).toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("table", { name: "Registered videos" })).toBeInTheDocument();
  });

  it("explains empty regions with a heading and an action", () => {
    render(<EmptyState icon={icons.videos} title="No videos" action={<Button>Browse</Button>}>Nothing yet.</EmptyState>);
    expect(screen.getByRole("heading", { name: "No videos" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Browse" })).toBeInTheDocument();
  });

  it("keeps one empty-state group in both layouts: icon, title, text, then action", () => {
    const { rerender, container } = render(
      <EmptyState icon={icons.videos} title="No videos" action={<Button>Browse</Button>}>Nothing yet.</EmptyState>,
    );
    const parts = () => [...container.firstElementChild!.children].map((child) => child.className);
    const order = ["empty-state__icon", "type-subheading", "empty-state__text type-secondary", "empty-state__action"];
    expect(container.firstElementChild).toHaveClass("empty-state", "empty-state--inline");
    expect(parts()).toEqual(order);
    rerender(<EmptyState layout="page" icon={icons.videos} title="No videos" action={<Button>Browse</Button>}>Nothing yet.</EmptyState>);
    expect(container.firstElementChild).toHaveClass("empty-state", "empty-state--page");
    expect(parts()).toEqual(order);
  });

  it("announces loading politely", () => {
    render(<LoadingState label="Loading videos…" />);
    expect(screen.getByRole("status")).toHaveTextContent("Loading videos…");
  });

  it("shows an em dash for missing descriptions", () => {
    render(<DescriptionList items={[{ term: "Duration", value: null }]} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });
});

describe("NoResults", () => {
  it("offers Clear filters only when a search or filter is active", async () => {
    const onClear = vi.fn();
    const { rerender } = render(<NoResults onClear={onClear} />);
    expect(screen.getByRole("heading", { name: "No results found" })).toBeInTheDocument();
    expect(screen.getByText("Try a different search.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(onClear).toHaveBeenCalledOnce();
    rerender(<NoResults />);
    expect(screen.queryByRole("button", { name: "Clear filters" })).not.toBeInTheDocument();
  });
});

describe("Popover", () => {
  function renderAt(top: number) {
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({ top, bottom: top + 32 } as DOMRect);
    vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(200);
    render(<Popover trigger={(props) => <button type="button" {...props}>Open</button>}>Menu</Popover>);
    return userEvent.click(screen.getByRole("button", { name: "Open" }));
  }

  it("opens below its trigger when there is room", async () => {
    await renderAt(100);
    expect(screen.getByText("Menu")).not.toHaveClass("popover__panel--up");
    vi.restoreAllMocks();
  });

  it("opens above its trigger near the bottom of the window", async () => {
    await renderAt(window.innerHeight - 60);
    expect(screen.getByText("Menu")).toHaveClass("popover__panel--up");
    vi.restoreAllMocks();
  });
});

describe("table column alignment", () => {
  it("centres headers and values of compact columns together, keeping names left and actions right", () => {
    const rule = /\.table--fit th:not\(\.cell-primary, \.cell-flex, \.cell-select, \.cell-actions\),\s*\.table--fit td:not\(\.cell-primary, \.cell-flex, \.cell-select, \.cell-actions\)\s*{\s*text-align: center;/;
    expect(tableCss).toMatch(rule);
    // Badges and status text centre as one unit with their icon.
    expect(badgeCss).toMatch(/\.badge\s*{\s*display: inline-flex;/);
    expect(statusCss).toMatch(/\.status-text\s*{\s*display: inline-flex;/);
  });
});
