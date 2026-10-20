import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import tableCss from "./Table.css?raw";
import { Table } from "./Table";

describe("table", () => {
  it("defaults to browser sizing and opts into fit sizing", () => {
    const { rerender } = render(<Table label="Items"><tbody /></Table>);
    expect(screen.getByRole("table", { name: "Items" })).toHaveClass("table", "table--auto", "table--min-md");
    rerender(<Table label="Items" columns="fit" minWidth="lg"><tbody /></Table>);
    expect(screen.getByRole("table", { name: "Items" })).toHaveClass("table--fit", "table--min-lg");
  });

  it("sits on the page without a card and scrolls inside its own region", () => {
    const region = /\.table-region \{([^}]*)\}/.exec(tableCss)![1]!;
    expect(region).toMatch(/overflow-x: auto/);
    expect(region).not.toMatch(/border|background|radius/);
    expect(tableCss).not.toMatch(/tr:last-child td \{\s*border-bottom: 0/);
  });

  it("keeps fit columns compact and lets only the primary and flexible columns grow and truncate", () => {
    expect(tableCss).toMatch(/\.table--fit th,\s*\.table--fit td \{\s*width: 1%;\s*white-space: nowrap;/);
    expect(tableCss).toMatch(/\.table\.table--fit \.cell-primary,\s*\.table\.table--fit \.cell-flex \{\s*width: auto;\s*max-width: 0;/);
  });

  it("wraps names on narrow screens so touch users can read them", () => {
    expect(tableCss).toMatch(/@media \(max-width: 640px\) \{\s*\.table \.cell-primary \.truncate \{[^}]*white-space: normal;/);
  });

  it("keeps hover, selected and focus states", () => {
    expect(tableCss).toMatch(/tbody tr:hover td/);
    expect(tableCss).toMatch(/tbody tr\.is-selected td/);
    expect(tableCss).toMatch(/\.table-region:focus-visible/);
  });
});
