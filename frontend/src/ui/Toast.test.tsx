import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Button } from "./Button";
import { TOAST_DURATION_MS, ToastProvider, useToast, type ToastInput } from "./Toast";

afterEach(() => vi.useRealTimers());

function Trigger({ toasts }: { toasts: ToastInput[] }) {
  const show = useToast();
  return <Button onClick={() => toasts.forEach(show)}>Go</Button>;
}

function setup(toasts: ToastInput[]) {
  render(
    <ToastProvider>
      <Trigger toasts={toasts} />
    </ToastProvider>,
  );
  return screen.getByRole("button", { name: "Go" });
}

describe("toasts", () => {
  it("announces success politely and errors assertively", async () => {
    const go = setup([{ tone: "success", title: "Saved" }, { tone: "danger", title: "Failed", description: "Try again." }]);
    await userEvent.click(go);
    expect(screen.getByRole("status")).toHaveTextContent("Saved");
    expect(screen.getByRole("alert")).toHaveTextContent("FailedTry again.");
    expect(screen.getByRole("region", { name: "Notifications" })).toBeInTheDocument();
  });

  it("shows one toast for repeats of the same notification", async () => {
    const go = setup([{ tone: "danger", title: "Failed" }, { tone: "danger", title: "Failed" }]);
    await userEvent.click(go);
    await userEvent.click(go);
    expect(screen.getAllByRole("alert")).toHaveLength(1);
  });

  it("can be dismissed", async () => {
    const go = setup([{ tone: "info", title: "Note" }]);
    await userEvent.click(go);
    await userEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByText("Note")).not.toBeInTheDocument();
  });

  it("dismisses routine results quickly and keeps errors longer", () => {
    vi.useFakeTimers();
    const go = setup([{ tone: "success", title: "Saved" }, { tone: "danger", title: "Failed" }]);
    act(() => go.click());
    act(() => vi.advanceTimersByTime(TOAST_DURATION_MS.success + 1));
    expect(screen.queryByText("Saved")).not.toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
    act(() => vi.advanceTimersByTime(TOAST_DURATION_MS.danger));
    expect(screen.queryByText("Failed")).not.toBeInTheDocument();
  });
});
