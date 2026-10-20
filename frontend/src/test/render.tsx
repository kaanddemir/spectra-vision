import { QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router";

import { createQueryClient } from "../api/queries";
import { routes } from "../app/routes";
import { ThemeProvider } from "../theme/ThemeProvider";

/** Render the whole application at `path` with fresh server state and no automatic retries. */
export function renderApp(path = "/videos") {
  const client = createQueryClient({ queries: { retry: false, gcTime: Infinity }, mutations: { retry: false } });
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  const user = userEvent.setup();
  const view = render(
    <ThemeProvider>
      <QueryClientProvider client={client}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    </ThemeProvider>,
  );
  return { ...view, router, client, user };
}
