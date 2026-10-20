import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { Link, NavLink, Outlet, ScrollRestoration } from "react-router";

import { useHealth } from "../api/queries";
import { Alert } from "../ui/Alert";
import { BrandMark } from "../ui/BrandMark";
import { Button } from "../ui/Button";
import { Icon } from "../ui/Icon";
import { icons } from "../ui/icons";
import { ThemeToggle } from "../ui/ThemeToggle";
import { ToastProvider } from "../ui/Toast";
import { ServerStatus } from "./ServerStatus";
import "./AppShell.css";

const NAVIGATION = [
  { to: "/videos", label: "Videos", icon: icons.videos },
  { to: "/library", label: "Library", icon: icons.library },
  { to: "/events", label: "Events", icon: icons.events },
];

export function AppShell() {
  return (
    <ToastProvider>
    <div className="app">
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <header className="app-header">
        <div className="app-header__inner">
          <Link to="/videos" className="app-header__brand" aria-label="Spectra home">
            <BrandMark className="app-header__mark" />
            <span className="app-header__name">Spectra</span>
          </Link>
          <nav aria-label="Main" className="app-nav">
            <ul>
              {NAVIGATION.map((item) => (
                <li key={item.to}>
                  <NavLink to={item.to} className="app-nav__link">
                    <Icon icon={item.icon} />
                    <span>{item.label}</span>
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
          <div className="app-header__tools">
            <ServerStatus />
            <ThemeToggle />
          </div>
        </div>
      </header>
      <ServerBanner />
      <main id="main" className="app-main" tabIndex={-1}>
        <Outlet />
      </main>
      <ScrollRestoration />
    </div>
    </ToastProvider>
  );
}

/**
 * The single report of an unreachable server, on every page; pages show nothing more for it. When the
 * server answers again, every request that failed meanwhile is repeated, so pages recover on their own.
 */
function ServerBanner() {
  const health = useHealth();
  const client = useQueryClient();
  const wasOffline = useRef(false);

  useEffect(() => {
    if (health.isError) wasOffline.current = true;
    else if (health.isSuccess && wasOffline.current) {
      wasOffline.current = false;
      void client.refetchQueries({ predicate: (query) => query.state.status === "error" });
    }
  }, [health.isError, health.isSuccess, client]);

  if (!health.isError) return null;
  const showingData = client
    .getQueryCache()
    .getAll()
    .some((query) => query.queryKey[0] !== "health" && query.state.data !== undefined);
  return (
    <div className="app-banner">
      <Alert
        tone="danger"
        title="Server unavailable"
        actions={
          <Button size="sm" icon={icons.refresh} loading={health.isFetching} onClick={() => void client.refetchQueries()}>
            Retry
          </Button>
        }
      >
        Check that Spectra is running.{showingData ? " Data shown may be out of date." : null}
      </Alert>
    </div>
  );
}
