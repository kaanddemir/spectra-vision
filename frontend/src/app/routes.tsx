import { Navigate, type RouteObject } from "react-router";

import { AnalysisPage } from "../features/analyses/AnalysisPage";
import { EventsPage } from "../features/events/EventsPage";
import { LibraryPage } from "../features/library/LibraryPage";
import { VideoDetailPage } from "../features/videos/VideoDetailPage";
import { VideosPage } from "../features/videos/VideosPage";
import { AppShell } from "./AppShell";
import { NotFoundPage } from "./NotFoundPage";

export const routes: RouteObject[] = [
  {
    element: <AppShell />,
    children: [
      { index: true, element: <Navigate to="/videos" replace /> },
      { path: "videos", element: <VideosPage /> },
      { path: "videos/:videoId", element: <VideoDetailPage /> },
      { path: "library", element: <LibraryPage /> },
      { path: "events", element: <EventsPage /> },
      { path: "analyses/:analysisId", element: <AnalysisPage /> },
      { path: "*", element: <NotFoundPage /> },
    ],
  },
];
