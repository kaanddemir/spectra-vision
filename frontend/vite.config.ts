import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The development server and the API both listen on the IPv4 loopback interface only.
// Requests to /api are forwarded unchanged: the Host header stays 127.0.0.1:5173 (an accepted
// local host name) and the page origin http://127.0.0.1:5173 must be allowed explicitly with
// `python -m spectra.api --allow-origin http://127.0.0.1:5173`. No CORS headers are involved.
const API_TARGET = process.env.SPECTRA_API ?? "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    // Same-origin only: no CORS headers, matching the API, which never sends them.
    cors: false,
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: false, xfwd: false },
    },
  },
  preview: { host: "127.0.0.1", port: 4173, strictPort: true },
  build: { outDir: "dist", emptyOutDir: true, sourcemap: false, assetsInlineLimit: 0 },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    restoreMocks: true,
    // Stylesheets are processed so tests can read them (the design-system test checks their rules).
    css: { include: [/.+/] },
  },
});
