import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The API runs separately (`pf serve`); in development the page reaches it through this proxy.
export default defineConfig({
  plugins: [react()],
  worker: { format: "es" },
  server: { proxy: { "/api": "http://127.0.0.1:8000", "/tiles": "http://127.0.0.1:8000", "/fonts": "http://127.0.0.1:8000" } },
});
