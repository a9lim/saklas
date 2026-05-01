import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { resolve } from "node:path";

// Build output goes directly into the saklas Python package so the
// committed dist/ ships in the wheel.  emptyOutDir=true wipes any
// stale assets between builds — the directory's only consumer is
// FastAPI's StaticFiles mount, never user-authored files.
export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: resolve(__dirname, "../saklas/web/dist"),
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        // Pin asset names so the SPA fallback in saklas/web/routes.py
        // doesn't have to deal with hash variance — the bundle replaces
        // itself on every build, and StaticFiles serves whatever's at
        // the path the index.html references.
        entryFileNames: "assets/saklas.js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name][extname]",
      },
    },
  },
  server: {
    // For `npm run dev`, proxy API + WS to the running saklas serve.
    proxy: {
      "/saklas": {
        target: "http://localhost:8000",
        ws: true,
      },
      "/v1": "http://localhost:8000",
      "/api": "http://localhost:8000",
    },
  },
});
