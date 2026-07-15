/// <reference types="vite/client" />
import { mount } from "svelte";
import App from "./App.svelte";
import { installTooltipLayer } from "./lib/tooltips";
// Side-effect CSS imports — Vite extracts these into the bundled CSS.
// Imported here (not from App.svelte) so svelte-check, which runs sans
// Vite's CSS plugin, doesn't trip on missing module declarations.
// The triple-slash above pulls in vite/client's ``*.css`` ambient
// declaration so svelte-check sees it.
import "./lib/style/fonts.css";
import "./lib/style/tokens.css";
import "./lib/style/global.css";

// Vite-bundled entry.  Mounts the dashboard onto #app via Svelte 5's
// ``mount`` API (the legacy ``new App({target})`` form was removed in
// Svelte 5).  Component-scoped CSS comes through automatically.
//
// /styleguide mounts the living design-system page instead of the app —
// the server's SPA fallback already serves index.html on any path, and
// mounting StyleGuide (not App) means no WS/bootstrap runs there.
import StyleGuide from "./styleguide/StyleGuide.svelte";

const target = document.getElementById("app");
if (!target) throw new Error("saklas web: #app element missing in index.html");

const isStyleGuide =
  window.location.pathname.replace(/\/+$/, "") === "/styleguide";
// Installed once outside either page root so every current/future `title`
// authoring surface uses Saklas tooltip chrome, including drawers, portals,
// the command palette, and the standalone styleguide.
installTooltipLayer();
const app = mount(isStyleGuide ? StyleGuide : App, { target });
export default app;
