/// <reference types="vite/client" />
import { mount } from "svelte";
import App from "./App.svelte";
// Side-effect CSS imports — Vite extracts these into the bundled CSS.
// Imported here (not from App.svelte) so svelte-check, which runs sans
// Vite's CSS plugin, doesn't trip on missing module declarations.
// The triple-slash above pulls in vite/client's ``*.css`` ambient
// declaration so svelte-check sees it.
import "./lib/style/tokens.css";
import "./lib/style/global.css";

// Vite-bundled entry.  Mounts the dashboard onto #app via Svelte 5's
// ``mount`` API (the legacy ``new App({target})`` form was removed in
// Svelte 5).  Component-scoped CSS comes through automatically.
const target = document.getElementById("app");
if (!target) throw new Error("saklas web: #app element missing in index.html");

const app = mount(App, { target });
export default app;
