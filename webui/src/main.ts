import { mount } from "svelte";
import App from "./App.svelte";

// Vite-bundled entry.  Mounts the dashboard onto #app via Svelte 5's
// ``mount`` API (the legacy ``new App({target})`` form was removed in
// Svelte 5).  Component-scoped CSS comes through automatically.
const target = document.getElementById("app");
if (!target) throw new Error("saklas web: #app element missing in index.html");

const app = mount(App, { target });
export default app;
