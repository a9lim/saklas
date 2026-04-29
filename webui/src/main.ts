import App from "./App.svelte";

// Vite-bundled entry.  Mounts the dashboard onto #app and pulls the
// global stylesheet through Svelte's component-scoped CSS.
const target = document.getElementById("app");
if (!target) throw new Error("saklas web: #app element missing in index.html");

const app = new App({ target });
export default app;
