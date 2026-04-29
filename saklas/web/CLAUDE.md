# web/

Static analytics dashboard mounted at `/` when `saklas serve --web` is passed.  Opt-in only — production API deployments don't pay the StaticFiles mount cost unless they ask for it.

## Layout

```
saklas/web/
  __init__.py        # public re-exports: register_web_routes, dist_path
  routes.py          # the actual mount logic + SPA fallback
  dist/              # COMMITTED build artifact, ships in the wheel
    index.html
    assets/saklas.js
    assets/saklas.css
```

The Svelte+Vite source lives at the repo's `webui/` directory (peer of `saklas/`).  `cd webui && npm run build` emits to `saklas/web/dist/` directly — no intermediate copy step.  CI gating on source-vs-dist drift is wired but disabled by default in `.github/workflows/ci.yml` because the committed bundle is the v1.6 source of truth.

## Mount

`register_web_routes(app)` mounts `/assets/*` on `StaticFiles` for the bundled CSS/JS, registers `GET /` to return `index.html`, and a catch-all `GET /{full_path:path}` that serves real files when present and falls back to `index.html` for SPA-owned routes (`/lab`, `/chat`, ...).  Caller (`server.create_app(..., web=True)`) is expected to register the dashboard last so the catch-all doesn't shadow `/v1/*`, `/api/*`, `/saklas/v1/*`.

`dist_path()` resolves the bundled assets through `importlib.resources` so it works for both editable and wheel installs.  `WebUINotBuilt` is raised on mount when the dist directory is empty — only fires in source-tree installs that haven't run `npm run build` yet.

## Wire protocol

The dashboard uses the existing native API.  Three protocol additions land alongside `--web`:

1. **WS `token` event gets optional `per_layer_scores`** — `dict[str, dict[str, float]]` (string-keyed for JSON), populated by `monitor.score_single_token_per_layer` when probes are loaded.  Drives the per-token × per-layer × per-probe heatmap inspector in the center panel.
2. **GET `/saklas/v1/sessions/{id}/vectors/{name}` carries `per_layer_norms`** — full-resolution `||baked||` per layer, stringified-key dict.  Drives the right-bottom layer-norms bar chart.
3. **GET `/saklas/v1/sessions/{id}/correlation[?names=a,b,c]`** — N×N magnitude-weighted cosine matrix across loaded vectors.  Drives the right-top correlation panel.

## Adding a panel

1. New `webui/src/panels/Foo.svelte`.
2. Add a writable store + `refreshFoo` thunk in `webui/src/lib/stores.ts`.
3. Add the panel + grid cell to `webui/src/App.svelte`.
4. `npm run build`, then commit the regenerated `saklas/web/dist/`.
5. New server data?  Add the route in `saklas/server/saklas_api.py` and document the message shape in this file.

## Out of scope for v1.6

* Alpha-sweep launcher panel (the `POST /sweep` endpoint exists; the UI is a follow-up).
* Multi-session support in the UI (server URL-paths support it; client assumes `default`).
* Raw hidden-state inspector (only per-probe-per-layer heatmap in v1.6).
* Auth / login flows (the underlying API key Bearer middleware applies; no dedicated UI surface).
