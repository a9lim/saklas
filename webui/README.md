# Saklas web UI

Svelte 5 + Vite source for the dashboard served by `saklas serve`. Production
builds go directly to `../saklas/web/dist/`; that committed directory is package
data and is the UI users receive from the wheel.

## Development

```bash
cd webui
npm ci

# In another terminal:
saklas serve <model>

npm run dev
```

Vite runs on `http://localhost:5173` and proxies `/saklas`, `/v1`, and `/api`
(including the native WebSocket) to `http://localhost:8000`.

Before committing a UI change:

```bash
npm run check
npm run build
git diff --exit-code ../saklas/web/dist
```

`npm run check` runs Svelte/TypeScript checks and the theme-token validator.
`npm run build` wipes and regenerates the committed production bundle. CI repeats
both commands and fails if the rebuilt bundle differs.

## Application shape

The desktop shell has three permanent work areas:

- `LoomSidebar` — conversation tree, filtering, branching, regeneration, and
  node actions.
- `Chat` — role-plan composer, authored/generated turns, token highlights, and
  drilldown entry points.
- `InspectorPanel` — sampling plus the subspace, manifold, SAE, and J-lens
  steering/probe instruments.

Below 1280 px those same areas become explicit `threads`, `chat`, and
`instruments` views. Dense tools open in a focus-trapped drawer. The command
palette (`⌘K` / `Ctrl+K`) is the global launcher.

## Source map

```text
src/
  main.ts                    app entry point
  App.svelte                 shell, compact navigation, drawer host
  panels/
    Chat.svelte              turn surface and role-plan composer
    RawBuffer.svelte         raw transcript surface
    loom/                    conversation-tree components and controller
    InspectorPanel.svelte    instrument-tab host
    SteeringRack.svelte      subspace/manifold steering cards
    ProbeRack.svelte         attached geometry probes
    SaePanel.svelte          SAE source, steer, and probe surface
    JLensPanel.svelte        J-lens source, steer, and probe surface
    CommandPalette.svelte    global launcher
  drawers/                   authoring, analysis, admin, and export tools
  lib/
    api.ts                   typed HTTP, SSE, and WebSocket clients
    types.ts                 shared wire and UI types
    stores.svelte.ts         cross-cutting session/tree/stream state
    stores/                  focused palette, drawer, toast, and input slices
    expression.ts            steering-expression parser/serializer
    manifolds/               diagnostics helpers and renderer
    charts/                  small quantitative primitives
    ui/                      shared buttons, tabs, chips, and cards
    style/                   tokens, fonts, and global rules
```

The server-owned wire contract is documented beside the implementation in
`../saklas/server/AGENTS.md`; dashboard-specific ownership and interaction
contracts live in `../saklas/web/AGENTS.md`. Prefer those sources over copying a
route inventory into this file.

## State and component rules

- Put state in the smallest matching file under `lib/stores/`. Use
  `stores.svelte.ts` only for state shared across the WebSocket, loom, chat, and
  instruments.
- Svelte collection state uses `SvelteMap`/`SvelteSet`. Replace stored objects
  when mutating them so subscribers observe the change.
- Add shared primitives under `lib/ui/`; do not duplicate button, tab, card, or
  drawer chrome inside a feature.
- Keep `lib/types.ts` aligned with the Pydantic/native WebSocket schemas. New UI
  code reads the canonical `measurements` envelope on each token; the pre-5.x
  top-level `captured` / per-token readout aliases are gone.
- Long-running generate/fit/train work uses one updateable toast or the existing
  progress surface, not a stream of transient notifications.

## Adding a drawer

1. Add `src/drawers/FooDrawer.svelte` and accept `params: unknown` through
   `$props()`.
2. Add its name to `DrawerName` in `lib/types.ts` and to `NARROW_DRAWERS` in
   `App.svelte` if it is a form or picker.
3. Export it from `drawers/index.ts` and add the host branch in `App.svelte`.
4. Add a command-palette entry in `lib/commands.ts` or an inline launcher owned by
   the relevant panel.
5. Run the check/build sequence above and commit the regenerated bundle.

## Visual system

The dashboard is dark-only. Hue identifies data space: subspace/chrome is
achromatic, manifold violet, SAE gold, J-lens and surprise blue, live/positive
green, and error/negative red. Roles do not carry hue. Gradients encode depth or
time only when direction is the data; glow is reserved for live state. The source
of truth is `src/lib/style/tokens.css`, with Recursive Sans/Mono axes defined in
`src/lib/style/fonts.css`.
