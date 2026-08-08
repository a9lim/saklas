// Cross-component state for the dashboard — the re-export barrel.
//
// Svelte 5 runes-based.  Each slice is a $state-backed object exported as
// a named const; components import the slice and read/write its fields
// directly — Svelte's compiler tracks dependencies automatically.  The
// implementation lives one directory down, one module per concern, and
// this file re-exports the union so every ``from "./lib/stores.svelte"``
// import site reads the same flat surface it always has.
//
// Map (roughly the order a generation touches them):
//
//   session      the loaded session — the mirror of GET /sessions/current
//   sampling     sampling controls + the wire payload they build
//   steering     the unified steer rack (subspace / manifold / atom terms)
//   probes       the unified probe rack + the transcript-highlight target
//   instruments  lens / SAE / geometry live state, sources, preparations
//   ws           the singleton socket, its dispatcher, the send primitives
//   loom         the client mirror of the server's LoomTree
//   loomUi       threads-column UI: modals, edge labels, filter, selection
//   chat         the chat-log projection + per-generation counters
//   ab           the A/B shadow generation + auto-regen recipe-override
//   pending      the deferred-mutation queue
//   persistence  the localStorage half of the presentation preferences
//   bootstrap    the mount-time fan-out over all of the above
//
//   drawers / inputHistory / palette / preparations / toasts — the five
//   slices that were already extracted.
//
// Slices import each other directly; nothing imports this barrel back.

export * from "./stores/ab.svelte";
export * from "./stores/bootstrap.svelte";
export * from "./stores/chat.svelte";
export * from "./stores/drawers.svelte";
export * from "./stores/inputHistory.svelte";
export * from "./stores/instruments.svelte";
export * from "./stores/loom.svelte";
export * from "./stores/loomUi.svelte";
export * from "./stores/pending.svelte";
export * from "./stores/persistence.svelte";
export * from "./stores/probes.svelte";
export * from "./stores/sampling.svelte";
export * from "./stores/session.svelte";
export * from "./stores/steering.svelte";
export * from "./stores/ws.svelte";
export { dismissToast, pushToast, toasts } from "./stores/toasts.svelte";
