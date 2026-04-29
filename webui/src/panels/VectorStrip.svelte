<script lang="ts">
  // One row per loaded vector.  Layout is a single horizontal flex strip
  // of compact controls — enable toggle, name, alpha slider, alpha
  // numeric, trigger pill, variant chip, ⋮ menu, ✕ — with the strip
  // dimmed + struck-through when disabled.  Every mutation goes through
  // the store actions in stores.svelte.ts so the canonical expression
  // and pending-action queue stay coherent.
  //
  // The α slider has a 0 detent: dragging through ±0.025 snaps to 0 so
  // the user can park at "off" without fighting the slider.  The numeric
  // input bypasses the snap (typing is precise on purpose).  Sign drives
  // the displayed-value color (green / red).

  import { onMount } from "svelte";
  import type { ProjectionSpec, Trigger, VectorRackEntry } from "../lib/types";
  import {
    setVectorAlpha,
    setVectorEnabled,
    setVectorTrigger,
    setVectorVariant,
    setVectorProjection,
    setVectorAblate,
    addVectorToRack,
    removeVectorFromRack,
    vectorRack,
  } from "../lib/stores.svelte";
  import { apiVectors } from "../lib/api";

  interface Props {
    name: string;
    entry: VectorRackEntry;
  }

  let { name, entry }: Props = $props();

  // ---------- α slider with 0 detent ----------

  /** Coerce the slider's raw value to the 0-detent snap.  ±0.025 collapses
   * to 0 so users can park the term off without fiddling. */
  function snapAlpha(raw: number): number {
    if (Math.abs(raw) <= 0.025) return 0;
    return raw;
  }

  function onSliderInput(ev: Event): void {
    const t = ev.currentTarget as HTMLInputElement;
    const raw = parseFloat(t.value);
    if (!Number.isFinite(raw)) return;
    setVectorAlpha(name, snapAlpha(raw));
  }

  function onNumericInput(ev: Event): void {
    const t = ev.currentTarget as HTMLInputElement;
    let raw = t.value.trim();
    // Permit leading +/- prefix; parseFloat handles "+0.3" only on
    // some browsers, so strip the explicit '+' first.
    if (raw.startsWith("+")) raw = raw.slice(1);
    const v = parseFloat(raw);
    if (!Number.isFinite(v)) return;
    // Numeric input is precise — no detent.
    setVectorAlpha(name, v);
  }

  function formatAlpha(a: number): string {
    if (a === 0) return "0.00";
    const sign = a > 0 ? "+" : "-";
    return `${sign}${Math.abs(a).toFixed(2)}`;
  }

  const alphaColor = $derived.by(() => {
    if (entry.alpha > 0) return "var(--accent-green)";
    if (entry.alpha < 0) return "var(--accent-red)";
    return "var(--fg-muted)";
  });

  // ---------- trigger cycle ----------

  // Match the canonical render order in steering_expr.py — BOTH is
  // default and short-form 'B'; aliases collapse on serialize so we
  // surface only the canonical five.
  const TRIGGER_ORDER: Trigger[] = [
    "BOTH",
    "BEFORE",
    "AFTER",
    "THINKING",
    "RESPONSE",
    "PROMPT",
    "GENERATED",
  ];

  const TRIGGER_ABBR: Record<Trigger, string> = {
    BOTH: "B",
    BEFORE: "Bf",
    AFTER: "Af",
    THINKING: "Th",
    RESPONSE: "Rs",
    PROMPT: "Pr",
    GENERATED: "Gn",
  };

  const TRIGGER_LABEL: Record<Trigger, string> = {
    BOTH: "both (default)",
    BEFORE: "before — before thinking + response",
    AFTER: "after — after-thinking response only",
    THINKING: "thinking — chain-of-thought only",
    RESPONSE: "response — generated response only",
    PROMPT: "prompt (alias of before)",
    GENERATED: "generated (alias of response)",
  };

  function cycleTrigger(): void {
    const idx = TRIGGER_ORDER.indexOf(entry.trigger);
    const next = TRIGGER_ORDER[(idx + 1) % TRIGGER_ORDER.length];
    setVectorTrigger(name, next);
  }

  // ---------- variant chip ----------

  // For v1 we just surface whatever variant the entry already carries —
  // no SAE catalog query yet.  Tooltip explains; click prompts for an
  // explicit variant string (raw / sae / sae-<release>) so power users
  // can flip without leaving the strip.  Unrecognized input no-ops.
  function pickVariant(): void {
    const cur = entry.variant;
    const raw = window.prompt(
      `Variant for ${name} — raw, sae, or sae-<release>:`,
      cur,
    );
    if (raw === null) return;
    const v = raw.trim();
    if (!v) return;
    if (v === "raw" || v === "sae" || v.startsWith("sae-")) {
      setVectorVariant(name, v as VectorRackEntry["variant"]);
    }
    // Silently ignore malformed input — typing is correctable.
  }

  // ---------- ⋮ menu ----------

  let menuOpen = $state(false);
  let menuRef: HTMLDivElement | null = $state(null);

  function onDocClick(ev: MouseEvent): void {
    if (!menuOpen) return;
    if (menuRef && !menuRef.contains(ev.target as Node)) menuOpen = false;
  }
  function onDocKey(ev: KeyboardEvent): void {
    if (ev.key === "Escape" && menuOpen) menuOpen = false;
  }

  onMount(() => {
    document.addEventListener("click", onDocClick);
    document.addEventListener("keydown", onDocKey);
    return () => {
      document.removeEventListener("click", onDocClick);
      document.removeEventListener("keydown", onDocKey);
    };
  });

  function toggleMenu(ev: MouseEvent): void {
    ev.stopPropagation();
    menuOpen = !menuOpen;
  }

  /** Prompt for a target name to project base onto.  Empty cancels;
   * existing target shown for easy edit.  No autocomplete from the
   * loaded-vector list yet — v1 keeps the input dumb. */
  function promptProjection(op: ProjectionSpec["op"]): string | null {
    const cur = entry.projection?.target ?? "";
    const verb = op === "~" ? "Project onto" : "Project orthogonal to";
    const raw = window.prompt(`${verb} (target concept name):`, cur);
    if (raw === null) return null;
    const target = raw.trim();
    return target || null;
  }

  function pickProjection(op: ProjectionSpec["op"]): void {
    menuOpen = false;
    // Toggle off when the same operator is already wired and the user
    // accepts the current target via empty input.
    if (entry.projection && entry.projection.op === op) {
      // Clear projection.
      setVectorProjection(name, null);
      return;
    }
    const target = promptProjection(op);
    if (target === null) return;
    setVectorProjection(name, { op, target });
  }

  function toggleAblate(): void {
    menuOpen = false;
    setVectorAblate(name, !entry.ablate);
  }

  function duplicate(): void {
    menuOpen = false;
    let candidate = `${name}-copy`;
    let n = 2;
    while (vectorRack.entries.has(candidate)) {
      candidate = `${name}-copy-${n++}`;
    }
    addVectorToRack(candidate, entry.alpha, entry.trigger);
    // Carry the rest of the entry shape over so duplicate is faithful.
    const fresh = vectorRack.entries.get(candidate);
    if (fresh) {
      fresh.variant = entry.variant;
      fresh.projection = entry.projection
        ? { op: entry.projection.op, target: entry.projection.target }
        : null;
      fresh.ablate = entry.ablate;
      fresh.enabled = entry.enabled;
    }
  }

  async function copyTermExpression(): Promise<void> {
    menuOpen = false;
    // Build the single-term expression by serializing a one-entry rack.
    // Imported lazily to avoid a circular load — serializeExpression
    // lives in lib/expression and reads only the Map shape.
    const { serializeExpression } = await import("../lib/expression");
    const oneRack = new Map<string, VectorRackEntry>();
    oneRack.set(name, { ...entry, enabled: true });
    const expr = serializeExpression(oneRack);
    try {
      await navigator.clipboard.writeText(expr);
    } catch {
      // Clipboard is gated on user gesture in some browsers; fall back
      // to a temporary textarea + execCommand-free path is impractical
      // for v1 — surface a tiny inline error instead.
      window.prompt("Copy expression:", expr);
    }
  }

  // ---------- removal ----------

  async function removeVector(): Promise<void> {
    // Local rack delete first so the UI updates instantly even if the
    // server call lags.  The server-side drop is best-effort — a 404
    // (already gone) or a transient network error doesn't roll back the
    // local mutation, since the user clearly meant "off my rack".
    removeVectorFromRack(name);
    try {
      await apiVectors.delete(name);
    } catch {
      /* ignore — the rack is the user-visible source of truth. */
    }
  }

  // ---------- enable toggle ----------

  function toggleEnabled(): void {
    setVectorEnabled(name, !entry.enabled);
  }

  // ---------- display fragments ----------

  const variantLabel = $derived(
    entry.variant === "raw" ? "raw" : entry.variant,
  );

  const projectionGlyph = $derived.by(() => {
    if (!entry.projection) return null;
    return `${entry.projection.op} ${entry.projection.target}`;
  });
</script>

<div
  class="strip"
  class:disabled={!entry.enabled}
  class:ablate={entry.ablate}
  role="row"
>
  <button
    type="button"
    class="enable"
    onclick={toggleEnabled}
    title={entry.enabled ? "Enabled — click to disable" : "Disabled — click to enable"}
    aria-pressed={entry.enabled}
    aria-label="Toggle steering for {name}"
  >
    {entry.enabled ? "●" : "○"}
  </button>

  <span class="name" title={name}>
    {#if entry.ablate}<span class="ablate-mark" aria-label="ablation">!</span>{/if}{name}
  </span>

  <input
    type="range"
    class="slider"
    min="-1"
    max="1"
    step="0.05"
    value={entry.alpha}
    oninput={onSliderInput}
    aria-label="alpha for {name}"
    title="α for {name} — drag, ±0.025 snaps to 0"
  />

  <span class="alpha-display" style:color={alphaColor}>
    {formatAlpha(entry.alpha)}
  </span>

  <input
    type="number"
    class="alpha-input"
    min="-1"
    max="1"
    step="0.05"
    value={entry.alpha}
    oninput={onNumericInput}
    aria-label="alpha numeric for {name}"
    title="α numeric — accepts +/- prefix, no detent snap"
  />

  <button
    type="button"
    class="trigger-pill"
    onclick={cycleTrigger}
    title="trigger: {TRIGGER_LABEL[entry.trigger]} — click to cycle"
    aria-label="trigger for {name}: {entry.trigger}"
  >
    {TRIGGER_ABBR[entry.trigger]}
  </button>

  <button
    type="button"
    class="variant-chip"
    onclick={pickVariant}
    title="variant: {variantLabel} — click to change (raw, sae, sae-<release>)"
    aria-label="variant for {name}: {variantLabel}"
  >
    {variantLabel}
  </button>

  {#if projectionGlyph}
    <span class="projection-tag" title="projection: {projectionGlyph}">
      {projectionGlyph}
    </span>
  {/if}

  <div class="menu-wrap" bind:this={menuRef}>
    <button
      type="button"
      class="icon menu-btn"
      onclick={toggleMenu}
      aria-haspopup="menu"
      aria-expanded={menuOpen}
      aria-label="more actions for {name}"
      title="more actions"
    >
      ⋮
    </button>
    {#if menuOpen}
      <div class="menu" role="menu">
        <button
          type="button"
          role="menuitem"
          onclick={() => pickProjection("~")}
          disabled={entry.ablate}
        >
          {entry.projection?.op === "~"
            ? `clear projection (~ ${entry.projection.target})`
            : "project onto (~)…"}
        </button>
        <button
          type="button"
          role="menuitem"
          onclick={() => pickProjection("|")}
          disabled={entry.ablate}
        >
          {entry.projection?.op === "|"
            ? `clear projection (| ${entry.projection.target})`
            : "project orthogonal (|)…"}
        </button>
        <button type="button" role="menuitem" onclick={toggleAblate}>
          {entry.ablate ? "remove ablation (!)" : "ablate (!)"}
        </button>
        <hr />
        <button type="button" role="menuitem" onclick={duplicate}>
          duplicate
        </button>
        <button type="button" role="menuitem" onclick={copyTermExpression}>
          copy expression
        </button>
      </div>
    {/if}
  </div>

  <button
    type="button"
    class="icon remove"
    onclick={removeVector}
    aria-label="remove {name}"
    title="remove {name}"
  >
    ✕
  </button>
</div>

<style>
  .strip {
    display: flex;
    align-items: center;
    gap: 0.4em;
    min-height: 32px;
    padding: 0.25em 0.4em;
    border: 1px solid var(--border-dim);
    border-radius: 3px;
    background: var(--bg-alt);
    font-size: 0.85em;
  }
  .strip.disabled {
    opacity: 0.5;
  }
  .strip.disabled .name {
    text-decoration: line-through;
    color: var(--fg-muted);
  }
  .strip.ablate {
    border-color: var(--accent-purple);
  }

  .enable {
    background: transparent;
    border: 0;
    padding: 0 0.2em;
    color: var(--accent-green);
    font-size: 1em;
    line-height: 1;
  }
  .strip.disabled .enable {
    color: var(--fg-muted);
  }

  .name {
    flex: 0 1 auto;
    min-width: 5em;
    max-width: 10em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--fg-strong);
  }
  .ablate-mark {
    color: var(--accent-purple);
    font-weight: bold;
    margin-right: 0.1em;
  }

  .slider {
    flex: 1 1 auto;
    min-width: 80px;
    accent-color: var(--accent-blue);
    height: 14px;
  }

  .alpha-display {
    flex: 0 0 auto;
    min-width: 3.5em;
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-size: 0.85em;
  }

  .alpha-input {
    flex: 0 0 4.5em;
    width: 4.5em;
    background: var(--bg);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 0.1em 0.3em;
    font-size: 0.8em;
    font-variant-numeric: tabular-nums;
  }
  .alpha-input:focus {
    border-color: var(--accent-blue);
    outline: none;
  }

  .trigger-pill,
  .variant-chip {
    background: transparent;
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: 0.1em 0.4em;
    border-radius: 8px;
    font-size: 0.75em;
    line-height: 1.2;
    flex: 0 0 auto;
  }
  .trigger-pill:hover,
  .variant-chip:hover {
    background: var(--bg-elev);
    border-color: var(--fg-muted);
  }
  .variant-chip {
    color: var(--accent-blue);
  }

  .projection-tag {
    flex: 0 0 auto;
    color: var(--accent-yellow);
    font-size: 0.75em;
    border: 1px dashed var(--border);
    padding: 0.05em 0.3em;
    border-radius: 2px;
    max-width: 8em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .icon {
    background: transparent;
    border: 0;
    color: var(--fg-muted);
    font-size: 0.95em;
    line-height: 1;
    padding: 0.1em 0.35em;
    border-radius: 2px;
  }
  .icon:hover:not(:disabled) {
    color: var(--fg-strong);
    background: var(--bg-elev);
  }
  .remove:hover:not(:disabled) {
    color: var(--accent-red);
  }

  .menu-wrap {
    position: relative;
  }
  .menu {
    position: absolute;
    right: 0;
    top: calc(100% + 4px);
    min-width: 200px;
    background: var(--bg-alt);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.25em 0;
    z-index: var(--z-modal);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    display: flex;
    flex-direction: column;
  }
  .menu button {
    background: transparent;
    border: 0;
    text-align: left;
    padding: 0.4em 0.8em;
    color: var(--fg-strong);
    font-size: 0.85em;
  }
  .menu button:hover:not(:disabled) {
    background: var(--bg-elev);
    color: var(--accent-blue);
  }
  .menu button:disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
  }
  .menu hr {
    border: 0;
    border-top: 1px solid var(--border-dim);
    margin: 0.2em 0;
  }
</style>
