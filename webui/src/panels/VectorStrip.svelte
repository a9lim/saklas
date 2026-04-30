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

  // ---------- inline projection modal ----------
  //
  // Replaces the v1 ``window.prompt`` for picking a projection target.
  // Modal is local to the strip — backdrop covers the viewport, the
  // dialog itself is centered, Esc / click-outside cancel, Enter
  // confirms, the input autofocuses on open.  Preselects the current
  // target so a re-open lands on the existing value.

  let projectionPromptOp = $state<ProjectionSpec["op"] | null>(null);
  let projectionTargetDraft = $state("");
  let projectionInputRef: HTMLInputElement | null = $state(null);

  function pickProjection(op: ProjectionSpec["op"]): void {
    menuOpen = false;
    // Toggle off when the same operator is already wired — clicking the
    // same projection in the menu clears it without opening the dialog.
    if (entry.projection && entry.projection.op === op) {
      setVectorProjection(name, null);
      return;
    }
    projectionTargetDraft = entry.projection?.target ?? "";
    projectionPromptOp = op;
    // Autofocus the input once the modal mounts.  A microtask is
    // enough because Svelte 5 flushes the DOM before setTimeout.
    queueMicrotask(() => projectionInputRef?.focus());
  }

  function cancelProjection(): void {
    projectionPromptOp = null;
    projectionTargetDraft = "";
  }

  function confirmProjection(): void {
    const op = projectionPromptOp;
    if (op === null) return;
    const target = projectionTargetDraft.trim();
    projectionPromptOp = null;
    projectionTargetDraft = "";
    if (!target) return;
    setVectorProjection(name, { op, target });
  }

  function onProjectionKey(ev: KeyboardEvent): void {
    if (ev.key === "Enter") {
      ev.preventDefault();
      confirmProjection();
    } else if (ev.key === "Escape") {
      ev.preventDefault();
      cancelProjection();
    }
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

{#if projectionPromptOp !== null}
  <!-- Inline projection-target dialog.  Backdrop covers the viewport;
       click-outside / Escape cancels; Enter confirms.  Stops outer
       click propagation so clicking inside the dialog box doesn't trip
       the cancel handler. -->
  <div
    class="projection-backdrop"
    role="presentation"
    onclick={cancelProjection}
    onkeydown={onProjectionKey}
  >
    <div
      class="projection-modal"
      role="dialog"
      aria-modal="true"
      aria-label="Pick projection target"
      tabindex="-1"
      onclick={(ev) => ev.stopPropagation()}
      onkeydown={(ev) => ev.stopPropagation()}
    >
      <header class="projection-header">
        <span class="projection-title">
          {projectionPromptOp === "~"
            ? `project ${name} onto`
            : `project ${name} orthogonal to`}
        </span>
      </header>
      <input
        bind:this={projectionInputRef}
        bind:value={projectionTargetDraft}
        class="projection-input"
        placeholder="target concept name"
        spellcheck="false"
        autocomplete="off"
        onkeydown={onProjectionKey}
      />
      <footer class="projection-actions">
        <button
          type="button"
          class="projection-btn cancel"
          onclick={cancelProjection}
        >cancel</button>
        <button
          type="button"
          class="projection-btn confirm"
          onclick={confirmProjection}
          disabled={!projectionTargetDraft.trim()}
        >ok</button>
      </footer>
    </div>
  </div>
{/if}

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

  /* Enable / disable toggle — same ●/○ glyph the probe row uses, so the
   * two row families read as one visual system.  Colour is the blue
   * highlight accent for "active state on this row" parity with the
   * probe-rack selection glyph. */
  .enable {
    background: transparent;
    border: 0;
    padding: 0 0.2em;
    color: var(--accent-blue);
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
    /* Inherits 0.85em from .strip — matches the probe row's .value so
     * the two row families read as one visual system. */
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

  /* ----- projection modal ----- */
  .projection-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(1, 4, 9, 0.55);
    z-index: var(--z-modal);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .projection-modal {
    min-width: 360px;
    max-width: 480px;
    background: var(--bg-alt);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.8em 1em;
    display: flex;
    flex-direction: column;
    gap: 0.6em;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.5);
    font-family: var(--font-mono);
  }
  .projection-header {
    display: flex;
    align-items: baseline;
    gap: 0.4em;
  }
  .projection-title {
    color: var(--fg-strong);
    font-size: 0.9em;
  }
  .projection-input {
    background: var(--bg);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 0.4em 0.6em;
    font: inherit;
    font-size: 0.9em;
    font-family: var(--font-mono);
  }
  .projection-input:focus {
    outline: none;
    border-color: var(--accent-blue);
  }
  .projection-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.4em;
  }
  .projection-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--fg-strong);
    padding: 0.3em 0.9em;
    font: inherit;
    font-family: var(--font-mono);
    font-size: 0.85em;
    cursor: pointer;
    border-radius: 2px;
  }
  .projection-btn.cancel {
    color: var(--fg-dim);
  }
  .projection-btn.cancel:hover {
    color: var(--fg-strong);
    border-color: var(--fg-muted);
  }
  .projection-btn.confirm {
    color: var(--accent-blue);
    border-color: var(--accent-blue);
  }
  .projection-btn.confirm:hover:not(:disabled) {
    background: rgba(88, 166, 255, 0.12);
  }
  .projection-btn.confirm:disabled {
    color: var(--fg-muted);
    border-color: var(--border-dim);
    cursor: not-allowed;
  }
</style>
