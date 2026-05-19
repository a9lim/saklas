<script lang="ts">
  // One row per loaded steering vector.  The bipolar axis *is* the
  // layout: the negative pole flanks the slider on the left, the positive
  // on the right, so ``calm ◄──●──► angry`` reads as a control rather
  // than a bare name next to an unlabelled slider (docs/plans/
  // webui-overhaul.md §"The rack strip").  The canonical concept name
  // lives in the tooltip / aria-label; monopolar concepts show one pole.
  //
  // Every mutation goes through the store actions in stores.svelte.ts so
  // the canonical expression and pending-action queue stay coherent.
  //
  // The α slider has a 0 detent: dragging through ±0.025 snaps to 0 so
  // the user can park at "off" without fighting the slider.

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
  import { pushToast } from "../lib/stores/toasts.svelte";
  import { serializeExpression } from "../lib/expression";
  import { apiVectors } from "../lib/api";
  import { polesOf } from "../lib/concepts";
  import Slider from "../lib/Slider.svelte";

  interface Props {
    name: string;
    entry: VectorRackEntry;
  }

  let { name, entry }: Props = $props();

  // ---------- bipolar axis ----------

  const poles = $derived(polesOf(name));
  const monopolar = $derived(poles.negative === null);

  // ---------- α slider with 0 detent ----------

  /** Coerce the slider's raw value to the 0-detent snap.  ±0.025 collapses
   * to 0 so users can park the term off without fiddling. */
  function snapAlpha(raw: number): number {
    if (Math.abs(raw) <= 0.025) return 0;
    return raw;
  }

  function onSliderInput(v: number): void {
    if (!Number.isFinite(v)) return;
    setVectorAlpha(name, snapAlpha(v));
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
  // default; aliases collapse on serialize so we surface the canonical
  // five plus the two aliases.
  const TRIGGER_ORDER: Trigger[] = [
    "BOTH",
    "BEFORE",
    "AFTER",
    "THINKING",
    "RESPONSE",
    "PROMPT",
    "GENERATED",
  ];

  // Plain-language word shown at rest — no more decoding ``Bf`` / ``Af``.
  const TRIGGER_WORD: Record<Trigger, string> = {
    BOTH: "both",
    BEFORE: "before",
    AFTER: "after",
    THINKING: "thinking",
    RESPONSE: "response",
    PROMPT: "prompt",
    GENERATED: "generated",
  };

  const TRIGGER_LABEL: Record<Trigger, string> = {
    BOTH: "both: steer the whole turn (default)",
    BEFORE: "before: steer thinking and response",
    AFTER: "after: steer the after-thinking response only",
    THINKING: "thinking: steer the chain-of-thought only",
    RESPONSE: "response: steer the generated response only",
    PROMPT: "prompt (alias of before)",
    GENERATED: "generated (alias of response)",
  };

  function cycleTrigger(): void {
    const idx = TRIGGER_ORDER.indexOf(entry.trigger);
    const next = TRIGGER_ORDER[(idx + 1) % TRIGGER_ORDER.length];
    setVectorTrigger(name, next);
  }

  // ---------- variant dropdown ----------
  //
  // Replaces the v1 ``window.prompt``.  A small in-app menu offering
  // ``raw`` / ``sae`` plus the current ``sae-<release>`` when one is set,
  // so flipping the variant never leaves the saklas visual language.

  let variantOpen = $state(false);
  let variantRef: HTMLDivElement | null = $state(null);

  const variantOptions = $derived.by(() => {
    const opts: VectorRackEntry["variant"][] = ["raw", "sae"];
    if (
      entry.variant.startsWith("sae-") &&
      !opts.includes(entry.variant)
    ) {
      opts.push(entry.variant);
    }
    return opts;
  });

  function pickVariant(v: VectorRackEntry["variant"]): void {
    variantOpen = false;
    setVectorVariant(name, v);
  }

  // ---------- ⋮ menu ----------

  let menuOpen = $state(false);
  let menuRef: HTMLDivElement | null = $state(null);

  function onDocClick(ev: MouseEvent): void {
    const t = ev.target as Node;
    if (menuOpen && menuRef && !menuRef.contains(t)) menuOpen = false;
    if (variantOpen && variantRef && !variantRef.contains(t)) {
      variantOpen = false;
    }
  }
  function onDocKey(ev: KeyboardEvent): void {
    if (ev.key !== "Escape") return;
    if (menuOpen) menuOpen = false;
    if (variantOpen) variantOpen = false;
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
    variantOpen = false;
  }

  function toggleVariant(ev: MouseEvent): void {
    ev.stopPropagation();
    variantOpen = !variantOpen;
    menuOpen = false;
  }

  // ---------- inline projection modal ----------
  //
  // Replaces the v1 ``window.prompt`` for picking a projection target.
  // Modal is local to the strip — backdrop covers the viewport, the
  // dialog itself is centered, Esc / click-outside cancel, Enter
  // confirms, the input autofocuses on open.

  let projectionPromptOp = $state<ProjectionSpec["op"] | null>(null);
  let projectionTargetDraft = $state("");
  let projectionInputRef: HTMLInputElement | null = $state(null);

  function pickProjection(op: ProjectionSpec["op"]): void {
    menuOpen = false;
    // Toggle off when the same operator is already wired.
    if (entry.projection && entry.projection.op === op) {
      setVectorProjection(name, null);
      return;
    }
    projectionTargetDraft = entry.projection?.target ?? "";
    projectionPromptOp = op;
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
    const oneRack = new Map<string, VectorRackEntry>();
    oneRack.set(name, { ...entry, enabled: true });
    const expr = serializeExpression(oneRack);
    try {
      await navigator.clipboard.writeText(expr);
      pushToast(`copied: ${expr}`, { kind: "info", ttlMs: 3000 });
    } catch {
      // Clipboard is gated on user gesture in some browsers — surface an
      // in-app toast rather than a native prompt.
      pushToast("clipboard blocked by the browser", { kind: "warning" });
    }
  }

  // ---------- removal ----------

  async function removeVector(): Promise<void> {
    removeVectorFromRack(name);
    try {
      await apiVectors.delete(name);
    } catch {
      /* ignore — the rack is the user-visible source of truth. */
    }
  }

  function toggleEnabled(): void {
    setVectorEnabled(name, !entry.enabled);
  }

  // ---------- display fragments ----------

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
    title={entry.enabled ? "Enabled (click to disable)" : "Disabled (click to enable)"}
    aria-pressed={entry.enabled}
    aria-label="Toggle steering for {name}"
  >
    {entry.enabled ? "●" : "○"}
  </button>

  {#if entry.ablate}
    <span class="ablate-mark" title="ablation: concept removed from the residual stream">!</span>
  {/if}

  <!-- Bipolar axis frame.  The negative pole sits left of the slider, the
       positive right — dragging left/right now means something.
       Monopolar concepts render an empty left-pole slot rather than
       collapsing the grid, so a mixed mono/bipolar rack keeps slider
       starts + positive poles vertically aligned across rows. -->
  <div class="axis">
    <span class="pole neg" aria-hidden={monopolar}>
      {#if !monopolar}{poles.negative}{/if}
    </span>
    <Slider
      value={entry.alpha}
      min={monopolar ? 0 : -1}
      max={1}
      step={0.05}
      oninput={onSliderInput}
      ariaLabel="strength (α) for {name}"
      title="strength (α) for {name}: drag, ±0.025 snaps to 0"
    />
    <span class="pole pos" title="positive pole (drag right)">
      {poles.positive}
    </span>
  </div>

  <span
    class="alpha-display"
    style:color={alphaColor}
    title="strength (α): signed steering coefficient"
  >
    {formatAlpha(entry.alpha)}
  </span>

  <button
    type="button"
    class="trigger-pill"
    onclick={cycleTrigger}
    title="trigger: {TRIGGER_LABEL[entry.trigger]} (click to cycle)"
    aria-label="trigger for {name}: {entry.trigger}"
  >
    {TRIGGER_WORD[entry.trigger]}
  </button>

  <div class="variant-wrap" bind:this={variantRef}>
    <button
      type="button"
      class="variant-chip"
      onclick={toggleVariant}
      aria-haspopup="menu"
      aria-expanded={variantOpen}
      title="tensor variant: {entry.variant} (click to change)"
      aria-label="variant for {name}: {entry.variant}"
    >
      {entry.variant}
    </button>
    {#if variantOpen}
      <div class="variant-menu" role="menu">
        {#each variantOptions as v (v)}
          <button
            type="button"
            role="menuitemradio"
            aria-checked={entry.variant === v}
            class:active={entry.variant === v}
            onclick={() => pickVariant(v)}
          >
            {v}
          </button>
        {/each}
      </div>
    {/if}
  </div>

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
  <!-- Inline projection-target dialog. -->
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
    gap: var(--space-3);
    min-height: 32px;
    padding: var(--space-2) var(--space-3);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg-alt);
    font-size: var(--text-sm);
    transition: border-color var(--dur) var(--ease-out),
      opacity var(--dur) var(--ease-out);
  }
  .strip.disabled {
    opacity: 0.5;
  }
  .strip.ablate {
    border-color: var(--accent-purple);
  }

  /* Enable / disable toggle — same ●/○ glyph the probe row uses. */
  .enable {
    background: transparent;
    border: 0;
    padding: 0 var(--space-1);
    color: var(--accent-blue);
    font-size: var(--text);
    line-height: 1;
    flex: 0 0 auto;
  }
  .strip.disabled .enable {
    color: var(--fg-muted);
  }

  .ablate-mark {
    color: var(--accent-purple);
    font-weight: var(--weight-bold);
    flex: 0 0 auto;
  }

  /* The axis owns the strip's flexible width — poles + slider together.
   * Monopolar rows render an empty left-pole slot instead of collapsing
   * the grid, so a mono row sits on the same column proportions as the
   * bipolar rows around it and the slider starts at the same x. */
  .axis {
    display: grid;
    grid-template-columns: minmax(2.5em, 1fr) minmax(60px, 2.6fr) minmax(2.5em, 1fr);
    align-items: center;
    gap: var(--space-2);
    flex: 1 1 auto;
    min-width: 0;
  }
  .pole {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: var(--text-sm);
  }
  .strip.disabled .pole {
    text-decoration: line-through;
  }
  .pole.neg {
    color: var(--fg-muted);
    text-align: right;
  }
  .pole.pos {
    color: var(--fg-strong);
    text-align: left;
  }

  .alpha-display {
    flex: 0 0 auto;
    min-width: 3.5em;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  .trigger-pill,
  .variant-chip {
    background: transparent;
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: var(--space-1) var(--space-3);
    border-radius: var(--radius);
    font-size: var(--text-xs);
    line-height: 1.2;
    flex: 0 0 auto;
    cursor: pointer;
    transition: background var(--dur) var(--ease-out),
      border-color var(--dur) var(--ease-out);
  }
  .trigger-pill:hover,
  .variant-chip:hover {
    background: var(--bg-elev);
    border-color: var(--fg-muted);
  }
  .variant-chip {
    color: var(--accent-blue);
  }

  .variant-wrap {
    position: relative;
    flex: 0 0 auto;
  }
  .variant-menu {
    position: absolute;
    right: 0;
    top: calc(100% + 4px);
    min-width: 7em;
    background: var(--surface-hi);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-1) 0;
    z-index: var(--z-modal);
    box-shadow: var(--shadow-overlay);
    display: flex;
    flex-direction: column;
  }
  .variant-menu button {
    background: transparent;
    border: 0;
    text-align: left;
    padding: var(--space-2) var(--space-4);
    color: var(--fg-strong);
    font-size: var(--text-sm);
    cursor: pointer;
  }
  .variant-menu button:hover {
    background: var(--bg-elev);
    color: var(--accent-blue);
  }
  .variant-menu button.active {
    color: var(--accent-blue);
  }

  .projection-tag {
    flex: 0 0 auto;
    color: var(--accent-yellow);
    font-size: var(--text-xs);
    border: 1px solid var(--border);
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius);
    max-width: 8em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .icon {
    background: transparent;
    border: 0;
    color: var(--fg-muted);
    font-size: var(--text);
    line-height: 1;
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius);
    flex: 0 0 auto;
    transition: color var(--dur) var(--ease-out),
      background var(--dur) var(--ease-out);
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
    flex: 0 0 auto;
  }
  .menu {
    position: absolute;
    right: 0;
    top: calc(100% + 4px);
    min-width: 200px;
    background: var(--surface-hi);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-2) 0;
    z-index: var(--z-modal);
    box-shadow: var(--shadow-overlay);
    display: flex;
    flex-direction: column;
  }
  .menu button {
    background: transparent;
    border: 0;
    text-align: left;
    padding: var(--space-4) var(--space-5);
    color: var(--fg-strong);
    font-size: var(--text-sm);
    cursor: pointer;
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
    border-top: 1px solid var(--border);
    margin: var(--space-1) 0;
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
    background: var(--surface-hi);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-5) var(--space-6);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    box-shadow: var(--shadow-overlay);
    font-family: var(--font-mono);
  }
  .projection-header {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
  }
  .projection-title {
    color: var(--fg-strong);
    font-size: var(--text-sm);
  }
  .projection-input {
    background: var(--bg-elev);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-size: var(--text-sm);
    font-family: var(--font-mono);
  }
  .projection-input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .projection-actions {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-3);
  }
  .projection-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--fg-strong);
    padding: var(--space-2) var(--space-5);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
    border-radius: var(--radius);
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
    border-color: var(--accent);
  }
  .projection-btn.confirm:hover:not(:disabled) {
    background: var(--accent-subtle);
  }
  .projection-btn.confirm:disabled {
    color: var(--fg-muted);
    border-color: var(--border);
    cursor: not-allowed;
  }
</style>
