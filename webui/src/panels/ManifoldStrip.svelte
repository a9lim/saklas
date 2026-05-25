<script lang="ts">
  // One rack row per racked manifold — parallel to VectorStrip.  A
  // manifold term places generation at a point of a fitted manifold,
  // so the row carries: enable toggle, name, an expandable position
  // picker, a blend slider [0, 1], a trigger pill, and a remove ✕.
  //
  // Every mutation routes through the store actions so the canonical
  // expression and the pending-action queue stay coherent.

  import type { ManifoldRackEntry, Trigger } from "../lib/types";
  import {
    manifoldByName,
    setManifoldBlend,
    setManifoldCoords,
    setManifoldLabel,
    setManifoldTrigger,
    setManifoldEnabled,
    removeManifoldFromRack,
  } from "../lib/stores.svelte";
  import Slider from "../lib/Slider.svelte";
  import XYPad from "./manifold/XYPad.svelte";

  interface Props {
    name: string;
    entry: ManifoldRackEntry;
  }

  let { name, entry }: Props = $props();

  const info = $derived(manifoldByName(name));

  let expanded = $state(false);

  // ---------- trigger cycle (mirror VectorStrip) ----------

  const TRIGGER_ORDER: Trigger[] = [
    "BOTH",
    "BEFORE",
    "AFTER",
    "THINKING",
    "RESPONSE",
    "PROMPT",
    "GENERATED",
  ];
  const TRIGGER_WORD: Record<Trigger, string> = {
    BOTH: "both",
    BEFORE: "before",
    AFTER: "after",
    THINKING: "thinking",
    RESPONSE: "response",
    PROMPT: "prompt",
    GENERATED: "generated",
  };

  function cycleTrigger(): void {
    const idx = TRIGGER_ORDER.indexOf(entry.trigger);
    setManifoldTrigger(name, TRIGGER_ORDER[(idx + 1) % TRIGGER_ORDER.length]);
  }

  function onBlendInput(v: number): void {
    if (!Number.isFinite(v)) return;
    setManifoldBlend(name, v);
  }

  function onCoordsChange(coords: number[]): void {
    setManifoldCoords(name, coords);
  }

  function toggleEnabled(): void {
    setManifoldEnabled(name, !entry.enabled);
  }

  const coordsLabel = $derived(
    entry.coords.map((c) => (Number.isFinite(c) ? c.toFixed(2) : "0")).join(", "),
  );

  // Snap-to-node dropdown: the user can pick a labeled node from the
  // manifold's node list, which switches the term to label-form
  // (``persona%pirate``).  An empty selection clears the label binding
  // — the position becomes free-form, drag on the XYPad to author.
  function onSnapToNode(ev: Event): void {
    const t = ev.currentTarget as HTMLSelectElement;
    const val = t.value;
    setManifoldLabel(name, val === "" ? null : val);
  }

  // A manifold needs a tensor for the loaded model to actually steer.
  const fitted = $derived(info?.fitted_for_session === true);
  const stale = $derived(info?.stale === true);
</script>

<div class="strip" class:disabled={!entry.enabled} role="row">
  <div class="row-main">
    <button
      type="button"
      class="enable"
      onclick={toggleEnabled}
      aria-pressed={entry.enabled}
      title={entry.enabled ? "Enabled (click to disable)" : "Disabled (click to enable)"}
      aria-label="Toggle manifold {name}"
    >
      {entry.enabled ? "●" : "○"}
    </button>

    <button
      type="button"
      class="name"
      onclick={() => (expanded = !expanded)}
      title="manifold {name} — click to {expanded ? 'collapse' : 'edit position'}"
    >
      <span class="caret" aria-hidden="true">{expanded ? "▾" : "▸"}</span>
      <span class="name-text">{name}</span>
    </button>

    {#if !fitted}
      <span class="warn" title="no fitted tensor for the loaded model — fit it from the manifolds drawer">
        unfitted
      </span>
    {:else if stale}
      <span class="warn" title="the fitted tensor is stale — node geometry changed since the fit">
        stale
      </span>
    {/if}

    <span class="coords" title="authoring position: {coordsLabel}">
      {#if entry.label}
        <span class="node-pill" title="snapped to node '{entry.label}' (label-form)">%{entry.label}</span>
      {:else}
        {coordsLabel}
      {/if}
    </span>

    <button
      type="button"
      class="trigger-pill"
      onclick={cycleTrigger}
      title="trigger: {TRIGGER_WORD[entry.trigger]} (click to cycle)"
    >
      {TRIGGER_WORD[entry.trigger]}
    </button>

    <button
      type="button"
      class="icon remove"
      onclick={() => removeManifoldFromRack(name)}
      aria-label="remove manifold {name}"
      title="remove {name}"
    >
      ✕
    </button>
  </div>

  <div class="blend-row">
    <span class="blend-label" title="blend fraction — how strongly to pull onto the manifold">
      blend
    </span>
    <Slider
      value={entry.blend}
      min={0}
      max={1}
      step={0.05}
      oninput={onBlendInput}
      ariaLabel="blend fraction for {name}"
    />
    <span class="blend-val">{entry.blend.toFixed(2)}</span>
  </div>

  {#if expanded}
    <div class="picker">
      {#if info}
        {#if info.node_labels.length > 0}
          <label class="snap-row">
            <span class="snap-label">snap to node</span>
            <select
              class="snap-select"
              value={entry.label ?? ""}
              onchange={onSnapToNode}
              title="pick a node to switch to label-form, or '(free)' to drag the pad"
            >
              <option value="">(free position)</option>
              {#each info.node_labels as nl, i (nl)}
                {@const role = info.node_roles?.[i]}
                <option value={nl}>
                  {nl}{role ? ` [role=${role}]` : ""}
                </option>
              {/each}
            </select>
          </label>
        {/if}
        <XYPad manifold={info} coords={entry.coords} onchange={onCoordsChange} />
      {:else}
        <p class="picker-missing">
          manifold metadata unavailable — coordinates are still applied.
        </p>
      {/if}
    </div>
  {/if}
</div>

<style>
  .strip {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    /* Phase C.3: a left purple stripe + matching name color
     * differentiates manifold rows from vector rows at a glance.
     * Same accent the TUI's manifold rows use (ansi_magenta). */
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-purple);
    border-radius: var(--radius);
    background: var(--bg-alt);
    font-size: var(--text-sm);
    transition: border-color var(--dur) var(--ease-out),
      opacity var(--dur) var(--ease-out);
  }
  .strip.disabled {
    opacity: 0.5;
  }
  .row-main {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-width: 0;
  }
  .enable {
    background: transparent;
    border: 0;
    padding: 0 var(--space-1);
    color: var(--accent-purple);
    font-size: var(--text);
    line-height: 1;
    flex: 0 0 auto;
    cursor: pointer;
  }
  .strip.disabled .enable {
    color: var(--fg-muted);
  }
  .name {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    background: transparent;
    border: 0;
    /* Manifold rows use purple as the name color to set them apart
     * from vector rows (which use fg-strong / accent on hover). */
    color: var(--accent-purple);
    font: inherit;
    font-family: var(--font-mono);
    cursor: pointer;
    min-width: 0;
    flex: 1 1 auto;
    text-align: left;
  }
  .name:hover {
    color: var(--accent);
  }
  .caret {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    flex: 0 0 auto;
  }
  .name-text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .warn {
    flex: 0 0 auto;
    color: var(--accent-yellow);
    font-size: var(--text-2xs);
    border: 1px solid var(--accent-yellow);
    border-radius: var(--radius);
    padding: 0 var(--space-2);
  }
  .coords {
    flex: 0 0 auto;
    color: var(--fg-dim);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    max-width: 9em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .trigger-pill {
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
  .trigger-pill:hover {
    background: var(--bg-elev);
    border-color: var(--fg-muted);
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
    cursor: pointer;
    transition: color var(--dur) var(--ease-out);
  }
  .remove:hover {
    color: var(--accent-red);
  }
  .blend-row {
    display: grid;
    grid-template-columns: auto 1fr 3em;
    align-items: center;
    gap: var(--space-3);
  }
  .blend-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-transform: lowercase;
  }
  .blend-val {
    color: var(--fg-dim);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
  .picker {
    border-top: 1px solid var(--border);
    padding-top: var(--space-3);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .picker-missing {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .snap-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    color: var(--fg-strong);
    font-size: var(--text-xs);
  }
  .snap-label {
    color: var(--fg-muted);
    flex: 0 0 auto;
  }
  .snap-select {
    flex: 1 1 auto;
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-1) var(--space-2);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
  }
  .snap-select:focus {
    outline: none;
    border-color: var(--accent-purple);
  }
  .node-pill {
    background: color-mix(in srgb, var(--accent-purple) 14%, transparent);
    color: var(--accent-purple);
    padding: 0 var(--space-2);
    border-radius: var(--radius);
    font-variant-numeric: tabular-nums;
  }
</style>
