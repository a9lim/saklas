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
  import Select from "../lib/Select.svelte";
  import XYPad from "./manifold/XYPad.svelte";

  interface Props {
    name: string;
    entry: ManifoldRackEntry;
  }

  let { name, entry }: Props = $props();

  const info = $derived(manifoldByName(name));

  /** Display name — bare manifold name with the namespace prefix stripped
   *  (``default/personas`` → ``personas``).  Full name stays in the
   *  tooltip for disambiguation. */
  const displayName = $derived(name.split("/").pop() ?? name);

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
  //
  // ``setManifoldLabel`` already mirrors the catalog's
  // ``node_coords[idx]`` onto ``entry.coords`` so the XYPad readout
  // reflects the picked node's actual position.  The pad is rendered
  // ``locked`` while a label is bound — the engine takes the position
  // from the named node and ignores ``coords``, so dragging would
  // confuse rather than steer.
  function onSnapToNode(val: string): void {
    setManifoldLabel(name, val === "" ? null : val);
  }

  /** Build the node-list option set, prefixed with the "(free position)"
   *  escape hatch.  Recomputed reactively when ``info`` changes. */
  const snapOptions = $derived.by<{ value: string; label: string }[]>(() => {
    const labels = info?.node_labels ?? [];
    const roles = info?.node_roles;
    const opts = labels.map((nl, i) => {
      const role = roles?.[i];
      return { value: nl, label: nl + (role ? ` [role=${role}]` : "") };
    });
    return [{ value: "", label: "(free position)" }, ...opts];
  });

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
      {entry.enabled ? "◆" : "◇"}
    </button>

    <!-- Axis mirrors VectorStrip.axis: the name takes the neg-pole slot
         (right-aligned against the slider), the blend slider the middle,
         and the position readout the pos-pole slot — so a manifold row
         and a vector row share the same column lanes in the rack. -->
    <div class="axis">
      <span class="name" title="manifold {name}">{displayName}</span>
      <Slider
        value={entry.blend}
        min={0}
        max={1}
        step={0.05}
        oninput={onBlendInput}
        ariaLabel="blend fraction for {name}"
        title="blend fraction — how strongly to pull onto the manifold"
      />
      <span class="coords" title="authoring position: {coordsLabel}">
        {#if entry.label}
          <span class="node-pill" title="snapped to node '{entry.label}' (label-form)">%{entry.label}</span>
        {:else}
          {coordsLabel}
        {/if}
      </span>
    </div>

    <span class="blend-val" title="blend fraction">{entry.blend.toFixed(2)}</span>

    {#if !fitted}
      <span class="warn" title="no fitted tensor for the loaded model — fit it from the manifolds drawer">
        unfitted
      </span>
    {:else if stale}
      <span class="warn" title="the fitted tensor is stale — node geometry changed since the fit">
        stale
      </span>
    {/if}

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

  <div class="picker">
    {#if info}
      {#if info.node_labels.length > 0}
        <label class="snap-row">
          <span class="snap-label">snap to node</span>
          <span class="snap-select">
            <Select
              value={entry.label ?? ""}
              options={snapOptions}
              onchange={onSnapToNode}
              ariaLabel="snap to node"
              title="pick a node to switch to label-form, or '(free position)' to drag the pad"
            />
          </span>
        </label>
      {/if}
      <XYPad
        manifold={info}
        coords={entry.coords}
        onchange={onCoordsChange}
        locked={entry.label !== null}
      />
    {:else}
      <p class="picker-missing">
        manifold metadata unavailable — coordinates are still applied.
      </p>
    {/if}
  </div>
</div>

<style>
  .strip {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    /* A thin left purple stripe + matching name color differentiates
     * manifold rows from vector rows at a glance.  Matches the manifold
     * probe strip's 2px bar + ◆ glyph so the manifold family reads the
     * same across the steering and probe racks. */
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent-purple);
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
    /* Match VectorStrip's row height so a manifold header and a vector
     * row sit at the same height in the shared rack. */
    min-height: 32px;
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
  /* Axis grid mirrors VectorStrip.axis exactly so a manifold row and a
   * vector row in the same rack share the column lanes: name in the
   * neg-pole slot, blend slider in the middle, position readout in the
   * pos-pole slot. */
  .axis {
    display: grid;
    grid-template-columns: minmax(2.5em, 1fr) minmax(60px, 2.6fr) minmax(2.5em, 1fr);
    align-items: center;
    gap: var(--space-2);
    flex: 1 1 auto;
    min-width: 0;
  }
  .name {
    /* Manifold rows use purple as the name color to set them apart
     * from vector rows (which use fg-strong on the analogous label).
     * Right-aligned against the slider, mirroring the neg-pole. */
    color: var(--accent-purple);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
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
    color: var(--fg-dim);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    text-align: left;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
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
  /* Blend readout — the alpha-display analogue, same column geometry as
   * VectorStrip's α value so the two strips' numeric readouts line up
   * across the rack. */
  .blend-val {
    flex: 0 0 auto;
    min-width: 3.5em;
    text-align: right;
    color: var(--fg-muted);
    font-variant-numeric: tabular-nums;
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
  /* Layout host for the themed Select — Select owns its own theme. */
  .snap-select {
    flex: 1 1 auto;
    display: inline-flex;
  }
  .node-pill {
    background: color-mix(in srgb, var(--accent-purple) 14%, transparent);
    color: var(--accent-purple);
    padding: 0 var(--space-2);
    border-radius: var(--radius);
    font-variant-numeric: tabular-nums;
  }
</style>
