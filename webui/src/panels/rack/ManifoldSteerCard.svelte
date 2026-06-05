<script lang="ts">
  // Manifold steer card — the harmonised replacement for ManifoldStrip.
  // A manifold term places generation at a point of a fitted manifold.
  // Whether it reads as the *subspace* (flat) family or the *manifold*
  // (curved) family is decided by the catalog ``fit_mode``:
  //   - ``pca`` / ``baked`` → flat affine subspace (e.g. personas):
  //     accent ``--accent``, glyph ●/○, label-form only (NO free XYPad —
  //     a flat ``%`` manifold raises on free coords at the engine).
  //   - everything else (``spectral`` / ``authored``) → curved surface
  //     (e.g. pad): accent ``--accent-purple``, glyph ◆/◇, snap-to-node
  //     Select + XYPad + along slider.
  //
  // Composes the shared RackCard chrome: a statline on top (enable glyph ·
  // name · warn chip · trigger pill · ✕) with the position controls
  // stacked below.

  import type { ManifoldRackEntry, Trigger } from "../../lib/types";
  import {
    manifoldByName,
    setManifoldBlend,
    setManifoldOnto,
    setManifoldCoords,
    setManifoldLabel,
    setManifoldTrigger,
    setManifoldEnabled,
    removeManifoldFromRack,
  } from "../../lib/stores.svelte";
  import Slider from "../../lib/Slider.svelte";
  import Select from "../../lib/Select.svelte";
  import XYPad from "../manifold/XYPad.svelte";
  import RackCard from "./RackCard.svelte";

  interface Props {
    name: string;
    entry: ManifoldRackEntry;
  }

  let { name, entry }: Props = $props();

  const info = $derived(manifoldByName(name));

  /** Flat (affine) iff the fit is a discover-pca or baked direction —
   *  those are the subspace family; everything else is curved. */
  const flat = $derived(
    info?.fit_mode === "pca" || info?.fit_mode === "baked",
  );

  const accent = $derived(flat ? "--accent" : "--accent-purple");
  const enableGlyph = $derived(
    flat ? (entry.enabled ? "●" : "○") : (entry.enabled ? "◆" : "◇"),
  );

  /** Display name — bare manifold name with the namespace prefix stripped
   *  (``default/personas`` → ``personas``).  Full name stays in the
   *  tooltip for disambiguation. */
  const displayName = $derived(name.split("/").pop() ?? name);

  // ---------- trigger cycle (mirror VectorSteerCard) ----------

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

  function onOntoInput(v: number): void {
    if (!Number.isFinite(v)) return;
    setManifoldOnto(name, v);
  }

  function onCoordsChange(coords: number[]): void {
    setManifoldCoords(name, coords);
  }

  function toggleEnabled(): void {
    setManifoldEnabled(name, !entry.enabled);
  }

  // Snap-to-node dropdown: the user can pick a labeled node from the
  // manifold's node list, which switches the term to label-form
  // (``personas%pirate``).  An empty selection clears the label binding —
  // the position becomes free-form, drag on the XYPad to author (curved
  // only).  ``setManifoldLabel`` mirrors the catalog's ``node_coords[idx]``
  // onto ``entry.coords`` so any XYPad readout reflects the picked node.
  function onSnapToNode(val: string): void {
    setManifoldLabel(name, val === "" ? null : val);
  }

  /** Build the node-list option set.  A flat manifold is label-only at
   *  the engine, so it offers no "(free position)" escape hatch; a curved
   *  manifold prepends one. */
  const snapOptions = $derived.by<{ value: string; label: string }[]>(() => {
    const labels = info?.node_labels ?? [];
    const roles = info?.node_roles;
    const opts = labels.map((nl, i) => {
      const role = roles?.[i];
      return { value: nl, label: nl + (role ? ` [role=${role}]` : "") };
    });
    if (flat) return opts;
    return [{ value: "", label: "(free position)" }, ...opts];
  });

  const coordsLabel = $derived(
    entry.coords.map((c) => (Number.isFinite(c) ? c.toFixed(2) : "0")).join(", "),
  );

  // A manifold needs a tensor for the loaded model to actually steer.
  const fitted = $derived(info?.fitted_for_session === true);
  const stale = $derived(info?.stale === true);
</script>

<RackCard {accent} disabled={!entry.enabled}>
  {#snippet statline()}
    <button
      type="button"
      class="enable"
      class:off={!entry.enabled}
      onclick={toggleEnabled}
      aria-pressed={entry.enabled}
      title={entry.enabled ? "Enabled (click to disable)" : "Disabled (click to enable)"}
      aria-label="Toggle manifold {name}"
    >
      {enableGlyph}
    </button>

    <span class="name" class:struck={!entry.enabled} title="manifold {name}">
      {displayName}
    </span>

    {#if !fitted}
      <span class="warn" title="no fitted tensor for the loaded model — fit it from the manifolds drawer">
        unfitted
      </span>
    {:else if stale}
      <span class="warn" title="the fitted tensor is stale — node geometry changed since the fit">
        stale
      </span>
    {/if}

    <span class="spacer"></span>

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
  {/snippet}

  {#snippet body()}
    {#if info}
      {#if flat}
        <!-- Flat (affine) manifold (e.g. personas): label-only.  A free
             ``%`` position raises at the engine, so there is no XYPad —
             snap to a node, then slide ``along`` toward it. -->
        <label class="ctl-row">
          <span class="ctl-label">node</span>
          <span class="ctl-select">
            <Select
              value={entry.label ?? ""}
              options={snapOptions}
              onchange={onSnapToNode}
              ariaLabel="snap to node"
              title="pick a node to place this term at (label-form)"
            />
          </span>
        </label>
      {:else}
        <!-- Curved manifold (e.g. pad): optional snap-to-node + free XYPad
             position + along slider. -->
        {#if info.node_labels.length > 0}
          <label class="ctl-row">
            <span class="ctl-label">snap to node</span>
            <span class="ctl-select">
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
      {/if}

      <!-- along slider — how far to slide the in-subspace foot toward the
           target position.  Shared by flat + curved. -->
      <div class="along-row">
        <span class="along-label">along</span>
        <Slider
          value={entry.blend}
          min={0}
          max={1}
          step={0.05}
          oninput={onBlendInput}
          ariaLabel="along fraction for {name}"
          title="along — how far to slide toward the manifold position"
        />
        <span class="along-val" title="along fraction">{entry.blend.toFixed(2)}</span>
      </div>
      {#if !flat}
        <!-- onto — curved-only second coefficient: collapse the off-surface
             in-subspace residual onto the manifold.  Vacuous for flat/affine
             terms, so the row is hidden there. -->
        <div class="along-row">
          <span class="along-label">onto</span>
          <Slider
            value={entry.onto}
            min={0}
            max={1}
            step={0.05}
            oninput={onOntoInput}
            ariaLabel="onto fraction for {name}"
            title="onto — collapse the off-surface residual onto the manifold (curved only)"
          />
          <span class="along-val" title="onto fraction">{entry.onto.toFixed(2)}</span>
        </div>
      {/if}
    {:else}
      <p class="missing">
        manifold metadata unavailable — coordinates are still applied.
      </p>
    {/if}
  {/snippet}
</RackCard>

<style>
  /* ----- statline pieces ----- */
  .enable {
    background: transparent;
    border: 0;
    padding: 0 var(--space-1);
    color: var(--card-accent);
    font-size: var(--text);
    line-height: 1;
    flex: 0 0 auto;
    cursor: pointer;
  }
  .enable.off {
    color: var(--fg-muted);
  }
  .name {
    color: var(--card-accent);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .name.struck {
    text-decoration: line-through;
    color: var(--fg-muted);
  }
  .warn {
    flex: 0 0 auto;
    color: var(--accent-yellow);
    font-size: var(--text-2xs);
    border: 1px solid var(--accent-yellow);
    border-radius: var(--radius);
    padding: 0 var(--space-2);
  }
  .spacer {
    flex: 1 1 auto;
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

  /* ----- body controls ----- */
  .ctl-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    color: var(--fg-strong);
    font-size: var(--text-xs);
  }
  .ctl-label {
    color: var(--fg-muted);
    flex: 0 0 auto;
  }
  /* Layout host for the themed Select — Select owns its own theme. */
  .ctl-select {
    flex: 1 1 auto;
    display: inline-flex;
  }
  .along-row {
    display: grid;
    grid-template-columns: minmax(3em, auto) 1fr 3em;
    align-items: center;
    gap: var(--space-2);
  }
  .along-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-transform: lowercase;
  }
  .along-val {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
  .missing {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
</style>
