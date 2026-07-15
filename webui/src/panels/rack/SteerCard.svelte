<script lang="ts">
  // Unified steer card — one row for every steering term.  Every term is a
  // position on a fitted geometry; the card branches on ``entry.mode``:
  //
  //   subspace → a flat affine fit (a 2-node bipolar axis through the rank-8
  //              personas fan).  Subspace accent (●/○, --accent).
  //              statline: glyph · name · unfitted/stale warn · trigger · ✕
  //              body:     snap-to-node Select · XYPad   (NO per-card along —
  //                        the magnitude is the rack-level "subspace along"
  //                        master shared by every subspace term)
  //   manifold → a curved fit (e.g. emotions).  Manifold accent (◆/◇,
  //              --pillar-manifold).
  //              statline: glyph · name · unfitted/stale warn · trigger · ✕
  //              body:     snap-to-node Select · XYPad · along · onto
  //
  // The pre-4.0 ``~``/``|`` projection, ``!`` ablation, and ``:variant`` chip
  // are gone from the card (a ``%`` term can't carry projection/ablation;
  // variant survives on the entry for round-trip but isn't authored here).
  // ``s`` / ``m`` are the narrowed entry views the template renders behind
  // ``{#if s}`` / ``{#if m}`` so svelte-check enforces mode-correct access.

  import type { SteerEntry, SubspaceSteerEntry, ManifoldSteerEntry } from "../../lib/types";
  import {
    setSubspaceCoords,
    setSubspaceLabel,
    setSubspaceTrigger,
    setSubspaceEnabled,
    removeSubspaceFromRack,
    setManifoldBlend,
    setManifoldOnto,
    setManifoldCoords,
    setManifoldLabel,
    setManifoldTrigger,
    setManifoldEnabled,
    removeManifoldFromRack,
    manifoldByName,
  } from "../../lib/stores.svelte";
  import Slider from "../../lib/Slider.svelte";
  import Select from "../../lib/Select.svelte";
  import XYPad from "../manifold/XYPad.svelte";
  import RackCard from "./RackCard.svelte";
  import RackMarker from "./RackMarker.svelte";
  import { TRIGGER_LABEL, TRIGGER_WORD, nextTrigger } from "./triggers";

  interface Props {
    name: string;
    entry: SteerEntry;
  }

  let { name, entry }: Props = $props();

  // Narrowed views — exactly one is non-null per entry.
  const s = $derived<SubspaceSteerEntry | null>(
    entry.mode === "subspace" ? entry : null,
  );
  const m = $derived<ManifoldSteerEntry | null>(
    entry.mode === "manifold" ? entry : null,
  );

  // ---------- family chrome (accent + enable glyph) ----------

  const subspace = $derived(entry.mode === "subspace");
  const accent = $derived(subspace ? "--accent" : "--pillar-manifold");

  /** Display name — bare name with the namespace prefix stripped
   *  (``default/personas`` → ``personas``).  Full name stays in the tooltip. */
  const displayName = $derived(name.split("/").pop() ?? name);

  /** Catalog row — drives the node list, the XYPad bounds, and the
   *  fitted/stale chips. */
  const info = $derived(manifoldByName(name));
  const fitted = $derived(info?.fitted_for_session === true);
  const stale = $derived(info?.stale === true);

  // ---------- trigger cycle (shared vocabulary in ./triggers) ----------

  function cycleTrigger(): void {
    const next = nextTrigger(entry.trigger);
    if (entry.mode === "subspace") setSubspaceTrigger(name, next);
    else setManifoldTrigger(name, next);
  }

  function toggleEnabled(): void {
    if (entry.mode === "subspace") setSubspaceEnabled(name, !entry.enabled);
    else setManifoldEnabled(name, !entry.enabled);
  }

  function removeTerm(): void {
    // Remove from the rack only — never deletes the server-side artifact
    // (the RackDrawer's delete button owns that, behind a confirm).
    if (entry.mode === "subspace") removeSubspaceFromRack(name);
    else removeManifoldFromRack(name);
  }

  // ---------- position controls (shared snap + XYPad) ----------

  function onSnapToNode(val: string): void {
    const label = val === "" ? null : val;
    if (entry.mode === "subspace") setSubspaceLabel(name, label);
    else setManifoldLabel(name, label);
  }
  function onCoordsChange(coords: number[]): void {
    if (entry.mode === "subspace") setSubspaceCoords(name, coords);
    else setManifoldCoords(name, coords);
  }

  const snapOptions = $derived.by<{ value: string; label: string }[]>(() => {
    const labels = info?.node_labels ?? [];
    const roles = info?.node_roles;
    const opts = labels.map((nl, i) => {
      const role = roles?.[i];
      return { value: nl, label: nl + (role ? ` [role=${role}]` : "") };
    });
    return [{ value: "", label: "(free position)" }, ...opts];
  });

  const activeLabel = $derived(s?.label ?? m?.label ?? null);
  const activeCoords = $derived(s?.coords ?? m?.coords ?? []);

  // ---------- manifold-only controls ----------

  function onBlendInput(val: number): void {
    if (Number.isFinite(val)) setManifoldBlend(name, val);
  }
  function onOntoInput(val: number): void {
    if (Number.isFinite(val)) setManifoldOnto(name, val);
  }
</script>

<RackCard {accent} disabled={!entry.enabled}>
  {#snippet statline()}
    <button
      type="button"
      class="enable"
      class:off={!entry.enabled}
      onclick={toggleEnabled}
      title={entry.enabled ? "disable" : "enable"}
      aria-pressed={entry.enabled}
      aria-label="Toggle steering for {name}"
    >
      <RackMarker
        shape={subspace ? "circle" : "diamond"}
        filled={entry.enabled}
      />
    </button>

    <span class="name" class:struck={!entry.enabled} title={subspace ? `subspace ${name}` : `manifold ${name}`}>
      {displayName}
    </span>

    {#if !fitted && info}
      <span class="warn" title="fit required">
        unfitted
      </span>
    {:else if stale}
      <span class="warn" title="refit required">
        stale
      </span>
    {/if}

    <span class="spacer"></span>

    <button
      type="button"
      class="trigger-pill"
      onclick={cycleTrigger}
      title="trigger: {TRIGGER_LABEL[entry.trigger]}"
      aria-label="trigger for {name}: {entry.trigger}"
    >
      {TRIGGER_WORD[entry.trigger]}
    </button>

    <button
      type="button"
      class="icon remove"
      onclick={removeTerm}
      aria-label="remove {name}"
      title="remove {name}"
    >
      ✕
    </button>
  {/snippet}

  {#snippet body()}
    {#if info}
      {#if info.node_labels.length > 0}
        <label class="ctl-row">
          <span class="ctl-label">node</span>
          <span class="ctl-select">
            <Select
              value={activeLabel ?? ""}
              options={snapOptions}
              onchange={onSnapToNode}
              ariaLabel="snap to node"
              title="node or free position"
            />
          </span>
        </label>
      {/if}
      <XYPad
        manifold={info}
        coords={activeCoords}
        onchange={onCoordsChange}
        locked={activeLabel !== null}
      />
      {#if m}
        <!-- Curved manifold: per-card along + onto.  Subspace terms have no
             per-card along — they share the rack-level "subspace along". -->
        <div class="along-row">
          <span class="along-label">along</span>
          <Slider
            value={m.blend}
            min={0}
            max={1}
            step={0.05}
            oninput={onBlendInput}
            ariaLabel="along fraction for {name}"
            title="along"
          />
          <span class="along-val" title="along fraction">{m.blend.toFixed(2)}</span>
        </div>
        <div class="along-row">
          <span class="along-label">onto</span>
          <Slider
            value={m.onto}
            min={0}
            max={1}
            step={0.05}
            oninput={onOntoInput}
            ariaLabel="onto fraction for {name}"
            title="onto"
          />
          <span class="along-val" title="onto fraction">{m.onto.toFixed(2)}</span>
        </div>
      {:else}
        <p class="hint">shared rack magnitude</p>
      {/if}
    {:else}
      <p class="missing">metadata unavailable</p>
    {/if}
  {/snippet}
</RackCard>

<style>
  /* ----- statline pieces ----- */
  .enable {
    display: inline-grid;
    place-items: center;
    inline-size: 24px;
    block-size: 24px;
    margin: 0 -3px;
    background: transparent;
    border: 0;
    border-radius: var(--radius-sm);
    padding: 0;
    color: var(--card-accent);
    flex: 0 0 24px;
    cursor: pointer;
  }
  .enable.off {
    color: var(--fg-muted);
  }

  .name {
    color: var(--fg-strong);
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
    background: color-mix(in srgb, var(--accent-yellow) 15%, transparent);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: 0 var(--space-2);
  }

  .spacer {
    flex: 1 1 auto;
    min-width: 0;
  }

  .trigger-pill {
    min-height: var(--control-target);
    background: var(--glass);
    color: var(--fg-strong);
    border: 1px solid transparent;
    padding: var(--space-1) var(--space-3);
    border-radius: var(--radius);
    font-size: var(--text-xs);
    line-height: 1.2;
    flex: 0 0 auto;
    cursor: pointer;
    transition: background var(--dur) var(--ease-out);
  }
  .trigger-pill:hover {
    background: var(--glass-strong);
  }

  .icon {
    min-width: var(--control-target);
    min-height: var(--control-target);
    background: transparent;
    border: 0;
    color: var(--fg-muted);
    font-size: var(--text);
    line-height: 1;
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius);
    flex: 0 0 auto;
    cursor: pointer;
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

  /* ----- body: position controls ----- */
  .ctl-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    color: var(--fg-strong);
    font-size: var(--text-xs);
  }
  .ctl-label {
    color: var(--fg-muted);
    flex: 0 0 auto;
  }
  .ctl-select {
    flex: 1 1 auto;
    display: inline-flex;
    min-width: 0;
    max-width: 100%;
  }
  .along-row {
    display: grid;
    grid-template-columns: minmax(3em, auto) minmax(0, 1fr) 3em;
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
  .hint,
  .missing {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
</style>
