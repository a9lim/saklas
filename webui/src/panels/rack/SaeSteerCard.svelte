<script lang="ts">
  // SAE steer card — one racked ``α sae/<id>`` decoder-row atom, wearing
  // the same RackCard chrome as the concept and lens steer cards.  SAE
  // family: gold accent, ▲/△ marker.
  //
  //   statline : ▲/△ enable toggle · feature id · trigger pill · ✕
  //   body     : one α slider row (per-card, like the lens atoms)

  import type { SaeSteerEntry } from "../../lib/types";
  import Slider from "../../lib/Slider.svelte";
  import {
    removeSaeFromRack,
    setSaeAlpha,
    setSaeEnabled,
    setSaeTrigger,
  } from "../../lib/stores.svelte";
  import RackCard from "./RackCard.svelte";
  import RackMarker from "./RackMarker.svelte";
  import { TRIGGER_LABEL, TRIGGER_WORD, nextTrigger } from "./triggers";

  interface Props {
    name: string;
    entry: SaeSteerEntry;
  }

  let { name, entry }: Props = $props();

  const id = $derived(name.slice("sae/".length));
  function cycleTrigger(): void {
    setSaeTrigger(name, nextTrigger(entry.trigger));
  }
</script>

<RackCard accent="--pillar-sae" disabled={!entry.enabled}>
  {#snippet statline()}
    <button
      type="button"
      class="enable"
      class:off={!entry.enabled}
      onclick={() => setSaeEnabled(name, !entry.enabled)}
      title={entry.enabled ? "Enabled (click to disable)" : "Disabled (click to enable)"}
      aria-pressed={entry.enabled}
      aria-label="Toggle steering for {name}"
    >
      <RackMarker shape="triangle" filled={entry.enabled} />
    </button>

    <span class="name" class:struck={!entry.enabled} title="SAE decoder-row atom {name}">
      {id}
    </span>

    <span class="spacer"></span>

    <button
      type="button"
      class="trigger-pill"
      onclick={cycleTrigger}
      title="trigger: {TRIGGER_LABEL[entry.trigger]} (click to cycle)"
      aria-label="trigger for {name}: {entry.trigger}"
    >
      {TRIGGER_WORD[entry.trigger]}
    </button>

    <button
      type="button"
      class="icon remove"
      onclick={() => removeSaeFromRack(name)}
      aria-label="remove {name}"
      title="remove {name}"
    >
      ✕
    </button>
  {/snippet}

  {#snippet body()}
    <div class="alpha-row">
      <span class="alpha-label">α</span>
      <Slider
        value={entry.alpha}
        min={0}
        max={1}
        step={0.05}
        ariaLabel="alpha for {name}"
        title="α — push coefficient along the feature's decoder row"
        oninput={(v) => Number.isFinite(v) && setSaeAlpha(name, v)}
      />
      <span class="alpha-val" title="push coefficient">{entry.alpha.toFixed(2)}</span>
    </div>
  {/snippet}
</RackCard>

<style>
  /* ----- statline pieces (mirrors SteerCard / JLensSteerCard) ----- */
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

  .spacer {
    flex: 1 1 auto;
    min-width: 0;
  }

  .trigger-pill {
    min-height: 24px;
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
    min-width: 24px;
    min-height: 24px;
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

  /* ----- body: α row (the manifold card's along-row shape) ----- */
  .alpha-row {
    display: grid;
    grid-template-columns: minmax(3em, auto) minmax(0, 1fr) 3em;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .alpha-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .alpha-val {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
</style>
