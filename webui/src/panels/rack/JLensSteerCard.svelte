<script lang="ts">
  // J-lens steer card — one racked ``α jlens/<word>`` token atom, wearing
  // the same RackCard chrome as the concept steer cards.  J-lens family:
  // blue accent, ■/□ marker.
  //
  //   statline : ■/□ enable toggle · word · trigger pill · ✕
  //   body     : one α slider row (per-card, NOT the shared subspace
  //              along — lens atoms run hotter than concept vectors, so
  //              each token needs its own dial; ≈0.3 is the sweet spot)

  import type { JLensSteerEntry } from "../../lib/types";
  import Slider from "../../lib/Slider.svelte";
  import {
    removeJLensFromRack,
    setJLensAlpha,
    setJLensEnabled,
    setJLensTrigger,
  } from "../../lib/stores.svelte";
  import RackCard from "./RackCard.svelte";
  import RackMarker from "./RackMarker.svelte";
  import { TRIGGER_LABEL, TRIGGER_WORD, nextTrigger } from "./triggers";

  interface Props {
    name: string;
    entry: JLensSteerEntry;
  }

  let { name, entry }: Props = $props();

  const word = $derived(name.slice("jlens/".length));
  function cycleTrigger(): void {
    setJLensTrigger(name, nextTrigger(entry.trigger));
  }
</script>

<RackCard accent="--accent-blue" disabled={!entry.enabled}>
  {#snippet statline()}
    <button
      type="button"
      class="enable"
      class:off={!entry.enabled}
      onclick={() => setJLensEnabled(name, !entry.enabled)}
      title={entry.enabled ? "disable" : "enable"}
      aria-pressed={entry.enabled}
      aria-label="Toggle steering for {name}"
    >
      <RackMarker shape="square" filled={entry.enabled} />
    </button>

    <span class="name" class:struck={!entry.enabled} title="j-lens token atom {name}">
      {word}
    </span>

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
      onclick={() => removeJLensFromRack(name)}
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
        title="coefficient"
        oninput={(v) => Number.isFinite(v) && setJLensAlpha(name, v)}
      />
      <span class="alpha-val" title="coefficient">{entry.alpha.toFixed(2)}</span>
    </div>
  {/snippet}
</RackCard>

<style>
  /* ----- statline pieces (mirrors SteerCard) ----- */
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
