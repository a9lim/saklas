<script lang="ts">
  // SAE probe card — one feature of the resident SAE, pinned (a
  // ``sae/<id>`` readout probe — persistent, gate-able) or unpinned (a
  // live discovery row from the per-step top-k).  Both card kinds are the
  // same shape, exactly like the lens workspace cards — pinning just makes
  // the card persistent:
  //
  //   statline : ▲ (pinned, click to unpin) / △ (unpinned, click to pin) ·
  //              id · label · L<layer> · activation-history sparkline
  //   body     : the activation bar (scaled to the card's own history max —
  //              SAE activations carry no absolute 0→1 scale)

  import Bar from "../../lib/charts/Bar.svelte";
  import Sparkline from "../../lib/charts/Sparkline.svelte";
  import { detachProbe } from "../../lib/stores.svelte";
  import { pushToast } from "../../lib/stores/toasts.svelte";
  import RackCard from "./RackCard.svelte";

  interface Props {
    /** Feature index into the resident SAE's dictionary. */
    id: number;
    /** Optional human label (e.g. from Neuronpedia metadata). */
    label?: string | null;
    /** The resident SAE's hook layer — identity context, not a per-card fit. */
    layer: number | null;
    /** Latest activation (live step, or the settled end-of-gen aggregate). */
    value: number;
    /** Recent activation history driving the sparkline + the bar scale. */
    series: number[];
    pinned: boolean;
    /** Disables the pin glyph while the panel validates another feature. */
    busy?: boolean;
    /** Unpinned cards only — the panel owns validation + attach. */
    onpin?: (id: number) => void;
  }

  let {
    id,
    label = null,
    layer,
    value,
    series,
    pinned,
    busy = false,
    onpin,
  }: Props = $props();

  const name = $derived(`sae/${id}`);
  const scale = $derived(Math.max(...series, value, 1));

  let unpinBusy = $state(false);

  async function onUnpin(): Promise<void> {
    if (unpinBusy) return;
    unpinBusy = true;
    try {
      await detachProbe(name);
      pushToast(`unpinned ${name}`, { kind: "info" });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      pushToast(`unpin ${name} failed — ${msg}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      unpinBusy = false;
    }
  }
</script>

<RackCard accent="--pillar-sae" disabled={false}>
  {#snippet statline()}
    {#if pinned}
      <button
        type="button"
        class="pin-glyph"
        disabled={unpinBusy}
        onclick={() => void onUnpin()}
        title="Pinned (click to unpin)"
        aria-label="Unpin probe {name}"
        aria-pressed="true"
      >▲</button>
    {:else}
      <button
        type="button"
        class="pin-glyph unpinned"
        disabled={busy}
        onclick={() => onpin?.(id)}
        title="Pin as a persistent, gate-able probe"
        aria-label="Pin probe {name}"
        aria-pressed="false"
      >△</button>
    {/if}

    <span class="name" title="probe {name} — feature activation on the resident SAE">
      {id}{label ? ` · ${label}` : ""}
    </span>

    {#if layer !== null}
      <span class="layer" title="the resident SAE's hook layer">L{layer}</span>
    {/if}

    <span class="spacer"></span>

    <Sparkline points={series} width={56} height={14} color="var(--card-accent)" />
  {/snippet}

  {#snippet body()}
    <!-- Activation: the card's own history max sets the scale (the
         @when:sae/<id> gate channel reads the same raw activation). -->
    <div class="reading">
      <span
        class="row-label"
        title="activation — the feature's raw activation this step; the @when:sae/{id} gate channel"
      >activation</span>
      <div class="bar-cell" aria-hidden="true">
        <Bar value={Math.max(value, 0)} max={scale} width={160} height={8} color="var(--card-accent)" />
      </div>
      <span class="filler" aria-hidden="true"></span>
      <span class="value">{value.toFixed(2)}</span>
    </div>
  {/snippet}
</RackCard>

<style>
  /* ----- statline (mirrors JLensProbeCard / JLensTokenCard) ----- */
  .pin-glyph {
    background: transparent;
    border: 0;
    padding: 0 var(--space-1);
    color: var(--card-accent);
    font-size: var(--text);
    line-height: 1;
    flex: 0 0 auto;
    cursor: pointer;
    transition: color var(--dur-fast) var(--ease-out);
  }
  /* Unpin is a detach — keep the red destructive-action hover (mirrors
     the steer card's ✕). */
  .pin-glyph:hover:not(:disabled) {
    color: var(--accent-red);
  }
  .pin-glyph.unpinned:hover:not(:disabled) {
    color: var(--fg-strong);
  }
  .pin-glyph:disabled {
    cursor: default;
    opacity: 0.5;
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
  .layer {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }
  .spacer {
    flex: 1 1 auto;
    min-width: 0;
  }

  /* ----- body: reading row — ProbeCard's EXACT four-column grid
     (label · bar · nearest-or-empty · value), so the bar column aligns
     pixel-for-pixel with the other pillars' cards across the tab
     switch. ----- */
  .reading {
    display: grid;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    min-height: 24px;
    grid-template-columns: minmax(2.5em, 1fr) minmax(60px, 2.6fr) minmax(2.5em, 1fr) 3.5em;
  }
  .filler {
    min-width: 0;
  }
  .row-label {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .bar-cell {
    min-width: 0;
  }
  .bar-cell :global(.bar) {
    width: 100%;
    height: 8px;
    display: block;
  }
  .value {
    color: var(--fg-muted);
    font-variant-numeric: tabular-nums;
    min-width: 3.5em;
    text-align: right;
    flex: 0 0 auto;
  }
</style>
