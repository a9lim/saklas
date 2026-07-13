<script lang="ts">
  // J-lens probe card — one pinned ``jlens/<word>`` token probe, wearing
  // the same RackCard chrome as the probe tab's cards.  A lens probe is a
  // READOUT-channel probe (not a linear probe): the reading is the token's
  // standing in ``softmax(W_U · norm(J_l h))`` over the workspace band —
  // the workspace card's quantity, so a pinned card is the *same shape* as
  // its unpinned discovery sibling (JLensTokenCard), just persistent and
  // gate-able:
  //
  //   statline : ■ pin toggle (click to unpin) · word · @com ±spread ·
  //              strength-history sparkline
  //   body     : the strength bar (mean band probability, 0→1 absolute,
  //              the @when:jlens/<word> gate channel — the ONE readout
  //              channel), then the per-layer strength strip (p_l per
  //              layer, cell color normalized to the card's own max —
  //              same convention as the workspace token cards).

  import type { ProbeRackEntry } from "../../lib/types";
  import Bar from "../../lib/charts/Bar.svelte";
  import Sparkline from "../../lib/charts/Sparkline.svelte";
  import { detachProbe } from "../../lib/stores.svelte";
  import { pushToast } from "../../lib/stores/toasts.svelte";
  import RackCard from "./RackCard.svelte";
  import ProbePinButton from "./ProbePinButton.svelte";
  import LayerStrip from "./LayerStrip.svelte";
  import ProbeReadingRow from "./ProbeReadingRow.svelte";

  interface Props {
    name: string;
    entry: ProbeRackEntry;
  }

  let { name, entry }: Props = $props();

  const word = $derived(name.slice("jlens/".length));

  // ---------- latest reading: live during gen, settled (aggregate) after ----------
  const latest = $derived(entry.aggregate ?? entry.reading);
  /** Mean band probability (axis 0 — the gate channel + workspace number). */
  const strength = $derived(latest?.coords?.[0] ?? entry.current ?? 0);

  const sparkline = $derived(entry.sparkline ?? []);

  const depthCom = $derived(latest?.depth_com?.[0] ?? null);
  const depthSpread = $derived(latest?.depth_spread?.[0] ?? null);

  // ---------- per-layer strength strip (the store's axis-0 per-layer map) ----------
  const layerKeys = $derived<string[]>(
    entry.perLayer
      ? Object.keys(entry.perLayer).sort((a, b) => Number(a) - Number(b))
      : [],
  );

  /** Color scale — the card's own max p (absolute p spans orders of
   *  magnitude, so a fixed 0→1 scale would render near-black strips).
   *  Same convention as the workspace token cards. */
  const cellScale = $derived(
    Math.max(...layerKeys.map((l) => entry.perLayer?.[l] ?? 0), 1e-12),
  );

  function cellTooltip(layer: string): string {
    const v = entry.perLayer?.[layer];
    if (typeof v !== "number" || !Number.isFinite(v)) {
      return `L${layer} · —`;
    }
    return `L${layer} · p ${v.toPrecision(3)}`;
  }

  const layerCells = $derived(
    layerKeys.map((layer) => ({
      layer: Number(layer),
      value: entry.perLayer?.[layer],
      title: cellTooltip(layer),
    })),
  );

  function fmtCoord(v: number): string {
    return Number.isFinite(v) ? v.toFixed(2) : "0.00";
  }

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

<RackCard accent="--accent-blue" disabled={false}>
  {#snippet statline()}
    <ProbePinButton
      shape="square"
      pinned={true}
      disabled={unpinBusy}
      onclick={() => void onUnpin()}
      title="unpin"
      ariaLabel={`Unpin probe ${name}`}
    />

    <span class="name" title="probe {name}">
      {word}
    </span>

    {#if depthCom !== null}
      <span
        class="com"
        title="depth ± spread"
      >@{fmtCoord(depthCom)}{depthSpread !== null ? ` ±${fmtCoord(depthSpread)}` : ""}</span>
    {/if}

    <span class="spacer"></span>

    <Sparkline points={sparkline} width={56} height={14} color="var(--card-accent)" />
  {/snippet}

  {#snippet body()}
    <!-- Strength: mean band probability, absolute 0→1 (the gate channel). -->
    <ProbeReadingRow ariaLabel={`Strength ${strength.toFixed(2)}`}>
      {#snippet left()}
        <span
          class="row-label"
          title="mean band probability"
        >strength</span>
      {/snippet}
      {#snippet bar()}
        <Bar value={strength} max={1} width={160} height={8} color="var(--card-accent)" />
      {/snippet}
      {#snippet middle()}<span aria-hidden="true"></span>{/snippet}
      {#snippet right()}<span class="value">{strength.toFixed(2)}</span>{/snippet}
    </ProbeReadingRow>

    <!-- Per-layer strength strip with L endcaps. -->
    <LayerStrip
      cells={layerCells}
      scale={cellScale}
      ariaLabel={`Per-layer strength for ${name}`}
      positiveColor="var(--layer-cell-lens)"
    />
  {/snippet}
</RackCard>

<style>
  /* ----- statline (mirrors JLensTokenCard) ----- */
  .name {
    color: var(--fg-strong);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .com {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }
  .spacer {
    flex: 1 1 auto;
    min-width: 0;
  }

  /* ----- body: reading-row content; ProbeReadingRow owns geometry. ----- */
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
  .value {
    color: var(--fg-muted);
    font-variant-numeric: tabular-nums;
    min-width: 3.5em;
    text-align: right;
    flex: 0 0 auto;
  }

</style>
