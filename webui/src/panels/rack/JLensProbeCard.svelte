<script lang="ts">
  // J-lens probe card — one workspace token, pinned (a ``jlens/<word>``
  // token probe: persistent, gate-able) or unpinned (a live discovery row
  // from the per-step aggregate).  Both card kinds are the same shape —
  // pinning just makes the card persistent — so one component renders
  // both, exactly like the SAE pillar's feature cards.
  //
  // A lens probe is a READOUT-channel probe (not a linear probe): the
  // reading is the token's standing in ``softmax(W_U · norm(J_l h))``
  // over all fitted layers.
  //
  //   statline : ■ (pinned, click to unpin) / □ (unpinned, click to pin) ·
  //              token · @com ±spread · strength-history sparkline
  //   body     : the strength bar (mean fitted-layer probability, 0→1
  //              absolute — the ``@when:jlens/<word>`` gate channel, the
  //              ONE readout channel), then the per-layer strength strip
  //              (``p_l`` per layer, cell colour normalized to the card's
  //              own max because absolute p spans orders of magnitude; an
  //              empty cell means the token sat below that layer's top-k
  //              cutoff, and the full ranking lives in the token
  //              drilldown's j-lens tab).

  import Bar from "../../lib/charts/Bar.svelte";
  import Sparkline from "../../lib/charts/Sparkline.svelte";
  import { detachProbe, highlightState } from "../../lib/stores.svelte";
  import { pushToast } from "../../lib/stores/toasts.svelte";
  import RackCard from "./RackCard.svelte";
  import ProbePinButton from "./ProbePinButton.svelte";
  import ProbeHighlightButton from "./ProbeHighlightButton.svelte";
  import LayerStrip from "./LayerStrip.svelte";
  import ProbeReadingRow from "./ProbeReadingRow.svelte";

  interface Props {
    /** Raw vocabulary token text (untrimmed on a discovery row; a pinned
     *  probe's word is already bare). */
    token: string;
    /** Mean fitted-layer probability, 0..1. */
    strength: number;
    /** Probability-mass-weighted depth centre of mass, 0..1; null hides
     *  the chip. */
    com: number | null;
    /** Depth spread around the CoM; null omits the ± part. */
    spread: number | null;
    /** Recent strength history (0 where the token fell below top-k). */
    series: number[];
    /** Per-layer strength cells in layer order; ``p === null`` means the
     *  token sat below that layer's top-k cutoff. */
    cells: { layer: number; p: number | null }[];
    pinned: boolean;
    /** Pin in flight — disables the action. */
    busy?: boolean;
    /** Unpinned cards only — the panel owns validation + attach. */
    onpin?: (word: string) => void;
  }

  let {
    token, strength, com, spread, series, cells, pinned, busy = false, onpin,
  }: Props = $props();

  const display = $derived(token.trim() || JSON.stringify(token));
  /** Whitespace-only tokens have no pinnable single-token word. */
  const pinnable = $derived(token.trim().length > 0);
  const name = $derived(`jlens/${display}`);
  const isHighlight = $derived(highlightState.target === name);

  /** Colour scale — the card's own max p (absolute p spans orders of
   *  magnitude, so a fixed 0→1 scale would render near-black strips). */
  const cellScale = $derived(Math.max(...cells.map((c) => c.p ?? 0), 1e-12));

  const layerCells = $derived(
    cells.map((cell) => ({
      layer: cell.layer,
      value: cell.p,
      title: cell.p === null
        ? `L${cell.layer} · below top-k`
        : `L${cell.layer} · p ${cell.p.toPrecision(3)}`,
    })),
  );

  function fmtCoord(v: number): string {
    return Number.isFinite(v) ? v.toFixed(2) : "0.00";
  }

  let unpinBusy = $state(false);

  async function onUnpin(): Promise<void> {
    if (unpinBusy) return;
    unpinBusy = true;
    // Unpinning destroys this keyed card. Capture its reactive identity before
    // the request so the completion path never reads a derived owned by the
    // now-destroyed component effect (Svelte's `derived_inert` warning).
    const probeName = name;
    try {
      await detachProbe(probeName);
      pushToast(`unpinned ${probeName}`, { kind: "info" });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      pushToast(`unpin ${probeName} failed — ${msg}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      unpinBusy = false;
    }
  }
</script>

<RackCard accent="--pillar-lens" disabled={false} active={pinned && isHighlight}>
  {#snippet statline()}
    {#if pinned}
      <ProbePinButton
        shape="square"
        pinned={true}
        disabled={unpinBusy}
        onclick={() => void onUnpin()}
        title="unpin"
        ariaLabel={`Unpin probe ${name}`}
      />
    {:else}
      <ProbePinButton
        shape="square"
        pinned={false}
        disabled={busy || !pinnable}
        onclick={() => onpin?.(display)}
        title="pin"
        ariaLabel={`Pin probe ${name}`}
      />
    {/if}

    <span
      class="name"
      title={pinned ? `probe ${name}` : `"${token}" — aggregate workspace token`}
    >
      {display}
    </span>

    {#if com !== null}
      <span
        class="com"
        title="depth ± spread"
      >@{fmtCoord(com)}{spread !== null ? ` ±${fmtCoord(spread)}` : ""}</span>
    {/if}

    <span class="spacer"></span>

    {#if pinned}
      <ProbeHighlightButton {name} />
    {/if}

    <Sparkline points={series} width={56} height={14} color="var(--card-accent)" />
  {/snippet}

  {#snippet body()}
    <!-- Strength: mean fitted-layer probability, absolute 0→1. -->
    <ProbeReadingRow ariaLabel={`Strength ${strength.toFixed(2)}`}>
      {#snippet left()}
        <span
          class="row-label"
          title="mean fitted-layer probability"
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
      ariaLabel={`Per-layer strength for ${display}`}
      emptyMessage={pinned ? undefined : "no layer data"}
      positiveColor="var(--layer-cell-lens)"
    />
  {/snippet}
</RackCard>

<style>
  /* ----- statline (mirrors SaeProbeCard) ----- */
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
