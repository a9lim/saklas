<script lang="ts">
  // SAE probe card — one feature of the resident SAE, pinned (a
  // ``sae/<id>`` readout probe — persistent, gate-able) or unpinned (a
  // live discovery row from the per-step top-k).  Both card kinds are the
  // same shape, exactly like the lens workspace cards — pinning just makes
  // the card persistent:
  //
  //   statline : ▲ (pinned, click to unpin) / △ (unpinned, click to pin) ·
  //              id · label · L<layer> · history sparkline
  //   body     : the strength bar — ``activation / maxActApprox``, the
  //              normalized 0..1 unit every SAE surface reads (the lens
  //              cards' absolute-scale convention).  Features without
  //              Neuronpedia metadata fall back to the raw activation on
  //              the panel-shared scale, so bars and numbers always rank
  //              identically across visible cards.

  import Bar from "../../lib/charts/Bar.svelte";
  import Sparkline from "../../lib/charts/Sparkline.svelte";
  import { detachProbe } from "../../lib/stores.svelte";
  import { pushToast } from "../../lib/stores/toasts.svelte";
  import RackCard from "./RackCard.svelte";
  import ProbePinButton from "./ProbePinButton.svelte";
  import ProbeReadingRow from "./ProbeReadingRow.svelte";

  interface Props {
    /** Feature index into the resident SAE's dictionary. */
    id: number;
    /** Optional human label (e.g. from Neuronpedia metadata). */
    label?: string | null;
    /** The resident SAE's hook layer — identity context, not a per-card fit. */
    layer: number | null;
    /** Latest reading (live step, or the settled end-of-gen aggregate).
     *  Raw activation for discovery cards; a pinned card's probe channel,
     *  which is already strength when the probe has ``max_act``
     *  (``valueIsStrength``). */
    value: number;
    /** Neuronpedia ``maxActApprox`` — the strength unit; null = no
     *  metadata (bar falls back to the raw panel scale). */
    maxAct?: number | null;
    /** True when ``value`` (and ``series``) are already normalized
     *  server-side (pinned probes with metadata). */
    valueIsStrength?: boolean;
    /** Panel-shared raw scale for metadata-less cards — the max raw
     *  reading across visible cards, so their bars stay comparable. */
    fallbackScale?: number;
    /** Recent reading history driving the sparkline (same unit as
     *  ``value``; the sparkline is shape-only, so the unit cancels). */
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
    maxAct = null,
    valueIsStrength = false,
    fallbackScale = 1,
    series,
    pinned,
    busy = false,
    onpin,
  }: Props = $props();

  const name = $derived(`sae/${id}`);
  /** Normalized 0..1 strength when the unit is known; null keeps raw. */
  const strength = $derived(
    maxAct != null && maxAct > 0
      ? (valueIsStrength ? value : value / maxAct)
      : null,
  );
  /** Raw activation view (reconstructed for a normalized pinned card). */
  const rawValue = $derived(
    valueIsStrength && maxAct != null ? value * maxAct : value,
  );

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
      <ProbePinButton
        shape="triangle"
        pinned={true}
        disabled={unpinBusy}
        onclick={() => void onUnpin()}
        title="unpin"
        ariaLabel={`Unpin probe ${name}`}
      />
    {:else}
      <ProbePinButton
        shape="triangle"
        pinned={false}
        disabled={busy}
        onclick={() => onpin?.(id)}
        title="pin"
        ariaLabel={`Pin probe ${name}`}
      />
    {/if}

    <span class="name" title="probe {name}">
      {id}{label ? ` · ${label}` : ""}
    </span>

    {#if layer !== null}
      <span class="layer" title="hook layer">L{layer}</span>
    {/if}

    <span class="spacer"></span>

    <Sparkline points={series} width={56} height={14} color="var(--card-accent)" />
  {/snippet}

  {#snippet body()}
    {#if strength !== null}
      <!-- Strength: activation / maxActApprox — absolute 0..1 scale, the
           same convention as the lens cards; the @when:sae/<id> gate
           channel reads this unit. -->
      <ProbeReadingRow ariaLabel={`Strength ${strength.toFixed(2)}`}>
        {#snippet left()}
          <span
            class="row-label"
            title="normalized activation"
          >strength</span>
        {/snippet}
        {#snippet bar()}
          <Bar value={Math.max(strength, 0)} max={1} width={160} height={8} color="var(--card-accent)" />
        {/snippet}
        {#snippet middle()}<span aria-hidden="true"></span>{/snippet}
        {#snippet right()}<span class="value">{strength.toFixed(2)}</span>{/snippet}
      </ProbeReadingRow>
    {:else}
      <!-- No Neuronpedia metadata (offline / unlisted feature): raw
           activation on the panel-shared scale, so bars still rank
           consistently with the numbers across cards. -->
      <ProbeReadingRow ariaLabel={`Activation ${rawValue.toFixed(2)}`}>
        {#snippet left()}
          <span
            class="row-label"
            title="raw activation"
          >activation</span>
        {/snippet}
        {#snippet bar()}
          <Bar value={Math.max(rawValue, 0)} max={Math.max(fallbackScale, 1)} width={160} height={8} color="var(--card-accent)" />
        {/snippet}
        {#snippet middle()}<span aria-hidden="true"></span>{/snippet}
        {#snippet right()}<span class="value">{rawValue.toFixed(2)}</span>{/snippet}
      </ProbeReadingRow>
    {/if}
  {/snippet}
</RackCard>

<style>
  /* ----- statline (mirrors JLensProbeCard / JLensTokenCard) ----- */
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
