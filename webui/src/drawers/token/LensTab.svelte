<script lang="ts">
  // J-lens tab — pinned ``jlens/<word>`` probe readings (when captured
  // live), the layer-aggregated chip row, then the all-fitted-layer
  // readout matrix: each row ranks softmax(W_U · norm(J_l h)) at the
  // forward that produced this token — what that layer's residual was
  // disposed to make the model say.

  import type {
    LensTokenReadoutJSON,
    ProbeReadingJSON,
  } from "../../lib/types";
  import type { ReplayReadout } from "./readout.svelte";
  import InstrumentHeader from "./InstrumentHeader.svelte";
  import EmptyState from "./EmptyState.svelte";
  import PinnedReadings from "./PinnedReadings.svelte";

  let {
    readout,
    steered = $bindable(),
    jlensFitted,
    hasReplayContext,
    pinned,
    modelId,
  }: {
    readout: ReplayReadout<LensTokenReadoutJSON>;
    steered: boolean;
    jlensFitted: boolean;
    hasReplayContext: boolean;
    /** Live-captured pinned-probe readings from the token's envelope. */
    pinned: Record<string, ProbeReadingJSON> | null;
    /** For the unfitted-state CLI hint. */
    modelId: string | null;
  } = $props();

  const showToggle = $derived(
    (readout.data?.steering ?? null) !== null || !steered,
  );

  const columnCount = $derived(
    Math.max(0, ...(readout.data?.layers.map((row) => row.tokens.length) ?? [])),
  );

  function cellStyle(logprob: number): string {
    const p = Math.min(1, Math.exp(logprob));
    const pct = Math.round(p * 60);
    return `background: color-mix(in srgb, var(--pillar-lens) ${pct}%, transparent);`;
  }

  function cellTitle(layer: number, t: { token: string; logprob: number }): string {
    const p = Math.exp(t.logprob);
    const pTxt = p >= 0.001 ? p.toFixed(4) : p.toExponential(2);
    return `L${layer} · ${JSON.stringify(t.token)} · p=${pTxt} · logprob=${t.logprob.toFixed(3)}`;
  }

  function cellText(t: { token: string }): string {
    const trimmed = t.token.trim();
    return trimmed.length > 0 ? trimmed : JSON.stringify(t.token);
  }
</script>

{#if readout.loading}
  <EmptyState title="computing…" />
{:else if readout.error}
  <EmptyState title={`readout: ${readout.error}`} />
{:else if readout.data}
  <InstrumentHeader
    origin={readout.origin}
    source={readout.source}
    steering={readout.data.steering}
    bind:steered
    {showToggle}
  />
  {#if pinned && Object.keys(pinned).length > 0}
    <PinnedReadings readings={pinned} accent="var(--pillar-lens)" />
  {/if}
  {#if (readout.data.aggregate ?? []).length > 0}
    <!-- Layer-aggregated view of the same logits across all layers:
         strength = mean probability, com = probability-mass-weighted
         depth center of mass (0 = first block, 1 = last). -->
    <div class="lens-agg" role="list" aria-label="Aggregate lens tokens">
      <span class="lens-agg-label">aggregate</span>
      {#each readout.data.aggregate ?? [] as chip, i (i)}
        <span
          class="lens-agg-chip"
          role="listitem"
          title={`"${chip.token}" — strength ${chip.strength.toFixed(3)} · com ${chip.com.toFixed(2)} ±${chip.spread.toFixed(2)}`}
        >
          <span class="lens-agg-tok">{chip.token.trim() || JSON.stringify(chip.token)}</span>
          <span class="lens-agg-com">@{chip.com.toFixed(2)}</span>
        </span>
      {/each}
    </div>
  {/if}
  <div class="grid-scroll">
    <table class="lens-table">
      <thead>
        <tr>
          <th class="corner">L \ rank</th>
          {#each { length: columnCount } as _, i (i)}
            <th class="num">{i + 1}</th>
          {/each}
        </tr>
      </thead>
      <tbody>
        {#each readout.data.layers as row (row.layer)}
          <tr>
            <th class="row-label" title={`Layer ${row.layer}`}>
              L{row.layer}
            </th>
            {#each row.tokens as cell (cell.id)}
              <td
                class="lens-cell"
                class:hit={cell.id === readout.data.token_id}
                style={cellStyle(cell.logprob)}
                title={cellTitle(row.layer, cell)}
              >
                {cellText(cell)}
              </td>
            {/each}
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
{:else if !jlensFitted}
  <EmptyState title="no J-LENS fit">
    <code>saklas lens fit {modelId ?? "<model>"}</code>
  </EmptyState>
{:else if !hasReplayContext}
  <EmptyState
    title="no raw decode record"
    detail="replay needs a loom node generated with raw-decode capture in this session"
  />
{:else}
  <EmptyState title="no readout" />
{/if}

<style>
  .lens-agg {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: var(--space-2);
    margin-bottom: var(--space-4);
  }
  .lens-agg-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-right: var(--space-2);
  }
  .lens-agg-chip {
    display: inline-flex;
    align-items: baseline;
    gap: var(--space-2);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    background: color-mix(in srgb, var(--pillar-lens) 16%, var(--glass));
    border: 1px solid transparent;
    border-radius: var(--radius-pill);
    padding: 1px var(--space-4);
  }
  .lens-agg-tok {
    color: var(--pillar-lens);
  }
  .lens-agg-com {
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
  }

  .grid-scroll {
    overflow: auto;
    max-height: 100%;
    border-radius: var(--radius);
    background: var(--bg);
  }
  .lens-table {
    border-collapse: separate;
    border-spacing: 0;
    font-variant-numeric: tabular-nums;
    font-size: var(--text-sm);
  }
  .lens-table th,
  .lens-table td {
    padding: var(--space-1) var(--space-3);
    text-align: left;
    background: var(--bg);
  }
  .lens-table thead th {
    position: sticky;
    top: 0;
    z-index: 2;
    color: var(--fg-muted);
    font-family: var(--font-ui);
    font-weight: var(--weight-medium);
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    box-shadow: var(--shadow-sticky);
  }
  .lens-table .corner {
    position: sticky;
    left: 0;
    z-index: 3;
    box-shadow: var(--shadow-sticky), var(--shadow-sticky-inline);
  }
  .lens-table .row-label {
    position: sticky;
    left: 0;
    z-index: 1;
    text-align: right;
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-weight: var(--weight-normal);
    font-size: var(--text-xs);
    box-shadow: var(--shadow-sticky-inline);
    white-space: nowrap;
  }
  .lens-cell {
    font-family: var(--font-mono);
    color: var(--fg-strong);
    white-space: nowrap;
    max-width: 12ch;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .lens-cell.hit {
    outline: 1px solid var(--accent);
    outline-offset: -1px;
  }
</style>
