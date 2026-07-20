<script lang="ts">
  // J-lens tab — pinned ``jlens/<word>`` probe readings (when captured
  // live), the layer-aggregated chip row, then the all-fitted-layer
  // readout matrix: each row ranks softmax(W_U · norm(J_l h)) at the
  // forward that produced this token — what that layer's residual was
  // disposed to make the model say.

  import type {
    LensAggregateTokenJSON,
    LensTokenReadoutJSON,
    ProbeReadingJSON,
  } from "../../lib/types";
  import Bar from "../../lib/charts/Bar.svelte";
  import LayerStrip from "../../panels/rack/LayerStrip.svelte";
  import ProbeReadingRow from "../../panels/rack/ProbeReadingRow.svelte";
  import RackCard from "../../panels/rack/RackCard.svelte";
  import type { ReplayReadout } from "./readout.svelte";
  import EmptyState from "./EmptyState.svelte";
  import PinnedReadings from "./PinnedReadings.svelte";
  import InstrumentHeader from "./InstrumentHeader.svelte";
  import DetailSection from "./DetailSection.svelte";

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

  const layerCount = $derived(readout.data?.layers.length ?? 0);

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

  function displayToken(token: string): string {
    return token.trim() || JSON.stringify(token);
  }

  interface AggregateCell {
    layer: number;
    value: number | null;
    title: string;
  }

  function aggregateCells(chip: LensAggregateTokenJSON): AggregateCell[] {
    return (readout.data?.layers ?? []).map((row) => {
      const hit = row.tokens.find((token) => token.token === chip.token)
        ?? row.tokens.find((token) => token.token.trim() === chip.token.trim());
      const value = hit ? Math.exp(hit.logprob) : null;
      return {
        layer: row.layer,
        value,
        title: value == null
          ? `L${row.layer} · below top-${columnCount}`
          : `L${row.layer} · p ${value.toPrecision(3)}`,
      };
    });
  }

  function aggregateScale(cells: AggregateCell[]): number {
    return Math.max(...cells.map((cell) => cell.value ?? 0), 1e-12);
  }

  function visibleProbability(logprob: number): string {
    const p = Math.exp(logprob);
    return p >= 0.001 ? p.toFixed(3) : p.toExponential(1);
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
    accent="var(--pillar-lens)"
  />
  {#if pinned && Object.keys(pinned).length > 0}
    <PinnedReadings readings={pinned} accent="--pillar-lens" shape="square" />
  {/if}
  {#if (readout.data.aggregate ?? []).length > 0}
    <DetailSection
      title="AGGREGATE WORKSPACE"
      count={`${readout.data.aggregate?.length ?? 0} tokens`}
      description="The main J-lens card grammar, expanded with every fitted layer retained on the token."
      accent="var(--pillar-lens)"
    >
      <div class="aggregate-grid" role="list" aria-label="Aggregate lens tokens">
        {#each readout.data.aggregate ?? [] as chip, i (i)}
          {@const cells = aggregateCells(chip)}
          {@const hitCount = cells.filter((cell) => cell.value != null).length}
          <div role="listitem">
            <RackCard
              accent="--pillar-lens"
              disabled={false}
              active={chip.token.trim() === readout.data.token_text.trim()}
            >
              {#snippet statline()}
                <span class="agg-rank">#{i + 1}</span>
                <code class="agg-token">{displayToken(chip.token)}</code>
                <span class="agg-depth">@{chip.com.toFixed(2)} ±{chip.spread.toFixed(2)}</span>
                {#if chip.token.trim() === readout.data?.token_text.trim()}
                  <span class="generated">generated</span>
                {/if}
                <span class="spacer"></span>
                <span class="agg-strength">{chip.strength.toFixed(3)}</span>
              {/snippet}
              {#snippet body()}
                <ProbeReadingRow ariaLabel={`Strength ${chip.strength.toFixed(3)}`}>
                  {#snippet left()}<span class="row-label">strength</span>{/snippet}
                  {#snippet bar()}
                    <Bar value={chip.strength} max={1} color="var(--pillar-lens)" />
                  {/snippet}
                  {#snippet middle()}<span class="row-context">{hitCount}/{layerCount} layers</span>{/snippet}
                  {#snippet right()}<span class="row-value">{chip.strength.toFixed(3)}</span>{/snippet}
                </ProbeReadingRow>
                <LayerStrip
                  {cells}
                  scale={aggregateScale(cells)}
                  positiveColor="var(--layer-cell-lens)"
                  ariaLabel={`Per-layer strength for ${displayToken(chip.token)}`}
                />
              {/snippet}
            </RackCard>
          </div>
        {/each}
      </div>
    </DetailSection>
  {/if}
  <DetailSection
    title="LAYER × VOCABULARY"
    count={`${readout.data.layers.length} layers × ${columnCount} ranks`}
    description="The complete matrix is preserved: token text and probability are visible in every retained cell, with the generated token outlined."
    accent="var(--pillar-lens)"
  >
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
                  <span class="cell-token">{cellText(cell)}</span>
                  <span class="cell-prob">p {visibleProbability(cell.logprob)}</span>
                </td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </DetailSection>
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
  .aggregate-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-3);
  }
  .agg-rank,
  .agg-depth,
  .agg-strength,
  .generated {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .agg-rank,
  .agg-strength {
    color: var(--pillar-lens);
    font-family: var(--font-mono);
  }
  .agg-token {
    color: var(--fg);
    background: transparent;
    font-size: var(--text-sm);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .generated {
    color: var(--fg);
    background: color-mix(in srgb, var(--pillar-lens) 12%, transparent);
    border-radius: var(--radius-sm);
    padding: 1px var(--space-2);
  }
  .spacer {
    flex: 1 1 auto;
    min-width: 0;
  }
  .row-label,
  .row-context,
  .row-value {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
  }
  .row-label,
  .row-value {
    text-align: right;
  }

  .grid-scroll {
    overflow: auto;
    max-height: 520px;
    border-radius: var(--radius-lg);
    background: var(--bg);
    box-shadow: var(--shadow-rack);
  }
  .lens-table {
    border-collapse: separate;
    border-spacing: 0;
    font-variant-numeric: tabular-nums;
    font-size: var(--text-sm);
  }
  .lens-table th,
  .lens-table td {
    padding: var(--space-2) var(--space-3);
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
    min-width: 92px;
    max-width: 13ch;
    overflow: hidden;
    vertical-align: top;
  }
  .cell-token,
  .cell-prob {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .cell-prob {
    margin-top: var(--space-1);
    color: var(--fg-muted);
    font-size: var(--text-2xs);
  }
  .lens-cell.hit {
    outline: 2px solid var(--pillar-lens);
    outline-offset: -2px;
  }
  @media (max-width: 760px) {
    .aggregate-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
