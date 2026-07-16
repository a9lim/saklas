<script lang="ts">
  // Logits tab — the ranked top-K alternatives captured at this
  // position, each with a probability bar (absolute unit: a flat bar row
  // means genuine uncertainty), and the logit fork that regenerates the
  // node as a sibling with the alternative swapped in.

  import Bar from "../../lib/charts/Bar.svelte";
  import Button from "../../lib/ui/Button.svelte";
  import { samplingState, sendFork, closeDrawer } from "../../lib/stores.svelte";
  import type { TokenScore } from "../../lib/types";
  import EmptyState from "./EmptyState.svelte";

  let {
    token,
    nodeId,
  }: {
    token: TokenScore;
    nodeId: string | null;
  } = $props();

  interface RankRow {
    rank: number;
    id: number;
    text: string;
    logprob: number;
    /** ``exp(logprob)`` — the post-sampler probability. */
    p: number;
    /** ``logprob - logprob_rank1`` — zero for rank 1. */
    delta: number;
    /** True iff this row's ``id`` matches the chosen token id. */
    chosen: boolean;
  }

  /** Server emits ``top_alts`` in descending-logprob order, so
   *  ``index + 1`` is the rank. */
  const rankRows = $derived.by<RankRow[]>(() => {
    const alts = token.topAlts;
    if (!alts || alts.length === 0) return [];
    const lp0 = alts[0]?.logprob ?? 0;
    return alts.map((a, i) => ({
      rank: i + 1,
      id: a.id,
      text: a.text,
      logprob: a.logprob,
      p: Math.exp(a.logprob),
      delta: a.logprob - lp0,
      chosen: token.tokenId != null && a.id === token.tokenId,
    }));
  });

  let branchingRank = $state<number | null>(null);
  let branchError = $state<string | null>(null);

  function fmtLogprob(v: number | null | undefined): string {
    if (v == null || !Number.isFinite(v)) return "—";
    return v.toFixed(3);
  }
  function fmtProb(p: number): string {
    if (!Number.isFinite(p)) return "—";
    if (p >= 0.001) return p.toFixed(4);
    return p.toExponential(2);
  }
  function fmtDelta(d: number, rank: number): string {
    if (rank === 1) return "—";
    if (!Number.isFinite(d)) return "—";
    return d.toFixed(3);
  }

  /** Flip the SamplingStrip's alts toggle from the empty state.  Takes
   *  effect on the next generation — the current token's alts weren't
   *  captured and can't be recovered. */
  function enableAlts(): void {
    if (samplingState.return_top_k === 0) {
      samplingState.return_top_k = 8;
    }
  }

  /** Logit fork — replay the node's raw decode prefix with this row's
   *  token swapped in and resample the continuation as a sibling. */
  async function branchFromAlt(row: RankRow): Promise<void> {
    branchError = null;
    if (!nodeId) {
      branchError = "no generated loom node is available for this token";
      return;
    }
    if (token.rawIndex == null) {
      branchError =
        "this token has no raw-decode index; forking needs a node " +
        "generated with raw-decode capture in this session";
      return;
    }
    branchingRank = row.rank;
    try {
      await sendFork(nodeId, token.rawIndex, row.id);
      closeDrawer();
    } catch (e) {
      branchError = e instanceof Error ? e.message : String(e);
    } finally {
      branchingRank = null;
    }
  }
</script>

{#if rankRows.length > 0}
  <div class="grid-scroll">
    <table class="logits-table">
      <thead>
        <tr>
          <th class="num">rank</th>
          <th class="tok">token</th>
          <th class="pbar" aria-label="probability bar"></th>
          <th class="num">p</th>
          <th class="num">logprob</th>
          <th class="num">Δ top</th>
          <th class="num">branch</th>
        </tr>
      </thead>
      <tbody>
        {#each rankRows as row (row.rank)}
          <tr class:chosen={row.chosen}>
            <td class="num">
              {row.chosen ? "* " : ""}{row.rank}
            </td>
            <td class="tok">
              <code>{JSON.stringify(row.text)}</code>
            </td>
            <td class="pbar">
              <Bar
                value={row.p}
                max={1}
                width={72}
                height={6}
                color="color-mix(in srgb, var(--accent) 55%, transparent)"
              />
            </td>
            <td class="num">{fmtProb(row.p)}</td>
            <td class="num">{fmtLogprob(row.logprob)}</td>
            <td class="num">{fmtDelta(row.delta, row.rank)}</td>
            <td class="num">
              <Button
                size="sm"
                disabled={row.chosen || branchingRank !== null}
                onclick={() => branchFromAlt(row)}
                title="fork with token"
              >
                {branchingRank === row.rank ? "…" : "fork"}
              </Button>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  {#if branchError}
    <p class="branch-error">{branchError}</p>
  {/if}
{:else if token.logprob != null}
  <EmptyState title={`logprob ${fmtLogprob(token.logprob)} · no alternatives captured`}>
    <Button onclick={enableAlts} disabled={samplingState.return_top_k > 0}>
      {samplingState.return_top_k > 0 ? "alts on next run" : "enable alts"}
    </Button>
  </EmptyState>
{:else}
  <EmptyState title="no logprob data">
    <Button onclick={enableAlts} disabled={samplingState.return_top_k > 0}>
      {samplingState.return_top_k > 0 ? "alts on next run" : "enable alts"}
    </Button>
  </EmptyState>
{/if}

<style>
  .grid-scroll {
    overflow: auto;
    max-height: 100%;
    border-radius: var(--radius);
    background: var(--bg);
  }
  .logits-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-variant-numeric: tabular-nums;
    font-size: var(--text-sm);
  }
  .logits-table th,
  .logits-table td {
    padding: var(--space-2) var(--space-4);
    text-align: left;
    background: var(--bg);
  }
  .logits-table thead th {
    position: sticky;
    top: 0;
    z-index: 1;
    color: var(--fg-muted);
    font-family: var(--font-ui);
    font-weight: var(--weight-medium);
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    box-shadow: var(--shadow-sticky);
  }
  .logits-table td {
    font-family: var(--font-mono);
  }
  .logits-table td.num,
  .logits-table th.num {
    text-align: right;
    width: 1%;
    white-space: nowrap;
  }
  .logits-table td.pbar,
  .logits-table th.pbar {
    width: 1%;
    white-space: nowrap;
    line-height: 0;
  }
  .logits-table td.tok code {
    color: var(--fg-strong);
    background: transparent;
    word-break: break-all;
  }
  .logits-table tbody tr:hover td {
    background: color-mix(in srgb, var(--bg-hover) 60%, var(--bg));
  }
  /* Chosen row gets a soft accent wash + a heavier color so it reads at
     a glance. */
  .logits-table tr.chosen td,
  .logits-table tbody tr.chosen:hover td {
    background: var(--accent-subtle);
    color: var(--fg-strong);
  }
  .branch-error {
    color: var(--accent-red);
    font-size: var(--text-sm);
    margin: var(--space-4) 0 0;
  }
</style>
