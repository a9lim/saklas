<script lang="ts">
  // Logits tab — the ranked top-K alternatives captured at this
  // position, each with a probability bar (absolute unit: a flat bar row
  // means genuine uncertainty), and the logit fork that regenerates the
  // node as a sibling with the alternative swapped in.

  import Bar from "../../lib/charts/Bar.svelte";
  import Button from "../../lib/ui/Button.svelte";
  import ProbeReadingRow from "../../panels/rack/ProbeReadingRow.svelte";
  import RackCard from "../../panels/rack/RackCard.svelte";
  import { samplingState, sendFork, closeDrawer } from "../../lib/stores.svelte";
  import type { TokenScore } from "../../lib/types";
  import EmptyState from "./EmptyState.svelte";
  import DetailSummary from "./DetailSummary.svelte";
  import DetailSection from "./DetailSection.svelte";

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

  const chosenRow = $derived(rankRows.find((row) => row.chosen) ?? null);
  const tokenProbability = $derived(
    token.logprob != null && Number.isFinite(token.logprob)
      ? Math.exp(token.logprob)
      : null,
  );
  const capturedMass = $derived(rankRows.reduce((sum, row) => sum + row.p, 0));
  const probabilityMargin = $derived(
    rankRows.length > 1 ? rankRows[0].p - rankRows[1].p : null,
  );
  const tokenPerplexity = $derived(
    token.logprob != null && Number.isFinite(token.logprob)
      ? Math.exp(-token.logprob)
      : null,
  );

  const summaryMetrics = $derived([
    {
      label: "chosen probability",
      value: tokenProbability == null ? "—" : fmtProb(tokenProbability),
      detail: token.logprob == null ? "not captured" : `logp ${fmtLogprob(token.logprob)}`,
    },
    {
      label: "chosen rank",
      value: chosenRow ? `#${chosenRow.rank}` : "—",
      detail: rankRows.length > 0 ? `of ${rankRows.length} retained` : "no alternatives retained",
    },
    {
      label: "captured mass",
      value: rankRows.length > 0 ? `${(capturedMass * 100).toFixed(1)}%` : "—",
      detail: "sum of retained probabilities",
    },
    {
      label: probabilityMargin == null ? "token perplexity" : "top-two margin",
      value: probabilityMargin == null
        ? (tokenPerplexity == null ? "—" : tokenPerplexity.toFixed(2))
        : probabilityMargin.toFixed(4),
      detail: probabilityMargin == null ? "exp(-logp)" : "absolute probability gap",
    },
  ]);

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

<DetailSummary
  accent="var(--pillar-lens)"
  eyebrow="logits"
  title="Sampling decision"
  description="The exact post-temperature, post-top-p, post-top-k distribution from which this token was selected."
  metrics={summaryMetrics}
  origin="captured"
  source="sampler"
  steering={null}
  steered={true}
  showToggle={false}
/>

<DetailSection
  title="RANKED ALTERNATIVES"
  count={rankRows.length > 0 ? `${rankRows.length} retained` : "capture unavailable"}
  description="Every card uses the same absolute probability scale; branch swaps that token into the raw decode prefix and resamples a sibling."
  accent="var(--pillar-lens)"
>
  {#if rankRows.length > 0}
    <div class="logit-grid" aria-label="Ranked token alternatives">
      {#each rankRows as row (row.rank)}
        <RackCard accent="--pillar-lens" disabled={false} active={row.chosen}>
          {#snippet statline()}
            <span class="rank">#{row.rank}</span>
            <code class="token">{JSON.stringify(row.text)}</code>
            <span class="token-id">id {row.id}</span>
            {#if row.chosen}<span class="chosen">generated</span>{/if}
            <span class="spacer"></span>
            <span class="prob">p {fmtProb(row.p)}</span>
          {/snippet}
          {#snippet body()}
            <ProbeReadingRow ariaLabel={`Probability ${fmtProb(row.p)}`}>
              {#snippet left()}<span class="row-label">probability</span>{/snippet}
              {#snippet bar()}
                <Bar value={row.p} max={1} color="var(--pillar-lens)" />
              {/snippet}
              {#snippet middle()}
                <span class="row-context">logp {fmtLogprob(row.logprob)}</span>
              {/snippet}
              {#snippet right()}<span class="row-value">{fmtProb(row.p)}</span>{/snippet}
            </ProbeReadingRow>
            <div class="row-meta">
              <span>Δ top <b>{fmtDelta(row.delta, row.rank)}</b></span>
              <span>token id <b>{row.id}</b></span>
              <span class="spacer"></span>
              <Button
                size="sm"
                disabled={row.chosen || branchingRank !== null}
                onclick={() => branchFromAlt(row)}
                title="fork with token"
              >
                {branchingRank === row.rank ? "branching…" : row.chosen ? "selected" : "branch"}
              </Button>
            </div>
          {/snippet}
        </RackCard>
      {/each}
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
</DetailSection>

<style>
  .logit-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-3);
  }
  .rank,
  .token-id,
  .prob,
  .chosen {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .rank,
  .prob {
    color: var(--pillar-lens);
    font-family: var(--font-mono);
  }
  .token {
    color: var(--fg);
    background: transparent;
    font-size: var(--text-sm);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .chosen {
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
  .row-value,
  .row-meta {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
  }
  .row-label,
  .row-value,
  .row-meta b {
    font-family: var(--font-mono);
  }
  .row-label,
  .row-value {
    text-align: right;
  }
  .row-meta {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    min-height: var(--control-target);
  }
  .row-meta b {
    color: var(--fg-dim);
    font-weight: var(--weight-normal);
  }
  .branch-error {
    color: var(--accent-red);
    font-size: var(--text-sm);
    margin: var(--space-4) 0 0;
  }
  @media (max-width: 760px) {
    .logit-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
