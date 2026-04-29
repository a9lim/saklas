<script lang="ts">
  // Single-line generation footer.  Mirrors the TUI's status footer:
  //   ● gen 47/512 [████░░░░░░] · 23 t/s · 2.1s · ppl 8.3
  //
  // Idle state collapses to "○ idle" before the first generation lands —
  // matches the TUI behavior of hiding stats until they have something
  // real to report.

  import Bar from "../lib/charts/Bar.svelte";
  import { genStatus, geometricMeanPpl } from "../lib/stores.svelte";

  // Live elapsed counter — ticks while gen is active, freezes on done so
  // the user can still read the final timing after the generation lands.
  let nowMs = $state(performance.now());
  $effect(() => {
    if (!genStatus.active) return;
    const id = setInterval(() => {
      nowMs = performance.now();
    }, 100);
    return () => clearInterval(id);
  });

  const elapsedSec = $derived.by(() => {
    if (!genStatus.startedAt) return 0;
    const end = genStatus.active ? nowMs : nowMs;
    return Math.max(0, (end - genStatus.startedAt) / 1000);
  });

  const tokPerSec = $derived.by(() => {
    if (genStatus.active && elapsedSec > 0) {
      return genStatus.tokensSoFar / elapsedSec;
    }
    return genStatus.tokPerSec;
  });

  const ppl = $derived(geometricMeanPpl(genStatus));

  // Have-anything: only render the full strip once a generation has at
  // least started.  "Active" is the obvious signal, but a finished gen
  // with startedAt set should also keep its trailing stats visible.
  const hasRun = $derived(genStatus.startedAt !== null);
</script>

<footer class="status-footer" aria-label="Generation status">
  {#if !hasRun && !genStatus.active}
    <span class="dot idle" aria-hidden="true">○</span>
    <span class="text">idle</span>
  {:else}
    <span class="dot {genStatus.active ? 'live' : 'done'}" aria-hidden="true">●</span>
    <span class="text">gen {genStatus.tokensSoFar}/{genStatus.maxTokens || "?"}</span>
    <span class="sep" aria-hidden="true">·</span>
    <span class="bar-wrap" aria-label="progress">
      <Bar
        value={genStatus.tokensSoFar}
        max={genStatus.maxTokens || Math.max(genStatus.tokensSoFar, 1)}
        width={120}
        height={6}
        color={genStatus.active ? "var(--accent-green)" : "var(--fg-muted)"}
      />
    </span>
    <span class="sep" aria-hidden="true">·</span>
    <span class="text">{tokPerSec.toFixed(1)} t/s</span>
    <span class="sep" aria-hidden="true">·</span>
    <span class="text">{elapsedSec.toFixed(1)}s</span>
    {#if ppl !== null && Number.isFinite(ppl)}
      <span class="sep" aria-hidden="true">·</span>
      <span class="text">ppl {ppl.toFixed(2)}</span>
    {/if}
    {#if !genStatus.active && genStatus.finishReason}
      <span class="sep" aria-hidden="true">·</span>
      <span class="text muted">{genStatus.finishReason}</span>
    {/if}
  {/if}
</footer>

<style>
  .status-footer {
    display: flex;
    align-items: center;
    gap: 0.5em;
    padding: 0.3em 1em;
    background: var(--bg-deep);
    border-top: 1px solid var(--border);
    color: var(--fg-dim);
    font-size: var(--font-size-small);
    font-family: var(--font-mono);
    min-height: 24px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .dot.live {
    color: var(--accent-green);
  }
  .dot.done {
    color: var(--fg-muted);
  }
  .dot.idle {
    color: var(--fg-muted);
  }
  .sep {
    color: var(--fg-muted);
  }
  .text.muted {
    color: var(--fg-muted);
  }
  .bar-wrap {
    display: inline-flex;
    align-items: center;
  }
</style>
