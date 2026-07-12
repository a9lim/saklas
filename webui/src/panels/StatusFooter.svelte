<script lang="ts">
  // Single-line generation footer.  Mirrors the TUI's status footer:
  //   ● gen 47/512 [████░░░░░░] · 23 t/s · 2.1s · ppl 8.3
  //
  // Idle state collapses to "○ idle" before the first generation lands —
  // matches the TUI behavior of hiding stats until they have something
  // real to report.

  import Bar from "../lib/charts/Bar.svelte";
  import {
    genStatus,
    geometricMeanPpl,
    pendingActions,
  } from "../lib/stores.svelte";

  // Pending-queue badge — counts the items waiting in the FIFO queue.
  // Under the v2.x queue semantics, drain is automatic on every WS
  // ``done`` event; the per-bubble ``×`` in the chat-side
  // PendingBubbles strip handles cancellation, so there's no "apply
  // now" button here anymore.  The badge stays as a status readout.
  const pendingCount = $derived(pendingActions.queue.length);
  const pendingTitle = $derived(
    pendingCount === 1
      ? "1 item queued; drains automatically on the next done event"
      : `${pendingCount} items queued; drain automatically on each done event`,
  );

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
  const finishLabel = $derived.by(() => {
    if (!genStatus.finishReason || genStatus.finishReason === "stop") return null;
    if (genStatus.finishReason === "length") return "token limit";
    if (genStatus.finishReason === "cancelled") return "stopped";
    return genStatus.finishReason;
  });
</script>

<footer class="status-footer" aria-label="Generation status">
  {#if !hasRun && !genStatus.active}
    <span class="dot idle" aria-hidden="true">○</span>
    <span class="text">idle</span>
  {:else}
    <span class="dot {genStatus.active ? 'live' : 'done'}" aria-hidden="true">●</span>
    {#if genStatus.active}
      <span class="text">gen {genStatus.tokensSoFar}/{genStatus.maxTokens || "?"}</span>
      <span class="sep" aria-hidden="true">·</span>
      <span class="bar-wrap" aria-label="progress">
        <Bar
          value={genStatus.tokensSoFar}
          max={genStatus.maxTokens || Math.max(genStatus.tokensSoFar, 1)}
          width={120}
          height={6}
          color="var(--accent-green)"
        />
      </span>
    {:else}
      <span class="text done-label">done · {genStatus.tokensSoFar} tokens</span>
    {/if}
    <span class="sep" aria-hidden="true">·</span>
    <span class="text">{tokPerSec.toFixed(1)} t/s</span>
    <span class="sep" aria-hidden="true">·</span>
    <span class="text">{elapsedSec.toFixed(1)}s</span>
    {#if ppl !== null && Number.isFinite(ppl)}
      <span class="sep" aria-hidden="true">·</span>
      <span
        class="text"
        title="predictive-distribution entropy perplexity, geometrically averaged across generated tokens"
      >entropy ppl {ppl.toFixed(2)}</span>
    {/if}
    {#if !genStatus.active && finishLabel}
      <span class="sep" aria-hidden="true">·</span>
      <span class="text muted">{finishLabel}</span>
    {/if}
  {/if}

  {#if pendingCount > 0}
    <span class="pending-badge" title={pendingTitle}>
      {pendingCount} queued
    </span>
  {/if}
</footer>

<style>
  /* Embedded in the chat column, directly above the input row — a thin
   * status line.  Horizontal padding is zero so it aligns with the log
   * and input box; borderless — the gap above already separates it from
   * the log. */
  .status-footer {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-2) 0;
    color: var(--fg-dim);
    font-size: var(--text-sm);
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    min-height: 22px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .dot.live {
    color: var(--live);
    border-radius: 50%;
    animation: dot-pulse 1.6s var(--ease-in-out) infinite;
  }
  /* Reduced-motion is handled by the global kill switch in global.css. */
  @keyframes dot-pulse {
    0% {
      box-shadow: 0 0 0 0 color-mix(in srgb, var(--live) 40%, transparent);
    }
    70% {
      box-shadow: 0 0 0 6px transparent;
    }
    100% {
      box-shadow: 0 0 0 0 transparent;
    }
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
  .done-label {
    color: var(--fg-strong);
  }
  .bar-wrap {
    display: inline-flex;
    align-items: center;
  }

  /* Pending-queue badge — status readout pushed to the right edge.
   * Display-only; per-item cancel lives on the PendingBubbles strip
   * above the composer. */
  .pending-badge {
    margin-left: auto;
    background: var(--glass-strong);
    color: var(--fg-dim);
    border: 1px solid transparent;
    padding: var(--space-1) var(--space-4);
    border-radius: var(--radius-pill);
    font-size: var(--text-sm);
    font-family: var(--font-ui);
  }
</style>
