<script lang="ts">
  // Context ribbon — a windowed strip of the current segment's tokens
  // around the inspected position.  Orientation + navigation in one:
  // tokens tint with the active highlight probe (same scoreToRgb ramp as
  // the transcript), the inspected token wears a ring, and clicking any
  // token moves the cursor there.  Ribbon tokens are mouse targets only
  // (tabindex -1) so the drawer's focus order stays walkable.

  import { highlightState, highlightScale } from "../../lib/stores.svelte";
  import {
    SURPRISE_TARGET,
    surpriseScore,
    probeScoreForTarget,
    scoreToRgb,
    highlightHue,
  } from "../../lib/tokens";
  import type { TokenScore } from "../../lib/types";

  let {
    tokens,
    index,
    onjump,
  }: {
    tokens: TokenScore[];
    index: number;
    onjump: (i: number) => void;
  } = $props();

  /** Tokens rendered either side of the inspected one. */
  const WINDOW = 60;

  const start = $derived(Math.max(0, index - WINDOW));
  const end = $derived(Math.min(tokens.length, index + WINDOW + 1));
  const slice = $derived(
    tokens.slice(start, end).map((tok, k) => ({ tok, i: start + k })),
  );

  function tint(tok: TokenScore): string {
    const target = highlightState.target;
    if (!target) return "";
    const score =
      target === SURPRISE_TARGET
        ? surpriseScore(tok.logprob)
        : (probeScoreForTarget(tok, target) ?? tok.score);
    const bg = scoreToRgb(score, highlightScale(target), highlightHue(target));
    return bg === "transparent" ? "" : `background-color: ${bg}`;
  }

  /** One-line rendering — newlines become a visible return glyph. */
  function label(text: string): string {
    return text.replace(/\n/g, "⏎");
  }

  let box = $state<HTMLElement | null>(null);
  $effect(() => {
    void index;
    void slice;
    box
      ?.querySelector(".current")
      ?.scrollIntoView({ inline: "center", block: "nearest" });
  });
</script>

<div class="ribbon" bind:this={box} role="group" aria-label="Token context">
  {#if start > 0}
    <span class="more" aria-hidden="true">…{start}</span>
  {/if}
  {#each slice as { tok, i } (i)}
    <button
      type="button"
      class="rtok"
      class:current={i === index}
      style={tint(tok)}
      tabindex="-1"
      aria-current={i === index}
      title={`token ${i + 1} / ${tokens.length}`}
      onclick={() => onjump(i)}
    >{label(tok.text)}</button>
  {/each}
  {#if end < tokens.length}
    <span class="more" aria-hidden="true">{tokens.length - end}…</span>
  {/if}
</div>

<style>
  .ribbon {
    display: flex;
    align-items: baseline;
    overflow-x: auto;
    white-space: nowrap;
    padding: var(--space-2) var(--space-6) var(--space-3);
    scrollbar-color: var(--glass-strong) transparent;
    scrollbar-width: thin;
  }
  .rtok {
    background: transparent;
    border: none;
    border-radius: var(--radius-sm);
    padding: var(--space-1) 0;
    margin: 0;
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    line-height: 1.4;
    white-space: pre;
    cursor: pointer;
    flex: 0 0 auto;
  }
  .rtok:hover {
    color: var(--fg);
  }
  .rtok.current {
    color: var(--fg-strong);
    outline: 1px solid var(--accent);
    outline-offset: -1px;
  }
  .more {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
    padding: 0 var(--space-3);
    flex: 0 0 auto;
  }
</style>
