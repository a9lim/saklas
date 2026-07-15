<script lang="ts">
  // Shared transcript-highlight action for pinned readout-channel probes.
  // J-LENS and SAE cards expose it only after pinning because only then does
  // the server persist and stream that feature's per-token strength channel.

  import {
    highlightState,
    setHighlightTarget,
  } from "../../lib/stores.svelte";

  let { name }: { name: string } = $props();

  const active = $derived(highlightState.target === name);

  function toggle(): void {
    setHighlightTarget(active ? null : name);
  }
</script>

<button
  type="button"
  class="highlight-action"
  class:on={active}
  aria-pressed={active}
  aria-label={active
    ? `Deselect ${name} as transcript highlight target`
    : `Select ${name} as transcript highlight target`}
  title="highlight"
  onclick={toggle}
>{active ? "highlighted" : "highlight"}</button>

<style>
  .highlight-action {
    min-height: 24px;
    padding: var(--space-1) var(--space-3);
    color: var(--fg-muted);
    background: var(--glass);
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    font-size: var(--text-xs);
    flex: 0 0 auto;
  }
  .highlight-action:hover,
  .highlight-action.on {
    color: var(--card-accent);
    background: color-mix(in srgb, var(--card-accent) 10%, var(--glass));
  }
</style>
