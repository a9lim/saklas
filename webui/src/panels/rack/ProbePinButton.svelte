<script lang="ts">
  // Shared pin/unpin control for J-lens and SAE discovery/persistent cards.
  // The shape is the pillar identity; the hit target, baseline, hover, and
  // disabled behavior are deliberately identical.

  import RackMarker, { type RackMarkerShape } from "./RackMarker.svelte";

  let {
    shape,
    pinned,
    disabled = false,
    onclick,
    ariaLabel,
    title,
  }: {
    shape: RackMarkerShape;
    pinned: boolean;
    disabled?: boolean;
    onclick: () => void;
    ariaLabel: string;
    title: string;
  } = $props();
</script>

<button
  type="button"
  class="pin"
  class:pinned
  {disabled}
  {onclick}
  {title}
  aria-label={ariaLabel}
  aria-pressed={pinned}
>
  <RackMarker {shape} filled={pinned} />
</button>

<style>
  .pin {
    display: inline-grid;
    place-items: center;
    inline-size: 24px;
    block-size: 24px;
    margin: 0 -3px;
    padding: 0;
    color: var(--fg-muted);
    background: transparent;
    border: 0;
    border-radius: var(--radius-sm);
    flex: 0 0 24px;
    cursor: pointer;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out);
  }
  .pin.pinned {
    color: var(--card-accent);
  }
  .pin:hover:not(:disabled) {
    color: var(--card-accent);
    background: color-mix(in srgb, var(--card-accent) 8%, transparent);
  }
  .pin.pinned:hover:not(:disabled) {
    color: var(--accent-red);
    background: color-mix(in srgb, var(--accent-red) 8%, transparent);
  }
  .pin:disabled {
    cursor: default;
    opacity: 0.5;
  }
</style>
