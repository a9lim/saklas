<script lang="ts">
  // v2 card material — the Observatory surface every rack card, probe
  // card, and turn sits on: translucent fill, hairline, lit from above,
  // soft drop. ``accent`` is the pillar hue; it feeds the children as
  // ``--card-accent`` (the same custom property the legacy RackCard
  // exposes, so card innards port without renames) and, when ``active``,
  // tints a hue ring.

  import type { Snippet } from "svelte";

  interface Props {
    children: Snippet;
    /** Pillar hue — any CSS color; chrome white when unset. */
    accent?: string;
    /** Active selection: hue ring. */
    active?: boolean;
    /** Dimmed (disabled term, detached probe). */
    disabled?: boolean;
  }

  let { children, accent, active = false, disabled = false }: Props =
    $props();
</script>

<div class="sk-card" class:active class:disabled style:--card-accent={accent}>
  {@render children()}
</div>

<style>
  /* Borderless: fill and a neutral drop carry the card; the border slot
   * exists only so the active state ring can fade in without reflow. */
  .sk-card {
    --card-accent: var(--accent);
    border-radius: var(--radius-lg);
    background: var(--glass);
    border: 1px solid transparent;
    box-shadow: var(--shadow-card);
    backdrop-filter: blur(var(--blur-glass));
    padding: var(--space-5) var(--space-6);
    transition:
      border-color var(--dur) var(--ease-out),
      box-shadow var(--dur) var(--ease-out),
      opacity var(--dur) var(--ease-out);
  }

  .sk-card.active {
    border-color: color-mix(in srgb, var(--card-accent) 40%, transparent);
    box-shadow: var(--shadow-card-active);
  }

  .sk-card.disabled {
    opacity: 0.55;
  }
</style>
