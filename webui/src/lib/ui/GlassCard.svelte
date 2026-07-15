<script lang="ts">
  // v2 card material — the Observatory surface every rack card, probe
  // card, and turn sits on: translucent fill, hairline, lit from above,
  // soft drop. ``accent`` is the pillar hue; it feeds the children as
  // ``--card-accent`` (the same custom property the legacy RackCard
  // exposes, so card innards port without renames) and, when ``active``,
  // tints the ring + casts a faint hue glow (the B layer — glow marks
  // what is alive right now).

  import type { Snippet } from "svelte";

  interface Props {
    children: Snippet;
    /** Pillar hue — any CSS color; chrome white when unset. */
    accent?: string;
    /** Alive right now: hue ring + faint glow. */
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
  /* Borderless: fill + top-light + drop carry the card; the border slot
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
    box-shadow:
      inset 0 1px 0 rgba(255, 255, 255, 0.05),
      0 0 0 1px color-mix(in srgb, var(--card-accent) 20%, transparent),
      0 0 24px color-mix(in srgb, var(--card-accent) 7%, transparent),
      0 8px 28px rgba(0, 0, 0, 0.4);
  }

  .sk-card.disabled {
    opacity: 0.55;
  }
</style>
