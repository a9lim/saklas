<script lang="ts">
  // v2 button — the one button. Three variants on one skeleton:
  //   solid  — accent-filled, dark text; the primary action of a surface
  //   ghost  — borderless glass fill + wash on hover; the default workhorse
  //   danger — red ghost; destructive affordances
  //
  // ``accent`` retints a button to a pillar hue (pass the CSS color) —
  // solid fills with it, ghost/danger tint text + hover wash. Deliberately
  // dumb otherwise: no loading state, no icon slot magic — compose in the
  // caller via the children snippet.

  import type { Snippet } from "svelte";

  interface Props {
    children: Snippet;
    variant?: "solid" | "ghost" | "danger";
    size?: "sm" | "md";
    /** Pillar/state hue override — any CSS color. */
    accent?: string;
    disabled?: boolean;
    title?: string;
    type?: "button" | "submit";
    onclick?: (ev: MouseEvent) => void;
  }

  let {
    children,
    variant = "ghost",
    size = "md",
    accent,
    disabled = false,
    title,
    type = "button",
    onclick,
  }: Props = $props();
</script>

<button
  class="sk-btn {variant} {size}"
  style:--btn-accent={accent}
  {disabled}
  {title}
  {type}
  {onclick}
>
  {@render children()}
</button>

<style>
  .sk-btn {
    --btn-accent: var(--accent);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-3);
    border-radius: var(--radius);
    font-weight: var(--weight-medium);
    line-height: 1;
    white-space: nowrap;
    border: 1px solid transparent;
    transition:
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out),
      color var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }
  .sk-btn:active:not(:disabled) {
    transform: scale(0.97);
  }
  .sk-btn:disabled {
    cursor: not-allowed;
    opacity: 0.45;
  }

  .md {
    font-size: var(--text-sm);
    padding: 7px 14px;
  }
  .sm {
    min-height: 24px;
    font-size: var(--text-xs);
    padding: 4px 10px;
    border-radius: var(--radius-sm);
  }

  /* solid — lit from above: a faint white top-light over the accent so
   * the fill reads as material (gradient across, never along, a data
   * axis — buttons carry no data axis, so this is pure material). */
  .solid {
    background:
      linear-gradient(180deg, rgba(255, 255, 255, 0.18), transparent 55%),
      var(--btn-accent);
    color: var(--text-on-accent);
    border-color: transparent;
  }
  .solid:hover:not(:disabled) {
    filter: brightness(1.08);
  }

  /* Borderless doctrine: the control floats UP on a glass fill — shape
   * without an outline. Hover deepens the wash toward the accent. */
  .ghost {
    background: var(--glass);
    color: var(--fg-dim);
    border-color: transparent;
  }
  .ghost:hover:not(:disabled) {
    background: color-mix(in srgb, var(--btn-accent) 10%, var(--glass));
    color: var(--fg);
  }

  .danger {
    background: color-mix(in srgb, var(--accent-red) 7%, var(--glass));
    color: var(--accent-red);
    border-color: transparent;
  }
  .danger:hover:not(:disabled) {
    background: color-mix(in srgb, var(--accent-red) 14%, var(--glass));
  }

  .sk-btn:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }
</style>
