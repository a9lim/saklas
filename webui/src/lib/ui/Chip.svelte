<script lang="ts">
  // v2 chip — the recipe-bar term, the depth badge, the role tag. A small
  // mono capsule washed in its hue (gradient = material, per policy).
  //
  // ``color`` is the hue (pillar or state — pass the CSS color; defaults
  // to chrome white). ``onremove`` grows the trailing ×; ``onclick`` makes
  // the body interactive (recipe chips jump to their card). Dumb about
  // content — the caller renders the label via children.

  import type { Snippet } from "svelte";

  interface Props {
    children: Snippet;
    /** Hue — any CSS color; chrome white when unset. */
    color?: string;
    /** Dimmed, e.g. a disabled term. */
    muted?: boolean;
    title?: string;
    onclick?: (ev: MouseEvent) => void;
    /** Grows a trailing × that fires independently of onclick. */
    onremove?: (ev: MouseEvent) => void;
  }

  let { children, color, muted = false, title, onclick, onremove }: Props =
    $props();
</script>

<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<!-- tabindex/role/keydown are all conditional on ``onclick`` — the span
     is interactive exactly when it's focusable; static analysis can't
     see the coupling. -->
<span
  class="sk-chip"
  class:muted
  class:clickable={!!onclick}
  style:--chip-c={color}
  {title}
  onclick={onclick}
  onkeydown={onclick
    ? (ev) => {
        if (ev.key === "Enter" || ev.key === " ") onclick(ev as unknown as MouseEvent);
      }
    : undefined}
  role={onclick ? "button" : undefined}
  tabindex={onclick ? 0 : undefined}
>
  <span class="body">{@render children()}</span>
  {#if onremove}
    <button
      class="x"
      aria-label="remove"
      onclick={(ev) => {
        ev.stopPropagation();
        onremove(ev);
      }}>×</button
    >
  {/if}
</span>

<style>
  .sk-chip {
    --chip-c: var(--accent);
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    line-height: 1;
    padding: 3px 9px;
    border-radius: var(--radius-sm);
    color: var(--chip-c);
    background: linear-gradient(
      120deg,
      color-mix(in srgb, var(--chip-c) 16%, transparent),
      color-mix(in srgb, var(--chip-c) 8%, transparent)
    );
    border: 1px solid color-mix(in srgb, var(--chip-c) 30%, transparent);
    white-space: nowrap;
    transition:
      border-color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out),
      opacity var(--dur-fast) var(--ease-out);
  }
  .sk-chip.muted {
    opacity: 0.45;
  }
  .sk-chip.clickable {
    cursor: pointer;
  }
  .sk-chip.clickable:hover {
    border-color: color-mix(in srgb, var(--chip-c) 55%, transparent);
  }
  .sk-chip:focus-visible {
    outline: 2px solid color-mix(in srgb, var(--chip-c) 45%, transparent);
    outline-offset: 2px;
  }

  .body {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }

  .x {
    background: none;
    border: none;
    padding: 0 0 0 2px;
    margin: 0;
    font-size: var(--text-sm);
    line-height: 1;
    color: color-mix(in srgb, var(--chip-c) 65%, transparent);
  }
  .x:hover {
    color: var(--chip-c);
  }
</style>
