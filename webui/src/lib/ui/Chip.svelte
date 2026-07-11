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
    /** Inactive term — readable, with a hollow/struck state treatment. */
    muted?: boolean;
    title?: string;
    onclick?: (ev: MouseEvent) => void;
    /** Grows a trailing × that fires independently of onclick. */
    onremove?: (ev: MouseEvent) => void;
    /** Accessible name for the trailing remove action. */
    removeLabel?: string;
  }

  let {
    children,
    color,
    muted = false,
    title,
    onclick,
    onremove,
    removeLabel = "Remove chip",
  }: Props = $props();
</script>

<span
  class="sk-chip"
  class:muted
  class:clickable={!!onclick}
  style:--chip-c={color}
  {title}
  role={onremove ? "group" : undefined}
>
  {#if onclick}
    <button type="button" class="body body-button" onclick={onclick}>
      {@render children()}
    </button>
  {:else}
    <span class="body">{@render children()}</span>
  {/if}
  {#if onremove}
    <button
      class="x"
      type="button"
      aria-label={removeLabel}
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
    min-width: 0;
    max-width: 100%;
    gap: var(--space-2);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    line-height: 1;
    padding: 1px 3px 1px 9px;
    border-radius: var(--radius-sm);
    color: var(--chip-c);
    /* Borderless: the hue wash IS the chip — a touch deeper than the old
     * outlined version so the shape holds without its hairline. */
    background: linear-gradient(
      120deg,
      color-mix(in srgb, var(--chip-c) 20%, transparent),
      color-mix(in srgb, var(--chip-c) 10%, transparent)
    );
    border: 1px solid transparent;
    white-space: nowrap;
    transition:
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out);
  }
  .sk-chip.muted {
    background: color-mix(in srgb, var(--chip-c) 5%, transparent);
    border-color: color-mix(in srgb, var(--chip-c) 32%, transparent);
  }
  .sk-chip.muted .body {
    text-decoration: line-through;
    text-decoration-color: color-mix(in srgb, var(--chip-c) 70%, transparent);
    text-decoration-thickness: 1px;
  }
  .sk-chip.clickable {
    cursor: pointer;
  }
  .sk-chip.clickable:hover {
    background: linear-gradient(
      120deg,
      color-mix(in srgb, var(--chip-c) 28%, transparent),
      color-mix(in srgb, var(--chip-c) 14%, transparent)
    );
  }
  .body {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    min-height: 24px;
  }
  .body-button {
    background: none;
    border: 0;
    padding: 0;
    margin: 0;
    color: inherit;
    font: inherit;
    text-align: left;
    cursor: pointer;
  }
  .body-button:focus-visible,
  .x:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }

  .x {
    flex: none;
    min-width: 24px;
    min-height: 24px;
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
