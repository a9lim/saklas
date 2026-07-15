<script lang="ts">
  // Shared range slider — one consistent thumb / track across the whole
  // webui (sampling strip, steering strips, the steering picker).
  //
  // Deliberately dumb: it reports the raw value through ``oninput`` and
  // ``bind:value``; consumers own any snapping (e.g. the steering strip's
  // 0-detent).  The thumb tints to ``accent`` so a slider reads as the
  // same control wherever it appears.

  interface Props {
    /** Current value — bindable. */
    value: number;
    min?: number;
    max?: number;
    step?: number;
    disabled?: boolean;
    ariaLabel?: string;
    title?: string;
    /** Fired on every drag tick with the raw (un-snapped) value. */
    oninput?: (value: number) => void;
  }

  let {
    value = $bindable(),
    min = 0,
    max = 1,
    step = 0.01,
    disabled = false,
    ariaLabel,
    title,
    oninput,
  }: Props = $props();

  function handle(ev: Event): void {
    const v = parseFloat((ev.currentTarget as HTMLInputElement).value);
    if (!Number.isFinite(v)) return;
    value = v;
    oninput?.(v);
  }
</script>

<input
  class="sk-slider"
  type="range"
  {min}
  {max}
  {step}
  value={value}
  {disabled}
  {title}
  aria-label={ariaLabel}
  oninput={handle}
/>

<style>
  .sk-slider {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: var(--control-target);
    margin: 0;
    /* Borderless: the track is a recessed groove — fill only.  The thumb
     * keeps its --bg-deep cutout ring (a glyph stroke, not chrome). */
    background: transparent;
    border: 0;
    border-radius: var(--radius-pill);
    cursor: pointer;
  }
  .sk-slider::-webkit-slider-runnable-track {
    height: 4px;
    background: var(--input-well);
    border: 0;
    border-radius: var(--radius-pill);
  }
  .sk-slider::-moz-range-track {
    height: 4px;
    background: var(--input-well);
    border: 0;
    border-radius: var(--radius-pill);
  }
  .sk-slider:disabled {
    cursor: not-allowed;
    opacity: 0.5;
  }

  .sk-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 12px;
    height: 12px;
    margin-top: -4px;
    border-radius: 50%;
    background: var(--accent);
    border: 1px solid var(--bg-deep);
    cursor: pointer;
    transition: transform var(--dur-fast) var(--ease-out),
      box-shadow var(--dur-fast) var(--ease-out);
  }
  .sk-slider::-moz-range-thumb {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--accent);
    border: 1px solid var(--bg-deep);
    cursor: pointer;
    transition: transform var(--dur-fast) var(--ease-out),
      box-shadow var(--dur-fast) var(--ease-out);
  }
  .sk-slider:hover:not(:disabled)::-webkit-slider-thumb,
  .sk-slider:active:not(:disabled)::-webkit-slider-thumb {
    transform: scale(1.15);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent);
  }
  .sk-slider:hover:not(:disabled)::-moz-range-thumb,
  .sk-slider:active:not(:disabled)::-moz-range-thumb {
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent);
  }
  .sk-slider:disabled::-webkit-slider-thumb {
    background: var(--fg-muted);
  }
  .sk-slider:disabled::-moz-range-thumb {
    background: var(--fg-muted);
  }
  .sk-slider:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 0;
  }
</style>
