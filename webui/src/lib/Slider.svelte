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
    height: 4px;
    margin: 0;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: var(--radius-pill);
    cursor: pointer;
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
    border-radius: 50%;
    background: var(--accent-blue);
    border: 1px solid var(--bg-deep);
    cursor: pointer;
    transition: transform var(--dur-fast) var(--ease-out);
  }
  .sk-slider::-moz-range-thumb {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--accent-blue);
    border: 1px solid var(--bg-deep);
    cursor: pointer;
  }
  .sk-slider:hover:not(:disabled)::-webkit-slider-thumb {
    transform: scale(1.15);
  }
  .sk-slider:disabled::-webkit-slider-thumb {
    background: var(--fg-muted);
  }
  .sk-slider:disabled::-moz-range-thumb {
    background: var(--fg-muted);
  }
  .sk-slider:focus-visible {
    outline: 2px solid var(--accent-glow);
    outline-offset: 3px;
  }
</style>
