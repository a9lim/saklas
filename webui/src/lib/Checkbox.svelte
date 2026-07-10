<script lang="ts">
  // Themed checkbox — square box with `--radius`, accent fill on
  // checked, ✓ glyph rendered via CSS so we don't depend on a font.
  // Drop-in for ``<input type="checkbox">`` everywhere the webui has
  // them inline-next-to-label.
  //
  // The component renders the box only — wrap it in a host ``<label>``
  // (or pass ``label`` for a quick inline one) the way the existing
  // ``.check`` / ``.axis-check`` rows do.

  interface Props {
    checked: boolean;
    disabled?: boolean;
    ariaLabel?: string;
    title?: string;
    /** Optional inline label rendered next to the box.  Most callsites
     *  use their own ``<span>``-after-the-input layout; that still
     *  works — just leave this empty and place the checkbox where the
     *  native input used to sit. */
    label?: string;
    onchange?: (checked: boolean) => void;
  }

  let {
    checked = $bindable(),
    disabled = false,
    ariaLabel,
    title,
    label,
    onchange,
  }: Props = $props();

  function toggle(): void {
    if (disabled) return;
    checked = !checked;
    onchange?.(checked);
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (disabled) return;
    if (ev.key === " " || ev.key === "Enter") {
      ev.preventDefault();
      toggle();
    }
  }
</script>

<span class="sk-checkbox-row">
  <button
    type="button"
    role="checkbox"
    class="sk-checkbox"
    class:is-checked={checked}
    class:is-disabled={disabled}
    aria-checked={checked}
    aria-label={ariaLabel ?? label}
    {disabled}
    {title}
    onclick={toggle}
    onkeydown={onKeydown}
  >
    {#if checked}
      <span class="sk-checkbox-glyph" aria-hidden="true">✓</span>
    {/if}
  </button>
  {#if label}
    <span class="sk-checkbox-label" class:is-disabled={disabled}>{label}</span>
  {/if}
</span>

<style>
  .sk-checkbox-row {
    display: inline-flex;
    align-items: center;
    gap: var(--space-3);
  }

  .sk-checkbox {
    flex: 0 0 auto;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    padding: 0;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out);
  }
  .sk-checkbox:hover:not(.is-disabled) {
    border-color: var(--accent-glow);
  }
  .sk-checkbox:focus-visible {
    outline: 2px solid var(--accent-glow);
    outline-offset: 2px;
  }
  .sk-checkbox.is-checked {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.18), transparent 55%),
      var(--accent);
    border-color: var(--accent);
  }
  .sk-checkbox.is-checked:hover:not(.is-disabled) {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.18), transparent 55%),
      var(--accent-light);
    border-color: var(--accent-light);
  }
  .sk-checkbox.is-disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .sk-checkbox-glyph {
    font-size: 10px;
    line-height: 1;
    color: var(--text-on-accent);
    font-weight: var(--weight-bold);
    pointer-events: none;
  }

  .sk-checkbox-label {
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
  }
  .sk-checkbox-label.is-disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
  }
</style>
