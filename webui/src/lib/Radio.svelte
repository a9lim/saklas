<script lang="ts" generics="T extends string | number">
  // Themed radio button.  Used singly (one ``<Radio />`` per option,
  // sharing a ``group`` binding) or via the matching ``RadioGroup``
  // wrapper.  Same primitives as ``Checkbox`` — square turned circular,
  // ✓ glyph swapped for a centred dot.
  //
  // The component is generic over value type so callsites can keep
  // strongly-typed enums.

  interface Props {
    /** This radio's value — committed into ``group`` on click. */
    value: T;
    /** The shared selection — bind to the same variable across the
     *  group of radios. */
    group: T;
    disabled?: boolean;
    ariaLabel?: string;
    title?: string;
    label?: string;
    name?: string;
    onchange?: (value: T) => void;
  }

  let {
    value,
    group = $bindable(),
    disabled = false,
    ariaLabel,
    title,
    label,
    name,
    onchange,
  }: Props = $props();

  const selected = $derived(group === value);

  function pick(): void {
    if (disabled || selected) return;
    group = value;
    onchange?.(value);
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (disabled) return;
    if (ev.key === " " || ev.key === "Enter") {
      ev.preventDefault();
      pick();
    }
  }
</script>

<span class="sk-radio-row">
  <button
    type="button"
    role="radio"
    class="sk-radio"
    class:is-selected={selected}
    class:is-disabled={disabled}
    aria-checked={selected}
    aria-label={ariaLabel ?? label}
    data-name={name}
    {disabled}
    {title}
    onclick={pick}
    onkeydown={onKeydown}
  >
    {#if selected}
      <span class="sk-radio-dot" aria-hidden="true"></span>
    {/if}
  </button>
  {#if label}
    <span class="sk-radio-label" class:is-disabled={disabled}>{label}</span>
  {/if}
</span>

<style>
  .sk-radio-row {
    display: inline-flex;
    align-items: center;
    gap: var(--space-3);
  }

  .sk-radio {
    flex: 0 0 auto;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    padding: 0;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 50%;
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out);
  }
  .sk-radio:hover:not(.is-disabled) {
    border-color: var(--accent-glow);
  }
  .sk-radio:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }
  .sk-radio.is-selected {
    border-color: var(--accent);
  }
  .sk-radio.is-disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .sk-radio-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    pointer-events: none;
    transition: background var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }

  .sk-radio-label {
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
  }
  .sk-radio-label.is-disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
  }
</style>
