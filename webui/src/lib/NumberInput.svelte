<script lang="ts">
  // Themed numeric input — strips the OS spinner buttons, exposes a
  // hover-revealed ▴/▾ pair on the right edge.  Keyboard ↑/↓ still
  // works natively because the underlying ``<input type="number">`` is
  // unchanged; the visible steppers just give mouse users the same
  // affordance.
  //
  // Drop-in for ``<input type="number">``: same ``min`` / ``max`` /
  // ``step`` / ``placeholder`` props, ``bind:value`` to a number (with
  // ``allowEmpty`` if the caller treats blank as ``null``).

  interface Props {
    /** Bindable numeric value.  When ``allowEmpty`` is true and the
     *  field is blank the binding receives ``null``. */
    value: number | null;
    min?: number;
    max?: number;
    step?: number;
    placeholder?: string;
    disabled?: boolean;
    ariaLabel?: string;
    title?: string;
    /** Allow a blank state — emits ``null`` on empty.  Default false. */
    allowEmpty?: boolean;
    /** Fires on every keystroke (after numeric coercion).  Bind:value
     *  is updated regardless. */
    oninput?: (value: number | null) => void;
    /** Fires only when the input commits (blur or Enter), mirroring the
     *  native ``change`` event.  Use this when the parent should defer
     *  expensive side-effects (network PATCH, etc.) until the user
     *  finishes typing. */
    onchange?: (value: number | null) => void;
    /** Forwarded to the inner ``<input>`` — modal-style Enter-to-commit
     *  hosts can hook this without an extra wrapper. */
    onkeydown?: (ev: KeyboardEvent) => void;
  }

  let {
    value = $bindable(),
    min,
    max,
    step = 1,
    placeholder,
    disabled = false,
    ariaLabel,
    title,
    allowEmpty = false,
    oninput,
    onchange,
    onkeydown,
  }: Props = $props();

  let inputEl: HTMLInputElement | null = $state(null);

  /** Imperative focus — bound parents can call ``ref.focus()`` to drop
   *  the cursor into the input, same as a native ``HTMLInputElement``. */
  export function focus(): void {
    inputEl?.focus();
  }
  /** Imperative select — mirrors ``HTMLInputElement.select`` so the
   *  modal-style "focus + select existing value for fast retype" pattern
   *  still works after the chrome swap. */
  export function select(): void {
    inputEl?.select();
  }

  function clamp(v: number | null): number | null {
    if (v === null) return null;
    let next = v;
    if (min !== undefined && next < min) next = min;
    if (max !== undefined && next > max) next = max;
    return next;
  }

  function emitInput(v: number | null): void {
    const next = clamp(v);
    value = next;
    oninput?.(next);
  }

  function emitChange(v: number | null): void {
    const next = clamp(v);
    value = next;
    onchange?.(next);
  }

  function onInput(ev: Event): void {
    const raw = (ev.currentTarget as HTMLInputElement).value;
    if (raw === "") {
      emitInput(allowEmpty ? null : 0);
      return;
    }
    const v = Number(raw);
    if (!Number.isFinite(v)) return;
    emitInput(v);
  }

  function onChange(ev: Event): void {
    const raw = (ev.currentTarget as HTMLInputElement).value;
    if (raw === "") {
      emitChange(allowEmpty ? null : 0);
      return;
    }
    const v = Number(raw);
    if (!Number.isFinite(v)) return;
    emitChange(v);
  }

  function nudge(direction: 1 | -1): void {
    if (disabled) return;
    const base = value ?? 0;
    let next = base + direction * step;
    // Round to ``step``'s decimal precision so 0.1 nudges stay clean.
    const decimals = (step.toString().split(".")[1] ?? "").length;
    next = Number(next.toFixed(decimals));
    // Stepper button is treated as a commit — fires both oninput and
    // onchange so hosts that only listen on commit also see it.
    emitInput(next);
    onchange?.(clamp(next));
    inputEl?.focus();
  }
</script>

<span class="sk-number" class:is-disabled={disabled}>
  <input
    bind:this={inputEl}
    type="number"
    class="sk-number-input"
    value={value === null ? "" : value}
    {min}
    {max}
    {step}
    {placeholder}
    {disabled}
    {title}
    aria-label={ariaLabel}
    oninput={onInput}
    onchange={onChange}
    {onkeydown}
  />
  {#if !disabled}
    <span class="sk-number-steppers" aria-hidden="true">
      <button
        type="button"
        class="sk-number-step"
        tabindex="-1"
        onclick={() => nudge(1)}
      >▴</button>
      <button
        type="button"
        class="sk-number-step"
        tabindex="-1"
        onclick={() => nudge(-1)}
      >▾</button>
    </span>
  {/if}
</span>

<style>
  .sk-number {
    position: relative;
    display: inline-flex;
    align-items: stretch;
    width: 100%;
  }

  /* Strip the native spinner across browsers. */
  .sk-number-input {
    flex: 1 1 0;
    min-width: 0;
    width: 100%;
    padding: var(--space-2) var(--space-3);
    padding-right: var(--space-6); /* room for the steppers */
    /* Borderless input: recessed well fill; ring on focus only. */
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    -moz-appearance: textfield;
    appearance: textfield;
    transition: border-color var(--dur-fast) var(--ease-out);
  }
  .sk-number-input::-webkit-outer-spin-button,
  .sk-number-input::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
  .sk-number-input:focus-visible {
    outline: 2px solid var(--accent-glow);
    outline-offset: 1px;
    border-color: var(--accent-glow);
  }
  .sk-number-input:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .sk-number-steppers {
    position: absolute;
    top: 1px;
    bottom: 1px;
    right: 1px;
    display: flex;
    flex-direction: column;
    width: 14px;
    border-left: 1px solid transparent;
    opacity: 0;
    transition: opacity var(--dur-fast) var(--ease-out);
    pointer-events: none;
  }
  .sk-number:hover .sk-number-steppers,
  .sk-number:focus-within .sk-number-steppers {
    opacity: 1;
    pointer-events: auto;
  }

  .sk-number-step {
    flex: 1 1 0;
    padding: 0;
    background: transparent;
    color: var(--fg-muted);
    border: 0;
    font-size: 8px;
    line-height: 1;
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease-out),
      color var(--dur-fast) var(--ease-out);
  }
  .sk-number-step:hover {
    background: var(--bg-hover);
    color: var(--accent);
  }
  .sk-number-step:active {
    background: var(--accent-glow);
  }

  .sk-number.is-disabled {
    opacity: 0.6;
  }
</style>
