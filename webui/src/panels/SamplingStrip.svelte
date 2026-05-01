<script lang="ts">
  // SamplingStrip: T / top-p / top-k / max / seed sliders + thinking toggle +
  // session-default-vs-next-message radio + system-prompt drawer button.
  //
  // Two modes via ``samplingState.oneShotOverride``:
  // - ``true``  (next-message-only): mutations write to ``samplingState`` only.
  //             ``sendGenerate`` reads the slice and packs it as a per-call
  //             ``SamplingConfig`` override; the server's session defaults
  //             stay untouched.
  // - ``false`` (session-default):  mutations write to ``samplingState`` AND
  //             PATCH the server.  Subsequent generations send ``sampling: null``
  //             so the server uses the (now-updated) session defaults.
  //
  // Thinking is a special case — it's not a sampling field, it lives on the
  // session and on each generate as ``thinking: bool|null``.  Same dual mode:
  // session-default mode PATCHes; one-shot mode mutates local state which
  // ``sendGenerate`` forwards as the WS ``thinking`` field.
  //
  // Empty seed = null = no per-call seed pin (model RNG).  The 🎲 button
  // fills with a fresh ``Math.floor(Math.random() * 2**31)`` integer.

  import {
    samplingState,
    sessionState,
    setSampling,
    patchSessionDefaults,
    openDrawer,
  } from "../lib/stores.svelte";

  // ------------------------------------------------------------------- consts

  // Placeholder defaults shown while the session info hasn't landed yet.
  // These never reach the server — the strip is disabled in this state.
  const PLACEHOLDER = {
    temperature: 1.0,
    top_p: 1.0,
    top_k: 1024,
    max_tokens: 512,
  };

  const TEMP_MIN = 0;
  const TEMP_MAX = 2;
  const TEMP_STEP = 0.05;
  const TOP_P_MIN = 0;
  const TOP_P_MAX = 1;
  const TOP_P_STEP = 0.01;
  const TOP_K_MIN = 1;
  const TOP_K_MAX = 4096;
  const MAX_TOK_MIN = 1;
  const MAX_TOK_MAX = 8192;
  const SEED_MAX = 2 ** 31; // exclusive — ``Math.random() * 2**31`` matches the prompt.

  // ------------------------------------------------------------------- ready

  /** True once session info has loaded — gates control enable state. */
  const ready = $derived(sessionState.info !== null);

  /** True iff thinking is supported for this model.  ``supports_thinking``
   * comes off the session info and may flip once the model loads. */
  const thinkingSupported = $derived(
    sessionState.info?.supports_thinking ?? false,
  );

  // ------------------------------------------------------------------- views
  //
  // Each control's *display* value reads ``samplingState`` first (which the
  // store's bootstrap populates from session config) and falls back to a
  // placeholder when it's still null.  Writes go through ``onChange`` —
  // which decides between session-default PATCH and one-shot local-only.

  const tempView = $derived(samplingState.temperature ?? PLACEHOLDER.temperature);
  const topPView = $derived(samplingState.top_p ?? PLACEHOLDER.top_p);
  const topKView = $derived(samplingState.top_k ?? PLACEHOLDER.top_k);
  const maxView = $derived(samplingState.max_tokens || PLACEHOLDER.max_tokens);
  const seedView = $derived(
    samplingState.seed === null ? "" : String(samplingState.seed),
  );
  const thinkingView = $derived(samplingState.thinking ?? false);

  // ------------------------------------------------------------------- writes

  /** PATCH the server with a single field when in session-default mode.
   * Errors surface as a console.warn — the strip itself stays usable
   * (local state already updated; user can retry). */
  async function persistDefault(
    body: Partial<{
      temperature: number;
      top_p: number;
      top_k: number;
      max_tokens: number;
      thinking: boolean;
    }>,
  ): Promise<void> {
    try {
      await patchSessionDefaults(body);
    } catch (e) {
      console.warn("[sampling] patch failed", e);
    }
  }

  function onTemp(ev: Event): void {
    const v = Number((ev.currentTarget as HTMLInputElement).value);
    setSampling("temperature", v);
    if (!samplingState.oneShotOverride) void persistDefault({ temperature: v });
  }

  function onTopP(ev: Event): void {
    const v = Number((ev.currentTarget as HTMLInputElement).value);
    setSampling("top_p", v);
    if (!samplingState.oneShotOverride) void persistDefault({ top_p: v });
  }

  function onTopK(ev: Event): void {
    const raw = Number((ev.currentTarget as HTMLInputElement).value);
    if (!Number.isFinite(raw)) return;
    const v = Math.max(TOP_K_MIN, Math.min(TOP_K_MAX, Math.floor(raw)));
    setSampling("top_k", v);
    if (!samplingState.oneShotOverride) void persistDefault({ top_k: v });
  }

  function onMax(ev: Event): void {
    const raw = Number((ev.currentTarget as HTMLInputElement).value);
    if (!Number.isFinite(raw)) return;
    const v = Math.max(MAX_TOK_MIN, Math.min(MAX_TOK_MAX, Math.floor(raw)));
    setSampling("max_tokens", v);
    if (!samplingState.oneShotOverride) void persistDefault({ max_tokens: v });
  }

  function onSeed(ev: Event): void {
    const raw = (ev.currentTarget as HTMLInputElement).value.trim();
    if (raw === "") {
      setSampling("seed", null);
      return;
    }
    const v = Number(raw);
    if (!Number.isFinite(v)) return;
    setSampling("seed", Math.floor(v));
    // Note: there's no PATCH-able ``seed`` on the session — seed is always
    // a per-call SamplingConfig field.  Persisting nothing in default mode
    // is intentional: a "default seed" would lock every generation to the
    // same number, which isn't what the user means by "session default".
  }

  function rollSeed(): void {
    const v = Math.floor(Math.random() * SEED_MAX);
    setSampling("seed", v);
  }

  function clearSeed(): void {
    setSampling("seed", null);
  }

  function onThinking(ev: Event): void {
    const v = (ev.currentTarget as HTMLInputElement).checked;
    setSampling("thinking", v);
    if (!samplingState.oneShotOverride) void persistDefault({ thinking: v });
  }

  function setOneShot(oneShot: boolean): void {
    setSampling("oneShotOverride", oneShot);
  }

  function openSystemPrompt(): void {
    openDrawer("system_prompt");
  }
</script>

<section class="sampling-strip" aria-label="sampling controls">
  <!-- Temperature -->
  <label class="control" title="Sampling temperature (0=greedy, 2=chaos)">
    <span class="label">T</span>
    <input
      type="range"
      min={TEMP_MIN}
      max={TEMP_MAX}
      step={TEMP_STEP}
      value={tempView}
      disabled={!ready}
      oninput={onTemp}
      aria-label="temperature"
    />
    <span class="value">{tempView.toFixed(2)}</span>
  </label>

  <!-- Top-p -->
  <label class="control" title="Top-p (nucleus) cumulative probability cutoff">
    <span class="label">P</span>
    <input
      type="range"
      min={TOP_P_MIN}
      max={TOP_P_MAX}
      step={TOP_P_STEP}
      value={topPView}
      disabled={!ready}
      oninput={onTopP}
      aria-label="top-p"
    />
    <span class="value">{topPView.toFixed(2)}</span>
  </label>

  <!-- Top-k -->
  <label class="control narrow" title="Top-k hard cap on candidate vocab size">
    <span class="label">K</span>
    <input
      type="number"
      min={TOP_K_MIN}
      max={TOP_K_MAX}
      step="1"
      value={topKView}
      disabled={!ready}
      onchange={onTopK}
      aria-label="top-k"
    />
  </label>

  <!-- Max tokens -->
  <label class="control narrow" title="Maximum tokens to generate">
    <span class="label">max</span>
    <input
      type="number"
      min={MAX_TOK_MIN}
      max={MAX_TOK_MAX}
      step="1"
      value={maxView}
      disabled={!ready}
      onchange={onMax}
      aria-label="max tokens"
    />
  </label>

  <!-- Seed -->
  <div class="control seed" title="RNG seed — empty means model picks">
    <span class="label">seed</span>
    <input
      type="number"
      min="0"
      step="1"
      placeholder="—"
      value={seedView}
      disabled={!ready}
      onchange={onSeed}
      aria-label="seed"
    />
    <button
      type="button"
      class="icon-btn"
      disabled={!ready}
      onclick={rollSeed}
      title="Random seed"
      aria-label="Random seed"
    >
      🎲
    </button>
    {#if samplingState.seed !== null}
      <button
        type="button"
        class="icon-btn"
        disabled={!ready}
        onclick={clearSeed}
        title="Clear seed (back to model default)"
        aria-label="Clear seed"
      >
        ✕
      </button>
    {/if}
  </div>

  <!-- Thinking toggle -->
  <label
    class="control toggle"
    title={thinkingSupported
      ? "Force chain-of-thought thinking on/off (overrides auto)"
      : "This model doesn't support thinking mode"}
  >
    <span class="label">think</span>
    <input
      type="checkbox"
      checked={thinkingView}
      disabled={!ready || !thinkingSupported}
      onchange={onThinking}
      aria-label="thinking mode"
    />
  </label>

  <!-- System prompt button -->
  <button
    type="button"
    class="sys-btn"
    disabled={!ready}
    onclick={openSystemPrompt}
    title="Edit system prompt"
  >
    <span aria-hidden="true">⚙</span> system prompt
  </button>

  <!-- Mode toggle -->
  <fieldset class="mode" aria-label="sampling apply mode">
    <legend class="sr-only">apply mode</legend>
    <label class="mode-opt" title="Persist changes to the session defaults">
      <input
        type="radio"
        name="sampling-mode"
        checked={!samplingState.oneShotOverride}
        disabled={!ready}
        onchange={() => setOneShot(false)}
      />
      <span>session default</span>
    </label>
    <label class="mode-opt" title="Apply only to the next message; session defaults untouched">
      <input
        type="radio"
        name="sampling-mode"
        checked={samplingState.oneShotOverride}
        disabled={!ready}
        onchange={() => setOneShot(true)}
      />
      <span>next message only</span>
    </label>
  </fieldset>
</section>

<style>
  .sampling-strip {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.6em 1em;
    padding: 0.5em 0.75em;
    background: var(--bg-alt);
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: var(--font-size-small);
    color: var(--fg-strong);
  }

  .control {
    display: inline-flex;
    align-items: center;
    gap: 0.35em;
    white-space: nowrap;
  }

  .control.narrow input[type="number"] {
    width: 5em;
  }

  .control.seed input[type="number"] {
    width: 7em;
  }

  .label {
    color: var(--fg-dim);
    font-size: var(--font-size-tiny);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    min-width: 1.6em;
    text-align: right;
  }

  .value {
    color: var(--fg-strong);
    font-variant-numeric: tabular-nums;
    min-width: 2.5em;
    text-align: left;
  }

  /* Sliders — GitHub-dark, custom thumb on both webkit and gecko. */
  input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    width: 8em;
    height: 4px;
    background: var(--bg-elev);
    border-radius: 2px;
    border: 1px solid var(--border-dim);
    cursor: pointer;
    margin: 0;
  }
  input[type="range"]:disabled {
    cursor: not-allowed;
    opacity: 0.5;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--accent-blue);
    border: 1px solid var(--bg-deep);
    cursor: pointer;
  }
  input[type="range"]::-moz-range-thumb {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--accent-blue);
    border: 1px solid var(--bg-deep);
    cursor: pointer;
  }
  input[type="range"]:disabled::-webkit-slider-thumb {
    background: var(--fg-muted);
  }
  input[type="range"]:disabled::-moz-range-thumb {
    background: var(--fg-muted);
  }

  /* Number inputs */
  input[type="number"] {
    background: var(--bg);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0.15em 0.35em;
    font-family: var(--font-mono);
    font-size: var(--font-size-small);
    font-variant-numeric: tabular-nums;
  }
  input[type="number"]:focus {
    outline: none;
    border-color: var(--accent-blue);
  }
  input[type="number"]:disabled {
    color: var(--fg-muted);
    border-color: var(--border-dim);
    cursor: not-allowed;
  }
  /* Trim Firefox / Chrome spinners — they steal width and the design tokens
   * already make values legible. */
  input[type="number"]::-webkit-outer-spin-button,
  input[type="number"]::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
  input[type="number"] {
    -moz-appearance: textfield;
    appearance: textfield;
  }

  /* Checkbox sits flush with its label. */
  input[type="checkbox"] {
    accent-color: var(--accent-blue);
    cursor: pointer;
  }
  input[type="checkbox"]:disabled {
    cursor: not-allowed;
  }

  /* Tiny inline buttons (🎲, ✕) */
  .icon-btn {
    background: transparent;
    color: var(--fg-strong);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0.05em 0.4em;
    font-size: var(--font-size-small);
    line-height: 1.2;
  }
  .icon-btn:hover:not(:disabled) {
    background: var(--bg-elev);
    border-color: var(--fg-muted);
  }
  .icon-btn:disabled {
    color: var(--fg-muted);
    border-color: var(--border-dim);
    cursor: not-allowed;
  }

  .sys-btn {
    background: transparent;
    color: var(--fg-strong);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0.2em 0.6em;
    font-family: var(--font-mono);
    font-size: var(--font-size-small);
    line-height: 1.3;
  }
  .sys-btn:hover:not(:disabled) {
    background: var(--bg-elev);
    border-color: var(--fg-muted);
    color: var(--accent-blue);
  }
  .sys-btn:disabled {
    color: var(--fg-muted);
    border-color: var(--border-dim);
    cursor: not-allowed;
  }

  /* Mode radio set — pulled to the right end of the strip on desktop, wraps
   * back to row 2 on narrow viewports along with the rest of the controls. */
  .mode {
    display: inline-flex;
    align-items: center;
    gap: 0.6em;
    margin-left: auto;
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0.15em 0.5em;
    background: var(--bg);
  }
  .mode-opt {
    display: inline-flex;
    align-items: center;
    gap: 0.3em;
    color: var(--fg-dim);
    cursor: pointer;
  }
  .mode-opt input[type="radio"] {
    accent-color: var(--accent-green);
    cursor: pointer;
  }
  .mode-opt input[type="radio"]:checked + span {
    color: var(--accent-green);
  }
  .mode-opt input[type="radio"]:disabled {
    cursor: not-allowed;
  }

  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  /* Narrow viewports — strip wraps to two rows; the mode radio drops below
   * by losing its auto-margin push (already wrapping fills the row). */
  @media (max-width: 900px) {
    .sampling-strip {
      gap: 0.5em 0.8em;
    }
    .mode {
      margin-left: 0;
    }
    input[type="range"] {
      width: 6em;
    }
  }
</style>
