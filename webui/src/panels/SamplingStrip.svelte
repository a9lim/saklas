<script lang="ts">
  // SamplingStrip: T / top-p / top-k / max / pres / freq / seed + a
  // thinking toggle + an alts (top-K capture) count + advanced /
  // system-prompt drawer buttons.
  //
  // Every edit applies immediately.  temperature / top-p / top-k /
  // max-tokens / thinking PATCH the session defaults as the user moves
  // them; seed and the advanced extras (penalties, stop strings, logit
  // bias, return_top_k) have no PATCH path, so ``sendGenerate`` packs
  // them onto each call's ``SamplingConfig``.  Either way the value the
  // strip shows is the value the next generation uses.
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
  import Slider from "../lib/Slider.svelte";

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
  const PENALTY_MIN = -2;
  const PENALTY_MAX = 2;
  const ALTS_MAX = 256;
  const SEED_MAX = 2 ** 31; // exclusive — ``Math.random() * 2**31`` matches the prompt.

  // ------------------------------------------------------------------- ready

  /** True once session info has loaded — gates control enable state. */
  const ready = $derived(sessionState.info !== null);

  /** True iff thinking is supported for this model.  ``supports_thinking``
   * comes off the session info and may flip once the model loads. */
  const thinkingSupported = $derived(
    sessionState.info?.supports_thinking ?? false,
  );

  /** True iff the user can actually turn thinking off (the chat template
   *  has an ``enable_thinking`` switch).  Forced-thinking models leave
   *  the toggle locked and read-only so the user knows clicking it
   *  is a no-op.  Older servers omit the field; default to ``true``
   *  so we don't lock controls against backends that pre-date the
   *  field. */
  const thinkingOptional = $derived(
    sessionState.info?.thinking_is_optional ?? true,
  );

  /** Tri-state for the title attribute and disabled gate. */
  const thinkingForced = $derived(
    thinkingSupported && !thinkingOptional,
  );

  // ------------------------------------------------------------------- views
  //
  // Each control's *display* value reads ``samplingState`` first (which the
  // store's bootstrap populates from session config) and falls back to a
  // placeholder when it's still null.

  const tempView = $derived(samplingState.temperature ?? PLACEHOLDER.temperature);
  const topPView = $derived(samplingState.top_p ?? PLACEHOLDER.top_p);
  const topKView = $derived(samplingState.top_k ?? PLACEHOLDER.top_k);
  const maxView = $derived(samplingState.max_tokens || PLACEHOLDER.max_tokens);
  const presenceView = $derived(samplingState.presence_penalty);
  const frequencyView = $derived(samplingState.frequency_penalty);
  const seedView = $derived(
    samplingState.seed === null ? "" : String(samplingState.seed),
  );
  const thinkingView = $derived(samplingState.thinking ?? false);

  // ------------------------------------------------------------------- writes

  /** PATCH the server with a single field.  Errors surface as a
   * console.warn — the strip itself stays usable (local state already
   * updated; user can retry). */
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

  function onTemp(v: number): void {
    setSampling("temperature", v);
    void persistDefault({ temperature: v });
  }

  function onTopP(v: number): void {
    setSampling("top_p", v);
    void persistDefault({ top_p: v });
  }

  function onTopK(ev: Event): void {
    const raw = Number((ev.currentTarget as HTMLInputElement).value);
    if (!Number.isFinite(raw)) return;
    const v = Math.max(TOP_K_MIN, Math.min(TOP_K_MAX, Math.floor(raw)));
    setSampling("top_k", v);
    void persistDefault({ top_k: v });
  }

  function onMax(ev: Event): void {
    const raw = Number((ev.currentTarget as HTMLInputElement).value);
    if (!Number.isFinite(raw)) return;
    const v = Math.max(MAX_TOK_MIN, Math.min(MAX_TOK_MAX, Math.floor(raw)));
    setSampling("max_tokens", v);
    void persistDefault({ max_tokens: v });
  }

  function onPenalty(
    key: "presence_penalty" | "frequency_penalty",
    ev: Event,
  ): void {
    const raw = Number((ev.currentTarget as HTMLInputElement).value);
    if (!Number.isFinite(raw)) return;
    const v = Math.max(PENALTY_MIN, Math.min(PENALTY_MAX, raw));
    setSampling(key, v);
    // No session-PATCH path: the penalties ride on the per-call sampling
    // payload only, same as the (now-removed) advanced-drawer control.
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
    // There's no PATCH-able ``seed`` on the session — seed always rides
    // per-call (``buildSamplingPayload``).  A set seed pins every
    // generation to that number; clear it (✕) to unpin.
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
    void persistDefault({ thinking: v });
  }

  /** Logit-pass: number of top-K alternative tokens to capture per
   *  position.  ``0`` disables capture; ``> 0`` lights up the token
   *  drilldown's logits tab + the inline ``surprise`` highlight.  No
   *  PATCH — the server's session-PATCH endpoint doesn't accept
   *  ``return_top_k``; it rides the WS sampling payload directly (see
   *  ``stores.svelte.ts::sendGenerate``).  Effective on the next
   *  generation; running gens keep their captured shape. */
  function onAlts(ev: Event): void {
    const raw = Number((ev.currentTarget as HTMLInputElement).value);
    if (!Number.isFinite(raw)) return;
    const v = Math.max(0, Math.min(ALTS_MAX, Math.floor(raw)));
    setSampling("return_top_k", v);
  }

  function openSystemPrompt(): void {
    openDrawer("system_prompt");
  }

  function openAdvanced(): void {
    openDrawer("advanced_sampling");
  }
</script>

<section class="sampling-strip" aria-label="sampling controls">
  <!-- Temperature -->
  <label class="control" title="Sampling temperature (0=greedy, 2=chaos)">
    <span class="label">T</span>
    <span class="slider-cell">
      <Slider
        value={tempView}
        min={TEMP_MIN}
        max={TEMP_MAX}
        step={TEMP_STEP}
        disabled={!ready}
        oninput={onTemp}
        ariaLabel="temperature"
      />
    </span>
    <span class="value">{tempView.toFixed(2)}</span>
  </label>

  <!-- Top-p -->
  <label class="control" title="Top-p (nucleus) cumulative probability cutoff">
    <span class="label">P</span>
    <span class="slider-cell">
      <Slider
        value={topPView}
        min={TOP_P_MIN}
        max={TOP_P_MAX}
        step={TOP_P_STEP}
        disabled={!ready}
        oninput={onTopP}
        ariaLabel="top-p"
      />
    </span>
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

  <!-- Presence penalty -->
  <label
    class="control narrow"
    title="Presence penalty: discourages tokens already present (−2…2)"
  >
    <span class="label">pres</span>
    <input
      type="number"
      min={PENALTY_MIN}
      max={PENALTY_MAX}
      step="0.05"
      value={presenceView}
      disabled={!ready}
      onchange={(ev) => onPenalty("presence_penalty", ev)}
      aria-label="presence penalty"
    />
  </label>

  <!-- Frequency penalty -->
  <label
    class="control narrow"
    title="Frequency penalty: discourages tokens by repeat count (−2…2)"
  >
    <span class="label">freq</span>
    <input
      type="number"
      min={PENALTY_MIN}
      max={PENALTY_MAX}
      step="0.05"
      value={frequencyView}
      disabled={!ready}
      onchange={(ev) => onPenalty("frequency_penalty", ev)}
      aria-label="frequency penalty"
    />
  </label>

  <!-- Seed -->
  <div class="control seed" title="RNG seed: empty means the model picks">
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
    class:forced={thinkingForced}
    title={!thinkingSupported
      ? "This model doesn't support thinking"
      : !thinkingOptional
        ? "This model always thinks"
        : "Force chain-of-thought thinking on/off (overrides auto)"}
  >
    <span class="label">think{thinkingForced ? " (forced)" : ""}</span>
    <input
      type="checkbox"
      checked={thinkingForced ? true : thinkingView}
      disabled={!ready || !thinkingSupported || thinkingForced}
      onchange={onThinking}
      aria-label="thinking mode"
    />
  </label>

  <!-- Top-K alternatives count (logit-pass).  0 disables capture; >0
       populates the drilldown logits tab + the inline surprise highlight
       mode.  Default 8 per Decision 1. -->
  <label
    class="control narrow"
    title="Top-K alternative tokens to capture per position (0 disables; feeds the drilldown logits tab + surprise highlight)"
  >
    <span class="label">alts</span>
    <input
      type="number"
      min="0"
      max={ALTS_MAX}
      step="1"
      value={samplingState.return_top_k}
      disabled={!ready}
      onchange={onAlts}
      aria-label="top-K alternatives to capture"
    />
  </label>

  <button
    type="button"
    class="sys-btn"
    disabled={!ready}
    onclick={openAdvanced}
    title="Open stop strings, logit bias, and numeric top-K alternatives"
  >
    advanced
  </button>

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
</section>

<style>
  .sampling-strip {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-4) var(--space-5);
    padding: var(--space-3) var(--space-5);
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    color: var(--fg-strong);
  }

  .control {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
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
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0;
    min-width: 1.6em;
    text-align: right;
  }

  .value {
    color: var(--fg-strong);
    font-variant-numeric: tabular-nums;
    min-width: 2.5em;
    text-align: left;
  }

  /* Forced-thinking toggle: locked-on visual.  The checkbox is disabled
     so the browser already dims it; we keep the label dim too so the
     "(forced)" suffix reads as informational rather than interactive. */
  .control.toggle.forced .label {
    color: var(--fg-dim);
    font-style: italic;
  }

  /* Fixed-width host for the shared <Slider> inside the inline strip. */
  .slider-cell {
    display: flex;
    width: 8em;
  }

  /* Number inputs */
  input[type="number"] {
    background: var(--bg);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-1) var(--space-2);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    font-variant-numeric: tabular-nums;
  }
  input[type="number"]:focus {
    outline: none;
    border-color: var(--accent);
  }
  input[type="number"]:disabled {
    color: var(--fg-muted);
    border-color: var(--border);
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
    accent-color: var(--accent);
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
    border-radius: var(--radius);
    padding: var(--space-1) var(--space-3);
    font-size: var(--text-sm);
    line-height: 1.2;
  }
  .icon-btn:hover:not(:disabled) {
    background: var(--bg-elev);
    border-color: var(--fg-muted);
  }
  .icon-btn:disabled {
    color: var(--fg-muted);
    border-color: var(--border);
    cursor: not-allowed;
  }

  .sys-btn {
    background: transparent;
    color: var(--fg-strong);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-1) var(--space-4);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    line-height: 1.3;
  }
  .sys-btn:hover:not(:disabled) {
    background: var(--bg-elev);
    border-color: var(--fg-muted);
    color: var(--accent);
  }
  .sys-btn:disabled {
    color: var(--fg-muted);
    border-color: var(--border);
    cursor: not-allowed;
  }

  /* Narrow viewports — strip wraps to two rows. */
  @media (max-width: 900px) {
    .sampling-strip {
      gap: var(--space-3) var(--space-5);
    }
    .slider-cell {
      width: 6em;
    }
  }
</style>
