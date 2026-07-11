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
  import NumberInput from "../lib/NumberInput.svelte";
  import Checkbox from "../lib/Checkbox.svelte";

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

  // (The per-message role boxes moved to the composer's cast row —
  // Chat.svelte ``speaking as`` / ``reply as`` chips, same client state.)

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
  /** Bindable raw seed value for the NumberInput.  null = no per-call
   *  seed pin (the placeholder dash shows). */
  const seedView = $derived<number | null>(samplingState.seed);
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

  function onTopK(raw: number | null): void {
    if (raw === null) return;
    const v = Math.max(TOP_K_MIN, Math.min(TOP_K_MAX, Math.floor(raw)));
    setSampling("top_k", v);
    void persistDefault({ top_k: v });
  }

  function onMax(raw: number | null): void {
    if (raw === null) return;
    const v = Math.max(MAX_TOK_MIN, Math.min(MAX_TOK_MAX, Math.floor(raw)));
    setSampling("max_tokens", v);
    void persistDefault({ max_tokens: v });
  }

  function onPenalty(
    key: "presence_penalty" | "frequency_penalty",
    raw: number | null,
  ): void {
    if (raw === null) return;
    const v = Math.max(PENALTY_MIN, Math.min(PENALTY_MAX, raw));
    setSampling(key, v);
    // No session-PATCH path: the penalties ride on the per-call sampling
    // payload only, same as the (now-removed) advanced-drawer control.
  }

  function onSeed(raw: number | null): void {
    if (raw === null) {
      setSampling("seed", null);
      return;
    }
    setSampling("seed", Math.floor(raw));
    // There's no PATCH-able ``seed`` on the session — seed always rides
    // per-call (``buildSamplingPayload``).  A set seed pins every
    // generation to that number; empty the field to unpin.
  }

  function onThinking(v: boolean): void {
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
  function onAlts(raw: number | null): void {
    if (raw === null) return;
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
  <!-- Row 1: temperature + top-p sliders -->
  <div class="row sliders">
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
  </div>

  <!-- Row 2: top-k, repetition (frequency) penalty, presence penalty -->
  <div class="row">
    <label class="control" title="Top-k hard cap on candidate vocab size">
      <span class="label">K</span>
      <span class="num-cell">
        <NumberInput
          value={topKView}
          min={TOP_K_MIN}
          max={TOP_K_MAX}
          step={1}
          disabled={!ready}
          onchange={onTopK}
          ariaLabel="top-k"
        />
      </span>
    </label>

    <label
      class="control"
      title="Repetition (frequency) penalty: discourages tokens by repeat count (−2…2)"
    >
      <span class="label">rep</span>
      <span class="num-cell">
        <NumberInput
          value={frequencyView}
          min={PENALTY_MIN}
          max={PENALTY_MAX}
          step={0.05}
          disabled={!ready}
          onchange={(v) => onPenalty("frequency_penalty", v)}
          ariaLabel="repetition (frequency) penalty"
        />
      </span>
    </label>

    <label
      class="control"
      title="Presence penalty: discourages tokens already present (−2…2)"
    >
      <span class="label">pres</span>
      <span class="num-cell">
        <NumberInput
          value={presenceView}
          min={PENALTY_MIN}
          max={PENALTY_MAX}
          step={0.05}
          disabled={!ready}
          onchange={(v) => onPenalty("presence_penalty", v)}
          ariaLabel="presence penalty"
        />
      </span>
    </label>
  </div>

  <!-- (Per-message role labels moved to the composer's cast row.) -->

  <!-- Row 4: max tokens, alts (top-K capture), seed -->
  <div class="row">
    <label class="control" title="Maximum tokens to generate">
      <span class="label">max</span>
      <span class="num-cell">
        <NumberInput
          value={maxView}
          min={MAX_TOK_MIN}
          max={MAX_TOK_MAX}
          step={1}
          disabled={!ready}
          onchange={onMax}
          ariaLabel="max tokens"
        />
      </span>
    </label>

    <!-- Top-K alternatives count (logit-pass).  0 disables capture; >0
         populates the drilldown logits tab + the inline surprise highlight
         mode.  Default 8 per Decision 1. -->
    <label
      class="control"
      title="Top-K alternative tokens to capture per position (0 disables; feeds the drilldown logits tab + surprise highlight)"
    >
      <span class="label">alts</span>
      <span class="num-cell">
        <NumberInput
          value={samplingState.return_top_k}
          min={0}
          max={ALTS_MAX}
          step={1}
          disabled={!ready}
          onchange={onAlts}
          ariaLabel="top-K alternatives to capture"
        />
      </span>
    </label>

    <label class="control" title="RNG seed: empty means the model picks (clear the field to unpin)">
      <span class="label">seed</span>
      <span class="num-cell">
        <NumberInput
          value={seedView}
          min={0}
          step={1}
          placeholder="—"
          allowEmpty
          disabled={!ready}
          onchange={onSeed}
          ariaLabel="seed"
        />
      </span>
    </label>
  </div>

  <!-- Row 5: advanced / system-prompt drawers + thinking toggle -->
  <div class="row actions">
    <button
      type="button"
      class="sys-btn"
      disabled={!ready}
      onclick={openAdvanced}
      title="Open stop strings, logit bias, and numeric top-K alternatives"
    >
      advanced
    </button>

    <button
      type="button"
      class="sys-btn"
      disabled={!ready}
      onclick={openSystemPrompt}
      title="Edit system prompt"
    >
      <span aria-hidden="true">⚙</span> system prompt
    </button>

    <label
      class="control toggle"
      class:forced={thinkingForced}
      title={!thinkingSupported
        ? "This model doesn't support thinking"
        : !thinkingOptional
          ? "This model always thinks"
          : "Force chain-of-thought thinking on/off (overrides auto)"}
    >
      <span class="label think-label">think{thinkingForced ? " (forced)" : ""}</span>
      <Checkbox
        checked={thinkingForced ? true : thinkingView}
        disabled={!ready || !thinkingSupported || thinkingForced}
        onchange={onThinking}
        ariaLabel="thinking mode"
      />
    </label>
  </div>
</section>

<style>
  .sampling-strip {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    padding: var(--space-3) var(--space-5);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    color: var(--fg-strong);
  }

  /* One logical group of controls per row, laid out as equal grid
   * columns.  Because both 3-up rows (K/rep/pres and max/alts/seed) get
   * the same three 1fr tracks — and both 2-up rows (sliders, roles) the
   * same two — every control lines up vertically across rows and each
   * row's right edge is flush.  minmax(0, 1fr) keeps the tracks equal
   * even when a cell's content (the seed's 🎲 / ✕) would otherwise widen
   * it; the field inside shrinks instead. */
  .row {
    display: grid;
    grid-auto-flow: column;
    grid-auto-columns: minmax(0, 1fr);
    align-items: center;
    gap: var(--space-5);
    width: 100%;
  }
  .row > .control {
    min-width: 0;
  }
  /* The action row sizes its buttons / toggle to content and spreads
   * them, so the toggle's right edge still lands flush. */
  .row.actions {
    display: flex;
    justify-content: space-between;
  }
  .row.actions > * {
    flex: 0 0 auto;
  }
  .control {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    white-space: nowrap;
    min-width: 0;
  }

  /* Boxed inputs fill their (grown) control after the fixed-width label.
   * The inner <input> is width:100%, so the cell owns the sizing. */
  .slider-cell,
  .num-cell {
    display: inline-flex;
    flex: 1 1 0;
    min-width: 0;
  }

  .label {
    flex: 0 0 auto;
    color: var(--fg-dim);
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0;
    /* Fixed floor so the short labels reserve a consistent gutter and the
     * boxes start at the same offset within each control. */
    min-width: 3em;
    text-align: right;
  }

  .value {
    flex: 0 0 auto;
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
  /* The think label hugs its checkbox rather than reserving the numeric
   * gutter — it sits in the action row, not a field column. */
  .think-label {
    min-width: 0;
    text-align: left;
  }

  .sys-btn {
    background: var(--glass);
    color: var(--fg-strong);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-1) var(--space-4);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    line-height: 1.3;
  }
  .sys-btn:hover:not(:disabled) {
    background: var(--glass-strong);
    color: var(--accent);
  }
  .sys-btn:disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
  }
  /* (Role-input styles moved to Chat.svelte's cast row.) */
</style>
