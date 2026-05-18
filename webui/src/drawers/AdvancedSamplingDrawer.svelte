<script lang="ts">
  import {
    closeDrawer,
    openDrawer,
    samplingState,
    setSampling,
  } from "../lib/stores.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  function setNumber<K extends "presence_penalty" | "frequency_penalty" | "return_top_k">(
    key: K,
    ev: Event,
  ): void {
    const raw = Number((ev.currentTarget as HTMLInputElement).value);
    if (!Number.isFinite(raw)) return;
    if (key === "return_top_k") {
      setSampling(key, Math.max(0, Math.min(256, Math.floor(raw))));
    } else {
      setSampling(key, raw);
    }
  }

  const logitBiasValid = $derived.by(() => {
    const raw = samplingState.logit_bias_text.trim();
    if (!raw) return true;
    try {
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === "object" && !Array.isArray(parsed);
    } catch {
      return raw
        .split(/\r?\n/)
        .filter(Boolean)
        .every((line) =>
          /^\s*-?\d+\s*[:=,\s]\s*-?\d+(?:\.\d+)?\s*$/.test(line),
        );
    }
  });
</script>

<section class="drawer-shell" aria-label="Advanced sampling drawer">
  <header class="header">
    <div>
      <span class="title">advanced sampling</span>
      <p>per-run controls for stop strings, penalties, top alternatives, and logit bias</p>
    </div>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}>×</button>
  </header>

  <div class="body">
    <section class="panel">
      <h3>distribution capture</h3>
      <label class="field inline">
        <span>top alternatives K</span>
        <input
          type="number"
          min="0"
          max="256"
          value={samplingState.return_top_k}
          oninput={(ev) => setNumber("return_top_k", ev)}
        />
      </label>
      <p class="hint">0 disables alternate-token capture; 8 is the lightweight logits-drilldown default.</p>
    </section>

    <section class="panel grid2">
      <label class="field">
        <span>presence penalty</span>
        <input
          type="number"
          step="0.05"
          value={samplingState.presence_penalty}
          oninput={(ev) => setNumber("presence_penalty", ev)}
        />
      </label>
      <label class="field">
        <span>frequency penalty</span>
        <input
          type="number"
          step="0.05"
          value={samplingState.frequency_penalty}
          oninput={(ev) => setNumber("frequency_penalty", ev)}
        />
      </label>
    </section>

    <section class="panel">
      <h3>stop sequences</h3>
      <textarea
        rows="5"
        value={samplingState.stop_sequences}
        oninput={(ev) =>
          setSampling("stop_sequences", (ev.currentTarget as HTMLTextAreaElement).value)}
        placeholder={"one stop sequence per line\n###\n<|eot_id|>"}
      ></textarea>
    </section>

    <section class="panel">
      <h3>logit bias</h3>
      <textarea
        rows="7"
        class:invalid={!logitBiasValid}
        value={samplingState.logit_bias_text}
        oninput={(ev) =>
          setSampling("logit_bias_text", (ev.currentTarget as HTMLTextAreaElement).value)}
        placeholder={'{"198": -4, "220": 1.5}\n\nor:\n198: -4\n220: 1.5'}
      ></textarea>
      <p class:error={!logitBiasValid} class="hint">
        {logitBiasValid
          ? "JSON object or one token_id: bias pair per line."
          : "Could not parse logit bias. Use JSON or token_id: number lines."}
      </p>
    </section>

    <section class="panel">
      <h3>scope</h3>
      <div class="segmented">
        <button
          type="button"
          class:active={!samplingState.oneShotOverride}
          onclick={() => setSampling("oneShotOverride", false)}
        >session defaults</button>
        <button
          type="button"
          class:active={samplingState.oneShotOverride}
          onclick={() => setSampling("oneShotOverride", true)}
        >next message</button>
      </div>
      <button type="button" class="secondary" onclick={() => openDrawer("system_prompt")}>
        open system prompt
      </button>
    </section>
  </div>
</section>

<style>
  .drawer-shell {
    min-height: 0;
    display: flex;
    flex-direction: column;
    background: var(--bg-alt);
  }
  .header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    padding: 1rem 1.1rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  .title {
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0;
    font-size: 0.75rem;
    font-weight: 700;
  }
  p {
    margin: 0.3rem 0 0;
    color: var(--fg-muted);
  }
  .close {
    background: transparent;
    border: 0;
    color: var(--fg-muted);
    font-size: 1.25rem;
  }
  .body {
    display: grid;
    gap: 0.8rem;
    padding: 1rem;
    overflow: auto;
  }
  .panel {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    padding: 0.9rem;
  }
  .grid2 {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.75rem;
  }
  h3 {
    margin: 0 0 0.6rem;
    color: var(--fg);
    font-size: 0.9rem;
    letter-spacing: 0;
  }
  .field {
    display: grid;
    gap: 0.35rem;
    color: var(--fg-dim);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0;
  }
  .field.inline {
    grid-template-columns: 1fr 7rem;
    align-items: center;
  }
  input,
  textarea {
    width: 100%;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg-deep);
    color: var(--fg);
    padding: 0.55rem;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    letter-spacing: 0;
  }
  textarea {
    resize: vertical;
    line-height: 1.45;
  }
  input:focus,
  textarea:focus {
    outline: none;
    border-color: var(--accent);
  }
  .invalid {
    border-color: var(--accent-red);
  }
  .hint {
    font-size: 0.74rem;
    line-height: 1.35;
  }
  .error {
    color: var(--accent-red);
  }
  .segmented {
    display: grid;
    grid-template-columns: 1fr 1fr;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    margin-bottom: 0.75rem;
  }
  .segmented button,
  .secondary {
    border: 0;
    background: transparent;
    color: var(--fg-strong);
    padding: 0.55rem 0.7rem;
  }
  .segmented button + button {
    border-left: 1px solid var(--border);
  }
  .segmented button.active {
    background: rgba(72, 138, 203, 0.12);
    color: var(--accent);
  }
  .secondary {
    width: 100%;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: rgba(255, 255, 255, 0.03);
  }
</style>
