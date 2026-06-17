<script lang="ts">
  import {
    closeDrawer,
    samplingState,
    setSampling,
    genUiMode,
    setGenUiMode,
    sessionState,
  } from "../lib/stores.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  // Render mode — two-state: chat renders bubbles + roles, raw renders a
  // single flat completion buffer.  The mode is seeded from the model's
  // ``is_base_model`` flag the first time the model is seen, then it's a
  // plain user toggle.  Toggling never mutates the loom tree.
  const RENDER_MODES: { value: "chat" | "raw"; label: string }[] = [
    { value: "chat", label: "chat" },
    { value: "raw", label: "raw" },
  ];
  const isBaseModel = $derived(sessionState.info?.is_base_model === true);

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
      <p>per-run controls for stop strings and logit bias</p>
    </div>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}>×</button>
  </header>

  <div class="body">
    <section class="panel">
      <h3>render mode</h3>
      <div class="mode-row" role="group" aria-label="Render mode">
        {#each RENDER_MODES as m (m.label)}
          <button
            type="button"
            class="mode-opt"
            class:active={genUiMode.mode === m.value}
            onclick={() => setGenUiMode(m.value)}
          >{m.label}</button>
        {/each}
      </div>
      <p class="hint">
        the {genUiMode.mode} surface is active. this is a
        {isBaseModel ? "base" : "chat"} model, so {isBaseModel ? "raw" : "chat"}
        is its default. switching mode never changes the loom tree.
      </p>
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
    gap: var(--space-6);
    padding: var(--space-6) var(--space-6);
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  .title {
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0;
    font-size: var(--text-xs);
    font-weight: var(--weight-bold);
  }
  p {
    margin: var(--space-2) 0 0;
    color: var(--fg-muted);
  }
  .close {
    background: transparent;
    border: 0;
    color: var(--fg-muted);
    font-size: var(--text-md);
  }
  .body {
    display: grid;
    gap: var(--space-5);
    padding: var(--space-6);
    overflow: auto;
  }
  .panel {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    padding: var(--space-6);
  }
  h3 {
    margin: 0 0 var(--space-4);
    color: var(--fg);
    font-size: var(--text);
    letter-spacing: 0;
  }
  textarea {
    width: 100%;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg-deep);
    color: var(--fg);
    padding: var(--space-4);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    letter-spacing: 0;
    resize: vertical;
    line-height: 1.45;
  }
  textarea:focus {
    outline: none;
    border-color: var(--accent);
  }
  .invalid {
    border-color: var(--accent-red);
  }
  .hint {
    font-size: var(--text-xs);
    line-height: 1.35;
    color: var(--fg-muted);
  }
  .error {
    color: var(--accent-red);
  }
  .mode-row {
    display: flex;
    gap: var(--space-1);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-1);
    margin-bottom: var(--space-3);
  }
  .mode-opt {
    flex: 1 1 0;
    background: transparent;
    color: var(--fg-muted);
    border: 0;
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    text-transform: lowercase;
    cursor: pointer;
    transition:
      background var(--dur) var(--ease-out),
      color var(--dur) var(--ease-out);
  }
  .mode-opt:hover {
    color: var(--fg);
  }
  .mode-opt.active {
    background: var(--accent-subtle);
    color: var(--accent);
  }
</style>
