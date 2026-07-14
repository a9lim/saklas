<script lang="ts">
  // Recipe builder — visual rack editor plus an advanced full-grammar mode.
  // The visual rack stays the approachable surface for positions; the text
  // mode makes projection, ablation, probe gates, and mixed expressions
  // genuinely authorable instead of merely documenting syntax the UI drops.
  import {
    addSubspaceToRack,
    closeDrawer,
    currentSteeringExpression,
    openDrawer,
    removeSubspaceFromRack,
    removeManifoldFromRack,
    setSubspaceTrigger,
    setSubspaceEnabled,
    setManifoldTrigger,
    setManifoldEnabled,
    setSubspaceAlong,
    steerRack,
    vectorsState,
    applyCustomSteeringExpression,
    useVisualSteeringRack,
    setJLensTrigger,
    setJLensEnabled,
    removeJLensFromRack,
    setSaeTrigger,
    setSaeEnabled,
    removeSaeFromRack,
  } from "../lib/stores.svelte";
  import { apiSessions } from "../lib/api";
  import type { SteerEntry, Trigger } from "../lib/types";
  import Select from "../lib/Select.svelte";
  import Slider from "../lib/Slider.svelte";
  import Checkbox from "../lib/Checkbox.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  let newTerm = $state("");
  let copied = $state(false);
  let rawDraft = $state(currentSteeringExpression());
  let validating = $state(false);
  let expressionError: string | null = $state(null);

  const entries = $derived(
    [...steerRack.entries.entries()].sort((a, b) => a[0].localeCompare(b[0])),
  );
  const hasSubspace = $derived(entries.some(([, e]) => e.mode === "subspace"));
  const customActive = $derived(steerRack.customExpression !== null);
  const allNames = $derived.by(() => {
    const names = new Set<string>([...vectorsState.names, ...steerRack.profiles.keys()]);
    return [...names].sort();
  });

  const triggers: { value: Trigger; label: string }[] = [
    { value: "BOTH", label: "both" },
    { value: "BEFORE", label: "prompt" },
    { value: "AFTER", label: "after thinking" },
    { value: "THINKING", label: "thinking" },
    { value: "RESPONSE", label: "response" },
  ];

  function add(): void {
    const name = newTerm.trim();
    if (!name) return;
    addSubspaceToRack(name);
    newTerm = "";
  }

  function setTrig(name: string, entry: SteerEntry, t: Trigger): void {
    if (entry.mode === "subspace") setSubspaceTrigger(name, t);
    else if (entry.mode === "manifold") setManifoldTrigger(name, t);
    else if (entry.mode === "jlens") setJLensTrigger(name, t);
    else setSaeTrigger(name, t);
  }
  function setEn(name: string, entry: SteerEntry, v: boolean): void {
    if (entry.mode === "subspace") setSubspaceEnabled(name, v);
    else if (entry.mode === "manifold") setManifoldEnabled(name, v);
    else if (entry.mode === "jlens") setJLensEnabled(name, v);
    else setSaeEnabled(name, v);
  }
  function remove(name: string, entry: SteerEntry): void {
    if (entry.mode === "subspace") removeSubspaceFromRack(name);
    else if (entry.mode === "manifold") removeManifoldFromRack(name);
    else if (entry.mode === "jlens") removeJLensFromRack(name);
    else removeSaeFromRack(name);
  }
  function onAlongInput(v: number): void {
    if (Number.isFinite(v)) setSubspaceAlong(v);
  }

  async function copyExpression(): Promise<void> {
    try {
      await navigator.clipboard.writeText(rawDraft);
      copied = true;
      setTimeout(() => (copied = false), 1200);
    } catch {
      copied = false;
    }
  }

  async function useExpression(): Promise<void> {
    validating = true;
    expressionError = null;
    try {
      const result = await apiSessions.validateSteering(rawDraft);
      if (!result.valid) {
        expressionError = result.error ?? "This expression cannot be applied.";
        return;
      }
      rawDraft = result.expression;
      applyCustomSteeringExpression(result.expression);
    } catch (e) {
      expressionError = e instanceof Error ? e.message : String(e);
    } finally {
      validating = false;
    }
  }

  function useVisual(): void {
    useVisualSteeringRack();
    rawDraft = currentSteeringExpression();
    expressionError = null;
  }
</script>

<section class="drawer-shell" aria-label="Recipe builder drawer">
  <header class="header">
    <div>
      <span class="title">recipe builder</span>
    </div>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}>✕</button>
  </header>

  <div class="body">
    <section class="expression-card">
      <div class="expression-editor">
        <span class="label">expression</span>
        <textarea
          bind:value={rawDraft}
          rows="4"
          aria-label="Full steering expression"
          placeholder="0.5 personas%pirate + 0.2 !verbose.concise@response"
        ></textarea>
        <p class="muted">
          Full grammar: positions, push, <code>!</code> ablation,
          <code>~</code>/<code>|</code> projection, triggers, and probe gates.
          Applying switches from the visual rack to this exact expression.
        </p>
        {#if expressionError}
          <p class="error" role="alert">{expressionError}</p>
        {/if}
      </div>
      <div class="actions">
        <button type="button" onclick={copyExpression}>{copied ? "copied" : "copy"}</button>
        <button
          type="button"
          class="primary"
          disabled={validating}
          onclick={() => void useExpression()}
        >{validating ? "checking…" : "use expression"}</button>
        {#if customActive}
          <button type="button" onclick={useVisual}>use visual rack</button>
        {/if}
      </div>
    </section>

    {#if !customActive}
      <section class="add-card">
        <input list="recipe-concepts" bind:value={newTerm} placeholder="concept" onkeydown={(ev) => { if (ev.key === "Enter") add(); }} />
        <datalist id="recipe-concepts">
          {#each allNames as name (name)}
            <option value={name}></option>
          {/each}
        </datalist>
        <button type="button" onclick={add}>add</button>
        <button type="button" onclick={() => openDrawer("subspace")}>browse…</button>
      </section>

    {#if hasSubspace}
      <section class="along-card">
        <span class="label">shared along</span>
        <div class="along-row">
          <Slider
            value={steerRack.subspaceAlong}
            min={0}
            max={2}
            step={0.05}
            oninput={onAlongInput}
            ariaLabel="shared subspace along"
          />
          <strong>{steerRack.subspaceAlong.toFixed(2)}</strong>
        </div>
      </section>
    {/if}

    <section class="terms">
      {#if entries.length === 0}
        <div class="empty">no terms</div>
      {:else}
        {#each entries as [name, entry] (name)}
          <article class="term" class:disabled={!entry.enabled}>
            <header>
              <span class="enable">
                <Checkbox
                  checked={entry.enabled}
                  onchange={(v) => setEn(name, entry, v)}
                  ariaLabel="enabled"
                />
                <span>{name}</span>
                <span class="mode-badge mode-{entry.mode}">{entry.mode}</span>
              </span>
              <button type="button" class="remove" aria-label={`Remove ${name}`} onclick={() => remove(name, entry)}>×</button>
            </header>

            <label class="field">
              <span>trigger</span>
              <Select
                value={entry.trigger}
                options={triggers}
                onchange={(v) => setTrig(name, entry, v)}
                ariaLabel="trigger"
              />
            </label>
          </article>
        {/each}
      {/if}
    </section>
    {:else}
      <section class="custom-note">
        Advanced expression mode is active. Use “visual rack” above to return
        to card-based authoring; adding a card will also leave this mode.
      </section>
    {/if}
  </div>
</section>

<style>
  .drawer-shell { display: flex; flex-direction: column; min-height: 0; background: transparent; }
  .header { display: flex; justify-content: space-between; gap: var(--space-6); padding: var(--space-5) var(--space-6); background: var(--surface); }
  .title { color: var(--accent); letter-spacing: 0; font-size: var(--text-md); font-weight: var(--weight-medium); }
  .remove { background: transparent; border: 0; color: var(--fg-muted); font-size: var(--text-md); }
  .close {
    background: var(--glass);
    color: var(--fg-muted);
    border: 1px solid transparent;
    border-radius: 50%;
    width: 26px;
    height: 26px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font: inherit;
    font-size: var(--text-md);
    line-height: 1;
    cursor: pointer;
    flex: none;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out);
  }
  .close:hover {
    color: var(--fg);
    background: var(--glass-strong);
  }
  .body { display: grid; gap: var(--space-5); padding: var(--space-6); overflow: auto; }
  .expression-card, .add-card, .along-card, .term {
    border-radius: var(--radius);
    background: var(--glass);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
    padding: var(--space-6);
  }
  .expression-card { display: grid; gap: var(--space-4); }
  .expression-editor { min-width: 0; }
  .label, .field span { display: block; color: var(--fg-muted); font-size: var(--text-xs); text-transform: uppercase; letter-spacing: 0; margin-bottom: var(--space-2); }
  code { color: var(--accent-amber); font-family: var(--font-mono); white-space: pre-wrap; word-break: break-word; }
  .actions, .add-card { display: flex; gap: var(--space-4); align-items: center; flex-wrap: wrap; }
  button { border: 1px solid transparent; border-radius: var(--radius); background: var(--glass-strong); color: var(--fg); padding: var(--space-4) var(--space-5); }
  button:hover:not(:disabled) { background: color-mix(in srgb, var(--accent) 10%, var(--glass-strong)); color: var(--accent); }
  button.primary { background: color-mix(in srgb, var(--accent) 18%, var(--glass-strong)); color: var(--accent); }
  button:disabled { opacity: 0.5; }
  .add-card input { flex: 1; }
  input { border: 1px solid transparent; border-radius: var(--radius); background: var(--input-well); color: var(--fg); padding: var(--space-4); font-family: var(--font-mono); font-size: var(--text-xs); }
  input:focus { outline: none; border-color: var(--accent); }
  textarea { width: 100%; resize: vertical; min-height: 6.5rem; box-sizing: border-box; border: 1px solid transparent; border-radius: var(--radius); background: var(--input-well); color: var(--fg); padding: var(--space-4); font-family: var(--font-mono); font-size: var(--text-xs); line-height: 1.5; }
  textarea:focus { outline: 2px solid var(--focus-ring); outline-offset: 1px; border-color: var(--accent); }
  .along-row { display: grid; grid-template-columns: 1fr 4rem; gap: var(--space-5); align-items: center; }
  .along-row strong { color: var(--accent); font-family: var(--font-mono); text-align: right; }
  .muted { color: var(--fg-muted); font-size: var(--text-xs); margin: var(--space-3) 0 0; }
  .error { color: var(--accent-red); font-size: var(--text-xs); margin: var(--space-3) 0 0; white-space: pre-wrap; }
  .custom-note { color: var(--fg-muted); background: var(--glass); border-radius: var(--radius); padding: var(--space-6); font-size: var(--text-xs); line-height: 1.5; }
  .terms { display: grid; gap: var(--space-5); }
  .term { display: grid; gap: var(--space-5); }
  .term.disabled { opacity: 0.58; }
  .term header { display: flex; align-items: center; justify-content: space-between; gap: var(--space-5); }
  .enable { display: flex; align-items: center; gap: var(--space-4); color: var(--fg); }
  .enable span { font-weight: var(--weight-bold); }
  .mode-badge { font-weight: var(--weight-normal); font-size: var(--text-2xs); text-transform: uppercase; border: 1px solid transparent; border-radius: var(--radius); padding: 0 var(--space-2); background: var(--glass); color: var(--fg-muted); }
  .mode-badge.mode-subspace { background: var(--accent-subtle); color: var(--accent); }
  .mode-badge.mode-manifold { background: color-mix(in srgb, var(--accent-purple) 16%, var(--glass)); color: var(--accent-purple); }
  .field { display: grid; gap: var(--space-1); }
  .empty { display: grid; place-items: center; min-height: 10rem; color: var(--fg-muted); background: var(--bg); border-radius: var(--radius); }
</style>
