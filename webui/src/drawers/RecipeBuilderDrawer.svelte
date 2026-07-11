<script lang="ts">
  // Recipe builder — a list editor over the unified steer rack.  Since 4.1
  // every term is a position on a fitted geometry; per-card coefficient,
  // projection, ablation, and variant editing moved out (subspace terms share
  // one "subspace along" master, positions are authored on the cards).  What
  // survives here: the canonical-expression readout, an add-by-name field, the
  // shared subspace-along master, and a per-term trigger / enable / remove
  // list across both families.
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
  } from "../lib/stores.svelte";
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

  const entries = $derived(
    [...steerRack.entries.entries()].sort((a, b) => a[0].localeCompare(b[0])),
  );
  const hasSubspace = $derived(entries.some(([, e]) => e.mode === "subspace"));
  const expression = $derived(currentSteeringExpression());
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
    else setManifoldTrigger(name, t);
  }
  function setEn(name: string, entry: SteerEntry, v: boolean): void {
    if (entry.mode === "subspace") setSubspaceEnabled(name, v);
    else setManifoldEnabled(name, v);
  }
  function remove(name: string, entry: SteerEntry): void {
    if (entry.mode === "subspace") removeSubspaceFromRack(name);
    else removeManifoldFromRack(name);
  }
  function onAlongInput(v: number): void {
    if (Number.isFinite(v)) setSubspaceAlong(v);
  }

  async function copyExpression(): Promise<void> {
    try {
      await navigator.clipboard.writeText(expression);
      copied = true;
      setTimeout(() => (copied = false), 1200);
    } catch {
      copied = false;
    }
  }
</script>

<section class="drawer-shell" aria-label="Recipe builder drawer">
  <header class="header">
    <div>
      <span class="title">recipe builder</span>
      <p>canonical expression, the shared subspace-along, and per-term triggers</p>
    </div>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}>×</button>
  </header>

  <div class="body">
    <section class="expression-card">
      <div>
        <span class="label">canonical expression</span>
        <code>{expression || "unsteered"}</code>
      </div>
      <div class="actions">
        <button type="button" onclick={copyExpression}>{copied ? "copied" : "copy"}</button>
      </div>
    </section>

    <section class="add-card">
      <input list="recipe-concepts" bind:value={newTerm} placeholder="add concept or ns/concept" onkeydown={(ev) => { if (ev.key === "Enter") add(); }} />
      <datalist id="recipe-concepts">
        {#each allNames as name (name)}
          <option value={name}></option>
        {/each}
      </datalist>
      <button type="button" onclick={add}>add term</button>
      <button type="button" onclick={() => openDrawer("subspace")}>browse…</button>
    </section>

    {#if hasSubspace}
      <section class="along-card">
        <span class="label">subspace along (shared)</span>
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
        <p class="muted">one slide magnitude for every flat term — relative weight lives in each term's position (set on its card)</p>
      </section>
    {/if}

    <section class="terms">
      {#if entries.length === 0}
        <div class="empty">no active steering terms, add a concept or manifold to start</div>
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
  </div>
</section>

<style>
  .drawer-shell { display: flex; flex-direction: column; min-height: 0; background: transparent; }
  .header { display: flex; justify-content: space-between; gap: var(--space-6); padding: var(--space-5) var(--space-6); background: var(--surface); }
  .title { color: var(--accent); letter-spacing: 0; font-size: var(--text-md); font-weight: var(--weight-medium); }
  .header p { margin: var(--space-2) 0 0; color: var(--fg-muted); }
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
  .expression-card { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: var(--space-5); align-items: center; }
  .label, .field span { display: block; color: var(--fg-muted); font-size: var(--text-xs); text-transform: uppercase; letter-spacing: 0; margin-bottom: var(--space-2); }
  code { color: var(--accent-amber); font-family: var(--font-mono); white-space: pre-wrap; word-break: break-word; }
  .actions, .add-card { display: flex; gap: var(--space-4); align-items: center; }
  button { border: 1px solid transparent; border-radius: var(--radius); background: var(--glass-strong); color: var(--fg); padding: var(--space-4) var(--space-5); }
  button:hover { background: color-mix(in srgb, var(--accent) 10%, var(--glass-strong)); color: var(--accent); }
  .add-card input { flex: 1; }
  input { border: 1px solid transparent; border-radius: var(--radius); background: var(--input-well); color: var(--fg); padding: var(--space-4); font-family: var(--font-mono); font-size: var(--text-xs); }
  input:focus { outline: none; border-color: var(--accent); }
  .along-row { display: grid; grid-template-columns: 1fr 4rem; gap: var(--space-5); align-items: center; }
  .along-row strong { color: var(--accent); font-family: var(--font-mono); text-align: right; }
  .muted { color: var(--fg-muted); font-size: var(--text-xs); margin: var(--space-3) 0 0; }
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
