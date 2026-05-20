<script lang="ts">
  import {
    addVectorToRack,
    closeDrawer,
    currentSteeringExpression,
    openDrawer,
    removeVectorFromRack,
    setVectorAblate,
    setVectorAlpha,
    setVectorEnabled,
    setVectorProjection,
    setVectorTrigger,
    setVectorVariant,
    vectorRack,
    vectorsState,
  } from "../lib/stores.svelte";
  import type { Trigger, Variant } from "../lib/types";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  let newVector = $state("");
  let copied = $state(false);

  const entries = $derived([...vectorRack.entries.entries()]);
  const expression = $derived(currentSteeringExpression());
  const allNames = $derived.by(() => {
    const names = new Set<string>([...vectorsState.names, ...vectorRack.profiles.keys()]);
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
    const name = newVector.trim();
    if (!name) return;
    addVectorToRack(name);
    newVector = "";
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

  function setProjection(name: string, op: "~" | "|", target: string): void {
    const t = target.trim();
    setVectorProjection(name, t ? { op, target: t } : null);
  }
</script>

<section class="drawer-shell" aria-label="Recipe builder drawer">
  <header class="header">
    <div>
      <span class="title">recipe builder</span>
      <p>visual editor for coefficients, variants, projections, ablations, and triggers</p>
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
        <button type="button" onclick={() => openDrawer("merge")}>merge…</button>
      </div>
    </section>

    <section class="add-card">
      <input list="recipe-concepts" bind:value={newVector} placeholder="add concept or ns/concept" onkeydown={(ev) => { if (ev.key === "Enter") add(); }} />
      <datalist id="recipe-concepts">
        {#each allNames as name (name)}
          <option value={name}></option>
        {/each}
      </datalist>
      <button type="button" onclick={add}>add term</button>
      <button type="button" onclick={() => openDrawer("vector_picker")}>browse…</button>
    </section>

    <section class="terms">
      {#if entries.length === 0}
        <div class="empty">no active steering terms, add a vector to start building a recipe</div>
      {:else}
        {#each entries as [name, entry] (name)}
          <article class="term" class:disabled={!entry.enabled}>
            <header>
              <label class="enable">
                <input
                  type="checkbox"
                  checked={entry.enabled}
                  onchange={(ev) => setVectorEnabled(name, (ev.currentTarget as HTMLInputElement).checked)}
                />
                <span>{name}</span>
              </label>
              <button type="button" class="remove" aria-label={`Remove ${name}`} onclick={() => removeVectorFromRack(name)}>×</button>
            </header>

            <div class="alpha">
              <label>
                <span>alpha</span>
                <input
                  type="range"
                  min="-1"
                  max="1"
                  step="0.01"
                  value={entry.alpha}
                  oninput={(ev) => setVectorAlpha(name, Number((ev.currentTarget as HTMLInputElement).value))}
                />
                <strong>{entry.alpha.toFixed(2)}</strong>
              </label>
            </div>

            <div class="control-grid">
              <label class="field">
                <span>trigger</span>
                <select value={entry.trigger} onchange={(ev) => setVectorTrigger(name, (ev.currentTarget as HTMLSelectElement).value as Trigger)}>
                  {#each triggers as trigger (trigger.value)}
                    <option value={trigger.value}>{trigger.label}</option>
                  {/each}
                </select>
              </label>

              <label class="field">
                <span>variant</span>
                <input
                  value={entry.variant}
                  placeholder="raw | sae | sae-release"
                  oninput={(ev) => {
                    const raw = (ev.currentTarget as HTMLInputElement).value.trim();
                    if (raw === "raw" || raw === "sae" || raw.startsWith("sae-")) {
                      setVectorVariant(name, raw as Variant);
                    }
                  }}
                />
              </label>

              <label class="field">
                <span>projection</span>
                <select
                  value={entry.projection?.op ?? ""}
                  onchange={(ev) => {
                    const op = (ev.currentTarget as HTMLSelectElement).value as "~" | "|" | "";
                    if (!op) setVectorProjection(name, null);
                    else setProjection(name, op, entry.projection?.target ?? "");
                  }}
                  disabled={entry.ablate}
                >
                  <option value="">none</option>
                  <option value="~">keep shared (~)</option>
                  <option value="|">remove shared (|)</option>
                </select>
              </label>

              <label class="field">
                <span>projection target</span>
                <input
                  list="recipe-concepts"
                  value={entry.projection?.target ?? ""}
                  disabled={entry.ablate}
                  oninput={(ev) => {
                    const target = (ev.currentTarget as HTMLInputElement).value;
                    setProjection(name, entry.projection?.op ?? "|", target);
                  }}
                />
              </label>
            </div>

            <label class="ablate">
              <input
                type="checkbox"
                checked={entry.ablate}
                onchange={(ev) => setVectorAblate(name, (ev.currentTarget as HTMLInputElement).checked)}
              />
              <span>mean-ablate this concept instead of steering toward it</span>
            </label>
          </article>
        {/each}
      {/if}
    </section>
  </div>
</section>

<style>
  .drawer-shell { display: flex; flex-direction: column; min-height: 0; background: var(--bg-alt); }
  .header { display: flex; justify-content: space-between; gap: var(--space-6); padding: var(--space-6) var(--space-6); border-bottom: 1px solid var(--border); background: var(--surface); }
  .title { color: var(--accent); text-transform: uppercase; letter-spacing: 0; font-size: var(--text-xs); font-weight: var(--weight-bold); }
  .header p { margin: var(--space-2) 0 0; color: var(--fg-muted); }
  .close, .remove { background: transparent; border: 0; color: var(--fg-muted); font-size: var(--text-md); }
  .body { display: grid; gap: var(--space-5); padding: var(--space-6); overflow: auto; }
  .expression-card, .add-card, .term { border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); padding: var(--space-6); }
  .expression-card { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: var(--space-5); align-items: center; }
  .label, .field span { display: block; color: var(--fg-muted); font-size: var(--text-xs); text-transform: uppercase; letter-spacing: 0; margin-bottom: var(--space-2); }
  code { color: var(--accent-amber); font-family: var(--font-mono); white-space: pre-wrap; word-break: break-word; }
  .actions, .add-card { display: flex; gap: var(--space-4); align-items: center; }
  button { border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-elev); color: var(--fg); padding: var(--space-4) var(--space-5); }
  button:hover { border-color: var(--accent); color: var(--accent); }
  .add-card input { flex: 1; }
  input, select { border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-deep); color: var(--fg); padding: var(--space-4); font-family: var(--font-mono); font-size: var(--text-xs); }
  input:focus, select:focus { outline: none; border-color: var(--accent); }
  .terms { display: grid; gap: var(--space-5); }
  .term { display: grid; gap: var(--space-5); }
  .term.disabled { opacity: 0.58; }
  .term header { display: flex; align-items: center; justify-content: space-between; gap: var(--space-5); }
  .enable, .ablate { display: flex; align-items: center; gap: var(--space-4); color: var(--fg); }
  .enable span { font-weight: var(--weight-bold); }
  .alpha label { display: grid; grid-template-columns: auto 1fr 4rem; gap: var(--space-5); align-items: center; color: var(--fg-muted); text-transform: uppercase; letter-spacing: 0; font-size: var(--text-xs); }
  .alpha strong { color: var(--accent); font-family: var(--font-mono); text-align: right; }
  input[type="range"] { padding: 0; accent-color: var(--accent); }
  .control-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-5); }
  .field { display: grid; gap: var(--space-1); }
  .ablate { color: var(--fg-muted); font-size: var(--text-xs); }
  .empty { display: grid; place-items: center; min-height: 10rem; color: var(--fg-muted); border: 1px solid var(--border); border-radius: var(--radius); }
</style>
