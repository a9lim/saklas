<script lang="ts">
  // TemplateLabDrawer — the standalone templated-completion artifact.
  //
  // Two tabs:
  //   * score — pick a template, optionally steer, run; render the per-context
  //     restricted-choice value distribution (the logit read). With a steering
  //     expression this is the distributional before/after.
  //   * build — author a template: a slot token, candidate values, and one or
  //     more multi-turn contexts (history turns + the slotted final assistant
  //     turn). The slot lives only in the assistant turn.
  //
  // Reached from the workspace rail's "manifolds → templates…" entry. Templates
  // also feed a manifold fit (`saklas manifold from-template`).

  import { onMount } from "svelte";
  import { ApiError, apiTemplates } from "../lib/api";
  import { closeDrawer } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type {
    ChoiceScores,
    TemplateContextSpec,
    TemplateSummary,
    TemplateTurn,
  } from "../lib/types";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => { void _drawerProps.params; });

  function describeError(e: unknown): string {
    if (e instanceof ApiError) {
      const detail =
        e.body && typeof e.body === "object" && "detail" in (e.body as object)
          ? String((e.body as { detail: unknown }).detail)
          : e.message;
      return `${e.status}: ${detail}`;
    }
    return e instanceof Error ? e.message : String(e);
  }

  type Tab = "score" | "build";
  let tab: Tab = $state("score");

  // ----- shared: template catalog --------------------------------------
  let templates: TemplateSummary[] = $state([]);
  let loading = $state(false);

  async function loadTemplates(): Promise<void> {
    loading = true;
    try {
      templates = (await apiTemplates.list()).templates;
    } catch (e) {
      pushToast(`couldn't load templates: ${describeError(e)}`, { kind: "error" });
    } finally {
      loading = false;
    }
  }
  onMount(loadTemplates);

  // ----- score tab -----------------------------------------------------
  let selectedKey = $state("");
  let steerExpr = $state("");
  let scoring = $state(false);
  let scoreBy: "sum" | "mean" = $state("sum");
  let baseline: ChoiceScores[] | null = $state(null);
  let steered: ChoiceScores[] | null = $state(null);
  let scoredKey = $state("");

  const selectedTemplate = $derived(
    templates.find((t) => `${t.namespace}/${t.name}` === selectedKey) ?? null,
  );

  function probOf(c: { prob_sum: number; prob_mean: number }): number {
    return scoreBy === "sum" ? c.prob_sum : c.prob_mean;
  }

  async function runScore(): Promise<void> {
    if (!selectedTemplate) return;
    scoring = true;
    baseline = null;
    steered = null;
    const { namespace, name } = selectedTemplate;
    try {
      baseline = (await apiTemplates.score(namespace, name, null)).contexts;
      const expr = steerExpr.trim();
      if (expr) {
        steered = (await apiTemplates.score(namespace, name, expr)).contexts;
      }
      scoredKey = selectedKey;
    } catch (e) {
      pushToast(`scoring failed: ${describeError(e)}`, { kind: "error" });
    } finally {
      scoring = false;
    }
  }

  /** Rows for one context: label + baseline prob + optional steered prob,
   *  sorted by the active baseline probability descending. */
  function rows(ctxIdx: number): {
    label: string;
    base: number;
    steer: number | null;
  }[] {
    const b = baseline?.[ctxIdx];
    if (!b) return [];
    const s = steered?.[ctxIdx] ?? null;
    const out = b.choices.map((c, i) => ({
      label: c.label,
      base: probOf(c),
      steer: s ? probOf(s.choices[i]) : null,
    }));
    out.sort((x, y) => Math.max(y.base, y.steer ?? -1) - Math.max(x.base, x.steer ?? -1));
    return out;
  }

  // ----- build tab -----------------------------------------------------
  let bName = $state("");
  let bSlot = $state("[DAY]");
  let bValuesText = $state("");
  let bContexts = $state<TemplateContextSpec[]>([
    { turns: [{ role: "user", content: "" }], assistant: "" },
  ]);
  let building = $state(false);

  const bValues = $derived(
    bValuesText
      .split(/[\n,]+/)
      .map((v) => v.trim())
      .filter(Boolean),
  );

  function addContext(): void {
    bContexts = [
      ...bContexts,
      { turns: [{ role: "user", content: "" }], assistant: "" },
    ];
  }
  function removeContext(i: number): void {
    bContexts = bContexts.filter((_, idx) => idx !== i);
  }
  function addTurn(ci: number): void {
    const ctx = bContexts[ci];
    const lastRole = ctx.turns[ctx.turns.length - 1]?.role;
    const role: TemplateTurn["role"] = lastRole === "user" ? "assistant" : "user";
    ctx.turns = [...ctx.turns, { role, content: "" }];
    bContexts = [...bContexts];
  }
  function removeTurn(ci: number, ti: number): void {
    bContexts[ci].turns = bContexts[ci].turns.filter((_, idx) => idx !== ti);
    bContexts = [...bContexts];
  }

  const buildValidation = $derived.by(() => {
    const errs: string[] = [];
    if (!bName.trim()) errs.push("name required");
    if (!bSlot.trim()) errs.push("slot required");
    if (bValues.length < 2) errs.push("≥ 2 values");
    const slot = bSlot.trim();
    bContexts.forEach((c, i) => {
      const turns = c.turns.filter((t) => t.content.trim());
      if (!turns.length) errs.push(`context ${i + 1}: needs a history turn`);
      else if (turns[turns.length - 1].role !== "user")
        errs.push(`context ${i + 1}: last turn must be user`);
      if (turns.some((t) => slot && t.content.includes(slot)))
        errs.push(`context ${i + 1}: slot must not appear in a history turn`);
      const n = slot ? c.assistant.split(slot).length - 1 : 0;
      if (n !== 1) errs.push(`context ${i + 1}: slot must appear once in the assistant turn`);
    });
    return errs;
  });

  async function submitBuild(ev: Event): Promise<void> {
    ev.preventDefault();
    if (buildValidation.length) return;
    building = true;
    try {
      await apiTemplates.create({
        namespace: "local",
        name: bName.trim(),
        slot: bSlot.trim(),
        values: bValues,
        contexts: bContexts.map((c) => ({
          turns: c.turns.filter((t) => t.content.trim()),
          assistant: c.assistant,
        })),
      });
      pushToast(`template ${bName.trim()} created`, { kind: "info" });
      await loadTemplates();
      selectedKey = `local/${bName.trim()}`;
      tab = "score";
    } catch (e) {
      pushToast(`create failed: ${describeError(e)}`, { kind: "error" });
    } finally {
      building = false;
    }
  }

  async function deleteTemplate(t: TemplateSummary): Promise<void> {
    try {
      await apiTemplates.delete(t.namespace, t.name);
      pushToast(`removed ${t.namespace}/${t.name}`, { kind: "info" });
      if (selectedKey === `${t.namespace}/${t.name}`) selectedKey = "";
      await loadTemplates();
    } catch (e) {
      pushToast(`delete failed: ${describeError(e)}`, { kind: "error" });
    }
  }
</script>

<section class="drawer-shell" aria-label="Template lab">
  <header class="header">
    <span class="title">templates</span>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}>✕</button>
  </header>

  <div class="tabs" role="tablist">
    <button class="tab" class:active={tab === "score"} role="tab"
      aria-selected={tab === "score"} onclick={() => (tab = "score")}>score</button>
    <button class="tab" class:active={tab === "build"} role="tab"
      aria-selected={tab === "build"} onclick={() => (tab = "build")}>build</button>
  </div>

  <div class="body">
    {#if tab === "score"}
      <p class="hint">
        The model's <strong>restricted-choice</strong> distribution over a
        template's values, per context — the logit read. Add a steering
        expression to see how steering reshapes it (the distributional
        before/after).
      </p>

      {#if loading}
        <p class="muted">loading…</p>
      {:else if templates.length === 0}
        <p class="muted">no templates yet — author one in the <strong>build</strong> tab.</p>
      {:else}
        <label class="field">
          <span class="label">template</span>
          <select bind:value={selectedKey} disabled={scoring}>
            <option value="">— pick a template —</option>
            {#each templates as t (`${t.namespace}/${t.name}`)}
              <option value={`${t.namespace}/${t.name}`}>
                {t.namespace}/{t.name} · {t.n_values} values × {t.n_contexts} ctx
              </option>
            {/each}
          </select>
        </label>

        <label class="field">
          <span class="label">steering <span class="optional">(optional)</span></span>
          <input type="text" placeholder="0.5 patient.hurried" bind:value={steerExpr}
            disabled={scoring} autocomplete="off" spellcheck="false" />
        </label>

        <div class="controls">
          <label class="byrow">
            <span class="label">rank by</span>
            <select bind:value={scoreBy}>
              <option value="sum">sum (joint)</option>
              <option value="mean">mean (length-norm)</option>
            </select>
          </label>
          <button type="button" class="primary" disabled={!selectedTemplate || scoring}
            onclick={runScore}>{scoring ? "scoring…" : "score"}</button>
        </div>

        {#if baseline && scoredKey === selectedKey}
          {#each baseline as _ctx, ci (ci)}
            <div class="ctx-card">
              <div class="ctx-head">context {ci + 1}{steered ? " — base → steered" : ""}</div>
              {#each rows(ci) as r (r.label)}
                <div class="bar-row">
                  <span class="bar-label" title={r.label}>{r.label}</span>
                  <div class="bars">
                    <div class="bar base" style={`width:${(r.base * 100).toFixed(1)}%`}></div>
                    {#if r.steer !== null}
                      <div class="bar steer" style={`width:${(r.steer * 100).toFixed(1)}%`}></div>
                    {/if}
                  </div>
                  <span class="bar-num">
                    {(r.base * 100).toFixed(0)}%{#if r.steer !== null}<span class="arrow">→</span>{(r.steer * 100).toFixed(0)}%{/if}
                  </span>
                </div>
              {/each}
            </div>
          {/each}
        {/if}
      {/if}

    {:else}
      <p class="hint">
        Author a template: a <strong>slot</strong> token, candidate
        <strong>values</strong> (one node per value), and one or more
        <strong>contexts</strong> — a multi-turn history ending on a user turn,
        plus the final assistant turn that carries the slot exactly once.
      </p>

      <form class="form" onsubmit={submitBuild}>
        <label class="field">
          <span class="label">name <span class="optional">(under local/)</span></span>
          <input type="text" placeholder="weekday" bind:value={bName} disabled={building}
            autocomplete="off" spellcheck="false" />
        </label>
        <label class="field">
          <span class="label">slot token</span>
          <input type="text" placeholder="[DAY]" bind:value={bSlot} disabled={building}
            autocomplete="off" spellcheck="false" />
        </label>
        <label class="field">
          <span class="label">values <span class="optional">(one per line or comma-separated)</span></span>
          <textarea rows="3" placeholder={"Monday\nTuesday\nWednesday"}
            bind:value={bValuesText} disabled={building}></textarea>
        </label>

        <fieldset class="contexts">
          <legend>contexts <span class="optional">({bContexts.length})</span></legend>
          {#each bContexts as ctx, ci (ci)}
            <div class="ctx-build">
              <div class="ctx-build-head">
                <span>context {ci + 1}</span>
                {#if bContexts.length > 1}
                  <button type="button" class="mini" onclick={() => removeContext(ci)}>remove</button>
                {/if}
              </div>
              {#each ctx.turns as turn, ti (ti)}
                <div class="turn-row">
                  <select bind:value={turn.role} disabled={building} aria-label="turn role">
                    <option value="user">user</option>
                    <option value="assistant">assistant</option>
                    <option value="system">system</option>
                  </select>
                  <input type="text" placeholder="turn content" bind:value={turn.content}
                    disabled={building} autocomplete="off" />
                  {#if ctx.turns.length > 1}
                    <button type="button" class="mini" onclick={() => removeTurn(ci, ti)}>×</button>
                  {/if}
                </div>
              {/each}
              <button type="button" class="mini add" onclick={() => addTurn(ci)}>+ turn</button>
              <label class="field assistant-field">
                <span class="label">assistant (slot here)</span>
                <input type="text" placeholder={`today is ${bSlot}`} bind:value={ctx.assistant}
                  disabled={building} autocomplete="off" />
              </label>
            </div>
          {/each}
          <button type="button" class="mini add" onclick={addContext}>+ context</button>
        </fieldset>

        {#if buildValidation.length}
          <ul class="errs">
            {#each buildValidation as e (e)}<li>{e}</li>{/each}
          </ul>
        {/if}

        <footer class="foot">
          <button type="button" class="secondary" onclick={closeDrawer}>cancel</button>
          <button type="submit" class="primary" disabled={building || buildValidation.length > 0}>
            {building ? "creating…" : "create template"}
          </button>
        </footer>
      </form>
    {/if}

    {#if templates.length > 0}
      <div class="catalog">
        <div class="catalog-head">installed</div>
        {#each templates as t (`${t.namespace}/${t.name}`)}
          <div class="cat-row">
            <span class="cat-name">{t.namespace}/{t.name}</span>
            <span class="cat-sub">{t.slot} · {t.n_values}×{t.n_contexts}</span>
            <button type="button" class="mini" onclick={() => deleteTemplate(t)}>delete</button>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</section>

<style>
  .drawer-shell { display: flex; flex-direction: column; height: 100%; }
  .header { display: flex; align-items: center; justify-content: space-between;
    padding: 0.75rem 1rem; border-bottom: 1px solid var(--border, #2a2a2a); }
  .title { font-weight: 600; }
  .close { background: none; border: none; color: inherit; cursor: pointer; font-size: 1rem; }
  .tabs { display: flex; gap: 0.25rem; padding: 0.5rem 1rem 0; }
  .tab { background: none; border: none; border-bottom: 2px solid transparent;
    color: var(--muted, #888); cursor: pointer; padding: 0.35rem 0.6rem; font-size: 0.85rem; }
  .tab.active { color: inherit; border-bottom-color: var(--accent, #fff); }
  .body { flex: 1; overflow-y: auto; padding: 1rem; display: flex; flex-direction: column; gap: 0.75rem; }
  .hint { font-size: 0.8rem; color: var(--muted, #999); margin: 0; line-height: 1.4; }
  .muted { color: var(--muted, #888); font-size: 0.85rem; }
  .field { display: flex; flex-direction: column; gap: 0.25rem; }
  .label { font-size: 0.75rem; color: var(--muted, #999); }
  .optional { opacity: 0.6; }
  input, select, textarea { background: var(--input-bg, #1a1a1a); color: inherit;
    border: 1px solid var(--border, #333); border-radius: 4px; padding: 0.4rem 0.5rem;
    font: inherit; font-size: 0.85rem; }
  textarea { resize: vertical; font-family: ui-monospace, monospace; }
  .controls { display: flex; align-items: flex-end; gap: 0.75rem; }
  .byrow { display: flex; flex-direction: column; gap: 0.25rem; flex: 1; }
  .primary, .secondary { border-radius: 4px; padding: 0.45rem 0.9rem; cursor: pointer;
    font: inherit; font-size: 0.85rem; border: 1px solid var(--border, #333); }
  .primary { background: var(--accent, #3a6); color: #fff; border-color: transparent; }
  .primary:disabled { opacity: 0.5; cursor: not-allowed; }
  .secondary { background: none; color: inherit; }
  .ctx-card { border: 1px solid var(--border, #2a2a2a); border-radius: 6px; padding: 0.6rem; }
  .ctx-head { font-size: 0.75rem; color: var(--muted, #999); margin-bottom: 0.4rem; }
  .bar-row { display: grid; grid-template-columns: 5.5rem 1fr 4.5rem; align-items: center;
    gap: 0.5rem; margin: 0.15rem 0; }
  .bar-label { font-size: 0.8rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .bars { display: flex; flex-direction: column; gap: 2px; }
  .bar { height: 7px; border-radius: 3px; min-width: 1px; }
  .bar.base { background: var(--accent, #4a8); }
  .bar.steer { background: var(--accent-purple, #a6f); }
  .bar-num { font-size: 0.72rem; color: var(--muted, #aaa); text-align: right; font-variant-numeric: tabular-nums; }
  .arrow { opacity: 0.5; margin: 0 0.2rem; }
  .form { display: flex; flex-direction: column; gap: 0.6rem; }
  .contexts { border: 1px solid var(--border, #2a2a2a); border-radius: 6px; padding: 0.5rem; margin: 0; }
  legend { font-size: 0.78rem; color: var(--muted, #999); padding: 0 0.35rem; }
  .ctx-build { border-top: 1px solid var(--border, #222); padding: 0.5rem 0; }
  .ctx-build:first-of-type { border-top: none; }
  .ctx-build-head { display: flex; justify-content: space-between; font-size: 0.75rem;
    color: var(--muted, #999); margin-bottom: 0.35rem; }
  .turn-row { display: grid; grid-template-columns: 6rem 1fr auto; gap: 0.3rem; margin: 0.2rem 0; }
  .assistant-field { margin-top: 0.35rem; }
  .mini { background: none; border: 1px solid var(--border, #333); color: var(--muted, #aaa);
    border-radius: 4px; padding: 0.15rem 0.45rem; cursor: pointer; font-size: 0.72rem; }
  .mini.add { margin-top: 0.35rem; }
  .errs { color: var(--danger, #e66); font-size: 0.78rem; margin: 0; padding-left: 1.1rem; }
  .foot { display: flex; justify-content: flex-end; gap: 0.5rem; }
  .catalog { border-top: 1px solid var(--border, #2a2a2a); padding-top: 0.6rem; margin-top: 0.4rem; }
  .catalog-head { font-size: 0.72rem; color: var(--muted, #888); text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 0.3rem; }
  .cat-row { display: grid; grid-template-columns: 1fr auto auto; align-items: center;
    gap: 0.5rem; padding: 0.2rem 0; font-size: 0.8rem; }
  .cat-sub { color: var(--muted, #888); font-size: 0.72rem; }
</style>
