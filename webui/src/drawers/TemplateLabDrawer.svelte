<script lang="ts">
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
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
  import SegmentedTabs from "../lib/ui/SegmentedTabs.svelte";
  import Button from "../lib/ui/Button.svelte";
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

  const TAB_ITEMS: Array<{ value: Tab; label: string; title: string }> = [
    { value: "score", label: "score", title: "Score a template's restricted-choice distribution" },
    { value: "build", label: "build", title: "Author a new template" },
  ];

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
    <div class="title">
      <span class="eyebrow">templates</span>
    </div>
    <DrawerCloseButton onclick={closeDrawer} />
  </header>

  <div class="toolbar">
    <SegmentedTabs items={TAB_ITEMS} bind:value={tab} ariaLabel="Template lab view" />
  </div>

  <div class="body">
    {#if tab === "score"}
      <p class="hint">restricted-choice probabilities</p>

      {#if loading}
        <p class="muted">loading…</p>
      {:else if templates.length === 0}
        <p class="muted">no templates</p>
      {:else}
        <label class="field">
          <span class="label">template</span>
          <select bind:value={selectedKey} disabled={scoring}>
            <option value="">select…</option>
            {#each templates as t (`${t.namespace}/${t.name}`)}
              <option value={`${t.namespace}/${t.name}`}>
                {t.namespace}/{t.name} · {t.n_values} values × {t.n_contexts} ctx
              </option>
            {/each}
          </select>
        </label>

        <label class="field">
          <span class="label">steering</span>
          <input type="text" placeholder="0.5 patient.hurried" bind:value={steerExpr}
            disabled={scoring} autocomplete="off" spellcheck="false" />
        </label>

        <div class="controls">
          <label class="byrow">
            <span class="label">rank by</span>
            <select bind:value={scoreBy}>
              <option value="sum">sum</option>
              <option value="mean">mean</option>
            </select>
          </label>
          <Button variant="solid" disabled={!selectedTemplate || scoring} onclick={runScore}>
            {scoring ? "scoring…" : "score"}
          </Button>
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
      <p class="hint">slot · values · contexts</p>

      <form class="form" onsubmit={submitBuild}>
        <label class="field">
          <span class="label">name</span>
          <input type="text" placeholder="weekday" bind:value={bName} disabled={building}
            autocomplete="off" spellcheck="false" />
        </label>
        <label class="field">
          <span class="label">slot token</span>
          <input type="text" placeholder="[DAY]" bind:value={bSlot} disabled={building}
            autocomplete="off" spellcheck="false" />
        </label>
        <label class="field">
          <span class="label">values</span>
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
                <span class="label">assistant · slot</span>
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
          <Button variant="ghost" onclick={closeDrawer}>cancel</Button>
          <Button
            type="submit"
            variant="solid"
            disabled={building || buildValidation.length > 0}
          >
            {building ? "creating…" : "create template"}
          </Button>
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
            <Button variant="danger" size="sm" onclick={() => deleteTemplate(t)}>delete</Button>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</section>

<style>
  /* v2 sheet interior — the host paints the sheet surface, so the root
   * stays transparent and chrome speaks sans (values/identifiers stay
   * mono). Templates carry no pillar hue — chrome stays achromatic. */
  .drawer-shell {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: transparent;
    color: var(--fg);
    font-family: var(--font-ui);
    font-size: var(--text);
  }
  .header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-5);
    padding: var(--space-5) var(--space-6);
  }
  .title {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }
  .eyebrow {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .toolbar {
    padding: var(--space-3) var(--space-6);
  }

  .body {
    flex: 1;
    overflow-y: auto;
    padding: var(--space-5) var(--space-6);
    display: flex;
    flex-direction: column;
    gap: var(--space-5);
  }
  .hint {
    font-size: var(--text-sm);
    color: var(--fg-muted);
    margin: 0;
    line-height: 1.5;
    max-width: 62ch;
  }
  .muted { color: var(--fg-muted); font-size: var(--text-sm); }
  .field { display: flex; flex-direction: column; gap: var(--space-2); }
  .label { font-size: var(--text-sm); color: var(--fg-muted); }
  .optional { color: var(--fg-subtle); }
  input,
  select,
  textarea {
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-4);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
  }
  input:focus,
  select:focus,
  textarea:focus {
    outline: none;
    border-color: var(--fg-muted);
  }
  textarea { resize: vertical; }
  .controls { display: flex; align-items: flex-end; gap: var(--space-5); }
  .byrow { display: flex; flex-direction: column; gap: var(--space-2); flex: 1; }

  /* Data well — the per-context restricted-choice distribution. */
  .ctx-card {
    border-radius: var(--radius);
    background: var(--bg);
    padding: var(--space-4);
  }
  .ctx-head {
    font-size: var(--text-xs);
    color: var(--fg-muted);
    margin-bottom: var(--space-3);
  }
  .bar-row {
    display: grid;
    grid-template-columns: 5.5em 1fr 4.5em;
    align-items: center;
    gap: var(--space-3);
    margin: var(--space-1) 0;
  }
  .bar-label {
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .bars { display: flex; flex-direction: column; gap: 2px; }
  .bar { height: 7px; border-radius: var(--radius-sm); min-width: 1px; }
  /* Achromatic before/after: the steered bar reads brighter above the
   * muted baseline — no pillar hue borrowed for a non-pillar surface. */
  .bar.base { background: var(--fg-muted); }
  .bar.steer { background: var(--accent); }
  .bar-num {
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    color: var(--fg-dim);
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .arrow { color: var(--fg-subtle); margin: 0 var(--space-1); }

  .form { display: flex; flex-direction: column; gap: var(--space-4); }
  .contexts {
    border-radius: var(--radius);
    background: var(--glass);
    box-shadow: var(--shadow-well);
    padding: var(--space-4);
    margin: 0;
  }
  legend {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0 var(--space-2);
  }
  .ctx-build {
    padding: var(--space-4) 0;
  }
  .ctx-build-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: var(--text-xs);
    color: var(--fg-muted);
    margin-bottom: var(--space-3);
  }
  .turn-row {
    display: grid;
    grid-template-columns: 6em 1fr auto;
    gap: var(--space-2);
    margin: var(--space-2) 0;
  }
  .assistant-field { margin-top: var(--space-3); }

  /* Icon-ish micro-buttons — dense-row pills, styled like the reference
   * .scrub-btn rather than the full Button component. */
  .mini {
    background: var(--glass);
    border: 1px solid transparent;
    color: var(--fg-muted);
    border-radius: var(--radius-pill);
    padding: var(--space-2) var(--space-4);
    cursor: pointer;
    font: inherit;
    font-size: var(--text-2xs);
    line-height: 1;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out);
  }
  .mini:hover {
    color: var(--fg);
    background: var(--glass-strong);
  }
  .mini.add { margin-top: var(--space-3); }
  .errs {
    color: var(--accent-red);
    font-size: var(--text-sm);
    margin: 0;
    padding-left: 1.1em;
  }
  .foot { display: flex; justify-content: flex-end; gap: var(--space-3); }

  .catalog {
    padding-top: var(--space-4);
  }
  .catalog-head {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: var(--space-3);
  }
  .cat-row {
    display: grid;
    grid-template-columns: 1fr auto auto;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-2) 0;
    font-size: var(--text-sm);
  }
  .cat-name { font-family: var(--font-mono); color: var(--fg); }
  .cat-sub {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
  }
</style>
