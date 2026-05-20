<script lang="ts">
  import { apiExperiments } from "../lib/api";
  import type { ExperimentFanResponse, WSSampling } from "../lib/types";
  import {
    chatLog,
    closeDrawer,
    loomNavigate,
    refreshLoomTree,
    samplingState,
    vectorRack,
    vectorsState,
  } from "../lib/stores.svelte";
  import { serializeExpression } from "../lib/expression";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  let prompt = $state(lastUserPrompt() ?? "");
  let xConcept = $state(firstConcept());
  let yConcept = $state("");
  let xGrid = $state("linspace(-1, 1, 9)");
  let yGrid = $state("0");
  let metric = $state("token_count");
  let busy = $state(false);
  let errorMsg: string | null = $state(null);
  let response: ExperimentFanResponse | null = $state(null);

  const conceptOptions = $derived.by(() => {
    const names = new Set<string>([
      ...vectorsState.names,
      ...vectorRack.entries.keys(),
      ...vectorRack.profiles.keys(),
    ]);
    return [...names].sort();
  });

  const metrics = $derived.by(() => {
    const names = new Set(["token_count", "tok_per_sec", "elapsed"]);
    for (const row of response?.rows ?? []) {
      for (const k of Object.keys(row.result.readings ?? {})) names.add(k);
    }
    return [...names];
  });

  const xAlphas = $derived(parseAlphaList(xGrid));
  const yAlphas = $derived(parseAlphaList(yGrid));

  function lastUserPrompt(): string | null {
    for (let i = chatLog.turns.length - 1; i >= 0; i--) {
      const t = chatLog.turns[i];
      if (t.role === "user" && t.text.trim()) return t.text;
    }
    return null;
  }

  function firstConcept(): string {
    return [...vectorRack.entries.keys()][0] ?? vectorsState.names[0] ?? "";
  }

  function parseAlphaList(raw: string): number[] {
    const text = raw.trim();
    if (!text) return [];
    const m = text.match(/^linspace\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(\d+)\s*\)$/i);
    if (m) {
      const start = Number(m[1]);
      const end = Number(m[2]);
      const n = Math.max(1, Math.min(41, Number(m[3])));
      if (n === 1) return [start];
      return Array.from({ length: n }, (_, i) => start + ((end - start) * i) / (n - 1));
    }
    return text
      .split(/[,\s]+/)
      .map(Number)
      .filter((n) => Number.isFinite(n));
  }

  function samplingPayload(): WSSampling {
    return {
      temperature: samplingState.temperature,
      top_p: samplingState.top_p,
      top_k: samplingState.top_k,
      max_tokens: samplingState.max_tokens,
      seed: samplingState.seed,
      return_top_k: samplingState.return_top_k || null,
      presence_penalty: samplingState.presence_penalty,
      frequency_penalty: samplingState.frequency_penalty,
    };
  }

  function baseSteeringExcluding(axisNames: string[]): string | null {
    const excluded = new Set(axisNames.map((name) => name.trim()).filter(Boolean));
    const base = new Map(
      [...vectorRack.entries.entries()].filter(([name]) => !excluded.has(name)),
    );
    return serializeExpression(base) || null;
  }

  async function run(): Promise<void> {
    errorMsg = null;
    response = null;
    const grid: Record<string, number[]> = {};
    if (!prompt.trim()) {
      errorMsg = "prompt is required";
      return;
    }
    if (!xConcept.trim() || xAlphas.length === 0) {
      errorMsg = "x-axis concept and alpha grid are required";
      return;
    }
    grid[xConcept.trim()] = xAlphas;
    if (yConcept.trim()) {
      if (yAlphas.length === 0) {
        errorMsg = "y-axis alpha grid is empty";
        return;
      }
      grid[yConcept.trim()] = yAlphas;
    }
    busy = true;
    try {
      response = await apiExperiments.fan({
        prompt,
        grid,
        base_steering: baseSteeringExcluding(Object.keys(grid)),
        sampling: samplingPayload(),
        thinking: samplingState.thinking ?? false,
      });
      await refreshLoomTree();
    } catch (e) {
      errorMsg = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  function valueFor(rowIdx: number, alphaX: number, alphaY: number): number | null {
    const row = response?.rows.find((r) => {
      const x = r.alpha_values[xConcept];
      const y = yConcept ? r.alpha_values[yConcept] : alphaY;
      return nearly(x, alphaX) && nearly(y, alphaY);
    });
    if (!row) return null;
    if (metric === "token_count") return row.result.token_count;
    if (metric === "tok_per_sec") return row.result.tok_per_sec;
    if (metric === "elapsed") return row.result.elapsed;
    return row.result.readings[metric] ?? null;
  }

  function rowFor(alphaX: number, alphaY: number) {
    return response?.rows.find((r) => {
      const x = r.alpha_values[xConcept];
      const y = yConcept ? r.alpha_values[yConcept] : alphaY;
      return nearly(x, alphaX) && nearly(y, alphaY);
    });
  }

  function nearly(a: number | undefined, b: number): boolean {
    return typeof a === "number" && Math.abs(a - b) < 1e-6;
  }

  function cellStyle(v: number | null): string {
    if (v === null) return "";
    if (metric === "elapsed" || metric === "token_count" || metric === "tok_per_sec") {
      const max = Math.max(
        1,
        ...(response?.rows.map((r) => {
          if (metric === "elapsed") return r.result.elapsed;
          if (metric === "tok_per_sec") return r.result.tok_per_sec;
          return r.result.token_count;
        }) ?? [1]),
      );
      const t = Math.min(1, Math.max(0, v / max));
      return `background: color-mix(in srgb, rgba(72, 138, 203, 0.12), rgba(72, 138, 203, 0.55) ${t * 100}%);`;
    }
    const t = Math.max(-1, Math.min(1, v));
    const hue = t >= 0 ? "72, 138, 203" : "218, 83, 79";
    return `background: rgba(${hue}, ${0.14 + Math.abs(t) * 0.48});`;
  }

  async function navigate(nodeId: string | null): Promise<void> {
    if (!nodeId) return;
    await loomNavigate(nodeId);
    closeDrawer();
  }
</script>

<section class="drawer-shell" aria-label="Experiment lab drawer">
  <header class="header">
    <div>
      <span class="title">experiment lab</span>
      <p>run alpha grids as loom siblings, then inspect the response surface</p>
    </div>
    <button type="button" class="close" aria-label="Close" onclick={closeDrawer}>×</button>
  </header>

  <div class="body">
    <section class="setup">
      <label class="field prompt">
        <span>prompt</span>
        <textarea bind:value={prompt} rows="4"></textarea>
      </label>

      <div class="axis-grid">
        <label class="field">
          <span>x concept</span>
          <input list="concepts" bind:value={xConcept} placeholder="honest.deceptive" />
        </label>
        <label class="field">
          <span>x alphas</span>
          <input bind:value={xGrid} placeholder="linspace(-1, 1, 9)" />
        </label>
        <label class="field">
          <span>y concept</span>
          <input list="concepts" bind:value={yConcept} placeholder="optional" />
        </label>
        <label class="field">
          <span>y alphas</span>
          <input bind:value={yGrid} placeholder="-0.5, 0, 0.5" />
        </label>
      </div>
      <datalist id="concepts">
        {#each conceptOptions as c (c)}
          <option value={c}></option>
        {/each}
      </datalist>

      <div class="runbar">
        <label class="field metric">
          <span>metric</span>
          <select bind:value={metric}>
            {#each metrics as m (m)}
              <option value={m}>{m}</option>
            {/each}
          </select>
        </label>
        <button type="button" class="primary" disabled={busy} onclick={run}>
          {busy ? "running grid…" : "run experiment"}
        </button>
      </div>
      {#if errorMsg}
        <p class="error">{errorMsg}</p>
      {/if}
    </section>

    <section class="results">
      <div class="result-head">
        <div>
          <h3>response surface</h3>
          <p>{response ? `${response.total} runs` : "no run yet"}</p>
        </div>
        {#if response}
          <code>{response.kind}</code>
        {/if}
      </div>

      {#if response}
        <div class="heatmap-wrap">
          <table class="heatmap">
            <thead>
              <tr>
                <th>{yConcept || "run"}</th>
                {#each xAlphas as ax (ax)}
                  <th>{ax.toFixed(2)}</th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each (yConcept ? yAlphas : [0]) as ay, yi (ay)}
                <tr>
                  <th>{yConcept ? ay.toFixed(2) : yi + 1}</th>
                  {#each xAlphas as ax (ax)}
                    {@const v = valueFor(yi, ax, ay)}
                    {@const row = rowFor(ax, ay)}
                    <td style={cellStyle(v)} title={row?.result.applied_steering ?? ""}>
                      <button type="button" onclick={() => navigate(row?.node_id ?? null)}>
                        <strong>{v === null ? "—" : v.toFixed(metric === "token_count" ? 0 : 2)}</strong>
                        <span>{row?.result.finish_reason ?? ""}</span>
                      </button>
                    </td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>

        <div class="run-list">
          {#each response.rows as row (row.idx)}
            <button type="button" onclick={() => navigate(row.node_id)}>
              <span class="idx">#{row.idx + 1}</span>
              <code>{Object.entries(row.alpha_values).map(([k, v]) => `${k}=${v}`).join(" · ")}</code>
              <span>{row.result.token_count} tok · {row.result.tok_per_sec.toFixed(1)} tok/s</span>
            </button>
          {/each}
        </div>
      {:else}
        <div class="empty">
          alpha grids become loom siblings; each cell can be opened as the active branch after the run.
        </div>
      {/if}
    </section>
  </div>
</section>

<style>
  .drawer-shell { display: flex; flex-direction: column; min-height: 0; background: var(--bg-alt); }
  .header { display: flex; justify-content: space-between; gap: var(--space-6); padding: var(--space-6) var(--space-6); border-bottom: 1px solid var(--border); background: var(--surface); }
  .title { color: var(--accent); text-transform: uppercase; letter-spacing: 0; font-size: var(--text-xs); font-weight: var(--weight-bold); }
  .header p, .result-head p { margin: var(--space-2) 0 0; color: var(--fg-muted); }
  .close { background: transparent; border: 0; color: var(--fg-muted); font-size: var(--text-md); }
  .body { display: grid; grid-template-columns: minmax(20rem, 0.78fr) minmax(24rem, 1fr); gap: var(--space-5); padding: var(--space-6); overflow: auto; }
  .setup, .results { border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); padding: var(--space-5); min-width: 0; }
  .field { display: grid; gap: var(--space-2); color: var(--fg-muted); font-size: var(--text-xs); text-transform: uppercase; letter-spacing: 0; }
  .prompt { margin-bottom: var(--space-5); }
  input, textarea, select { width: 100%; border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-deep); color: var(--fg); padding: var(--space-4); font-family: var(--font-mono); font-size: var(--text-xs); letter-spacing: 0; }
  textarea { resize: vertical; line-height: 1.45; }
  input:focus, textarea:focus, select:focus { outline: none; border-color: var(--accent); }
  .axis-grid { display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-5); }
  .runbar { display: grid; grid-template-columns: 1fr auto; align-items: end; gap: var(--space-5); margin-top: var(--space-5); }
  .primary { border: 1px solid var(--accent); border-radius: var(--radius); background: var(--accent); color: var(--text-on-accent); padding: var(--space-4) var(--space-5); font-weight: var(--weight-bold); }
  .primary:disabled { opacity: 0.55; cursor: wait; }
  .error { color: var(--accent-red); margin: var(--space-4) 0 0; }
  .result-head { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: var(--space-5); }
  h3 { margin: 0; color: var(--fg); font-size: var(--text); letter-spacing: 0; }
  code { color: var(--accent-amber); font-family: var(--font-mono); }
  .heatmap-wrap { overflow: auto; border: 1px solid var(--border); border-radius: var(--radius); }
  .heatmap { border-collapse: collapse; width: 100%; min-width: 32rem; }
  th { color: var(--fg-muted); font-size: var(--text-xs); text-transform: uppercase; letter-spacing: 0; background: var(--bg-elev); }
  th, td { border: 1px solid var(--border); padding: 0; }
  td button { width: 100%; min-height: 3.4rem; display: grid; gap: var(--space-1); place-items: center; border: 0; background: transparent; color: var(--fg); }
  td button:hover { outline: 1px solid var(--accent); outline-offset: -1px; }
  td span { color: var(--fg-muted); font-size: var(--text-2xs); }
  .run-list { display: grid; gap: var(--space-2); margin-top: var(--space-5); max-height: 16rem; overflow: auto; }
  .run-list button { display: grid; grid-template-columns: auto 1fr auto; gap: var(--space-4); align-items: center; border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-elev); color: var(--fg-strong); padding: var(--space-3); text-align: left; }
  .run-list button:hover { border-color: var(--accent); }
  .idx { color: var(--accent); font-family: var(--font-mono); }
  .empty { display: grid; place-items: center; min-height: 18rem; color: var(--fg-muted); text-align: center; border: 1px solid var(--border); border-radius: var(--radius); padding: var(--space-8); }
</style>
