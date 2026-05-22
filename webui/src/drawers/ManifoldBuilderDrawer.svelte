<script lang="ts">
  // Manifold authoring form — reached from "+ build manifold" in the
  // ManifoldDrawer.  Two steps in one scroll:
  //
  //   1. Domain — pick Box (1D / 2D / 3D, per-axis lo/hi + open/periodic)
  //      or Sphere (dimension).  The intrinsic dimension follows.
  //   2. Nodes — one per node: a label, ``intrinsic_dim`` coordinate
  //      inputs, and an expandable multi-statement editor.
  //
  // A live validation banner checks node count against
  // ``min_nodes = 2n+1`` and that every coordinate sits in the domain.
  // Save → apiManifolds.create, then back to the ManifoldDrawer.

  import { apiManifolds, ApiError } from "../lib/api";
  import {
    closeDrawer,
    openDrawer,
    refreshManifoldList,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type {
    AxisSpec,
    CreateManifoldRequest,
    ManifoldDomain,
  } from "../lib/types";

  let { params: _params }: { params?: unknown } = $props();
  $effect(() => { void _params; });

  // ---------- domain ----------

  type DomainKind = "box" | "sphere";
  let domainKind: DomainKind = $state("box");
  let boxDim = $state(2); // 1 | 2 | 3
  let sphereDim = $state(2);

  // Per-axis specs — three slots authored; only the first ``boxDim``
  // are used.  Defaults give a unit square that's easy to author on.
  interface AxisDraft {
    name: string;
    lo: number;
    hi: number;
    periodic: boolean;
  }
  let axisDrafts: AxisDraft[] = $state([
    { name: "x", lo: 0, hi: 1, periodic: false },
    { name: "y", lo: 0, hi: 1, periodic: false },
    { name: "z", lo: 0, hi: 1, periodic: false },
  ]);

  const intrinsicDim = $derived(domainKind === "box" ? boxDim : sphereDim);

  /** Build the wire ManifoldDomain from the form state. */
  function buildDomain(): ManifoldDomain {
    if (domainKind === "sphere") {
      return { type: "sphere", dim: sphereDim };
    }
    const axes: AxisSpec[] = axisDrafts.slice(0, boxDim).map((a) => ({
      name: a.name,
      periodic: a.periodic,
      period: a.hi - a.lo,
      lo: a.lo,
      hi: a.hi,
    }));
    return { type: "box", axes };
  }

  // ---------- nodes ----------

  interface NodeDraft {
    label: string;
    coords: number[];
    statements: string;
    expanded: boolean;
  }

  let nodes: NodeDraft[] = $state([]);

  /** Resize every node's coord array to the current intrinsic dim,
   *  preserving existing values, padding with zeros. */
  function reshapeNodeCoords(): void {
    const n = intrinsicDim;
    nodes = nodes.map((nd) => {
      const coords = nd.coords.slice(0, n);
      while (coords.length < n) coords.push(0);
      return { ...nd, coords };
    });
  }

  function addNode(): void {
    nodes = [
      ...nodes,
      {
        label: `node_${nodes.length + 1}`,
        coords: new Array(intrinsicDim).fill(0),
        statements: "",
        expanded: true,
      },
    ];
  }

  function removeNode(idx: number): void {
    nodes = nodes.filter((_, i) => i !== idx);
  }

  function setNodeField<K extends keyof NodeDraft>(
    idx: number,
    key: K,
    value: NodeDraft[K],
  ): void {
    nodes = nodes.map((nd, i) => (i === idx ? { ...nd, [key]: value } : nd));
  }

  function setNodeCoord(idx: number, ci: number, value: number): void {
    nodes = nodes.map((nd, i) => {
      if (i !== idx) return nd;
      const coords = nd.coords.slice();
      coords[ci] = value;
      return { ...nd, coords };
    });
  }

  // ---------- description ----------

  let manifoldName = $state("");
  let namespace = $state("local");
  let description = $state("");

  function slug(s: string): string {
    return s
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9._-]+/g, "_")
      .replace(/^[_.-]+|[_.-]+$/g, "");
  }

  // ---------- validation ----------

  const minNodes = $derived(2 * intrinsicDim + 1);

  /** Split a node's textarea body into trimmed non-empty statements. */
  function statementsOf(nd: NodeDraft): string[] {
    return nd.statements
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter(Boolean);
  }

  /** Check one coordinate vector against the domain.  Box: each coord
   *  in [lo, hi] (periodic axes accept anything — they wrap).  Sphere:
   *  no per-coord bound, the domain immerses the chart. */
  function coordsInDomain(coords: number[]): boolean {
    if (domainKind === "sphere") return true;
    for (let i = 0; i < boxDim; i++) {
      const a = axisDrafts[i];
      if (a.periodic) continue;
      const v = coords[i];
      if (!Number.isFinite(v)) return false;
      if (v < a.lo || v > a.hi) return false;
    }
    return true;
  }

  const validation = $derived.by<{ ok: boolean; messages: string[] }>(() => {
    const messages: string[] = [];
    if (!slug(manifoldName)) {
      messages.push("a manifold name is required");
    }
    if (domainKind === "box") {
      for (let i = 0; i < boxDim; i++) {
        const a = axisDrafts[i];
        if (a.hi <= a.lo) {
          messages.push(`axis "${a.name || i}" needs hi > lo`);
        }
      }
    }
    if (nodes.length < minNodes) {
      messages.push(
        `need at least ${minNodes} nodes for an ${intrinsicDim}D domain (have ${nodes.length})`,
      );
    }
    const seenLabels = new Set<string>();
    for (const nd of nodes) {
      const lbl = slug(nd.label);
      if (!lbl) {
        messages.push("every node needs a label");
      } else if (seenLabels.has(lbl)) {
        messages.push(`duplicate node label "${lbl}"`);
      } else {
        seenLabels.add(lbl);
      }
      if (!coordsInDomain(nd.coords)) {
        messages.push(`node "${nd.label}" has out-of-domain coordinates`);
      }
      if (statementsOf(nd).length === 0) {
        messages.push(`node "${nd.label}" needs at least one statement`);
      }
    }
    return { ok: messages.length === 0, messages };
  });

  let submitting = $state(false);

  async function save(): Promise<void> {
    if (!validation.ok || submitting) return;
    submitting = true;
    const req: CreateManifoldRequest = {
      namespace: slug(namespace) || "local",
      name: slug(manifoldName),
      description: description.trim(),
      domain: buildDomain(),
      nodes: nodes.map((nd) => ({
        label: slug(nd.label),
        coords: nd.coords.slice(0, intrinsicDim),
        statements: statementsOf(nd),
      })),
    };
    try {
      const r = await apiManifolds.create(req);
      await refreshManifoldList();
      const advisories = r.advisories ?? [];
      if (advisories.length > 0) {
        pushToast(
          `built ${req.namespace}/${req.name} — ${advisories.length} poisedness advisory`,
          { kind: "warning", detail: advisories.join("; "), ttlMs: 10000 },
        );
      } else {
        pushToast(`built manifold ${req.namespace}/${req.name}`, {
          kind: "info",
        });
      }
      closeDrawer();
      openDrawer("manifolds");
    } catch (e) {
      const msg =
        e instanceof ApiError
          ? e.body && typeof e.body === "object" && "detail" in (e.body as object)
            ? String((e.body as { detail: unknown }).detail)
            : e.message
          : e instanceof Error
            ? e.message
            : String(e);
      pushToast(`build failed — ${msg}`, { kind: "error", ttlMs: null });
    } finally {
      submitting = false;
    }
  }

  function cancel(): void {
    closeDrawer();
    openDrawer("manifolds");
  }

  function pickBoxDim(d: number): void {
    boxDim = d;
    domainKind = "box";
    reshapeNodeCoords();
  }
  function pickSphere(): void {
    domainKind = "sphere";
    reshapeNodeCoords();
  }
  function onSphereDim(d: number): void {
    sphereDim = d;
    reshapeNodeCoords();
  }
</script>

<section class="drawer-shell" aria-label="Build manifold">
  <header class="header">
    <span class="title">build manifold</span>
    <button type="button" class="close" aria-label="Close" onclick={cancel}
      >✕</button>
  </header>

  <div class="body">
    <p class="hint">
      author a steering manifold: pick a domain, place labelled nodes
      with a statement corpus each, then fit it from the manifolds
      drawer.
    </p>

    <!-- identity -->
    <div class="grid2">
      <label class="field">
        <span class="label">namespace</span>
        <input
          type="text"
          class="input"
          bind:value={namespace}
          placeholder="local"
          autocomplete="off"
          spellcheck="false"
        />
      </label>
      <label class="field">
        <span class="label">name <span class="req">required</span></span>
        <input
          type="text"
          class="input"
          bind:value={manifoldName}
          placeholder="e.g. circumplex"
          autocomplete="off"
          spellcheck="false"
        />
      </label>
    </div>
    <label class="field">
      <span class="label">description</span>
      <input
        type="text"
        class="input"
        bind:value={description}
        placeholder="what this manifold steers"
        autocomplete="off"
      />
    </label>

    <!-- domain step -->
    <section class="step">
      <h2 class="step-title">1 · domain</h2>
      <div class="domain-kind">
        <button
          type="button"
          class="kind-btn"
          class:active={domainKind === "box" && boxDim === 1}
          onclick={() => pickBoxDim(1)}
        >box 1D</button>
        <button
          type="button"
          class="kind-btn"
          class:active={domainKind === "box" && boxDim === 2}
          onclick={() => pickBoxDim(2)}
        >box 2D</button>
        <button
          type="button"
          class="kind-btn"
          class:active={domainKind === "box" && boxDim === 3}
          onclick={() => pickBoxDim(3)}
        >box 3D</button>
        <button
          type="button"
          class="kind-btn"
          class:active={domainKind === "sphere"}
          onclick={pickSphere}
        >sphere</button>
      </div>

      {#if domainKind === "box"}
        <div class="axes">
          {#each axisDrafts.slice(0, boxDim) as axis, i (i)}
            <div class="axis-card">
              <label class="axis-field name-field">
                <span class="mini-label">axis</span>
                <input
                  type="text"
                  class="input mini"
                  value={axis.name}
                  oninput={(ev) => {
                    axisDrafts[i].name = (ev.currentTarget as HTMLInputElement).value;
                  }}
                  spellcheck="false"
                />
              </label>
              <label class="axis-field">
                <span class="mini-label">lo</span>
                <input
                  type="number"
                  class="input mini"
                  value={axis.lo}
                  step="0.1"
                  oninput={(ev) => {
                    axisDrafts[i].lo = Number((ev.currentTarget as HTMLInputElement).value);
                  }}
                />
              </label>
              <label class="axis-field">
                <span class="mini-label">hi</span>
                <input
                  type="number"
                  class="input mini"
                  value={axis.hi}
                  step="0.1"
                  oninput={(ev) => {
                    axisDrafts[i].hi = Number((ev.currentTarget as HTMLInputElement).value);
                  }}
                />
              </label>
              <label class="axis-check">
                <input
                  type="checkbox"
                  checked={axis.periodic}
                  onchange={(ev) => {
                    axisDrafts[i].periodic = (ev.currentTarget as HTMLInputElement).checked;
                  }}
                />
                <span>periodic</span>
              </label>
            </div>
          {/each}
        </div>
      {:else}
        <label class="field sphere-field">
          <span class="label">sphere dimension (S^n)</span>
          <select
            class="input"
            value={sphereDim}
            onchange={(ev) =>
              onSphereDim(Number((ev.currentTarget as HTMLSelectElement).value))}
          >
            <option value={1}>S¹ — circle</option>
            <option value={2}>S² — sphere</option>
            <option value={3}>S³</option>
          </select>
        </label>
      {/if}
      <p class="dim-note">
        intrinsic dimension: <strong>{intrinsicDim}</strong> ·
        minimum nodes: <strong>{minNodes}</strong>
      </p>
    </section>

    <!-- node editor -->
    <section class="step">
      <h2 class="step-title">2 · nodes</h2>
      {#if nodes.length === 0}
        <p class="muted">no nodes yet — add at least {minNodes}.</p>
      {/if}
      <div class="node-list">
        {#each nodes as node, idx (idx)}
          <div class="node-card">
            <div class="node-head">
              <button
                type="button"
                class="node-expand"
                onclick={() => setNodeField(idx, "expanded", !node.expanded)}
                aria-expanded={node.expanded}
              >
                <span class="caret">{node.expanded ? "▾" : "▸"}</span>
              </button>
              <input
                type="text"
                class="input mini node-label"
                value={node.label}
                oninput={(ev) =>
                  setNodeField(idx, "label", (ev.currentTarget as HTMLInputElement).value)}
                placeholder="label"
                spellcheck="false"
              />
              <div class="node-coords">
                {#each node.coords as c, ci (ci)}
                  <input
                    type="number"
                    class="input mini coord"
                    value={c}
                    step="0.1"
                    title="coordinate {ci}"
                    oninput={(ev) =>
                      setNodeCoord(
                        idx,
                        ci,
                        Number((ev.currentTarget as HTMLInputElement).value),
                      )}
                  />
                {/each}
              </div>
              <button
                type="button"
                class="node-remove"
                onclick={() => removeNode(idx)}
                aria-label="remove node {node.label}"
                title="remove node"
              >✕</button>
            </div>
            {#if node.expanded}
              <textarea
                class="node-statements"
                rows="4"
                value={node.statements}
                oninput={(ev) =>
                  setNodeField(
                    idx,
                    "statements",
                    (ev.currentTarget as HTMLTextAreaElement).value,
                  )}
                placeholder={"one statement per line — the corpus pooled into this node's centroid"}
              ></textarea>
            {/if}
          </div>
        {/each}
      </div>
      <button type="button" class="add-node" onclick={addNode}>
        + add node
      </button>
    </section>

    {#if !validation.ok}
      <div class="validation" role="alert">
        <p class="validation-head">not ready to build:</p>
        <ul>
          {#each validation.messages as m (m)}
            <li>{m}</li>
          {/each}
        </ul>
      </div>
    {/if}

    <button
      type="button"
      class="save-btn"
      disabled={!validation.ok || submitting}
      onclick={save}
    >
      {submitting ? "building…" : "build manifold → return to list"}
    </button>
  </div>
</section>

<style>
  .drawer-shell {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text);
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-4) var(--space-5);
    border-bottom: 1px solid var(--border);
  }
  .title {
    color: var(--accent);
    letter-spacing: 0;
  }
  .close {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    font-size: var(--text);
    line-height: 1;
    padding: var(--space-2) var(--space-3);
    cursor: pointer;
  }
  .close:hover {
    color: var(--accent-red);
  }
  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--space-4) var(--space-5) var(--space-5);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-height: 0;
  }
  .hint {
    color: var(--fg-dim);
    font-size: var(--text-sm);
    margin: 0;
    line-height: 1.4;
  }
  .muted {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    margin: 0;
  }
  .grid2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-3);
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .label {
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
  .req {
    color: var(--accent-red);
    font-size: var(--text-xs);
    font-style: italic;
    margin-left: var(--space-2);
  }
  .input {
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
  }
  .input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .input.mini {
    padding: var(--space-1) var(--space-2);
    font-size: var(--text-sm);
  }

  .step {
    border-top: 1px solid var(--border);
    padding-top: var(--space-3);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .step-title {
    margin: 0;
    color: var(--accent);
    font-size: var(--text-sm);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: var(--weight-medium);
  }

  .domain-kind {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }
  .kind-btn {
    background: transparent;
    color: var(--fg-dim);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-4);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
    transition:
      background var(--dur) var(--ease-out),
      border-color var(--dur) var(--ease-out),
      color var(--dur) var(--ease-out);
  }
  .kind-btn:hover {
    color: var(--fg-strong);
  }
  .kind-btn.active {
    color: var(--accent);
    border-color: var(--accent);
    background: var(--accent-subtle);
  }

  .axes {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .axis-card {
    display: grid;
    grid-template-columns: 1.4fr 1fr 1fr auto;
    align-items: end;
    gap: var(--space-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-3);
  }
  .axis-field {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
  .mini-label {
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    text-transform: uppercase;
  }
  .axis-check {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    color: var(--fg-strong);
    font-size: var(--text-xs);
    padding-bottom: var(--space-1);
  }
  .axis-check input {
    accent-color: var(--accent);
  }
  .sphere-field {
    max-width: 18em;
  }
  .dim-note {
    margin: 0;
    color: var(--fg-dim);
    font-size: var(--text-xs);
  }
  .dim-note strong {
    color: var(--accent);
  }

  .node-list {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .node-card {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-3);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    background: var(--bg-deep);
  }
  .node-head {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .node-expand {
    background: transparent;
    border: 0;
    color: var(--fg-muted);
    cursor: pointer;
    font-size: var(--text-xs);
    padding: 0 var(--space-1);
  }
  .node-label {
    flex: 0 0 8em;
  }
  .node-coords {
    display: flex;
    gap: var(--space-1);
    flex: 1 1 auto;
  }
  .coord {
    width: 4.5em;
  }
  .node-remove {
    background: transparent;
    border: 0;
    color: var(--fg-muted);
    cursor: pointer;
    font-size: var(--text);
    padding: 0 var(--space-2);
  }
  .node-remove:hover {
    color: var(--accent-red);
  }
  .node-statements {
    width: 100%;
    box-sizing: border-box;
    background: var(--bg-elev);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    resize: vertical;
    line-height: 1.4;
  }
  .node-statements:focus {
    outline: none;
    border-color: var(--accent);
  }
  .add-node {
    background: var(--accent-subtle);
    color: var(--accent);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-4);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
    align-self: flex-start;
  }
  .add-node:hover {
    background: var(--accent-glow);
  }

  .validation {
    border: 1px solid var(--accent-yellow);
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-4);
    background: color-mix(in srgb, var(--accent-yellow) 8%, transparent);
  }
  .validation-head {
    margin: 0 0 var(--space-2);
    color: var(--accent-yellow);
    font-size: var(--text-sm);
  }
  .validation ul {
    margin: 0;
    padding-left: var(--space-5);
    color: var(--fg-dim);
    font-size: var(--text-sm);
  }

  .save-btn {
    background: var(--accent);
    color: var(--text-on-accent);
    border: 1px solid var(--accent);
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-5);
    font: inherit;
    font-family: var(--font-mono);
    cursor: pointer;
    transition: background var(--dur) var(--ease-out);
  }
  .save-btn:hover:not(:disabled) {
    background: var(--accent-light);
  }
  .save-btn:disabled {
    background: var(--bg-elev);
    color: var(--fg-muted);
    border-color: var(--border);
    cursor: not-allowed;
  }
</style>
