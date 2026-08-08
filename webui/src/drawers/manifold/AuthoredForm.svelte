<script lang="ts">
  // Custom-nodes authoring — the historical path.
  //
  // The user brings labelled corpora.  The ``auto-domain`` switch decides
  // what happens to the geometry:
  //
  //   * off — pick a domain (box 1D/2D/3D with per-axis lo/hi + periodic,
  //     or sphere), place every node at hand-supplied coordinates, and
  //     submit through ``apiManifolds.create``.  Poisedness needs
  //     ``2n+1`` nodes.
  //   * on  — no domain, no coordinates: submit through
  //     ``apiManifolds.createDiscover`` and let the fitter derive the
  //     layout per-model via the same pca / spectral hyperparams the
  //     auto-generated tab exposes.  Only ≥2 nodes are required.

  import { apiManifolds, ApiError } from "../../lib/api";
  import { closeDrawer, openDrawer, refreshManifoldList } from "../../lib/stores.svelte";
  import { pushToast } from "../../lib/stores/toasts.svelte";
  import type {
    AxisSpec,
    CreateDiscoverManifoldRequest,
    CreateManifoldRequest,
    ManifoldDomain,
  } from "../../lib/types";
  import Select from "../../lib/Select.svelte";
  import Checkbox from "../../lib/Checkbox.svelte";
  import NumberInput from "../../lib/NumberInput.svelte";
  import AdvancedSection from "../../lib/builder/AdvancedSection.svelte";
  import ValidationBlock from "../../lib/builder/ValidationBlock.svelte";
  import DiscoverTuningFields from "./DiscoverTuningFields.svelte";
  import FitMethodPicker from "./FitMethodPicker.svelte";
  import {
    defaultTuning,
    identitySlugs,
    slug,
    tuningHyperparams,
    tuningMessages,
    type ManifoldIdentity,
  } from "./shared";

  let { identity }: { identity: ManifoldIdentity } = $props();

  let autoDomain = $state(false);
  const tuning = $state(defaultTuning());
  let advancedOpen = $state(false);
  let submitting = $state(false);

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
  const minNodes = $derived(2 * intrinsicDim + 1);

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

  // ---------- nodes ----------

  interface NodeDraft {
    label: string;
    coords: number[];
    statements: string;
    /** Optional per-node assistant-role substitution.  Empty string =
     *  "use the standard assistant baseline" (the legacy default).
     *  Validated client-side against the same slug regex the engine
     *  uses (`[a-z0-9._-]+`).  Persona manifolds use this — each node's
     *  centroid is pooled under its role's chat-template substitution. */
    role: string;
    expanded: boolean;
  }

  let nodes: NodeDraft[] = $state([]);
  const ROLE_SLUG_RE = /^[a-z0-9._-]+$/;

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
        role: "",
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

  // ---------- validation ----------

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
    if (!slug(identity.name)) {
      messages.push("name required");
    }
    // Domain-shape validation only fires when the user is hand-authoring
    // coordinates.  auto-domain skips the box / sphere picker entirely
    // — the fitter derives the layout per-model.
    if (!autoDomain && domainKind === "box") {
      for (let i = 0; i < boxDim; i++) {
        const a = axisDrafts[i];
        if (a.hi <= a.lo) {
          messages.push(`axis "${a.name || i}": hi > lo`);
        }
      }
    }
    // Min-node count: hand-authored coords need ``2n+1`` for poisedness;
    // auto-domain only needs >=2 nodes (shared-structure requirement,
    // matching the auto-generated tab).
    if (autoDomain) {
      if (nodes.length < 2) {
        messages.push(`nodes: ${nodes.length} / 2`);
      }
    } else if (nodes.length < minNodes) {
      messages.push(`nodes: ${nodes.length} / ${minNodes}`);
    }
    const seenLabels = new Set<string>();
    for (const nd of nodes) {
      const lbl = slug(nd.label);
      if (!lbl) {
        messages.push("node label required");
      } else if (seenLabels.has(lbl)) {
        messages.push(`duplicate label "${lbl}"`);
      } else {
        seenLabels.add(lbl);
      }
      if (!autoDomain && !coordsInDomain(nd.coords)) {
        messages.push(`"${nd.label}": outside domain`);
      }
      if (statementsOf(nd).length === 0) {
        messages.push(`"${nd.label}": statement required`);
      }
      const r = nd.role.trim();
      if (r && !ROLE_SLUG_RE.test(r)) {
        messages.push(`"${nd.label}": invalid role "${r}"`);
      }
    }
    // auto-domain shares hyperparam validation with the auto-generated tab.
    if (autoDomain) messages.push(...tuningMessages(tuning));
    return { ok: messages.length === 0, messages };
  });

  // ---------- submit ----------

  async function save(): Promise<void> {
    if (!validation.ok || submitting) return;
    submitting = true;
    const { namespace, name, description } = identitySlugs(identity);
    // auto-domain split: bring-your-own-corpora discover (the fitter
    // derives coords per-model via pca / spectral) routes through
    // createDiscover; the historical authored path with hand-placed
    // coords keeps using create.
    if (autoDomain) {
      const req: CreateDiscoverManifoldRequest = {
        namespace,
        name,
        description,
        fit_mode: tuning.fitMode,
        hyperparams: tuningHyperparams(tuning),
        nodes: nodes.map((nd) => {
          const r = nd.role.trim();
          return {
            label: slug(nd.label),
            statements: statementsOf(nd),
            ...(r ? { role: r } : {}),
          };
        }),
      };
      try {
        await apiManifolds.createDiscover(req);
        await refreshManifoldList();
        pushToast(
          `built ${namespace}/${name} (auto-domain, ${tuning.fitMode} fit) — open the manifolds drawer to fit`,
          { kind: "info" },
        );
        closeDrawer();
        openDrawer("manifolds");
      } catch (e) {
        pushToast(`build failed — ${errorText(e)}`, {
          kind: "error",
          ttlMs: null,
        });
      } finally {
        submitting = false;
      }
      return;
    }
    const req: CreateManifoldRequest = {
      namespace,
      name,
      description,
      domain: buildDomain(),
      nodes: nodes.map((nd) => {
        const r = nd.role.trim();
        return {
          label: slug(nd.label),
          coords: nd.coords.slice(0, intrinsicDim),
          statements: statementsOf(nd),
          ...(r ? { role: r } : {}),
        };
      }),
    };
    try {
      const r = await apiManifolds.create(req);
      await refreshManifoldList();
      const advisories = r.advisories ?? [];
      if (advisories.length > 0) {
        pushToast(
          `built ${namespace}/${name} — ${advisories.length} poisedness advisory`,
          { kind: "warning", detail: advisories.join("; "), ttlMs: 10000 },
        );
      } else {
        pushToast(`built manifold ${namespace}/${name}`, { kind: "info" });
      }
      closeDrawer();
      openDrawer("manifolds");
    } catch (e) {
      pushToast(`build failed — ${errorText(e)}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      submitting = false;
    }
  }

  function errorText(e: unknown): string {
    if (e instanceof ApiError) {
      return e.body && typeof e.body === "object" && "detail" in (e.body as object)
        ? String((e.body as { detail: unknown }).detail)
        : e.message;
    }
    return e instanceof Error ? e.message : String(e);
  }
</script>

<div class="form-stack">
  <!-- auto-domain switch: when on, skip the box/sphere picker and the
       per-node coord inputs; the fitter derives the layout per-model
       via pca / spectral.  When off, hand-author coords as before. -->
  <span class="auto-domain-toggle">
    <Checkbox bind:checked={autoDomain} label="auto-domain" />
  </span>

  {#if autoDomain}
    <!-- fit-method picker — mirrors the auto-generated tab's choice. -->
    <FitMethodPicker {tuning} spectralNote="curved · best with ≥50 nodes" />
  {:else}
    <section class="step">
      <h2 class="step-title">domain</h2>
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
                <NumberInput
                  value={axis.lo}
                  step={0.1}
                  oninput={(v) => {
                    if (v !== null) axisDrafts[i].lo = v;
                  }}
                />
              </label>
              <label class="axis-field">
                <span class="mini-label">hi</span>
                <NumberInput
                  value={axis.hi}
                  step={0.1}
                  oninput={(v) => {
                    if (v !== null) axisDrafts[i].hi = v;
                  }}
                />
              </label>
              <span class="axis-check">
                <Checkbox
                  checked={axis.periodic}
                  label="periodic"
                  onchange={(v) => {
                    axisDrafts[i].periodic = v;
                  }}
                />
              </span>
            </div>
          {/each}
        </div>
      {:else}
        <label class="field sphere-field">
          <span class="label">sphere dim</span>
          <Select
            value={sphereDim}
            options={[
              { value: 1, label: "S¹ — circle" },
              { value: 2, label: "S² — sphere" },
              { value: 3, label: "S³" },
            ]}
            ariaLabel="Sphere dimension"
            onchange={onSphereDim}
          />
        </label>
      {/if}
      <p class="dim-note">
        dim <strong>{intrinsicDim}</strong> · min <strong>{minNodes}</strong> nodes
      </p>
    </section>
  {/if}

  <!-- node editor -->
  <section class="step">
    <h2 class="step-title">nodes</h2>
    {#if nodes.length === 0}
      <p class="muted">
        add ≥{autoDomain ? 2 : minNodes} nodes
      </p>
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
            {#if !autoDomain}
              <div class="node-coords">
                {#each node.coords as c, ci (ci)}
                  <span class="coord-cell">
                    <NumberInput
                      value={c}
                      step={0.1}
                      title="coordinate {ci}"
                      oninput={(v) => setNodeCoord(idx, ci, v ?? 0)}
                    />
                  </span>
                {/each}
              </div>
            {/if}
            <button
              type="button"
              class="node-remove"
              onclick={() => removeNode(idx)}
              aria-label="remove node {node.label}"
              title="remove node"
            >✕</button>
          </div>
          {#if node.expanded}
            <label class="node-role">
              <span class="label">role</span>
              <input
                type="text"
                class="input mini"
                value={node.role}
                oninput={(ev) =>
                  setNodeField(
                    idx,
                    "role",
                    (ev.currentTarget as HTMLInputElement).value,
                  )}
                placeholder="pirate"
                autocomplete="off"
                spellcheck="false"
              />
            </label>
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
              placeholder="one statement per line"
            ></textarea>
          {/if}
        </div>
      {/each}
    </div>
    <button type="button" class="add-node" onclick={addNode}>
      + add node
    </button>
  </section>

  {#if autoDomain}
    <AdvancedSection bind:expanded={advancedOpen}>
      <DiscoverTuningFields {tuning} />
    </AdvancedSection>
  {/if}

  <ValidationBlock verb="build" messages={validation.messages} />

  <button
    type="button"
    class="save-btn"
    disabled={!validation.ok || submitting}
    onclick={save}
  >
    {submitting ? "building…" : autoDomain ? `build · ${tuning.fitMode}` : "build"}
  </button>
</div>
