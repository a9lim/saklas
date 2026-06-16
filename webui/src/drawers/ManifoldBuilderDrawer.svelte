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

  import {
    apiManifolds,
    apiManifoldFitStream,
    apiManifoldGenerateStream,
    ApiError,
  } from "../lib/api";
  import {
    closeDrawer,
    openDrawer,
    refreshManifoldList,
  } from "../lib/stores.svelte";
  import {
    dismissToast,
    pushToast,
    updateToast,
  } from "../lib/stores/toasts.svelte";
  import type {
    AxisSpec,
    CreateDiscoverManifoldRequest,
    CreateManifoldRequest,
    CreateTemplatedManifoldRequest,
    GenerateManifoldRequest,
    ManifoldDomain,
  } from "../lib/types";
  import Select from "../lib/Select.svelte";
  import Checkbox from "../lib/Checkbox.svelte";
  import Radio from "../lib/Radio.svelte";
  import NumberInput from "../lib/NumberInput.svelte";
  import ModeTabs from "../lib/builder/ModeTabs.svelte";
  import AdvancedSection from "../lib/builder/AdvancedSection.svelte";
  import ValidationBlock from "../lib/builder/ValidationBlock.svelte";

  let { params: _params }: { params?: unknown } = $props();
  $effect(() => { void _params; });

  // ---------- mode picker ----------
  //
  // Two authoring paths share this drawer:
  //
  //   * Authored  — user picks a domain (box / sphere) and places nodes
  //                 at user-supplied coordinates.  The historical path.
  //   * Discover  — user hands the model a flat concept list; the K-tuple
  //                 generator produces per-concept corpora, then the
  //                 fitter derives coords per-model via PCA or spectral
  //                 embedding.  No coords to author; no domain to pick.
  //
  // Both build into the same on-disk manifold artifact; the mode shows
  // up as ``manifold.json::fit_mode``.  Inspector + steering paths are
  // unchanged from there on.
  //   * Templated — user gives a slot token, a list of values (one node
  //                 each), and a set of {user, assistant} chat-turn
  //                 templates with the slot in the assistant turn; the
  //                 slot is filled per value (deterministic, no model) and
  //                 the templates' user turns become the per-manifold
  //                 elicitation prompts. The tool for categories one
  //                 references rather than embodies (days, months, …).
  type AuthoringMode = "authored" | "discover" | "templated";
  let authoringMode: AuthoringMode = $state("authored");

  // ---------- custom-nodes auto-domain switch ----------
  //
  // The custom-nodes (authored) tab carries a single ``auto-domain``
  // toggle: when on, the user supplies labelled corpora only (no coords,
  // no domain picker), and submission routes through the discover-create
  // endpoint where the fitter derives coords per-model via the same
  // pca / spectral hyperparams the auto-generated tab exposes.  When
  // off, the historical authored flow runs unchanged (box / sphere
  // picker + per-node coord inputs + ``apiManifolds.create``).
  let autoDomain = $state(false);

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
    // Domain-shape validation only fires when the user is hand-authoring
    // coordinates.  auto-domain skips the box / sphere picker entirely
    // — the fitter derives the layout per-model.
    if (!autoDomain && domainKind === "box") {
      for (let i = 0; i < boxDim; i++) {
        const a = axisDrafts[i];
        if (a.hi <= a.lo) {
          messages.push(`axis "${a.name || i}" needs hi > lo`);
        }
      }
    }
    // Min-node count: hand-authored coords need ``2n+1`` for poisedness;
    // auto-domain only needs >=2 nodes (shared-structure requirement,
    // matching the auto-generated tab's discoverValidation).
    if (autoDomain) {
      if (nodes.length < 2) {
        messages.push(
          `need at least 2 nodes for auto-domain (have ${nodes.length})`,
        );
      }
    } else if (nodes.length < minNodes) {
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
      if (!autoDomain && !coordsInDomain(nd.coords)) {
        messages.push(`node "${nd.label}" has out-of-domain coordinates`);
      }
      if (statementsOf(nd).length === 0) {
        messages.push(`node "${nd.label}" needs at least one statement`);
      }
      const r = nd.role.trim();
      if (r && !ROLE_SLUG_RE.test(r)) {
        messages.push(
          `node "${nd.label}" role "${r}" must match [a-z0-9._-]+`,
        );
      }
    }
    // auto-domain shares hyperparam validation with the auto-generated tab.
    if (autoDomain) {
      if (discoverMaxDim < 1) messages.push("max_dim must be >= 1");
      if (
        (discoverFitMode === "pca" || discoverFitMode === "auto") &&
        (discoverVarThreshold <= 0 || discoverVarThreshold > 1)
      ) {
        messages.push("var_threshold must be in (0, 1]");
      }
    }
    return { ok: messages.length === 0, messages };
  });

  let submitting = $state(false);

  /** Per-tab AdvancedSection open state.  Kept separate so toggling one
   *  tab's advanced flyout doesn't carry across to the other. */
  let advancedAuthoredOpen = $state(false);
  let advancedDiscoverOpen = $state(false);

  async function save(): Promise<void> {
    if (!validation.ok || submitting) return;
    submitting = true;
    // auto-domain split: bring-your-own-corpora discover (the fitter
    // derives coords per-model via pca / spectral) routes through
    // createDiscover; the historical authored path with hand-placed
    // coords keeps using create.
    if (autoDomain) {
      const namespaceSlug = slug(namespace) || "local";
      const nameSlug = slug(manifoldName);
      const req: CreateDiscoverManifoldRequest = {
        namespace: namespaceSlug,
        name: nameSlug,
        description: description.trim(),
        fit_mode: discoverFitMode,
        hyperparams: buildDiscoverHyperparams(),
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
          `built ${namespaceSlug}/${nameSlug} (auto-domain, ${discoverFitMode} fit) — open the manifolds drawer to fit`,
          { kind: "info" },
        );
        closeDrawer();
        openDrawer("manifolds");
      } catch (e) {
        const msg = describeFitError(e);
        pushToast(`build failed — ${msg}`, { kind: "error", ttlMs: null });
      } finally {
        submitting = false;
      }
      return;
    }
    const req: CreateManifoldRequest = {
      namespace: slug(namespace) || "local",
      name: slug(manifoldName),
      description: description.trim(),
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

  // ============================================================ discover ===

  type DiscoverFitMode = "pca" | "spectral" | "auto";
  type DiscoverKind = "abstract" | "concrete";
  // ``auto`` is the friendly default — ``select_topology`` picks
  // flat / curved / periodic per-model, so a newcomer needn't know which
  // geometry their concepts want.  pca / spectral pin it for power users.
  let discoverFitMode: DiscoverFitMode = $state("auto");
  let discoverConceptsText = $state("");
  // A2 conversational corpus knobs: ``kind`` frames each concept's
  // system prompt (abstract → "someone {c}", concrete → "{article} {c}");
  // ``samplesPerPrompt`` is the in-character responses generated per
  // shared baseline prompt.
  let discoverKind: DiscoverKind = $state("abstract");
  let discoverSamplesPerPrompt = $state(1);
  let discoverMaxDim = $state(8);
  let discoverVarThreshold = $state(0.70);
  let discoverKNN: number | null = $state(null);
  let discoverBandwidth: number | null = $state(null);
  let discoverForce = $state(false);
  // Persona-manifold opt-in: when set, each concept slug doubles as the
  // matching node's assistant-role substitution at fit time, producing a
  // role-paired manifold (steering through it implies the nearest
  // node's role at decode time).  The slug regex matches the engine's
  // role validation — concepts that pass ``slug()`` are already in
  // ``[a-z0-9._-]+``.
  let discoverRolePerNode = $state(false);
  let discoverProgress: string | null = $state(null);
  let alsoFit = $state(true);

  function parseConcepts(text: string): string[] {
    return text
      .split(/[\s,]+/)
      .map((s) => s.trim())
      .filter(Boolean);
  }

  const discoverConcepts = $derived(parseConcepts(discoverConceptsText));

  const discoverValidation = $derived.by<{
    ok: boolean;
    messages: string[];
  }>(() => {
    const messages: string[] = [];
    if (!slug(manifoldName)) {
      messages.push("a manifold name is required");
    }
    if (discoverConcepts.length < 2) {
      messages.push(
        `need >= 2 concepts (have ${discoverConcepts.length}) — ` +
        "a manifold layout needs at least two nodes",
      );
    }
    const seen = new Set<string>();
    for (const c of discoverConcepts) {
      const s = slug(c);
      if (!s) {
        messages.push(`bad concept slug: "${c}"`);
      } else if (seen.has(s)) {
        messages.push(`duplicate concept: "${s}"`);
      } else {
        seen.add(s);
      }
    }
    if (discoverSamplesPerPrompt <= 0) {
      messages.push("samples_per_prompt must be > 0");
    }
    if (discoverMaxDim < 1) messages.push("max_dim must be >= 1");
    if (
      (discoverFitMode === "pca" || discoverFitMode === "auto") &&
      (discoverVarThreshold <= 0 || discoverVarThreshold > 1)
    ) {
      messages.push("var_threshold must be in (0, 1]");
    }
    return { ok: messages.length === 0, messages };
  });

  function buildDiscoverHyperparams(): Record<string, number> {
    // ``max_dim`` (the layout-dim cap) is the one knob every mode honors.
    // The rest are method-specific; ``auto`` accepts the union (the server
    // sanitizer drops whichever the resolved geometry doesn't consume), so
    // we forward every knob the user actually set and let the backend
    // fill data-driven defaults for the rest (median k-NN distance,
    // ``max(5, ceil(log K))``).
    const hp: Record<string, number> = { max_dim: discoverMaxDim };
    if (discoverFitMode === "pca" || discoverFitMode === "auto") {
      hp.var_threshold = discoverVarThreshold;
    }
    if (discoverFitMode === "spectral" || discoverFitMode === "auto") {
      if (discoverKNN !== null && discoverKNN > 0) hp.k_nn = discoverKNN;
      if (discoverBandwidth !== null && discoverBandwidth > 0) {
        hp.bandwidth = discoverBandwidth;
      }
    }
    return hp;
  }

  async function discoverSave(): Promise<void> {
    if (!discoverValidation.ok || submitting) return;
    submitting = true;
    discoverProgress = null;
    const namespaceSlug = slug(namespace) || "local";
    const nameSlug = slug(manifoldName);
    const req: GenerateManifoldRequest = {
      namespace: namespaceSlug,
      name: nameSlug,
      description: description.trim(),
      concepts: discoverConcepts.map((c) => slug(c)),
      kind: discoverKind,
      samples_per_prompt: discoverSamplesPerPrompt,
      fit_mode: discoverFitMode,
      hyperparams: buildDiscoverHyperparams(),
      force: discoverForce,
      role_per_node: discoverRolePerNode,
    };
    const toastId = pushToast(
      `generating ${namespaceSlug}/${nameSlug} corpora…`,
      { kind: "info", ttlMs: null },
    );
    try {
      await apiManifoldGenerateStream(req, (ev) => {
        if (ev.event === "progress") {
          const msg =
            ev.data && typeof ev.data === "object"
              ? (ev.data as { message?: string }).message
              : null;
          if (msg) {
            discoverProgress = msg;
            updateToast(toastId, { detail: msg });
          }
        }
      });
      dismissToast(toastId);
      if (alsoFit) {
        // Chain the fit immediately — the user opted into the
        // two-step.  The fit endpoint accepts the discover-mode
        // hyperparams as an override; passing them here keeps the
        // sidecar metadata in sync even if the folder already had
        // matching values from generate.
        const fitToastId = pushToast(
          `fitting ${namespaceSlug}/${nameSlug}…`,
          { kind: "info", ttlMs: null },
        );
        try {
          await apiManifoldFitStream(
            namespaceSlug,
            nameSlug,
            {
              fit_mode: discoverFitMode,
              hyperparams: buildDiscoverHyperparams(),
            },
            (ev) => {
              if (ev.event === "progress") {
                const msg =
                  ev.data && typeof ev.data === "object"
                    ? (ev.data as { message?: string }).message
                    : null;
                if (msg) {
                  discoverProgress = msg;
                  updateToast(fitToastId, { detail: msg });
                }
              }
            },
          );
          dismissToast(fitToastId);
          pushToast(
            `fit ${namespaceSlug}/${nameSlug} (${discoverFitMode})`,
            { kind: "info" },
          );
        } catch (e) {
          dismissToast(fitToastId);
          pushToast(`fit failed — ${describeFitError(e)}`, {
            kind: "error",
            ttlMs: null,
          });
        }
      } else {
        pushToast(
          `generated ${namespaceSlug}/${nameSlug} — open manifolds drawer to fit`,
          { kind: "info" },
        );
      }
      await refreshManifoldList();
      closeDrawer();
      openDrawer("manifolds");
    } catch (e) {
      dismissToast(toastId);
      const msg = describeFitError(e);
      pushToast(`generate failed — ${msg}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      submitting = false;
      discoverProgress = null;
    }
  }

  // ============================================================ templated ===
  //
  // Slot + values + chat-turn templates. The slot fills (deterministically,
  // no model) across every template's assistant turn, so each value's node
  // corpus is the slot-filled assistant turns; the templates' user turns
  // become the per-manifold elicitation prompts the fit pools against.
  type TemplatedFitMode = "pca" | "spectral" | "auto";
  let templatedFitMode: TemplatedFitMode = $state("auto");
  let templatedSlot = $state("[DAY]");
  let templatedValuesText = $state("");
  let templatedPairs = $state<{ user: string; assistant: string }[]>([
    { user: "what day is it?", assistant: "today is [DAY]" },
  ]);
  let templatedMaxDim: number | null = $state(null);
  let advancedTemplatedOpen = $state(false);

  const templatedValues = $derived(parseConcepts(templatedValuesText));

  function addTemplatedPair(): void {
    templatedPairs = [...templatedPairs, { user: "", assistant: "" }];
  }
  function removeTemplatedPair(i: number): void {
    templatedPairs = templatedPairs.filter((_, idx) => idx !== i);
  }
  function nonEmptyTemplatedPairs(): { user: string; assistant: string }[] {
    return templatedPairs.filter((p) => p.user.trim() || p.assistant.trim());
  }

  const templatedValidation = $derived.by<{
    ok: boolean;
    messages: string[];
  }>(() => {
    const messages: string[] = [];
    if (!slug(manifoldName)) messages.push("a manifold name is required");
    const sl = templatedSlot.trim();
    if (!sl) messages.push("a slot token is required (e.g. [DAY])");
    if (templatedValues.length < 2) {
      messages.push(
        `need >= 2 values (have ${templatedValues.length}) — one node per value`,
      );
    }
    const seen = new Set<string>();
    for (const v of templatedValues) {
      const s = slug(v);
      if (!s) messages.push(`value "${v}" has no valid label slug`);
      else if (seen.has(s)) messages.push(`duplicate value label: "${s}"`);
      else seen.add(s);
    }
    const pairs = nonEmptyTemplatedPairs();
    if (pairs.length === 0) messages.push("need at least one template");
    pairs.forEach((p, i) => {
      if (!p.user.trim()) messages.push(`template ${i + 1}: user turn is empty`);
      if (!p.assistant.trim()) {
        messages.push(`template ${i + 1}: assistant turn is empty`);
      } else if (sl && !p.assistant.includes(sl)) {
        messages.push(
          `template ${i + 1}: assistant turn must contain the slot ${sl}`,
        );
      }
      if (sl && p.user.includes(sl)) {
        messages.push(
          `template ${i + 1}: user turn must not contain the slot ` +
          "(user turns are shared common-mode across nodes)",
        );
      }
    });
    if (templatedMaxDim !== null && templatedMaxDim < 1) {
      messages.push("max_dim must be >= 1");
    }
    return { ok: messages.length === 0, messages };
  });

  async function templatedSave(): Promise<void> {
    if (!templatedValidation.ok || submitting) return;
    submitting = true;
    const namespaceSlug = slug(namespace) || "local";
    const nameSlug = slug(manifoldName);
    const hyperparams: Record<string, number> = {};
    if (templatedMaxDim !== null && templatedMaxDim >= 1) {
      hyperparams.max_dim = templatedMaxDim;
    }
    const req: CreateTemplatedManifoldRequest = {
      namespace: namespaceSlug,
      name: nameSlug,
      description: description.trim(),
      fit_mode: templatedFitMode,
      slot: templatedSlot.trim(),
      values: templatedValues,
      pairs: nonEmptyTemplatedPairs().map((p) => ({
        user: p.user,
        assistant: p.assistant,
      })),
      hyperparams,
    };
    const toastId = pushToast(
      `authoring ${namespaceSlug}/${nameSlug}…`,
      { kind: "info", ttlMs: null },
    );
    try {
      await apiManifolds.createTemplated(req);
      dismissToast(toastId);
      if (alsoFit) {
        const fitToastId = pushToast(
          `fitting ${namespaceSlug}/${nameSlug}…`,
          { kind: "info", ttlMs: null },
        );
        try {
          await apiManifoldFitStream(
            namespaceSlug,
            nameSlug,
            { fit_mode: templatedFitMode, hyperparams },
            (ev) => {
              if (ev.event === "progress") {
                const msg =
                  ev.data && typeof ev.data === "object"
                    ? (ev.data as { message?: string }).message
                    : null;
                if (msg) updateToast(fitToastId, { detail: msg });
              }
            },
          );
          dismissToast(fitToastId);
          pushToast(
            `fit ${namespaceSlug}/${nameSlug} (${templatedFitMode})`,
            { kind: "info" },
          );
        } catch (e) {
          dismissToast(fitToastId);
          pushToast(`fit failed — ${describeFitError(e)}`, {
            kind: "error",
            ttlMs: null,
          });
        }
      } else {
        pushToast(
          `authored ${namespaceSlug}/${nameSlug} — open manifolds drawer to fit`,
          { kind: "info" },
        );
      }
      await refreshManifoldList();
      closeDrawer();
      openDrawer("manifolds");
    } catch (e) {
      dismissToast(toastId);
      pushToast(`author failed — ${describeFitError(e)}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      submitting = false;
    }
  }

  function describeFitError(e: unknown): string {
    if (e instanceof ApiError) {
      return e.body && typeof e.body === "object" && "detail" in (e.body as object)
        ? String((e.body as { detail: unknown }).detail)
        : e.message;
    }
    return e instanceof Error ? e.message : String(e);
  }
</script>

<section class="drawer-shell" aria-label="Build manifold">
  <header class="header">
    <span class="title">build manifold</span>
    <button type="button" class="close" aria-label="Close" onclick={cancel}
      >✕</button>
  </header>

  <div class="body">
    <!-- mode picker — "custom nodes" (user-authored coords + corpora)
         vs "auto-generated" (LLM-authored corpora; coords derived). -->
    <ModeTabs
      bind:value={authoringMode}
      tabs={[
        { value: "discover", label: "auto-generated" },
        { value: "templated", label: "templated" },
        { value: "authored", label: "custom nodes" },
      ]}
      ariaLabel="Authoring mode"
    />

    {#if authoringMode === "authored"}
      <p class="hint">
        author a steering manifold: pick a domain, place labelled nodes
        with a statement corpus each, then fit it from the manifolds
        drawer.
      </p>
    {:else if authoringMode === "discover"}
      <p class="hint">
        hand the model a flat list of concepts; each one answers a shared
        set of baseline prompts in character (one corpus per node), then
        the fitter derives node coordinates per-model. leave the fit method
        on auto unless you know you want flat (pca) or curved (spectral).
        recommended at 20–48 concepts.
      </p>
    {:else}
      <p class="hint">
        give a slot token, a list of values (one node each), and a set of
        (user, assistant) templates with the slot in the assistant turn.
        the slot fills per value — deterministic, no model — and the
        templates' user turns become the elicitation prompts the fit pools
        against. the tool for categories you reference rather than embody
        (days, months, colours, directions); fit_mode auto detects cyclic
        layouts.
      </p>
    {/if}

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

    {#if authoringMode === "authored"}
    <!-- auto-domain switch: when on, skip the box/sphere picker and the
         per-node coord inputs; the fitter derives the layout per-model
         via pca / spectral.  When off, hand-author coords as before. -->
    <span class="auto-domain-toggle">
      <Checkbox
        bind:checked={autoDomain}
        label="auto-domain (let the fitter derive coords from corpora)"
      />
    </span>

    {#if autoDomain}
      <!-- fit-method picker — mirrors the auto-generated tab's choice. -->
      <section class="step">
        <h2 class="step-title">fit method</h2>
        <div class="radio-row">
          <Radio bind:group={discoverFitMode} value="auto" label="auto" />
          <Radio bind:group={discoverFitMode} value="pca" label="pca" />
          <Radio bind:group={discoverFitMode} value="spectral" label="spectral" />
        </div>
        <p class="dim-note">
          {#if discoverFitMode === "auto"}
            recommended — lets the fitter pick flat / curved per-model and
            detect periodic axes. the safe choice when you're not sure which
            geometry your concepts want.
          {:else if discoverFitMode === "pca"}
            flat linear layout — picks the smallest prefix whose cumulative
            variance crosses the threshold, capped at max_dim.
          {:else}
            laplacian eigenmaps — recovers curved-manifold topology that
            pca flattens. noisy below ~50 nodes.
          {/if}
        </p>
      </section>
    {:else}
    <!-- domain step -->
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
          <span class="label">sphere dimension (S^n)</span>
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
        intrinsic dimension: <strong>{intrinsicDim}</strong> ·
        minimum nodes: <strong>{minNodes}</strong>
      </p>
    </section>
    {/if}

    <!-- node editor -->
    <section class="step">
      <h2 class="step-title">nodes</h2>
      {#if nodes.length === 0}
        <p class="muted">
          no nodes yet — add at least {autoDomain ? 2 : minNodes}.
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
                <span class="label">
                  role <span class="opt">optional — role-augmented baseline</span>
                </span>
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
                  placeholder="e.g. pirate — leave empty for standard assistant baseline"
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

    {#if autoDomain}
      <AdvancedSection bind:expanded={advancedAuthoredOpen}>
        <div class="grid2">
          <label class="field">
            <span class="label">max_dim</span>
            <NumberInput
              value={discoverMaxDim}
              min={1}
              step={1}
              oninput={(v) => {
                if (v !== null) discoverMaxDim = v;
              }}
            />
          </label>
          {#if discoverFitMode === "pca" || discoverFitMode === "auto"}
            <label class="field">
              <span class="label">var_threshold</span>
              <NumberInput
                value={discoverVarThreshold}
                min={0}
                max={1}
                step={0.05}
                oninput={(v) => {
                  if (v !== null) discoverVarThreshold = v;
                }}
              />
            </label>
          {:else}
            <label class="field">
              <span class="label">k_nn (blank → auto)</span>
              <NumberInput
                value={discoverKNN}
                min={1}
                step={1}
                allowEmpty
                placeholder="max(5, ⌈log K⌉)"
                oninput={(v) => { discoverKNN = v; }}
              />
            </label>
          {/if}
        </div>
        {#if discoverFitMode === "spectral"}
          <label class="field">
            <span class="label">bandwidth σ (blank → median k-NN distance)</span>
            <NumberInput
              value={discoverBandwidth}
              min={0}
              step={0.01}
              allowEmpty
              placeholder="median(k-NN edges)"
              oninput={(v) => { discoverBandwidth = v; }}
            />
          </label>
        {/if}
      </AdvancedSection>
    {/if}

    <ValidationBlock verb="build" messages={validation.messages} />

    <button
      type="button"
      class="save-btn"
      disabled={!validation.ok || submitting}
      onclick={save}
    >
      {submitting
        ? "building…"
        : autoDomain
          ? `build manifold (auto-domain, ${discoverFitMode} fit) → return to list`
          : "build manifold → return to list"}
    </button>
    {:else if authoringMode === "discover"}
      <!-- auto-generated authoring: LLM-author corpora from a flat
           concept list, fitter derives coords. -->
      <section class="step">
        <h2 class="step-title">concepts</h2>
        <label class="field">
          <span class="label">
            concept list <span class="req">required (≥2)</span>
          </span>
          <textarea
            class="input"
            rows="4"
            placeholder={"pirate caveman assistant scholar robot\n(whitespace or comma-separated)"}
            bind:value={discoverConceptsText}
            spellcheck="false"
          ></textarea>
          <span class="dim-note">
            parsed: <strong>{discoverConcepts.length}</strong> concept(s)
          </span>
        </label>
        <div class="grid2">
          <div class="field">
            <span class="label">kind</span>
            <div class="radio-row">
              <Radio bind:group={discoverKind} value="abstract" label="abstract" />
              <Radio bind:group={discoverKind} value="concrete" label="concrete" />
            </div>
          </div>
          <label class="field">
            <span class="label">samples per prompt</span>
            <NumberInput
              value={discoverSamplesPerPrompt}
              min={1}
              step={1}
              oninput={(v) => {
                if (v !== null) discoverSamplesPerPrompt = v;
              }}
            />
          </label>
        </div>
        <p class="dim-note">
          each concept answers the shared baseline prompts in character —
          <strong>{discoverKind}</strong> framing,
          <strong>{discoverSamplesPerPrompt}</strong> response(s) per prompt.
        </p>
      </section>

      <section class="step">
        <h2 class="step-title">fit method</h2>
        <div class="radio-row">
          <Radio bind:group={discoverFitMode} value="auto" label="auto" />
          <Radio bind:group={discoverFitMode} value="pca" label="pca" />
          <Radio bind:group={discoverFitMode} value="spectral" label="spectral" />
        </div>
        <p class="dim-note">
          {#if discoverFitMode === "auto"}
            recommended — lets the fitter pick flat / curved per-model and
            detect periodic axes. the safe choice when you're not sure which
            geometry your concepts want.
          {:else if discoverFitMode === "pca"}
            flat linear layout — picks the smallest prefix whose cumulative
            variance crosses the threshold, capped at max_dim. a good fit
            for most concept heaps.
          {:else}
            laplacian eigenmaps — recovers curved-manifold topology
            that pca flattens. noisy below ~50 nodes.
          {/if}
        </p>
      </section>

      <AdvancedSection bind:expanded={advancedDiscoverOpen}>
        <div class="grid2">
          <label class="field">
            <span class="label">max_dim</span>
            <NumberInput
              value={discoverMaxDim}
              min={1}
              step={1}
              oninput={(v) => {
                if (v !== null) discoverMaxDim = v;
              }}
            />
          </label>
          {#if discoverFitMode === "pca" || discoverFitMode === "auto"}
            <label class="field">
              <span class="label">var_threshold</span>
              <NumberInput
                value={discoverVarThreshold}
                min={0}
                max={1}
                step={0.05}
                oninput={(v) => {
                  if (v !== null) discoverVarThreshold = v;
                }}
              />
            </label>
          {:else}
            <label class="field">
              <span class="label">k_nn (blank → auto)</span>
              <NumberInput
                value={discoverKNN}
                min={1}
                step={1}
                allowEmpty
                placeholder="max(5, ⌈log K⌉)"
                oninput={(v) => {
                  discoverKNN = v;
                }}
              />
            </label>
          {/if}
        </div>
        {#if discoverFitMode === "spectral"}
          <label class="field">
            <span class="label">bandwidth σ (blank → median k-NN distance)</span>
            <NumberInput
              value={discoverBandwidth}
              min={0}
              step={0.01}
              allowEmpty
              placeholder="median(k-NN edges)"
              oninput={(v) => {
                discoverBandwidth = v;
              }}
            />
          </label>
        {/if}
        <div class="check-stack">
          <Checkbox
            bind:checked={alsoFit}
            label="fit immediately after generating corpora"
          />
          <Checkbox
            bind:checked={discoverRolePerNode}
            label="persona manifold (use each concept slug as that node's role)"
          />
          {#if discoverRolePerNode}
            <p class="role-hint">
              Each node's centroid will be pooled with the chat template's
              assistant-role label replaced by the concept slug. The fitted
              manifold lives in persona-baseline activation space; steering
              through it implies the nearest node's role at decode time.
              Mistral-3 / talkie families don't support role substitution
              and raise at fit time.
            </p>
          {/if}
          <Checkbox
            bind:checked={discoverForce}
            label="overwrite an existing manifold with this name"
          />
        </div>
      </AdvancedSection>

      {#if discoverProgress}
        <p class="progress">{discoverProgress}</p>
      {/if}

      <ValidationBlock verb="generate" messages={discoverValidation.messages} />

      <button
        type="button"
        class="save-btn"
        disabled={!discoverValidation.ok || submitting}
        onclick={discoverSave}
      >
        {submitting
          ? "generating…"
          : alsoFit
            ? "generate corpora + fit"
            : "generate corpora"}
      </button>
    {:else}
      <!-- templated authoring: slot + values + chat-turn templates;
           deterministic fill (no model), then derive coords. -->
      <section class="step">
        <h2 class="step-title">slot + values</h2>
        <label class="field">
          <span class="label">slot token <span class="req">required</span></span>
          <input
            type="text"
            class="input"
            bind:value={templatedSlot}
            placeholder="[DAY]"
            autocomplete="off"
            spellcheck="false"
          />
          <span class="dim-note">
            the placeholder filled per value — lives in the assistant turn only
          </span>
        </label>
        <label class="field">
          <span class="label">
            values <span class="req">required (≥2)</span>
          </span>
          <textarea
            class="input"
            rows="3"
            placeholder={"Monday Tuesday Wednesday Thursday Friday Saturday Sunday\n(whitespace or comma-separated)"}
            bind:value={templatedValuesText}
            spellcheck="false"
          ></textarea>
          <span class="dim-note">
            parsed: <strong>{templatedValues.length}</strong> value(s) → one node each
          </span>
        </label>
      </section>

      <section class="step">
        <h2 class="step-title">templates</h2>
        <p class="dim-note">
          {templatedPairs.length} template(s) — each a (user, assistant) turn;
          the slot fills in the assistant turn. more templates = more samples
          per node (aim ~10–50).
        </p>
        {#each templatedPairs as pair, i (i)}
          <div class="pair-row">
            <div class="pair-fields">
              <input
                type="text"
                class="input"
                bind:value={pair.user}
                placeholder="user — e.g. what day is it?"
                spellcheck="false"
              />
              <input
                type="text"
                class="input"
                bind:value={pair.assistant}
                placeholder={`assistant — e.g. today is ${templatedSlot || "[SLOT]"}`}
                spellcheck="false"
              />
            </div>
            <button
              type="button"
              class="pair-remove"
              aria-label="Remove template"
              onclick={() => removeTemplatedPair(i)}
              disabled={templatedPairs.length <= 1}
            >✕</button>
          </div>
        {/each}
        <button type="button" class="add-pair-btn" onclick={addTemplatedPair}>
          + add template
        </button>
      </section>

      <section class="step">
        <h2 class="step-title">fit method</h2>
        <div class="radio-row">
          <Radio bind:group={templatedFitMode} value="auto" label="auto" />
          <Radio bind:group={templatedFitMode} value="pca" label="pca" />
          <Radio bind:group={templatedFitMode} value="spectral" label="spectral" />
        </div>
        <p class="dim-note">
          {#if templatedFitMode === "auto"}
            recommended — picks flat / curved per-model and detects periodic
            axes (the right default for cyclic categories like days / months).
          {:else if templatedFitMode === "pca"}
            flat linear layout — pins a straight affine arrangement of the
            values.
          {:else}
            laplacian eigenmaps — pins a curved RBF surface (use auto if you
            want periodic axes detected for cyclic categories).
          {/if}
        </p>
      </section>

      <AdvancedSection bind:expanded={advancedTemplatedOpen}>
        <label class="field">
          <span class="label">max_dim (blank → auto)</span>
          <NumberInput
            value={templatedMaxDim}
            min={1}
            step={1}
            allowEmpty
            placeholder="auto"
            oninput={(v) => {
              templatedMaxDim = v;
            }}
          />
        </label>
        <div class="check-stack">
          <Checkbox
            bind:checked={alsoFit}
            label="fit immediately after authoring"
          />
        </div>
      </AdvancedSection>

      <ValidationBlock verb="author" messages={templatedValidation.messages} />

      <button
        type="button"
        class="save-btn"
        disabled={!templatedValidation.ok || submitting}
        onclick={templatedSave}
      >
        {submitting ? "authoring…" : alsoFit ? "author + fit" : "author"}
      </button>
    {/if}
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
  .node-role {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    padding: var(--space-2) var(--space-3) 0;
  }
  .role-hint {
    margin: var(--space-1) 0 0 var(--space-5);
    color: var(--fg-dim);
    font-size: var(--text-xs);
    line-height: 1.4;
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
  /* Wraps the themed NumberInput in the node-coords flex row. */
  .coord-cell {
    flex: 0 0 auto;
    width: 4.5em;
    display: inline-flex;
  }
  /* Discover-tab radio pair (was .domain-kind / .kind-btn). */
  .radio-row {
    display: flex;
    gap: var(--space-5);
    padding: var(--space-1) 0 var(--space-3);
  }
  /* Discover-tab checkbox group at the bottom of the form. */
  .check-stack {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
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

  /* Templated-tab (user, assistant) template editor rows. */
  .pair-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: var(--space-2);
  }
  .pair-fields {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    flex: 1;
    min-width: 0;
  }
  .pair-remove {
    background: transparent;
    border: 0;
    color: var(--fg-muted);
    cursor: pointer;
    font-size: var(--text);
    padding: 0 var(--space-2);
  }
  .pair-remove:hover:not(:disabled) {
    color: var(--accent-red);
  }
  .pair-remove:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .add-pair-btn {
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
  .add-pair-btn:hover {
    background: var(--accent-glow);
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

  /* Spacing for the auto-domain checkbox row inside the custom-nodes tab. */
  .auto-domain-toggle {
    display: inline-flex;
    padding: var(--space-3) 0;
  }
  textarea.input {
    resize: vertical;
    line-height: 1.4;
    min-height: 4em;
  }
  .progress {
    margin: 0;
    color: var(--accent);
    font-size: var(--text-xs);
    font-style: italic;
  }
</style>
