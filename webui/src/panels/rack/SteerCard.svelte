<script lang="ts">
  // Unified steer card — one row for every steering term.  A steering vector
  // is the K=2 flat case of a manifold, so this single card replaces the
  // VectorSteerCard / ManifoldSteerCard split, branching on ``entry.mode``:
  //
  //   vector   → a bipolar (or monopolar) concept axis / pole-DiM term.
  //              statline: ●/○ · name · !ablate · variant chip · trigger
  //                        · projection tag · ⋮ menu · ✕
  //              body:     neg pole · Slider(α) · pos pole · signed α
  //   position → a placement on a fitted manifold (flat fan or curved
  //              surface).  Flat (pca/baked) wears the subspace accent (●/○,
  //              --accent); curved (spectral/authored) the manifold accent
  //              (◆/◇, --accent-purple).
  //              statline: glyph · name · unfitted/stale warn · trigger · ✕
  //              body:     snap-to-node Select · XYPad · along [· onto]
  //
  // The two modes carry disjoint control fields (the grammar forbids ``%``
  // composing with ``~``/``|``/``!``), so the bodies don't share affordances
  // — but the chrome, trigger pill, enable/remove, and the RackCard shell are
  // common.  ``v`` / ``p`` are the narrowed entry views the template renders
  // behind ``{#if v}`` / ``{#if p}`` so svelte-check enforces mode-correct
  // field access.

  import { onMount } from "svelte";
  import type { ProjectionSpec, Trigger } from "../../lib/types";
  import type { Variant } from "../../lib/types";
  import type { SteerEntry, VectorSteerEntry, PositionSteerEntry } from "../../lib/types";
  import {
    setVectorAlpha,
    setVectorEnabled,
    setVectorTrigger,
    setVectorVariant,
    setVectorProjection,
    setVectorAblate,
    addVectorToRack,
    removeVectorFromRack,
    setManifoldBlend,
    setManifoldOnto,
    setManifoldCoords,
    setManifoldLabel,
    setManifoldTrigger,
    setManifoldEnabled,
    removeManifoldFromRack,
    manifoldByName,
    steerRack,
  } from "../../lib/stores.svelte";
  import { pushToast } from "../../lib/stores/toasts.svelte";
  import { serializeExpression } from "../../lib/expression";
  import { apiVectors } from "../../lib/api";
  import { polesOf } from "../../lib/concepts";
  import Slider from "../../lib/Slider.svelte";
  import Select from "../../lib/Select.svelte";
  import XYPad from "../manifold/XYPad.svelte";
  import RackCard from "./RackCard.svelte";

  interface Props {
    name: string;
    entry: SteerEntry;
  }

  let { name, entry }: Props = $props();

  // Narrowed views — exactly one is non-null per entry.
  const v = $derived<VectorSteerEntry | null>(
    entry.mode === "vector" ? entry : null,
  );
  const p = $derived<PositionSteerEntry | null>(
    entry.mode === "position" ? entry : null,
  );

  // ---------- family chrome (accent + enable glyph) ----------

  /** Catalog row for a position term (drives flat/curved + node list). */
  const info = $derived(entry.mode === "position" ? manifoldByName(name) : null);
  /** Flat (affine) iff the fit is a discover-pca or baked direction. */
  const flat = $derived(info?.fit_mode === "pca" || info?.fit_mode === "baked");
  /** Subspace family = every vector term plus every flat position term. */
  const subspace = $derived(entry.mode === "vector" || flat);

  const accent = $derived(subspace ? "--accent" : "--accent-purple");
  const enableGlyph = $derived(
    subspace ? (entry.enabled ? "●" : "○") : (entry.enabled ? "◆" : "◇"),
  );

  /** Display name — bare name with the namespace prefix stripped
   *  (``default/personas`` → ``personas``).  Full name stays in the tooltip. */
  const displayName = $derived(name.split("/").pop() ?? name);

  // ---------- bipolar axis (vector mode) ----------

  const poles = $derived(polesOf(name));
  const monopolar = $derived(poles.negative === null);
  const alpha = $derived(v?.alpha ?? 0);

  function snapAlpha(raw: number): number {
    if (Math.abs(raw) <= 0.025) return 0;
    return raw;
  }
  function onSliderInput(val: number): void {
    if (!Number.isFinite(val)) return;
    setVectorAlpha(name, snapAlpha(val));
  }
  function formatAlpha(a: number): string {
    if (a === 0) return "0.00";
    const sign = a > 0 ? "+" : "-";
    return `${sign}${Math.abs(a).toFixed(2)}`;
  }
  const alphaColor = $derived.by(() => {
    if (alpha > 0) return "var(--accent-green)";
    if (alpha < 0) return "var(--accent-red)";
    return "var(--fg-muted)";
  });

  // ---------- trigger cycle (shared) ----------

  const TRIGGER_ORDER: Trigger[] = [
    "BOTH",
    "BEFORE",
    "AFTER",
    "THINKING",
    "RESPONSE",
    "PROMPT",
    "GENERATED",
  ];
  const TRIGGER_WORD: Record<Trigger, string> = {
    BOTH: "both",
    BEFORE: "before",
    AFTER: "after",
    THINKING: "thinking",
    RESPONSE: "response",
    PROMPT: "prompt",
    GENERATED: "generated",
  };
  const TRIGGER_LABEL: Record<Trigger, string> = {
    BOTH: "both: steer the whole turn (default)",
    BEFORE: "before: steer thinking and response",
    AFTER: "after: steer the after-thinking response only",
    THINKING: "thinking: steer the chain-of-thought only",
    RESPONSE: "response: steer the generated response only",
    PROMPT: "prompt (alias of before)",
    GENERATED: "generated (alias of response)",
  };

  function cycleTrigger(): void {
    const idx = TRIGGER_ORDER.indexOf(entry.trigger);
    const next = TRIGGER_ORDER[(idx + 1) % TRIGGER_ORDER.length];
    if (entry.mode === "vector") setVectorTrigger(name, next);
    else setManifoldTrigger(name, next);
  }

  function toggleEnabled(): void {
    if (entry.mode === "vector") setVectorEnabled(name, !entry.enabled);
    else setManifoldEnabled(name, !entry.enabled);
  }

  // ---------- variant dropdown (vector mode) ----------

  let variantOpen = $state(false);
  let variantRef: HTMLDivElement | null = $state(null);

  const variantOptions = $derived.by<Variant[]>(() => {
    const opts: Variant[] = ["raw", "sae", "role", "from"];
    if (v && !opts.includes(v.variant)) opts.push(v.variant);
    return opts;
  });

  function pickVariant(variant: Variant): void {
    variantOpen = false;
    setVectorVariant(name, variant);
  }

  // ---------- ⋮ menu (vector mode) ----------

  let menuOpen = $state(false);
  let menuRef: HTMLDivElement | null = $state(null);

  function onDocClick(ev: MouseEvent): void {
    const t = ev.target as Node;
    if (menuOpen && menuRef && !menuRef.contains(t)) menuOpen = false;
    if (variantOpen && variantRef && !variantRef.contains(t)) variantOpen = false;
  }
  function onDocKey(ev: KeyboardEvent): void {
    if (ev.key !== "Escape") return;
    if (menuOpen) menuOpen = false;
    if (variantOpen) variantOpen = false;
  }

  onMount(() => {
    document.addEventListener("click", onDocClick);
    document.addEventListener("keydown", onDocKey);
    return () => {
      document.removeEventListener("click", onDocClick);
      document.removeEventListener("keydown", onDocKey);
    };
  });

  function toggleMenu(ev: MouseEvent): void {
    ev.stopPropagation();
    menuOpen = !menuOpen;
    variantOpen = false;
  }
  function toggleVariant(ev: MouseEvent): void {
    ev.stopPropagation();
    variantOpen = !variantOpen;
    menuOpen = false;
  }

  // ---------- inline projection modal (vector mode) ----------

  let projectionPromptOp = $state<ProjectionSpec["op"] | null>(null);
  let projectionTargetDraft = $state("");
  let projectionInputRef: HTMLInputElement | null = $state(null);

  function pickProjection(op: ProjectionSpec["op"]): void {
    menuOpen = false;
    if (v?.projection && v.projection.op === op) {
      setVectorProjection(name, null);
      return;
    }
    projectionTargetDraft = v?.projection?.target ?? "";
    projectionPromptOp = op;
    queueMicrotask(() => projectionInputRef?.focus());
  }
  function cancelProjection(): void {
    projectionPromptOp = null;
    projectionTargetDraft = "";
  }
  function confirmProjection(): void {
    const op = projectionPromptOp;
    if (op === null) return;
    const target = projectionTargetDraft.trim();
    projectionPromptOp = null;
    projectionTargetDraft = "";
    if (!target) return;
    setVectorProjection(name, { op, target });
  }
  function onProjectionKey(ev: KeyboardEvent): void {
    if (ev.key === "Enter") {
      ev.preventDefault();
      confirmProjection();
    } else if (ev.key === "Escape") {
      ev.preventDefault();
      cancelProjection();
    }
  }

  function toggleAblate(): void {
    menuOpen = false;
    if (v) setVectorAblate(name, !v.ablate);
  }

  function duplicate(): void {
    menuOpen = false;
    if (!v) return;
    let candidate = `${name}-copy`;
    let n = 2;
    while (steerRack.entries.has(candidate)) candidate = `${name}-copy-${n++}`;
    addVectorToRack(candidate, v.alpha, v.trigger);
    const fresh = steerRack.entries.get(candidate);
    if (fresh && fresh.mode === "vector") {
      fresh.variant = v.variant;
      fresh.projection = v.projection
        ? { op: v.projection.op, target: v.projection.target }
        : null;
      fresh.ablate = v.ablate;
      fresh.enabled = v.enabled;
    }
  }

  async function copyTermExpression(): Promise<void> {
    menuOpen = false;
    const oneRack = new Map<string, SteerEntry>();
    oneRack.set(name, { ...entry, enabled: true });
    const expr = serializeExpression(oneRack);
    try {
      await navigator.clipboard.writeText(expr);
      pushToast(`copied: ${expr}`, { kind: "info", ttlMs: 3000 });
    } catch {
      pushToast("clipboard blocked by the browser", { kind: "warning" });
    }
  }

  const projectionGlyph = $derived.by(() => {
    if (!v?.projection) return null;
    return `${v.projection.op} ${v.projection.target}`;
  });

  // ---------- removal (shared) ----------

  async function removeTerm(): Promise<void> {
    if (entry.mode === "vector") {
      removeVectorFromRack(name);
      try {
        await apiVectors.delete(name);
      } catch {
        /* ignore — the rack is the user-visible source of truth. */
      }
    } else {
      removeManifoldFromRack(name);
    }
  }

  // ---------- position controls (position mode) ----------

  function onBlendInput(val: number): void {
    if (!Number.isFinite(val)) return;
    setManifoldBlend(name, val);
  }
  function onOntoInput(val: number): void {
    if (!Number.isFinite(val)) return;
    setManifoldOnto(name, val);
  }
  function onCoordsChange(coords: number[]): void {
    setManifoldCoords(name, coords);
  }
  function onSnapToNode(val: string): void {
    setManifoldLabel(name, val === "" ? null : val);
  }

  const snapOptions = $derived.by<{ value: string; label: string }[]>(() => {
    const labels = info?.node_labels ?? [];
    const roles = info?.node_roles;
    const opts = labels.map((nl, i) => {
      const role = roles?.[i];
      return { value: nl, label: nl + (role ? ` [role=${role}]` : "") };
    });
    return [{ value: "", label: "(free position)" }, ...opts];
  });

  // A manifold needs a tensor for the loaded model to actually steer.
  const fitted = $derived(info?.fitted_for_session === true);
  const stale = $derived(info?.stale === true);
</script>

<RackCard {accent} disabled={!entry.enabled}>
  {#snippet statline()}
    <button
      type="button"
      class="enable"
      class:off={!entry.enabled}
      onclick={toggleEnabled}
      title={entry.enabled ? "Enabled (click to disable)" : "Disabled (click to enable)"}
      aria-pressed={entry.enabled}
      aria-label="Toggle steering for {name}"
    >
      {enableGlyph}
    </button>

    <span class="name" class:struck={!entry.enabled} title={entry.mode === "vector" ? `concept ${name}` : `manifold ${name}`}>
      {entry.mode === "vector" ? name : displayName}
    </span>

    {#if v?.ablate}
      <span class="ablate-mark" title="ablation: concept removed from the residual stream">!</span>
    {/if}

    {#if p && !fitted}
      <span class="warn" title="no fitted tensor for the loaded model — fit it from the manifolds drawer">
        unfitted
      </span>
    {:else if p && stale}
      <span class="warn" title="the fitted tensor is stale — node geometry changed since the fit">
        stale
      </span>
    {/if}

    <span class="spacer"></span>

    {#if v}
      <div class="variant-wrap" bind:this={variantRef}>
        <button
          type="button"
          class="variant-chip"
          onclick={toggleVariant}
          aria-haspopup="menu"
          aria-expanded={variantOpen}
          title="tensor variant: {v.variant} (click to change)"
          aria-label="variant for {name}: {v.variant}"
        >
          {v.variant}
        </button>
        {#if variantOpen}
          <div class="variant-menu" role="menu">
            {#each variantOptions as opt (opt)}
              <button
                type="button"
                role="menuitemradio"
                aria-checked={v.variant === opt}
                class:active={v.variant === opt}
                onclick={() => pickVariant(opt)}
              >
                {opt}
              </button>
            {/each}
          </div>
        {/if}
      </div>
    {/if}

    <button
      type="button"
      class="trigger-pill"
      onclick={cycleTrigger}
      title="trigger: {TRIGGER_LABEL[entry.trigger]} (click to cycle)"
      aria-label="trigger for {name}: {entry.trigger}"
    >
      {TRIGGER_WORD[entry.trigger]}
    </button>

    {#if projectionGlyph}
      <span class="projection-tag" title="projection: {projectionGlyph}">
        {projectionGlyph}
      </span>
    {/if}

    {#if v}
      <div class="menu-wrap" bind:this={menuRef}>
        <button
          type="button"
          class="icon menu-btn"
          onclick={toggleMenu}
          aria-haspopup="menu"
          aria-expanded={menuOpen}
          aria-label="more actions for {name}"
          title="more actions"
        >
          ⋮
        </button>
        {#if menuOpen}
          <div class="menu" role="menu">
            <button
              type="button"
              role="menuitem"
              onclick={() => pickProjection("~")}
              disabled={v.ablate}
            >
              {v.projection?.op === "~"
                ? `clear projection (~ ${v.projection.target})`
                : "project onto (~)…"}
            </button>
            <button
              type="button"
              role="menuitem"
              onclick={() => pickProjection("|")}
              disabled={v.ablate}
            >
              {v.projection?.op === "|"
                ? `clear projection (| ${v.projection.target})`
                : "project orthogonal (|)…"}
            </button>
            <button type="button" role="menuitem" onclick={toggleAblate}>
              {v.ablate ? "remove ablation (!)" : "ablate (!)"}
            </button>
            <hr />
            <button type="button" role="menuitem" onclick={duplicate}>
              duplicate
            </button>
            <button type="button" role="menuitem" onclick={copyTermExpression}>
              copy expression
            </button>
          </div>
        {/if}
      </div>
    {/if}

    <button
      type="button"
      class="icon remove"
      onclick={removeTerm}
      aria-label="remove {name}"
      title="remove {name}"
    >
      ✕
    </button>
  {/snippet}

  {#snippet body()}
    {#if v}
      <!-- Bipolar axis frame.  Monopolar concepts render an empty left-pole
           slot rather than collapsing the grid, so a mixed mono/bipolar rack
           keeps slider starts + positive poles vertically aligned. -->
      <div class="axis" class:disabled={!entry.enabled}>
        <span class="pole neg" aria-hidden={monopolar}>
          {#if !monopolar}{poles.negative}{/if}
        </span>
        <Slider
          value={v.alpha}
          min={monopolar ? 0 : -1}
          max={1}
          step={0.05}
          oninput={onSliderInput}
          ariaLabel="strength (α) for {name}"
          title="strength (α) for {name}: drag, ±0.025 snaps to 0"
        />
        <span class="pole pos" title="positive pole (drag right)">
          {poles.positive}
        </span>
        <span
          class="alpha-display"
          style:color={alphaColor}
          title="strength (α): signed steering coefficient"
        >
          {formatAlpha(v.alpha)}
        </span>
      </div>
    {:else if p}
      <!-- Snap-to-node + free XYPad position, shared by flat + curved. -->
      {#if info}
        {#if info.node_labels.length > 0}
          <label class="ctl-row">
            <span class="ctl-label">snap to node</span>
            <span class="ctl-select">
              <Select
                value={p.label ?? ""}
                options={snapOptions}
                onchange={onSnapToNode}
                ariaLabel="snap to node"
                title="pick a node to switch to label-form, or '(free position)' to drag the pad"
              />
            </span>
          </label>
        {/if}
        <XYPad
          manifold={info}
          coords={p.coords}
          onchange={onCoordsChange}
          locked={p.label !== null}
        />
        <div class="along-row">
          <span class="along-label">along</span>
          <Slider
            value={p.blend}
            min={0}
            max={1}
            step={0.05}
            oninput={onBlendInput}
            ariaLabel="along fraction for {name}"
            title="along — how far to slide toward the manifold position"
          />
          <span class="along-val" title="along fraction">{p.blend.toFixed(2)}</span>
        </div>
        {#if !flat}
          <!-- onto — curved-only second coefficient: collapse the off-surface
               in-subspace residual onto the manifold.  Vacuous for flat. -->
          <div class="along-row">
            <span class="along-label">onto</span>
            <Slider
              value={p.onto}
              min={0}
              max={1}
              step={0.05}
              oninput={onOntoInput}
              ariaLabel="onto fraction for {name}"
              title="onto — collapse the off-surface residual onto the manifold (curved only)"
            />
            <span class="along-val" title="onto fraction">{p.onto.toFixed(2)}</span>
          </div>
        {/if}
      {:else}
        <p class="missing">
          manifold metadata unavailable — coordinates are still applied.
        </p>
      {/if}
    {/if}
  {/snippet}
</RackCard>

{#if projectionPromptOp !== null}
  <!-- Inline projection-target dialog. -->
  <div
    class="projection-backdrop"
    role="presentation"
    onclick={cancelProjection}
    onkeydown={onProjectionKey}
  >
    <div
      class="projection-modal"
      role="dialog"
      aria-modal="true"
      aria-label="Pick projection target"
      tabindex="-1"
      onclick={(ev) => ev.stopPropagation()}
      onkeydown={(ev) => ev.stopPropagation()}
    >
      <header class="projection-header">
        <span class="projection-title">
          {projectionPromptOp === "~"
            ? `project ${name} onto`
            : `project ${name} orthogonal to`}
        </span>
      </header>
      <input
        bind:this={projectionInputRef}
        bind:value={projectionTargetDraft}
        class="projection-input"
        placeholder="target concept name"
        spellcheck="false"
        autocomplete="off"
        onkeydown={onProjectionKey}
      />
      <footer class="projection-actions">
        <button
          type="button"
          class="projection-btn cancel"
          onclick={cancelProjection}
        >cancel</button>
        <button
          type="button"
          class="projection-btn confirm"
          onclick={confirmProjection}
          disabled={!projectionTargetDraft.trim()}
        >ok</button>
      </footer>
    </div>
  </div>
{/if}

<style>
  /* ----- statline pieces ----- */
  .enable {
    background: transparent;
    border: 0;
    padding: 0 var(--space-1);
    color: var(--card-accent);
    font-size: var(--text);
    line-height: 1;
    flex: 0 0 auto;
    cursor: pointer;
  }
  .enable.off {
    color: var(--fg-muted);
  }

  .name {
    color: var(--fg-strong);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .name.struck {
    text-decoration: line-through;
    color: var(--fg-muted);
  }

  .ablate-mark {
    color: var(--accent-purple);
    font-weight: var(--weight-bold);
    flex: 0 0 auto;
  }

  .warn {
    flex: 0 0 auto;
    color: var(--accent-yellow);
    font-size: var(--text-2xs);
    border: 1px solid var(--accent-yellow);
    border-radius: var(--radius);
    padding: 0 var(--space-2);
  }

  .spacer {
    flex: 1 1 auto;
    min-width: 0;
  }

  .trigger-pill,
  .variant-chip {
    background: transparent;
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: var(--space-1) var(--space-3);
    border-radius: var(--radius);
    font-size: var(--text-xs);
    line-height: 1.2;
    flex: 0 0 auto;
    cursor: pointer;
    transition: background var(--dur) var(--ease-out),
      border-color var(--dur) var(--ease-out);
  }
  .trigger-pill:hover,
  .variant-chip:hover {
    background: var(--bg-elev);
    border-color: var(--fg-muted);
  }
  .variant-chip {
    color: var(--card-accent);
  }

  .variant-wrap {
    position: relative;
    flex: 0 0 auto;
  }
  .variant-menu {
    position: absolute;
    right: 0;
    top: calc(100% + 4px);
    min-width: 7em;
    background: var(--surface-hi);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-1) 0;
    z-index: var(--z-modal);
    box-shadow: var(--shadow-overlay);
    display: flex;
    flex-direction: column;
  }
  .variant-menu button {
    background: transparent;
    border: 0;
    text-align: left;
    padding: var(--space-2) var(--space-4);
    color: var(--fg-strong);
    font-size: var(--text-sm);
    cursor: pointer;
  }
  .variant-menu button:hover {
    background: var(--bg-elev);
    color: var(--accent);
  }
  .variant-menu button.active {
    color: var(--accent);
  }

  .projection-tag {
    flex: 0 0 auto;
    color: var(--accent-yellow);
    font-size: var(--text-xs);
    border: 1px solid var(--border);
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius);
    max-width: 8em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .icon {
    background: transparent;
    border: 0;
    color: var(--fg-muted);
    font-size: var(--text);
    line-height: 1;
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius);
    flex: 0 0 auto;
    cursor: pointer;
    transition: color var(--dur) var(--ease-out),
      background var(--dur) var(--ease-out);
  }
  .icon:hover:not(:disabled) {
    color: var(--fg-strong);
    background: var(--bg-elev);
  }
  .remove:hover:not(:disabled) {
    color: var(--accent-red);
  }

  .menu-wrap {
    position: relative;
    flex: 0 0 auto;
  }
  .menu {
    position: absolute;
    right: 0;
    top: calc(100% + 4px);
    min-width: 200px;
    background: var(--surface-hi);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-2) 0;
    z-index: var(--z-modal);
    box-shadow: var(--shadow-overlay);
    display: flex;
    flex-direction: column;
  }
  .menu button {
    background: transparent;
    border: 0;
    text-align: left;
    padding: var(--space-4) var(--space-5);
    color: var(--fg-strong);
    font-size: var(--text-sm);
    cursor: pointer;
  }
  .menu button:hover:not(:disabled) {
    background: var(--bg-elev);
    color: var(--accent);
  }
  .menu button:disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
  }
  .menu hr {
    border: 0;
    border-top: 1px solid var(--border);
    margin: var(--space-1) 0;
  }

  /* ----- body: bipolar axis row (vector mode) ----- */
  .axis {
    display: grid;
    grid-template-columns: minmax(2.5em, 1fr) minmax(60px, 2.6fr) minmax(2.5em, 1fr) 3.5em;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .pole {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: var(--text-sm);
  }
  .axis.disabled .pole {
    text-decoration: line-through;
  }
  .pole.neg {
    color: var(--fg-muted);
    text-align: right;
  }
  .pole.pos {
    color: var(--fg-strong);
    text-align: left;
  }
  .alpha-display {
    flex: 0 0 auto;
    min-width: 3.5em;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  /* ----- body: position controls (position mode) ----- */
  .ctl-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    color: var(--fg-strong);
    font-size: var(--text-xs);
  }
  .ctl-label {
    color: var(--fg-muted);
    flex: 0 0 auto;
  }
  .ctl-select {
    flex: 1 1 auto;
    display: inline-flex;
  }
  .along-row {
    display: grid;
    grid-template-columns: minmax(3em, auto) 1fr 3em;
    align-items: center;
    gap: var(--space-2);
  }
  .along-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-transform: lowercase;
  }
  .along-val {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
  .missing {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }

  /* ----- projection modal ----- */
  .projection-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(1, 4, 9, 0.55);
    z-index: var(--z-modal);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .projection-modal {
    min-width: 360px;
    max-width: 480px;
    background: var(--surface-hi);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-5) var(--space-6);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    box-shadow: var(--shadow-overlay);
    font-family: var(--font-mono);
  }
  .projection-header {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
  }
  .projection-title {
    color: var(--fg-strong);
    font-size: var(--text-sm);
  }
  .projection-input {
    background: var(--bg-elev);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-size: var(--text-sm);
    font-family: var(--font-mono);
  }
  .projection-input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .projection-actions {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-3);
  }
  .projection-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--fg-strong);
    padding: var(--space-2) var(--space-5);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
    border-radius: var(--radius);
  }
  .projection-btn.cancel {
    color: var(--fg-dim);
  }
  .projection-btn.cancel:hover {
    color: var(--fg-strong);
    border-color: var(--fg-muted);
  }
  .projection-btn.confirm {
    color: var(--accent);
    border-color: var(--accent);
  }
  .projection-btn.confirm:hover:not(:disabled) {
    background: var(--accent-subtle);
  }
  .projection-btn.confirm:disabled {
    color: var(--fg-muted);
    border-color: var(--border);
    cursor: not-allowed;
  }
</style>
