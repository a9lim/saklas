<script lang="ts">
  // J-LENS — the inspector column's Jacobian-lens tab: the open- and
  // closed-vocabulary views of the workspace readout, card-based and
  // symmetric with the CAA tab (every row wears RackCard; the
  // j-lens family accent is blue, marker ■/□).
  //
  //   STEER    — one card per ``α jlens/<word>`` token atom in the ONE
  //              steering expression (the engine folds the lens direction
  //              over the workspace band with whitened shares, exactly
  //              like a concept vector).  Per-card α slider + trigger
  //              pill (lens atoms run hotter than concept vectors;
  //              default 0.3).
  //   PROBE    — one card per pinned ``jlens/<word>`` token probe
  //              (ordinary probe-rack entries): signed whitened-
  //              coordinate bar, sparkline, depth CoM, per-layer strip.
  //   WORKSPACE— the open-vocab live readout: one card per aggregate
  //              token (strength = mean band probability as an absolute
  //              0→1 bar, com = salience-weighted depth center of mass),
  //              each with a per-layer salience strip distilled from the
  //              same streamed matrix.  The full per-layer ranking lives
  //              in the transcript token drilldown's j-lens tab.
  //
  // With no fitted lens (``session_info.jlens_fitted``) the tab renders a
  // "fit j-lens" button instead — it kicks off the server's background fit
  // (``POST .../lens/fit``) and shows polled progress; the live readout
  // auto-enables when the artifact lands.  Every section needs the artifact.

  import Bar from "../lib/charts/Bar.svelte";
  import Select from "../lib/Select.svelte";
  import JLensProbeCard from "./rack/JLensProbeCard.svelte";
  import JLensSteerCard from "./rack/JLensSteerCard.svelte";
  import JLensTokenCard from "./rack/JLensTokenCard.svelte";
  import {
    addJLensToRack,
    activeProbeNames,
    attachProbe,
    checkLensFit,
    lensFitState,
    lensState,
    probeRack,
    sessionState,
    setLensWorkspaceSortMode,
    setLiveLens,
    setProbeSortMode,
    startLensFit,
    steerRack,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type {
    JLensSteerEntry,
    ProbeSortMode,
  } from "../lib/types";
  import type { LensWorkspaceSortMode } from "../lib/stores.svelte";

  const fitted = $derived(sessionState.info?.jlens_fitted === true);
  const liveOn = $derived(lensState.layers !== null);

  // Resume-visibility: a page reload mid-fit should pick the progress
  // polling back up (the fit runs server-side regardless of the client).
  $effect(() => {
    if (!fitted) void checkLensFit();
  });

  // ---------- STEER: jlens-mode rack entries (alphabetical) ----------
  const steerCards = $derived.by(() => {
    const arr = [...steerRack.entries.entries()].filter(
      (kv): kv is [string, JLensSteerEntry] => kv[1].mode === "jlens",
    );
    arr.sort((a, b) => a[0].localeCompare(b[0]));
    return arr;
  });

  let steerInput = $state("");

  function onAddSteer(ev: SubmitEvent): void {
    ev.preventDefault();
    const word = steerInput.trim();
    if (!word) return;
    // Single-token validation happens engine-side at the next generation
    // (MultiTokenWordError → stream error toast) — no dry-run endpoint.
    addJLensToRack(word);
    steerInput = "";
  }

  // ---------- PROBE: pinned jlens/ probe-rack entries ----------
  const pinnedCards = $derived.by(() => {
    const names = activeProbeNames().filter((n) => n.startsWith("jlens/"));
    return names.map((n) => ({ name: n, entry: probeRack.entries.get(n) }));
  });

  const probeSortMode = $derived(probeRack.sortMode);
  const PROBE_SORT_OPTIONS: { value: ProbeSortMode; label: string }[] = [
    { value: "name", label: "name" },
    { value: "value", label: "value" },
    { value: "change", label: "change" },
  ];

  let probeInput = $state("");
  let probeBusy = $state(false);

  async function pinWord(word: string): Promise<void> {
    const bare = word.trim().replace(/^jlens\//, "");
    if (!bare || probeBusy) return;
    const selector = `jlens/${bare}`;
    if (probeRack.active.includes(selector)) return;
    probeBusy = true;
    try {
      await attachProbe(selector);
      pushToast(`pinned ${selector}`, { kind: "info" });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      pushToast(`pin ${selector} failed — ${msg}`, { kind: "error" });
    } finally {
      probeBusy = false;
    }
  }

  function onAddProbe(ev: SubmitEvent): void {
    ev.preventDefault();
    void pinWord(probeInput);
    probeInput = "";
  }

  // ---------- WORKSPACE: aggregate token cards, user-sorted ----------

  interface AggRow {
    /** Raw vocabulary token text (untrimmed — the strip matches on it). */
    token: string;
    strength: number;
    com: number;
    spread: number;
    pinned: boolean;
  }

  const WORKSPACE_SORT_OPTIONS: {
    value: LensWorkspaceSortMode;
    label: string;
  }[] = [
    { value: "strength", label: "strength" },
    { value: "name", label: "name" },
    { value: "depth", label: "depth" },
  ];

  const aggRows = $derived.by((): AggRow[] => {
    const rows = lensState.aggregate;
    if (!rows || rows.length === 0) return [];
    const out = rows.map(([token, strength, com, spread]) => ({
      token,
      strength,
      com,
      spread,
      pinned: probeRack.active.includes(`jlens/${token.trim()}`),
    }));
    if (lensState.workspaceSortMode === "name") {
      out.sort((a, b) =>
        a.token.trim().localeCompare(b.token.trim()) || b.strength - a.strength,
      );
    } else if (lensState.workspaceSortMode === "depth") {
      out.sort((a, b) => a.com - b.com || b.strength - a.strength);
    } else {
      out.sort((a, b) => b.strength - a.strength);
    }
    return out;
  });

  function onToggleLive(): void {
    void setLiveLens(!liveOn);
  }
</script>

<div class="jlens" aria-label="Jacobian-lens inspector">
  {#if !fitted}
    <section class="section">
      <header class="header">
        <div class="header-text">
          <span class="title">J-LENS</span>
        </div>
      </header>
      {#if lensFitState.running}
        <div class="fit-progress" aria-label="Lens fit progress">
          <div class="fit-line">
            <span class="fit-msg">{lensFitState.message ?? "fitting…"}</span>
            {#if lensFitState.promptsTotal > 0}
              <span class="fit-count">
                {lensFitState.promptsDone}/{lensFitState.promptsTotal}
              </span>
            {/if}
          </div>
          <div class="fit-bar" aria-hidden="true">
            <Bar
              value={lensFitState.promptsDone}
              max={Math.max(lensFitState.promptsTotal, 1)}
              width={160}
              height={8}
              color="var(--accent-blue)"
            />
          </div>
          <p class="hint">
            the fit holds the model — generations error until it lands;
            an interrupted fit resumes from its last checkpoint
          </p>
        </div>
      {:else}
        <p class="hint">
          no Jacobian lens fitted for this model — the workspace readout,
          token atoms, and token probes all need the per-model artifact
        </p>
        {#if lensFitState.error}
          <p class="hint fit-error">last fit: {lensFitState.error}</p>
        {/if}
        <button
          type="button"
          class="fit-btn"
          onclick={() => void startLensFit()}
          title="Fit the Jacobian lens over the workspace band (~100 web-text prompts; hours of wall clock, checkpointed and resumable)"
        >
          fit j-lens
        </button>
        <p class="hint">
          ≈100 web-text prompts over the 40–90% workspace band —
          compute-bound (hours on Apple silicon), checkpointed every 25
          prompts, resumable; live readout turns on when it lands
        </p>
      {/if}
    </section>
  {:else}
    <!-- STEER — token-atom cards in the shared steering expression. -->
    <section class="section">
      <header class="header">
        <div class="header-text">
          <span class="title">STEER</span>
        </div>
        <span class="count">{steerCards.length} term{steerCards.length === 1 ? "" : "s"}</span>
      </header>

      {#if steerCards.length > 0}
        <div class="cards" role="list">
          {#each steerCards as [name, entry] (name)}
            <div role="listitem">
              <JLensSteerCard {name} {entry} />
            </div>
          {/each}
        </div>
      {/if}

      <form class="add-form" onsubmit={onAddSteer}>
        <input
          class="add-input"
          type="text"
          placeholder="word (single token)"
          bind:value={steerInput}
          aria-label="Add a J-lens steering token"
        />
        <button type="submit" class="add-btn" disabled={!steerInput.trim()}>
          + steer
        </button>
      </form>
    </section>

    <!-- PROBE — pinned closed-vocab token-probe cards. -->
    <section class="section">
      <header class="header">
        <div class="header-text">
          <span class="title">PROBE</span>
          <span class="count">{pinnedCards.length} pinned</span>
        </div>
        <label class="sort">
          <span class="sort-label">sort</span>
          <span class="sort-select">
            <Select
              value={probeSortMode}
              options={PROBE_SORT_OPTIONS}
              onchange={setProbeSortMode}
              ariaLabel="Sort J-lens probes by"
            />
          </span>
        </label>
      </header>

      {#if pinnedCards.length > 0}
        <div class="cards" role="list">
          {#each pinnedCards as { name, entry } (name)}
            {#if entry}
              <div role="listitem">
                <JLensProbeCard {name} {entry} />
              </div>
            {/if}
          {/each}
        </div>
      {/if}

      <form class="add-form" onsubmit={onAddProbe}>
        <input
          class="add-input"
          type="text"
          placeholder="word (single token)"
          bind:value={probeInput}
          aria-label="Pin a J-lens token probe"
        />
        <button
          type="submit"
          class="add-btn"
          disabled={probeBusy || !probeInput.trim()}
        >
          + pin
        </button>
      </form>
    </section>

    <!-- WORKSPACE — the open-vocab aggregate readout as token cards. -->
    <section class="section">
      <header class="header">
        <div class="header-text">
          <button
            type="button"
            class="toggle"
            class:on={liveOn}
            disabled={lensState.busy}
            onclick={onToggleLive}
            title={liveOn
              ? "Stop streaming the live workspace readout"
              : "Stream the layer-aggregated J-lens readout live during generation"}
          >
            {liveOn ? "live: on" : "live: off"}
          </button>
          <span class="title">WORKSPACE</span>
        </div>
        <label class="sort">
          <span class="sort-label">sort</span>
          <span class="sort-select">
            <Select
              value={lensState.workspaceSortMode}
              options={WORKSPACE_SORT_OPTIONS}
              onchange={setLensWorkspaceSortMode}
              ariaLabel="Sort workspace tokens by"
            />
          </span>
        </label>
      </header>

      {#if liveOn}
        {#if aggRows.length > 0}
          <div class="cards" role="list" aria-label="Aggregate lens tokens">
            {#each aggRows as row (row.token)}
              <div role="listitem">
                <JLensTokenCard
                  token={row.token}
                  strength={row.strength}
                  com={row.com}
                  spread={row.spread}
                  layers={lensState.layers ?? []}
                  readout={lensState.readout}
                  pinned={row.pinned}
                  busy={probeBusy}
                  onpin={pinWord}
                />
              </div>
            {/each}
          </div>
          <p class="hint drill-hint">
            click a transcript token for its full per-layer matrix
          </p>
        {:else}
          <p class="hint">streams on the next generation</p>
        {/if}
      {/if}
    </section>
  {/if}
</div>

<style>
  .jlens {
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow-y: auto;
  }

  /* Flat sections divided by hairlines, matching the rack chrome. */
  .section {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding: var(--space-5);
    border-bottom: 1px solid var(--border);
  }
  .section:last-child {
    border-bottom: 0;
  }

  /* Rack-style section header — underlined, matching ProbeRack /
     SteeringRack so the two tabs read as siblings. */
  .header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: var(--space-3);
  }
  .header-text {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    min-width: 0;
  }
  .title {
    font-weight: var(--weight-bold);
    color: var(--accent);
    font-size: var(--text-sm);
    text-transform: uppercase;
  }
  .count {
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
  .sort {
    display: inline-flex;
    align-items: center;
    gap: var(--space-3);
  }
  .sort-label {
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
  .sort-select {
    display: inline-flex;
    min-width: 8em;
  }

  .hint {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
  .drill-hint {
    font-size: var(--text-xs);
    color: var(--fg-dim);
  }

  /* ----- fit button + progress (the not-fitted state) ----- */
  .fit-btn {
    align-self: flex-start;
    background: color-mix(in srgb, var(--accent-blue) 10%, transparent);
    color: var(--accent-blue);
    border: 1px solid var(--accent-blue);
    border-radius: var(--radius);
    font-size: var(--text-sm);
    padding: var(--space-2) var(--space-5);
    cursor: pointer;
    transition: background var(--dur) var(--ease-out);
  }
  .fit-btn:hover {
    background: color-mix(in srgb, var(--accent-blue) 18%, transparent);
  }
  .fit-error {
    color: var(--accent-red);
  }
  .fit-progress {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .fit-line {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .fit-msg {
    color: var(--fg);
    font-size: var(--text-sm);
    font-family: var(--font-mono);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .fit-count {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }
  .fit-bar :global(.bar) {
    width: 100%;
    height: 8px;
    display: block;
  }

  /* Card stack — same rhythm as the probe rack's strips. */
  .cards {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  /* ----- add forms ----- */
  .add-form {
    display: flex;
    gap: var(--space-2);
  }
  .add-input {
    flex: 1 1 auto;
    min-width: 0;
    background: var(--bg);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    padding: 2px var(--space-3);
  }
  .add-input:focus-visible {
    outline: 1px solid var(--accent);
    outline-offset: -1px;
  }
  .add-btn {
    background: transparent;
    color: var(--fg-muted);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    font-size: var(--text-sm);
    padding: 1px var(--space-3);
    cursor: pointer;
    flex: 0 0 auto;
  }
  .add-btn:hover:not(:disabled) {
    color: var(--fg);
    border-color: var(--fg-muted);
  }
  .add-btn:disabled {
    opacity: 0.5;
    cursor: default;
  }

  /* ----- workspace live toggle ----- */
  .toggle {
    font-size: var(--text-sm);
    color: var(--fg-muted);
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 1px var(--space-3);
    cursor: pointer;
  }
  .toggle:hover:not(:disabled) {
    color: var(--fg);
    border-color: var(--fg-muted);
  }
  .toggle.on {
    color: var(--accent);
    border-color: var(--accent);
  }
  .toggle:disabled {
    opacity: 0.5;
    cursor: default;
  }
</style>
