<script lang="ts">
  // J-LENS — the inspector column's Jacobian-lens tab: two sections,
  // card-based and symmetric with the CAA tab (every row wears RackCard;
  // the j-lens family accent is blue, marker ■/□).
  //
  //   STEER — one card per ``α jlens/<word>`` token atom in the ONE
  //           steering expression (the engine folds the lens direction
  //           over the workspace band with whitened shares, exactly
  //           like a concept vector).  Per-card α slider + trigger
  //           pill (lens atoms run hotter than concept vectors;
  //           default 0.3).
  //   PROBE — the workspace readout, pinned and unpinned in ONE section
  //           (pinning a token just makes its card persistent and
  //           gate-able — both card kinds are the same shape: strength
  //           bar + per-layer strength strip; the pinned card's ■ unpins,
  //           the unpinned card's □ pins).  Pinned ``jlens/<word>`` token
  //           probes first, then the live open-vocab aggregate cards for
  //           the top-k tokens not already pinned.  The card list owns
  //           the scroll (header + add form stay anchored, like the CAA
  //           racks).  The header's live toggle is the lens live switch:
  //           off ⇒ no per-step lens computation — pinned probes settle
  //           to the end-of-gen aggregate, discovery cards go quiet.  The
  //           full per-layer ranking lives in the transcript drilldown.
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
    startLensFit,
    steerRack,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type { JLensSteerEntry, ProbeRackEntry } from "../lib/types";
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

  // ---------- PROBE: pinned probe cards + unpinned aggregate cards ----------
  // One merged section — pinning a workspace token just makes its card
  // persistent (and gate-able); both card families share the one sort
  // control (strength / name / depth).

  interface PinnedRow {
    name: string;
    entry: ProbeRackEntry;
    /** Sort keys mirroring the aggregate rows' (axis 0 = strength). */
    value: number;
    com: number;
  }

  interface AggRow {
    /** Raw vocabulary token text (untrimmed — the strip matches on it). */
    token: string;
    strength: number;
    com: number;
    spread: number;
    /** Recent strength history (0 where the token fell below top-k). */
    series: number[];
  }

  const SORT_OPTIONS: {
    value: LensWorkspaceSortMode;
    label: string;
  }[] = [
    { value: "strength", label: "strength" },
    { value: "name", label: "name" },
    { value: "depth", label: "depth" },
  ];

  const pinnedCards = $derived.by((): PinnedRow[] => {
    const names = activeProbeNames().filter((n) => n.startsWith("jlens/"));
    const rows: PinnedRow[] = [];
    for (const name of names) {
      const entry = probeRack.entries.get(name);
      if (!entry) continue;
      const latest = entry.aggregate ?? entry.reading;
      rows.push({
        name,
        entry,
        value: latest?.coords?.[0] ?? entry.current ?? 0,
        com: latest?.depth_com?.[0] ?? 0,
      });
    }
    if (lensState.workspaceSortMode === "name") {
      rows.sort((a, b) => a.name.localeCompare(b.name));
    } else if (lensState.workspaceSortMode === "depth") {
      rows.sort((a, b) => a.com - b.com || b.value - a.value);
    } else {
      rows.sort((a, b) => b.value - a.value || a.name.localeCompare(b.name));
    }
    return rows;
  });

  const aggRows = $derived.by((): AggRow[] => {
    const rows = lensState.aggregate;
    if (!rows || rows.length === 0) return [];
    const hist = lensState.aggHistory;
    // Pinned tokens already have a persistent card above — the aggregate
    // group carries only the unpinned remainder of the top-k.
    const out = rows
      .filter(([token]) => !probeRack.active.includes(`jlens/${token.trim()}`))
      .map(([token, strength, com, spread]) => ({
        token,
        strength,
        com,
        spread,
        series: hist.map(
          (frame) => frame.find(([t]) => t === token)?.[1] ?? 0,
        ),
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
    <section class="section steer">
      <header class="header">
        <div class="header-text">
          <span class="title">STEER</span>
        </div>
        <span class="count">{steerCards.length} term{steerCards.length === 1 ? "" : "s"}</span>
      </header>

      {#if steerCards.length > 0}
        <div class="cards steer-cards" role="list">
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

    <!-- PROBE — the merged workspace readout: pinned token-probe cards
         (persistent, gate-able) + the unpinned live aggregate cards.
         The card list owns the scroll; header + add form stay anchored
         (the CAA racks' fixed-chrome / scrollable-middle shape). -->
    <section class="section probe">
      <header class="header">
        <div class="header-text">
          <span class="title">PROBE</span>
          <button
            type="button"
            class="toggle"
            class:on={liveOn}
            disabled={lensState.busy}
            onclick={onToggleLive}
            title={liveOn
              ? "Stop the per-step lens readout (pinned probes settle to the end-of-gen aggregate)"
              : "Stream the J-lens readout live during generation (pinned probes + workspace top-k)"}
          >
            {liveOn ? "live: on" : "live: off"}
          </button>
          <span class="count">{pinnedCards.length} pinned</span>
        </div>
        <label class="sort">
          <span class="sort-label">sort</span>
          <span class="sort-select">
            <Select
              value={lensState.workspaceSortMode}
              options={SORT_OPTIONS}
              onchange={setLensWorkspaceSortMode}
              ariaLabel="Sort J-lens probe tokens by"
            />
          </span>
        </label>
      </header>

      <div class="scroll">
        {#if pinnedCards.length > 0}
          <div class="cards" role="list" aria-label="Pinned lens token probes">
            {#each pinnedCards as row (row.name)}
              <div role="listitem">
                <JLensProbeCard name={row.name} entry={row.entry} />
              </div>
            {/each}
          </div>
        {/if}

        {#if liveOn}
          {#if aggRows.length > 0}
            <div class="cards" role="list" aria-label="Workspace aggregate tokens">
              {#each aggRows as row (row.token)}
                <div role="listitem">
                  <JLensTokenCard
                    token={row.token}
                    strength={row.strength}
                    com={row.com}
                    spread={row.spread}
                    series={row.series}
                    layers={lensState.layers ?? []}
                    readout={lensState.readout}
                    pinned={false}
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
            <p class="hint">workspace top-k streams on the next generation</p>
          {/if}
        {:else}
          <p class="hint">
            live off — pinned probes report the end-of-gen aggregate only
          </p>
        {/if}
      </div>

      <form class="add-form anchored" onsubmit={onAddProbe}>
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
  {/if}
</div>

<style>
  /* Fixed-chrome column, matching the CAA rack-grid: STEER sizes to its
     content, PROBE takes the rest and scrolls internally so the header +
     add form stay visible (the ProbeRack header/strips/actions shape). */
  .jlens {
    display: flex;
    flex-direction: column;
    height: 100%;
    max-height: 100%;
    min-height: 0;
    overflow: hidden;
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
  .section.steer {
    flex: 0 1 auto;
    min-height: 0;
  }
  /* A pathological steer-card pile scrolls inside its own list rather
     than eating the probe section's share of the column. */
  .steer-cards {
    overflow-y: auto;
    max-height: 30vh;
  }
  .section.probe {
    flex: 1 1 0;
    min-height: 0;
    overflow: hidden;
  }
  /* The scrollable middle — cards + hints; header and add form stay put. */
  .scroll {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    flex: 1 1 0;
    min-height: 2.4rem;
    overflow-y: auto;
    padding-right: var(--space-1);
  }
  /* Anchored footer — same border-top treatment as the CAA racks'
     actions row. */
  .add-form.anchored {
    flex: 0 0 auto;
    border-top: 1px solid var(--border);
    padding-top: var(--space-3);
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
