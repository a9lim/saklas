<script lang="ts">
  // J-LENS — the inspector column's Jacobian-lens tab: the open- and
  // closed-vocabulary views of the workspace readout, card-based and
  // symmetric with the linear-probe tab (every row wears RackCard; the
  // j-lens family accent is blue, marker ■/□).
  //
  //   STEER    — one card per ``α jlens/<word>`` token atom in the ONE
  //              steering expression (the engine folds the lens direction
  //              over the workspace band with whitened shares, exactly
  //              like a concept vector).  Per-card α slider + trigger
  //              pill (lens atoms run hotter than concept vectors;
  //              default 0.3).
  //   PROBES   — one card per pinned ``jlens/<word>`` token probe
  //              (ordinary probe-rack entries): signed whitened-
  //              coordinate bar, sparkline, depth CoM, per-layer strip.
  //   WORKSPACE— the open-vocab live readout: one card per aggregate
  //              token (strength = mean band probability as an absolute
  //              0→1 bar, com = salience-weighted depth center of mass),
  //              each with a per-layer salience strip distilled from the
  //              same streamed matrix.  The full per-layer ranking lives
  //              in the transcript token drilldown's j-lens tab.
  //
  // Renders a fit hint when no Jacobian lens is fitted for the model
  // (``session_info.jlens_fitted``); every section needs the artifact.

  import JLensProbeCard from "./rack/JLensProbeCard.svelte";
  import JLensSteerCard from "./rack/JLensSteerCard.svelte";
  import JLensTokenCard from "./rack/JLensTokenCard.svelte";
  import {
    addJLensToRack,
    attachProbe,
    lensState,
    probeRack,
    sessionState,
    setLiveLens,
    steerRack,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type { JLensSteerEntry } from "../lib/types";

  const fitted = $derived(sessionState.info?.jlens_fitted === true);
  const liveOn = $derived(lensState.layers !== null);

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

  // ---------- PROBES: pinned jlens/ probe-rack entries ----------
  const pinnedCards = $derived.by(() => {
    const names = probeRack.active.filter((n) => n.startsWith("jlens/"));
    names.sort((a, b) => a.localeCompare(b));
    return names.map((n) => ({ name: n, entry: probeRack.entries.get(n) }));
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

  // ---------- WORKSPACE: aggregate token cards, strength-ranked ----------

  interface AggRow {
    /** Raw vocabulary token text (untrimmed — the strip matches on it). */
    token: string;
    strength: number;
    com: number;
    spread: number;
    pinned: boolean;
  }

  const aggRows = $derived.by((): AggRow[] => {
    const rows = lensState.aggregate;
    if (!rows || rows.length === 0) return [];
    return rows
      .map(([token, strength, com, spread]) => ({
        token,
        strength,
        com,
        spread,
        pinned: probeRack.active.includes(`jlens/${token.trim()}`),
      }))
      .sort((a, b) => b.strength - a.strength);
  });

  function onToggleLive(): void {
    void setLiveLens(!liveOn);
  }
</script>

<div class="jlens" aria-label="Jacobian-lens inspector">
  {#if !fitted}
    <section class="section">
      <header class="header">
        <span class="title">J-LENS</span>
      </header>
      <p class="hint">
        no Jacobian lens fitted for this model — run
        <code>saklas lens fit &lt;model&gt;</code>
      </p>
    </section>
  {:else}
    <!-- STEER — token-atom cards in the shared steering expression. -->
    <section class="section">
      <header class="header">
        <div class="header-text">
          <span class="title">STEER</span>
          <span class="subtitle">token atoms</span>
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

    <!-- PROBES — pinned closed-vocab token-probe cards. -->
    <section class="section">
      <header class="header">
        <div class="header-text">
          <span class="title">PROBES</span>
          <span class="subtitle">pinned tokens</span>
        </div>
        <span class="count">{pinnedCards.length} pinned</span>
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
          <span class="title">WORKSPACE</span>
          <span class="subtitle">aggregate readout</span>
        </div>
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
  .subtitle,
  .count {
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }

  .hint {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
  .hint code {
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    color: var(--fg);
  }
  .drill-hint {
    font-size: var(--text-xs);
    color: var(--fg-dim);
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
