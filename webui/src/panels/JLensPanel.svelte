<script lang="ts">
  // J-LENS — the inspector column's Jacobian-lens tab: the open- and
  // closed-vocabulary views of the workspace readout, parallel to the
  // linear-probe tab's STEER/PROBE racks.
  //
  //   STEER    — token chips compiling to ``α jlens/<word>`` terms in the
  //              ONE steering expression (the engine folds the lens
  //              direction over the workspace band with whitened shares,
  //              exactly like a concept vector).  Per-chip α (lens atoms
  //              run hotter than concept vectors; default 0.3).
  //   PROBES   — pinned token chips: ``jlens/<word>`` attached through the
  //              ordinary probe pipeline (rank-1 whitened coordinate,
  //              sparkline-backed in the store; the chip surfaces the live
  //              axis-0 value + the depth center of mass).
  //   WORKSPACE— the open-vocab live readout: the layer-aggregated chip
  //              list (strength = mean band probability, com =
  //              salience-weighted depth center of mass) as the primary
  //              surface, the per-layer matrix behind a disclosure.
  //              Clicking a readout chip pins it as a probe.
  //
  // Renders a fit hint when no Jacobian lens is fitted for the model
  // (``session_info.jlens_fitted``); every section needs the artifact.

  import Disclosure from "../lib/Disclosure.svelte";
  import Slider from "../lib/Slider.svelte";
  import {
    addJLensToRack,
    attachProbe,
    detachProbe,
    lensState,
    probeRack,
    removeJLensFromRack,
    sessionState,
    setJLensAlpha,
    setJLensEnabled,
    setLiveLens,
    steerRack,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type { JLensSteerEntry } from "../lib/types";

  const fitted = $derived(sessionState.info?.jlens_fitted === true);
  const liveOn = $derived(lensState.layers !== null);

  // ---------- STEER: jlens-mode rack entries (alphabetical) ----------
  const steerChips = $derived.by(() => {
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
  const pinnedChips = $derived.by(() => {
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

  async function onUnpin(name: string): Promise<void> {
    try {
      await detachProbe(name);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      pushToast(`detach ${name} failed — ${msg}`, { kind: "error" });
    }
  }

  /** Live axis-0 whitened coordinate + depth-CoM for a pinned chip —
   *  settled aggregate preferred, live reading during generation. */
  function chipStats(entry: (typeof pinnedChips)[number]["entry"]): {
    value: number | null;
    com: number | null;
  } {
    if (!entry) return { value: null, com: null };
    const reading = entry.aggregate ?? entry.reading;
    const value = entry.current ?? null;
    const com = reading?.depth_com?.[0] ?? null;
    return { value, com };
  }

  // ---------- WORKSPACE: aggregate chips + matrix disclosure ----------

  interface AggChip {
    token: string;
    strength: number;
    com: number;
    spread: number;
    /** Within-list brightness — strength over the list max. */
    weight: number;
  }

  const aggChips = $derived.by((): AggChip[] => {
    const rows = lensState.aggregate;
    if (!rows || rows.length === 0) return [];
    const top = Math.max(...rows.map(([, s]) => s)) || 1;
    return rows.map(([token, strength, com, spread]) => ({
      token: token.trim() || JSON.stringify(token),
      strength,
      com,
      spread,
      weight: strength / top,
    }));
  });

  interface LensRow {
    layer: number;
    tokens: { text: string; weight: number }[];
  }

  const matrixRows = $derived.by((): LensRow[] => {
    const layers = lensState.layers;
    if (!layers) return [];
    const readout = lensState.readout;
    return layers.map((layer) => {
      const pairs = readout?.[String(layer)] ?? [];
      if (pairs.length === 0) return { layer, tokens: [] };
      // Raw lens logits are uncalibrated across layers — normalise within
      // the row (top token full brightness) so each row reads as its own
      // ranked distribution.
      const max = Math.max(...pairs.map(([, s]) => s));
      const exps = pairs.map(([, s]) => Math.exp(s - max));
      const top = Math.max(...exps) || 1;
      return {
        layer,
        tokens: pairs.map(([text], i) => ({
          text: text.trim() || JSON.stringify(text),
          weight: exps[i] / top,
        })),
      };
    });
  });

  let matrixOpen = $state(false);

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
    <!-- STEER — token chips in the shared steering expression. -->
    <section class="section">
      <header class="header">
        <div class="header-text">
          <span class="title">STEER</span>
          <span class="subtitle">token atoms</span>
        </div>
        <span class="count">{steerChips.length} term{steerChips.length === 1 ? "" : "s"}</span>
      </header>

      {#each steerChips as [name, entry] (name)}
        <div class="chip-row" class:disabled={!entry.enabled}>
          <button
            type="button"
            class="chip steer-chip"
            class:off={!entry.enabled}
            title={entry.enabled
              ? `${name} — click to disable`
              : `${name} — click to enable`}
            onclick={() => setJLensEnabled(name, !entry.enabled)}
          >{name.slice("jlens/".length)}</button>
          <div class="chip-slider">
            <Slider
              value={entry.alpha}
              min={0}
              max={1}
              step={0.05}
              ariaLabel="alpha for {name}"
              title="α — push coefficient (lens atoms run hot; ≈0.3 is the sweet spot)"
              oninput={(v) => Number.isFinite(v) && setJLensAlpha(name, v)}
            />
          </div>
          <span class="chip-value">{entry.alpha.toFixed(2)}</span>
          <button
            type="button"
            class="icon remove"
            aria-label="Remove steering term {name}"
            title="Remove term"
            onclick={() => removeJLensFromRack(name)}
          >✕</button>
        </div>
      {/each}

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

    <!-- PROBES — pinned closed-vocab token probes. -->
    <section class="section">
      <header class="header">
        <div class="header-text">
          <span class="title">PROBES</span>
          <span class="subtitle">pinned tokens</span>
        </div>
        <span class="count">{pinnedChips.length} pinned</span>
      </header>

      {#each pinnedChips as { name, entry } (name)}
        {@const stats = chipStats(entry)}
        <div class="chip-row">
          <span
            class="chip probe-chip"
            title="{name} — whitened coordinate along the lens direction"
          >{name.slice("jlens/".length)}</span>
          <span class="chip-stats">
            {#if stats.value !== null}
              <span
                class="stat"
                class:pos={stats.value > 0}
                class:neg={stats.value < 0}
                title="live whitened coordinate (axis 0)"
              >{stats.value >= 0 ? "+" : ""}{stats.value.toFixed(2)}</span>
            {/if}
            {#if stats.com !== null}
              <span
                class="stat com"
                title="depth center of mass of the per-layer read (0 = first block, 1 = last)"
              >com {stats.com.toFixed(2)}</span>
            {/if}
            {#if stats.value === null && stats.com === null}
              <span class="stat empty">—</span>
            {/if}
          </span>
          <button
            type="button"
            class="icon remove"
            aria-label="Unpin probe {name}"
            title="Unpin probe"
            onclick={() => onUnpin(name)}
          >✕</button>
        </div>
      {/each}

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

    <!-- WORKSPACE — the open-vocab aggregate readout (+ matrix disclosure). -->
    <section class="section workspace">
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
        {#if aggChips.length > 0}
          <div class="agg-chips" aria-label="Aggregate lens tokens">
            {#each aggChips as chip, i (i)}
              <button
                type="button"
                class="chip agg-chip"
                style="opacity: {0.4 + 0.6 * chip.weight}"
                title={`"${chip.token}" — strength ${chip.strength.toFixed(3)} · com ${chip.com.toFixed(2)} ±${chip.spread.toFixed(2)} · click to pin as probe`}
                onclick={() => void pinWord(chip.token)}
              >
                <span class="agg-token">{chip.token}</span>
                <span class="agg-com">@{chip.com.toFixed(2)}</span>
              </button>
            {/each}
          </div>
        {:else}
          <p class="hint">streams on the next generation</p>
        {/if}

        <Disclosure bind:expanded={matrixOpen} summary="per-layer matrix" flush>
          <div class="rows" role="list">
            {#each matrixRows as row (row.layer)}
              <div class="row" role="listitem">
                <span class="layer">L{row.layer}</span>
                {#if row.tokens.length === 0}
                  <span class="empty">—</span>
                {:else}
                  <span class="tokens">
                    {#each row.tokens as tok, i (i)}
                      <span class="tok" style="opacity: {0.35 + 0.65 * tok.weight}"
                        >{tok.text}</span
                      >
                    {/each}
                  </span>
                {/if}
              </div>
            {/each}
          </div>
        </Disclosure>
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

  .header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
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

  /* ----- chips ----- */
  .chip {
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    color: var(--accent);
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 1px var(--space-3);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  button.chip {
    cursor: pointer;
  }
  button.chip:hover {
    border-color: var(--accent);
  }
  .steer-chip.off {
    color: var(--fg-muted);
    border-style: dashed;
  }
  .probe-chip {
    color: var(--fg-strong);
  }

  .chip-row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-width: 0;
  }
  .chip-row.disabled {
    opacity: 0.7;
  }
  .chip-slider {
    flex: 1 1 auto;
    min-width: 40px;
  }
  .chip-value {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }
  .chip-stats {
    display: flex;
    gap: var(--space-3);
    margin-left: auto;
    flex: 0 0 auto;
  }
  .stat {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    font-variant-numeric: tabular-nums;
  }
  .stat.pos {
    color: var(--accent-green);
  }
  .stat.neg {
    color: var(--accent-red);
  }
  .stat.com {
    color: var(--fg-muted);
  }
  .stat.empty {
    font-style: italic;
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
  }
  .icon:hover {
    color: var(--accent-red);
    background: var(--bg-elev);
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

  /* ----- workspace ----- */
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

  .agg-chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }
  .agg-chip {
    display: inline-flex;
    align-items: baseline;
    gap: var(--space-1);
  }
  .agg-token {
    color: var(--accent);
  }
  .agg-com {
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
  }

  .rows {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    max-height: 160px;
    overflow-y: auto;
  }
  .row {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    min-width: 0;
  }
  .layer {
    flex: 0 0 34px;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-family: var(--font-mono);
  }
  .tokens {
    display: flex;
    gap: var(--space-3);
    min-width: 0;
    overflow: hidden;
    white-space: nowrap;
  }
  .tok {
    color: var(--accent);
    font-size: var(--text-sm);
    font-family: var(--font-mono);
  }
  .empty {
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
</style>
