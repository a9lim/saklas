<script lang="ts">
  // SAE — the inspector column's sparse-autoencoder tab: two sections,
  // card-based and symmetric with the other three pillars (every row wears
  // RackCard; the SAE family accent is gold, marker ▲/△).
  //
  //   STEER — one card per ``α sae/<id>`` decoder-row atom in the ONE
  //           steering expression.  Per-card α slider + trigger pill.
  //   PROBE — pinned ``sae/<id>`` feature probes (persistent, gate-able)
  //           first, then the live discovery cards for the per-step top-k
  //           features not already pinned — both the same card shape,
  //           exactly like the lens workspace readout (the pinned card's ▲
  //           unpins, the unpinned card's △ pins).  The card list owns the
  //           scroll; header + add form stay anchored.  The header's live
  //           toggle is the SAE live switch: off ⇒ no per-step feature
  //           readout — pinned probes settle to the end-of-gen activation,
  //           discovery cards go quiet.
  //
  // With no resident SAE (``session_info.sae_loaded``) the tab renders the
  // release picker instead — loading pins one SAELens release + hook layer
  // for the session, so the absent channel advertises itself (the same
  // move as the lens tab's "fit j-lens" button).

  import SaeProbeCard from "./rack/SaeProbeCard.svelte";
  import SaeSteerCard from "./rack/SaeSteerCard.svelte";
  import { apiSae } from "../lib/api";
  import {
    activeProbeNames,
    addSaeToRack,
    attachProbe,
    loadSae,
    probeRack,
    saeState,
    sessionState,
    setLiveSae,
    steerRack,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type { SaeSteerEntry } from "../lib/types";

  const loaded = $derived(sessionState.info?.sae_loaded === true);
  const info = $derived(sessionState.info?.sae_info ?? null);
  let release = $state("");
  let releases = $state<{ release: string; layers: number[] }[]>([]);
  let discoverError = $state<string | null>(null);

  $effect(() => {
    if (loaded || releases.length > 0) return;
    void apiSae.releases().then((result) => {
      releases = result.releases;
      if (!release && releases.length > 0) release = releases[0].release;
    }).catch((error) => {
      discoverError = error instanceof Error ? error.message : String(error);
    });
  });

  // ---------- STEER: sae-mode rack entries (by feature id) ----------
  const steerCards = $derived.by(() => {
    const rows = [...steerRack.entries.entries()].filter(
      (row): row is [string, SaeSteerEntry] => row[1].mode === "sae",
    );
    rows.sort((a, b) => Number(a[0].slice(4)) - Number(b[0].slice(4)));
    return rows;
  });

  // ---------- PROBE: pinned probe cards + unpinned discovery cards ----------
  const pinned = $derived.by(() => activeProbeNames()
    .filter((name) => name.startsWith("sae/"))
    .map((name) => ({ name, entry: probeRack.entries.get(name) }))
    .filter((row) => row.entry !== undefined));

  const discovery = $derived(saeState.readout.filter(
    (row) => !probeRack.active.includes(`sae/${row.id}`),
  ));

  let steerInput = $state("");
  let probeInput = $state("");
  let featureBusy = $state(false);

  async function validateInput(raw: string): Promise<number | null> {
    const id = Number(raw.trim().replace(/^sae\//, ""));
    if (!Number.isInteger(id) || id < 0) {
      pushToast("feature id must be a non-negative integer", { kind: "error" });
      return null;
    }
    const validated = await apiSae.validateFeature(id);
    return validated.id;
  }

  async function addSteer(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    if (featureBusy) return;
    featureBusy = true;
    try {
      const id = await validateInput(steerInput);
      if (id !== null) {
        addSaeToRack(id);
        steerInput = "";
      }
    } catch (error) {
      pushToast(error instanceof Error ? error.message : String(error), { kind: "error" });
    } finally {
      featureBusy = false;
    }
  }

  async function pin(id: number): Promise<void> {
    if (featureBusy || probeRack.active.includes(`sae/${id}`)) return;
    featureBusy = true;
    try {
      await apiSae.validateFeature(id);
      await attachProbe(`sae/${id}`);
    } catch (error) {
      pushToast(error instanceof Error ? error.message : String(error), { kind: "error" });
    } finally {
      featureBusy = false;
    }
  }

  async function addProbe(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    if (featureBusy) return;
    featureBusy = true;
    try {
      const id = await validateInput(probeInput);
      if (id !== null) {
        await attachProbe(`sae/${id}`);
        probeInput = "";
      }
    } catch (error) {
      pushToast(error instanceof Error ? error.message : String(error), { kind: "error" });
    } finally {
      featureBusy = false;
    }
  }
</script>

<div class="sae" aria-label="Sparse-autoencoder inspector">
  {#if !loaded}
    <section class="section">
      <header class="header">
        <div class="header-text">
          <span class="title">SAE</span>
        </div>
      </header>
      <p class="hint">
        no resident SAE in this session — feature atoms, probes, and the
        live discovery readout all need one loaded SAELens release
      </p>
      <form class="load-form" onsubmit={(event) => { event.preventDefault(); void loadSae(release); }}>
        <input
          class="add-input"
          list="sae-releases"
          bind:value={release}
          placeholder="SAELens release"
          aria-label="SAE release"
        />
        <datalist id="sae-releases">
          {#each releases as row (row.release)}
            <option value={row.release}>{row.layers.map((layer) => `L${layer}`).join(", ")}</option>
          {/each}
        </datalist>
        <button class="add-btn" disabled={!release.trim() || saeState.loading}>
          {saeState.loading ? "loading…" : "load SAE"}
        </button>
      </form>
      <p class="hint">
        the selected hook layer stays resident; weights use the normal
        Hugging Face cache
      </p>
      {#if saeState.loadMessage}<p class="hint">{saeState.loadMessage}</p>{/if}
      {#if saeState.loadError}<p class="hint load-error">{saeState.loadError}</p>{/if}
      {#if discoverError}<p class="hint">registry suggestions unavailable: {discoverError}</p>{/if}
    </section>
  {:else}
    <!-- Identity strip — which SAE is resident (release · hook layer ·
         dictionary width).  Session identity, not a section. -->
    <div class="identity">
      <span class="release" title="resident SAELens release">{info?.release}</span>
      <span class="chip">L{info?.layer}</span>
      <span class="chip">{info?.width?.toLocaleString()} features</span>
    </div>

    <!-- STEER — decoder-row atom cards in the shared steering expression. -->
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
              <SaeSteerCard {name} {entry} />
            </div>
          {/each}
        </div>
      {/if}

      <form class="add-form" onsubmit={addSteer}>
        <input
          class="add-input"
          type="text"
          placeholder="feature id"
          bind:value={steerInput}
          aria-label="Add an SAE steering feature"
        />
        <button
          type="submit"
          class="add-btn"
          disabled={featureBusy || !steerInput.trim()}
        >
          + steer
        </button>
      </form>
    </section>

    <!-- PROBE — pinned feature probes + the live discovery top-k.  The
         card list owns the scroll; header + add form stay anchored (the
         other pillars' fixed-chrome / scrollable-middle shape). -->
    <section class="section probe">
      <header class="header">
        <div class="header-text">
          <span class="title">PROBE</span>
          <button
            type="button"
            class="toggle"
            class:on={saeState.live}
            disabled={saeState.busy}
            onclick={() => void setLiveSae(!saeState.live)}
            title={saeState.live
              ? "Stop the per-step feature readout (pinned probes settle to the end-of-gen activation)"
              : "Stream the SAE feature readout live during generation (pinned probes + discovery top-k)"}
          >
            {saeState.live ? "live: on" : "live: off"}
          </button>
          <span class="count">{pinned.length} pinned</span>
        </div>
      </header>

      <div class="scroll">
        {#if pinned.length > 0}
          <div class="cards" role="list" aria-label="Pinned SAE feature probes">
            {#each pinned as row (row.name)}
              {@const entry = row.entry!}
              {@const reading = entry.aggregate ?? entry.reading}
              <div role="listitem">
                <SaeProbeCard
                  id={Number(row.name.slice(4))}
                  label={entry.info.label}
                  layer={info?.layer ?? null}
                  value={reading?.coords?.[0] ?? entry.current ?? 0}
                  series={entry.sparkline}
                  pinned={true}
                />
              </div>
            {/each}
          </div>
        {/if}

        {#if saeState.live}
          {#if discovery.length > 0}
            <div class="cards" role="list" aria-label="Discovered SAE features">
              {#each discovery as feature (feature.id)}
                <div role="listitem">
                  <SaeProbeCard
                    id={feature.id}
                    label={feature.label}
                    layer={info?.layer ?? null}
                    value={feature.activation}
                    series={saeState.history.get(feature.id) ?? []}
                    pinned={false}
                    busy={featureBusy}
                    onpin={(id) => void pin(id)}
                  />
                </div>
              {/each}
            </div>
          {:else}
            <p class="hint">feature discovery streams on the next generation</p>
          {/if}
        {:else}
          <p class="hint">
            live off — pinned features settle to the end-of-generation activation
          </p>
        {/if}
      </div>

      <form class="add-form anchored" onsubmit={addProbe}>
        <input
          class="add-input"
          type="text"
          placeholder="feature id"
          bind:value={probeInput}
          aria-label="Pin an SAE feature probe"
        />
        <button
          type="submit"
          class="add-btn"
          disabled={featureBusy || !probeInput.trim()}
        >
          + pin
        </button>
      </form>
    </section>
  {/if}
</div>

<style>
  /* Fixed-chrome column, matching the other pillar tabs: STEER sizes to
     its content up to half the inspector, PROBE takes the rest and scrolls
     internally so the header + add form stay visible. */
  .sae {
    display: flex;
    flex-direction: column;
    height: 100%;
    max-height: 100%;
    min-height: 0;
    overflow: hidden;
  }

  /* Identity strip — borderless: the mono/muted register and its own
     padding separate it from the STEER section below. */
  .identity {
    display: flex;
    gap: var(--space-3);
    align-items: center;
    padding: var(--space-3) var(--space-5) 0;
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
  }
  .release {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .chip {
    background: var(--glass);
    border-radius: var(--radius);
    padding: 0 var(--space-2);
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }

  /* Flat borderless sections — typography + padding carry the divide,
     matching the rack chrome. */
  .section {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding: var(--space-5);
    min-height: 0;
  }
  .section.steer {
    flex: 0 1 auto;
    max-height: 50%;
    overflow: hidden;
  }
  /* A populated steer-card pile scrolls inside its own half-column cap
     rather than eating the probe section's share of the inspector. */
  .steer-cards {
    overflow-y: auto;
    min-height: 0;
  }
  .section.probe {
    flex: 1 1 0;
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
  /* Anchored footer — borderless, same padding treatment as the racks'
     actions row. */
  .add-form.anchored {
    flex: 0 0 auto;
    padding-top: var(--space-3);
  }

  /* Rack-style section header — borderless, matching ProbeRack /
     SteeringRack so the four tabs read as siblings. */
  .header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
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
    flex: 0 0 auto;
  }

  .hint {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
  .load-error {
    color: var(--accent-red);
  }

  /* Card stack — same rhythm as the other racks' strips. */
  .cards {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  /* ----- add / load forms ----- */
  .add-form,
  .load-form {
    display: flex;
    gap: var(--space-2);
  }
  .add-input {
    flex: 1 1 auto;
    min-width: 0;
    /* Borderless input: recessed well fill; ring on focus only. */
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    border-radius: var(--radius);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    padding: 2px var(--space-3);
    transition: border-color var(--dur-fast) var(--ease-out);
  }
  .add-input:focus-visible {
    outline: 2px solid var(--accent-glow);
    outline-offset: 1px;
    border-color: var(--accent-glow);
  }
  .add-btn {
    /* Family-tinted fill — the gold sibling of the racks' launchers. */
    background: color-mix(in srgb, var(--pillar-sae) 10%, transparent);
    color: var(--pillar-sae);
    border: 1px solid transparent;
    border-radius: var(--radius);
    font-size: var(--text-sm);
    padding: 1px var(--space-3);
    cursor: pointer;
    flex: 0 0 auto;
    transition: background var(--dur) var(--ease-out);
  }
  .add-btn:hover:not(:disabled) {
    background: color-mix(in srgb, var(--pillar-sae) 18%, transparent);
  }
  .add-btn:disabled {
    opacity: 0.5;
    cursor: default;
  }

  /* ----- live toggle — ProbeRack's glass treatment ----- */
  .toggle {
    font-size: var(--text-sm);
    color: var(--fg-muted);
    background: var(--glass);
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 1px var(--space-3);
    cursor: pointer;
  }
  .toggle:hover:not(:disabled) {
    color: var(--fg);
    background: var(--glass-strong);
  }
  .toggle.on {
    color: var(--accent);
    background: var(--accent-subtle);
  }
  .toggle:disabled {
    opacity: 0.5;
    cursor: default;
  }
</style>
