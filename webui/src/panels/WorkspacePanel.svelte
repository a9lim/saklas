<script lang="ts">
  // WORKSPACE — the live J-lens readout: what the model's intermediate
  // layers are *disposed to say* at each decode step, streamed on the WS
  // ``token`` frame's ``lens_readout`` channel while the live lens is
  // enabled (POST .../lens/live).  The webui sibling of the TUI's
  // ``/lens`` WORKSPACE section: one row per selected layer, the top-k
  // lens tokens for the latest step, brightness ∝ within-row weight.
  //
  // Renders nothing when no Jacobian lens is fitted for the model
  // (``session_info.jlens_fitted``); when fitted but off, just the header
  // with the toggle so the channel stays discoverable.

  import { lensState, sessionState, setLiveLens } from "../lib/stores.svelte";

  const fitted = $derived(sessionState.info?.jlens_fitted === true);
  const enabled = $derived(lensState.layers !== null);

  interface LensRow {
    layer: number;
    tokens: { text: string; weight: number }[];
  }

  const rows = $derived.by((): LensRow[] => {
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

  function onToggle(): void {
    void setLiveLens(!enabled);
  }
</script>

{#if fitted}
  <section class="workspace" aria-label="Workspace lens readout">
    <header class="header">
      <div class="header-text">
        <span class="title">WORKSPACE</span>
        <span class="subtitle">J-lens</span>
      </div>
      <button
        type="button"
        class="toggle"
        class:on={enabled}
        disabled={lensState.busy}
        onclick={onToggle}
        title={enabled
          ? "Stop streaming the live workspace readout"
          : "Stream the per-layer J-lens top-k live during generation"}
      >
        {enabled ? "live: on" : "live: off"}
      </button>
    </header>

    {#if enabled}
      <div class="rows" role="list">
        {#each rows as row (row.layer)}
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
        {#if !lensState.readout}
          <p class="hint">streams on the next generation</p>
        {/if}
      </div>
    {/if}
  </section>
{/if}

<style>
  /* Third flat section of the inspector — divided from the probe rack
   * above by its own border-top hairline; sized by content (the parent
   * grid row is ``auto``), never stealing rack budget when off. */
  .workspace {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding: var(--space-5);
    border-top: 1px solid var(--border);
    background: transparent;
    min-height: 0;
    overflow: hidden;
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
  /* Match the rack titles so the three sections read as siblings. */
  .title {
    font-weight: var(--weight-bold);
    color: var(--accent);
    font-size: var(--text-sm);
    text-transform: uppercase;
  }
  .subtitle {
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }

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
  .hint {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
</style>
