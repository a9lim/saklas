<script lang="ts">
  // Render a discover-mode manifold's PCA variance bars or spectral
  // eigenvalue spectrum, with the picked-k cut highlighted.  Used by
  // ManifoldDrawer's per-row inspector; safe to drop into any other
  // surface that has a ManifoldInfo on hand (the panel skips itself
  // gracefully on authored fits / fits with no diagnostics).

  import type { ManifoldInfo } from "../types";
  import {
    classifyDiagnostics,
    diagnosticsSummary,
    pcaBars,
    pickDiscoverFit,
    spectralBars,
    type DiagnosticsBar,
  } from "./diagnostics";

  let { manifold }: { manifold: ManifoldInfo } = $props();

  const fit = $derived(pickDiscoverFit(manifold.fitted));
  const diag = $derived(fit ? classifyDiagnostics(fit) : null);
  const bars: DiagnosticsBar[] = $derived(
    diag === null
      ? []
      : diag.kind === "pca"
        ? pcaBars(diag)
        : spectralBars(diag),
  );
</script>

{#if diag !== null}
  <div class="diag" data-kind={diag.kind}>
    <p class="summary">{diagnosticsSummary(diag)}</p>
    <ol class="bars" aria-label={`${diag.kind} diagnostics`}>
      {#each bars as bar (bar.index)}
        <li
          class="bar"
          class:picked={bar.picked}
          title={`#${bar.index} · ${bar.value.toPrecision(4)}${bar.picked ? " (kept)" : ""}`}
        >
          <span class="bar-fill" style="height: {Math.max(2, bar.frac * 100)}%"
          ></span>
          <span class="bar-label">{bar.index}</span>
        </li>
      {/each}
    </ol>
    {#if diag.kind === "spectral" && diag.component_count > 1}
      <p class="warn">
        graph disconnected ({diag.component_count} components) —
        increase k_nn or switch to pca
      </p>
    {/if}
    {#if fit && fit.hyperparams}
      <p class="hyperparams">
        hyperparams:
        {#each Object.entries(fit.hyperparams) as [k, v] (k)}
          <code>{k}={typeof v === "number" ? v.toString() : v}</code>
        {/each}
      </p>
    {/if}
  </div>
{:else if manifold.fit_mode === "pca" || manifold.fit_mode === "spectral"}
  <p class="muted">
    no diagnostics yet — discover-mode coords + diagnostics land after
    the first fit.
  </p>
{/if}

<style>
  .diag {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    padding: var(--space-3);
    background: var(--bg-deep);
    border: 1px solid var(--border);
    border-radius: var(--radius);
  }
  .summary {
    margin: 0;
    color: var(--fg-strong);
    font-size: var(--text-xs);
    font-family: var(--font-mono);
  }
  .bars {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 72px;
    border-bottom: 1px solid var(--border);
  }
  .bar {
    flex: 1 1 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
    height: 100%;
    position: relative;
    min-width: 6px;
  }
  .bar-fill {
    width: 100%;
    background: var(--fg-muted);
    border-radius: 1px 1px 0 0;
    transition: background var(--dur) var(--ease-out);
  }
  .bar.picked .bar-fill {
    background: var(--accent);
  }
  .bar-label {
    position: absolute;
    bottom: -1.2em;
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    font-family: var(--font-mono);
  }
  .hyperparams {
    margin: var(--space-2) 0 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }
  .hyperparams code {
    background: var(--bg-elev);
    padding: 0 var(--space-2);
    border-radius: var(--radius);
    color: var(--fg);
  }
  .warn {
    margin: 0;
    color: var(--accent-yellow);
    font-size: var(--text-xs);
  }
  .muted {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-style: italic;
  }
</style>
