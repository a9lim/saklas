<script lang="ts">
  // The recipe bar — the composed steering expression made permanently
  // visible.  Every racked term renders as a pillar-colored chip regardless
  // of which instrument tab is open: subspace white, manifold violet,
  // j-lens blue (sae gold, once the runtime lands).  Click a chip to jump
  // to its pillar; × removes the term; disabled terms show muted.  The ⧉
  // copies the canonical expression string — the exact text the WS
  // ``steering`` field carries.

  import Chip from "../lib/ui/Chip.svelte";
  import {
    steerRack,
    setInspectorTab,
    removeSubspaceFromRack,
    removeManifoldFromRack,
    removeJLensFromRack,
    removeSaeFromRack,
    currentSteeringExpression,
    openDrawer,
  } from "../lib/stores.svelte";
  import type { InspectorTab } from "../lib/stores.svelte";
  import {
    formatSubspaceTerm,
    formatManifoldTerm,
    formatJLensTerm,
    formatSaeTerm,
  } from "../lib/expression";
  import type { SteerEntry } from "../lib/types";
  import { pushToast } from "../lib/stores/toasts.svelte";

  interface ChipModel {
    name: string;
    text: string;
    color: string;
    tab: InspectorTab;
    /** Serialization-order group: subspace 0, jlens 1, manifold 2. */
    order: number;
    enabled: boolean;
    remove: () => void;
  }

  function chipFor(name: string, entry: SteerEntry): ChipModel {
    switch (entry.mode) {
      case "subspace":
        return {
          name,
          text: formatSubspaceTerm(name, entry, steerRack.subspaceAlong),
          color: "var(--pillar-subspace)",
          tab: "subspace",
          order: 0,
          enabled: entry.enabled,
          remove: () => removeSubspaceFromRack(name),
        };
      case "manifold":
        return {
          name,
          text: formatManifoldTerm(name, entry),
          color: "var(--pillar-manifold)",
          tab: "manifold",
          order: 2,
          enabled: entry.enabled,
          remove: () => removeManifoldFromRack(name),
        };
      case "jlens":
        return {
          name,
          text: formatJLensTerm(name, entry),
          color: "var(--pillar-lens)",
          tab: "lens",
          order: 1,
          enabled: entry.enabled,
          remove: () => removeJLensFromRack(name),
        };
      case "sae":
        return {
          name,
          text: formatSaeTerm(name, entry),
          color: "var(--pillar-sae)",
          tab: "sae",
          order: 1,
          enabled: entry.enabled,
          remove: () => removeSaeFromRack(name),
        };
    }
  }

  const chips = $derived.by(() => {
    const arr = [...steerRack.entries.entries()].map(([name, entry]) =>
      chipFor(name, entry),
    );
    arr.sort(
      (a, b) => a.order - b.order || a.name.localeCompare(b.name),
    );
    return arr;
  });

  const expression = $derived(currentSteeringExpression());
  const custom = $derived(steerRack.customExpression !== null);

  async function copyExpression(): Promise<void> {
    if (!expression) return;
    try {
      await navigator.clipboard.writeText(expression);
      pushToast("expression copied", { kind: "info" });
    } catch {
      pushToast("clipboard unavailable", { kind: "error" });
    }
  }
</script>

<div class="recipe" title={expression || "no active steering"}>
  <button
    type="button"
    class="lbl edit"
    title="edit steering recipe"
    aria-label="Edit steering recipe"
    onclick={() => openDrawer("recipe_builder")}
  >recipe</button>
  {#if custom}
    <button
      type="button"
      class="custom-expression"
      title="advanced expression — click to edit"
      onclick={() => openDrawer("recipe_builder")}
    >
      <span class="custom-badge">custom</span>
      <code>{expression || "unsteered"}</code>
    </button>
  {:else if chips.length === 0}
    <button
      type="button"
      class="empty edit-empty"
      onclick={() => openDrawer("recipe_builder")}
    >none · edit</button>
  {:else}
    <div class="chips">
      {#each chips as chip (chip.name)}
        <Chip
          color={chip.color}
          muted={!chip.enabled}
          title={chip.enabled ? chip.text : `${chip.text} (disabled)`}
          onclick={() => setInspectorTab(chip.tab)}
          onremove={chip.remove}
          removeLabel={`Remove ${chip.name} from steering recipe`}
        >
          {chip.text}
        </Chip>
      {/each}
    </div>
  {/if}
  {#if expression}
    <button
      type="button"
      class="copy"
      title="copy expression"
      aria-label="Copy steering expression"
      onclick={copyExpression}
    >⧉</button>
  {/if}
</div>

<style>
  /* A quiet glass well with the faintest lens-blue drift at its right
   * edge — material, not data (the chips carry the meaning). */
  .recipe {
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
    margin: var(--space-4) var(--space-4) 0;
    border-radius: var(--radius);
    border: 1px solid transparent;
    background: linear-gradient(
      100deg,
      var(--glass),
      var(--top-light-faint) 55%,
      color-mix(in srgb, var(--pillar-lens) 5%, transparent)
    );
  }
  .lbl {
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--fg-muted);
    padding-top: 3px;
    flex: none;
  }
  button.lbl,
  .edit-empty,
  .custom-expression {
    border: 0;
    background: transparent;
    cursor: pointer;
  }
  button.lbl:hover,
  .edit-empty:hover {
    color: var(--fg);
  }
  .empty {
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    color: var(--fg-muted);
    padding-top: 1px;
  }
  .edit-empty {
    padding: 1px 0 0;
  }
  .custom-expression {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex: 1 1 auto;
    min-width: 0;
    padding: 0;
    text-align: left;
  }
  .custom-expression code {
    color: var(--fg-strong);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .custom-badge {
    flex: none;
    color: var(--accent-amber);
    background: color-mix(in srgb, var(--accent-amber) 12%, transparent);
    border-radius: var(--radius-sm);
    padding: 1px var(--space-2);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    flex: 1;
    min-width: 0;
    /* Two chip rows before the bar scrolls — the expression stays
     * glanceable without eating the instrument column. */
    max-height: 60px;
    overflow-y: auto;
  }
  .copy {
    flex: none;
    background: transparent;
    border: 0;
    color: var(--fg-muted);
    font-size: var(--text-sm);
    line-height: 1;
    padding: 3px var(--space-2);
    border-radius: var(--radius-sm);
    transition: color var(--dur-fast) var(--ease-out);
  }
  .copy:hover {
    color: var(--fg);
    background: var(--bg-hover);
  }
</style>
