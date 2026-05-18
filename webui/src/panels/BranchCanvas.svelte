<script lang="ts">
  import {
    loomNavigate,
    loomTree,
    openDrawer,
    requestLoomModal,
  } from "../lib/stores.svelte";
  import type { LoomNodeJSON } from "../lib/types";

  const active = $derived(
    loomTree.active_node_id ? loomTree.nodes.get(loomTree.active_node_id) ?? null : null,
  );

  const pathNodes = $derived.by(() =>
    loomTree.activePath
      .map((id) => loomTree.nodes.get(id))
      .filter((n): n is LoomNodeJSON => Boolean(n))
      .filter((n) => !(n.parent_id === null && n.role === "system" && !n.text)),
  );

  const siblings = $derived.by(() => {
    if (!active?.parent_id) return [];
    const ids = loomTree.children_of.get(active.parent_id) ?? [];
    return ids
      .map((id) => loomTree.nodes.get(id))
      .filter((n): n is LoomNodeJSON => Boolean(n));
  });

  const children = $derived.by(() => {
    if (!active) return [];
    const ids = loomTree.children_of.get(active.id) ?? [];
    return ids
      .map((id) => loomTree.nodes.get(id))
      .filter((n): n is LoomNodeJSON => Boolean(n));
  });

  const comparableSiblings = $derived(
    siblings.filter((node) => node.role === "assistant"),
  );

  const comparableChildren = $derived(
    children.filter((node) => node.role === "assistant"),
  );

  function preview(text: string): string {
    const compact = text.replace(/\s+/g, " ").trim();
    return compact.length > 138 ? `${compact.slice(0, 138)}…` : compact || "(empty)";
  }

  function recipe(node: LoomNodeJSON): string {
    return node.applied_steering ?? node.recipe?.steering ?? "unsteered";
  }

  function compareSiblings(): void {
    if (!active?.parent_id) return;
    openDrawer("node_compare", {
      node_ids: comparableSiblings.map((node) => node.id),
      parent_id: active.parent_id,
    });
  }

  function compareChildren(): void {
    if (!active) return;
    openDrawer("node_compare", {
      node_ids: comparableChildren.map((node) => node.id),
      parent_id: active.id,
    });
  }

  function siblingCountSuffix(n: number): string {
    return n === 1 ? "" : "s";
  }

  function childLabel(n: number): string {
    return n === 1 ? "child" : "children";
  }
</script>

{#if loomTree.rev > 0 && active}
  <section class="branch-canvas" aria-label="Loom branch canvas">
    <header>
      <div>
        <span class="eyebrow">branch canvas</span>
        <strong>rev {loomTree.rev} · {pathNodes.length} path nodes · {siblings.length} sibling{siblingCountSuffix(siblings.length)} · {children.length} {childLabel(children.length)}</strong>
      </div>
      <div class="actions">
        <button type="button" onclick={() => requestLoomModal("regenerate", { nodeId: active.id, n: 3 })}>
          regenerate N
        </button>
        <button type="button" onclick={() => openDrawer("experiment_lab")}>
          fan out
        </button>
        <button type="button" onclick={compareSiblings} disabled={comparableSiblings.length < 2}>
          compare siblings
        </button>
        <button type="button" onclick={compareChildren} disabled={comparableChildren.length < 2}>
          compare children
        </button>
      </div>
    </header>

    <div class="lanes">
      <div class="path lane">
        <span class="lane-title">active path</span>
        <div class="cards">
          {#each pathNodes as node (node.id)}
            <button
              type="button"
              class="node-card"
              class:active={node.id === loomTree.active_node_id}
              onclick={() => loomNavigate(node.id)}
            >
              <span class="role">{node.role}</span>
              <strong>{preview(node.text)}</strong>
              {#if node.role === "assistant"}
                <code>{recipe(node)}</code>
              {/if}
            </button>
          {/each}
        </div>
      </div>

      <div class="siblings lane">
        <span class="lane-title">siblings under current parent</span>
        <div class="cards">
          {#each siblings as node (node.id)}
            <button
              type="button"
              class="node-card sibling"
              class:active={node.id === loomTree.active_node_id}
              onclick={() => loomNavigate(node.id)}
            >
              <span class="role">{node.role}</span>
              <strong>{preview(node.text)}</strong>
              <div class="metrics">
                <span>{node.finish_reason ?? "draft"}</span>
                {#if typeof node.mean_logprob === "number"}
                  <span>mean lp {node.mean_logprob.toFixed(2)}</span>
                {/if}
                {#if node.aggregate_readings}
                  <span>{Object.keys(node.aggregate_readings).length} probes</span>
                {/if}
              </div>
            </button>
          {/each}
        </div>
      </div>

      <div class="children lane">
        <span class="lane-title">children of active node</span>
        <div class="cards">
          {#if children.length === 0}
            <div class="empty-card">no children yet</div>
          {:else}
            {#each children as node (node.id)}
              <button
                type="button"
                class="node-card child"
                onclick={() => loomNavigate(node.id)}
              >
                <span class="role">{node.role}</span>
                <strong>{preview(node.text)}</strong>
                <div class="metrics">
                  <span>{node.finish_reason ?? "draft"}</span>
                  {#if typeof node.mean_logprob === "number"}
                    <span>mean lp {node.mean_logprob.toFixed(2)}</span>
                  {/if}
                  {#if node.aggregate_readings}
                    <span>{Object.keys(node.aggregate_readings).length} probes</span>
                  {/if}
                </div>
              </button>
            {/each}
          {/if}
        </div>
      </div>
    </div>
  </section>
{/if}

<style>
  .branch-canvas {
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    gap: 0.65rem;
    min-height: 15.5rem;
    max-height: 20rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    padding: 0.75rem;
    margin-bottom: 0.75rem;
    overflow: hidden;
  }

  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
  }

  .eyebrow,
  .lane-title,
  .role,
  .metrics span {
    color: var(--fg-muted);
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0;
  }

  header strong {
    display: block;
    margin-top: 0.2rem;
    color: var(--fg-strong);
    font-weight: 600;
    letter-spacing: 0;
  }

  .actions {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    justify-content: flex-end;
  }

  button {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: rgba(255, 255, 255, 0.025);
    color: var(--fg);
    padding: 0.45rem 0.6rem;
  }

  button:hover:not(:disabled) {
    border-color: var(--accent);
    color: var(--accent);
  }

  button:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .lanes {
    display: grid;
    grid-template-columns: minmax(0, 0.95fr) minmax(0, 1fr) minmax(0, 1fr);
    gap: 0.65rem;
    min-height: 0;
  }

  .lane {
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    gap: 0.4rem;
    min-height: 0;
  }

  .cards {
    display: flex;
    gap: 0.5rem;
    overflow: auto;
    padding-bottom: 0.25rem;
  }

  .node-card {
    flex: 0 0 13.5rem;
    display: grid;
    grid-template-rows: auto 1fr auto;
    align-content: start;
    gap: 0.35rem;
    min-height: 7.25rem;
    text-align: left;
    border-color: var(--border-dim);
    background: rgba(5, 8, 13, 0.56);
  }

  .empty-card {
    flex: 0 0 13.5rem;
    min-height: 7.25rem;
    display: grid;
    place-items: center;
    border: 1px dashed var(--border-dim);
    border-radius: var(--radius);
    color: var(--fg-muted);
    font-size: 0.74rem;
    background: rgba(5, 8, 13, 0.38);
  }

  .node-card.active {
    border-color: rgba(72, 138, 203, 0.72);
    box-shadow: inset 0 0 0 1px rgba(72, 138, 203, 0.22);
  }

  .node-card strong {
    color: var(--fg);
    font-size: 0.78rem;
    line-height: 1.35;
    font-weight: 600;
    letter-spacing: 0;
  }

  .node-card code {
    color: var(--accent-amber);
    font-size: 0.66rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .metrics {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
  }
</style>
