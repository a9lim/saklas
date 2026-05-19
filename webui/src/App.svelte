<script lang="ts">
  // saklas workbench shell.  The primary frame is a desktop research
  // cockpit: rail navigation, the threads (loom) column, central
  // chat/canvas, right-side inspector, and a wide drawer host for deep
  // tools.

  import { onMount } from "svelte";

  import WorkspaceRail from "./panels/WorkspaceRail.svelte";
  import InspectorPanel from "./panels/InspectorPanel.svelte";
  import Chat from "./panels/Chat.svelte";
  import LoomSidebar from "./panels/loom/LoomSidebar.svelte";
  import Toaster from "./lib/Toaster.svelte";

  import * as Drawers from "./drawers";
  import PackDrawer from "./drawers/PackDrawer.svelte";
  import MergeDrawer from "./drawers/MergeDrawer.svelte";
  import CloneDrawer from "./drawers/CloneDrawer.svelte";
  import TokenDrilldownDrawer from "./drawers/TokenDrilldownDrawer.svelte";

  import {
    bootstrap,
    ensureWebSocket,
    drawerState,
    closeDrawer,
    genStatus,
    sendStop,
    loomTree,
    loomUiState,
    loomRegenerateActive,
    requestLoomModal,
  } from "./lib/stores.svelte";

  import type { DrawerName } from "./lib/types";

  // Content-driven drawer sizing — forms and pickers get a narrow panel,
  // analysis views keep the wide one (docs/plans/webui-overhaul.md §8).
  const NARROW_DRAWERS: ReadonlySet<DrawerName> = new Set<DrawerName>([
    "vector_picker",
    "probe_picker",
    "load",
    "merge",
    "clone",
    "system_prompt",
    "save_conversation",
    "load_conversation",
  ]);

  type BootStatus = "loading" | "ready" | "failed";
  let bootStatus: BootStatus = $state("loading");
  let bootError: string | null = $state(null);

  async function runBootstrap(): Promise<void> {
    bootStatus = "loading";
    bootError = null;
    try {
      await bootstrap();
      // Open the WS eagerly so the first generate doesn't pay connect
      // latency.  Failure here is non-fatal — we'll re-attempt on send.
      try {
        await ensureWebSocket();
      } catch {
        /* ignore — sendGenerate will retry */
      }
      bootStatus = "ready";
    } catch (e) {
      bootError = e instanceof Error ? e.message : String(e);
      bootStatus = "failed";
    }
  }

  onMount(() => {
    void runBootstrap();
  });

  // Global keyboard accelerators.  Esc → stop (matches TUI).  Cmd/Ctrl-
  // Enter is left for the chat input to handle locally.
  //
  // Loom (phase 3): Ctrl/Cmd+R/E/B/N/D fire the corresponding tree op
  // via the sidebar's modal flow.  Browser Ctrl+B (bold) is suppressed
  // via ``preventDefault`` per Decision 9.
  async function onWindowKey(ev: KeyboardEvent) {
    // Escape priority (most-targeted close first):
    //   1. open loom modal / menu — let the sidebar's own Esc handler
    //      (LoomSidebar.svelte::onWindowKey) close it.  We DON'T
    //      preventDefault here so its listener still fires.
    //   2. open drawer — close it.
    //   3. fall-through: stop in-flight gen.
    //
    // The earlier order (gen-stop first) made Esc-during-stream-with-
    // modal-open stop the gen instead of closing the modal — surprising
    // for the n-way regen flow where a user might want to back out of
    // a follow-up modal without killing the stream.
    if (ev.key === "Escape") {
      if (loomUiState.modalRequest.kind !== null) {
        return;
      }
      if (drawerState.open !== null) {
        closeDrawer();
        ev.preventDefault();
        return;
      }
      if (genStatus.active) {
        sendStop();
        ev.preventDefault();
        return;
      }
    }

    if (loomTree.unavailable) return;
    const mod = ev.ctrlKey || ev.metaKey;
    if (!mod) return;
    // Shift+ctrl combos fall through to the browser; the loom shortcuts
    // use bare Cmd/Ctrl+key.
    if (ev.shiftKey) return;
    const k = ev.key.toLowerCase();

    if (k === "r") {
      ev.preventDefault();
      // Ctrl+R = regenerate active assistant (N=1, current rack).
      const active = loomTree.active_node_id;
      if (!active) return;
      const node = loomTree.nodes.get(active);
      if (node?.role === "assistant") {
        await loomRegenerateActive(1);
      } else {
        // Active is a user node — open the modal to let the user pick N
        // and confirm.
        requestLoomModal("regenerate", { nodeId: active, n: 1 });
      }
      return;
    }
    if (k === "e") {
      ev.preventDefault();
      const active = loomTree.active_node_id;
      if (!active) return;
      const node = loomTree.nodes.get(active);
      requestLoomModal("edit", { nodeId: active, text: node?.text ?? "" });
      return;
    }
    if (k === "b") {
      ev.preventDefault();
      const active = loomTree.active_node_id;
      if (!active) return;
      const node = loomTree.nodes.get(active);
      requestLoomModal("branch", { nodeId: active, text: node?.text ?? "" });
      return;
    }
    if (k === "n") {
      ev.preventDefault();
      requestLoomModal("navpicker", { nodeId: loomTree.active_node_id });
      return;
    }
    if (k === "d") {
      ev.preventDefault();
      const active = loomTree.active_node_id;
      if (!active) return;
      requestLoomModal("delete", { nodeId: active });
      return;
    }
  }
</script>

<svelte:window onkeydown={onWindowKey} />

{#if bootStatus === "failed"}
  <div class="boot-failed" role="alert">
    <h1>connection failed</h1>
    <p class="message">{bootError}</p>
    <p class="hint">
      saklas server unreachable.  Is <code>saklas serve</code> running?
    </p>
    <button type="button" class="retry" onclick={runBootstrap}>retry</button>
  </div>
{:else}
  <div class="shell" class:loading={bootStatus === "loading"}>
    <main class="layout">
      <section class="rail-zone" aria-label="Workspace navigation">
        <WorkspaceRail />
      </section>

      <section class="loom-zone" aria-label="Threads">
        <LoomSidebar />
      </section>

      <section class="chat-zone" aria-label="Chat">
        <Chat />
      </section>

      <section class="rack-zone" aria-label="Control rack">
        <InspectorPanel />
      </section>

      {#if drawerState.open !== null}
        <div
          class="drawer-backdrop"
          role="button"
          tabindex="-1"
          aria-label="Close drawer"
          onclick={closeDrawer}
          onkeydown={(ev) => {
            if (ev.key === "Enter" || ev.key === " ") closeDrawer();
          }}
        ></div>
        <aside
          class="drawer"
          class:narrow={NARROW_DRAWERS.has(drawerState.open)}
          aria-label="{drawerState.open} drawer"
        >
          {#if drawerState.open === "load"}
            <Drawers.Load params={drawerState.params} />
          {:else if drawerState.open === "vector_picker"}
            <Drawers.VectorPicker params={drawerState.params} />
          {:else if drawerState.open === "probe_picker"}
            <Drawers.ProbePicker params={drawerState.params} />
          {:else if drawerState.open === "save_conversation"}
            <Drawers.SaveConversation params={drawerState.params} />
          {:else if drawerState.open === "load_conversation"}
            <Drawers.LoadConversation params={drawerState.params} />
          {:else if drawerState.open === "compare"}
            <Drawers.Compare params={drawerState.params} />
          {:else if drawerState.open === "system_prompt"}
            <Drawers.SystemPrompt params={drawerState.params} />
          {:else if drawerState.open === "help"}
            <Drawers.Help params={drawerState.params} />
          {:else if drawerState.open === "export"}
            <Drawers.Export params={drawerState.params} />
          {:else if drawerState.open === "pack"}
            <PackDrawer params={drawerState.params} />
          {:else if drawerState.open === "merge"}
            <MergeDrawer params={drawerState.params} />
          {:else if drawerState.open === "clone"}
            <CloneDrawer params={drawerState.params} />
          {:else if drawerState.open === "token_drilldown"}
            <TokenDrilldownDrawer params={drawerState.params} />
          {:else if drawerState.open === "correlation"}
            <Drawers.Correlation params={drawerState.params} />
          {:else if drawerState.open === "layer_norms"}
            <Drawers.LayerNorms params={drawerState.params} />
          {:else if drawerState.open === "experiment_lab"}
            <Drawers.ExperimentLab params={drawerState.params} />
          {:else if drawerState.open === "activation_atlas"}
            <Drawers.ActivationAtlas params={drawerState.params} />
          {:else if drawerState.open === "recipe_builder"}
            <Drawers.RecipeBuilder params={drawerState.params} />
          {:else if drawerState.open === "advanced_sampling"}
            <Drawers.AdvancedSampling params={drawerState.params} />
          {:else if drawerState.open === "health"}
            <Drawers.Health params={drawerState.params} />
          {:else if drawerState.open === "session_admin"}
            <Drawers.SessionAdmin params={drawerState.params} />
          {:else if drawerState.open === "node_compare"}
            <Drawers.NodeCompare params={drawerState.params} />
          {:else if drawerState.open === "transcript"}
            <Drawers.Transcript params={drawerState.params} />
          {:else}
            <header class="drawer-header">
              <span class="drawer-title">{drawerState.open}</span>
              <button
                type="button"
                class="drawer-close"
                aria-label="Close"
                onclick={closeDrawer}
              >✕</button>
            </header>
            <div class="drawer-body">
              <p class="stub">unknown drawer: {drawerState.open}</p>
            </div>
          {/if}
        </aside>
      {/if}
    </main>

    <Toaster />
  </div>
{/if}

<style>
  .shell {
    display: grid;
    grid-template-rows: 1fr;
    height: 100vh;
    width: 100vw;
    min-width: 1280px;
    min-height: 720px;
    background: var(--bg);
    color: var(--fg);
    overflow: hidden;
  }
  .shell.loading {
    /* Slight desaturation so users can tell bootstrap hasn't finished
     * without us blocking the entire frame. */
    opacity: 0.85;
  }
  /* Four permanent columns: rail · threads · chat · rack.  The threads
   * (loom) column is 340px, the rack column 440px; min-width 1280px
   * keeps the chat column usable (1280 − 64 − 340 − 440 ≈ 436px floor). */
  .layout {
    display: grid;
    grid-template-columns: 64px 340px minmax(0, 1fr) 440px;
    grid-template-rows: 1fr;
    min-height: 0; /* let children scroll inside */
    position: relative; /* drawer sits over rack-zone via absolute pos */
    background: var(--grid-line);
    gap: 1px;
    /* Clip the drawer's translateX(100%) entry — without this the
     * offscreen-right starting position extends the page's horizontal
     * overflow by ~640px, the body scrolls during the 160ms animation,
     * and the chat/rack content visibly shifts left then snaps back. */
    overflow: hidden;
  }
  .rail-zone {
    background: var(--bg-deep);
    overflow: hidden;
    min-height: 0;
  }
  .loom-zone {
    background: var(--bg-alt);
    overflow: hidden;
    min-height: 0;
  }
  .chat-zone,
  .rack-zone {
    background: var(--bg);
    overflow: hidden;
    min-height: 0;
  }
  .chat-zone {
    display: flex;
    flex-direction: column;
    padding: var(--space-5);
  }
  /* Two-row grid: steering rack and probe rack.  Reference views
   * (correlation N×N, per-name layer norms) live in drawer overlays
   * launched from the workspace rail — keeping them out of the rack
   * zone gives both racks the full vertical budget.  Each rack handles
   * its own internal scroll so its actions row stays anchored. */
  .rack-zone {
    min-height: 0;
    overflow: hidden;
  }

  /* Drawer host — slides in from the right over the rack zone.  Backdrop
   * also covers the chat zone so the focus is unambiguous. */
  .drawer-backdrop {
    position: absolute;
    inset: 0;
    background: rgba(1, 4, 9, 0.55);
    z-index: var(--z-drawer);
    border: 0;
    cursor: pointer;
  }
  .drawer {
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    width: min(980px, 78%);
    background: var(--bg-alt);
    border-left: 1px solid var(--border);
    z-index: calc(var(--z-drawer) + 1);
    display: flex;
    flex-direction: column;
    box-shadow: var(--shadow-overlay);
    animation: drawer-in var(--dur) var(--ease-out);
  }
  /* Forms / pickers — sized to their content rather than the wide
   * analysis panel. */
  .drawer.narrow {
    width: min(480px, 92%);
  }
  @keyframes drawer-in {
    from {
      transform: translateX(100%);
    }
    to {
      transform: translateX(0);
    }
  }
  .drawer-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-3) var(--space-6);
    border-bottom: 1px solid var(--border);
  }
  .drawer-title {
    color: var(--accent-blue);
    font-size: var(--text);
    text-transform: lowercase;
    letter-spacing: 0;
  }
  .drawer-close {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    font-size: var(--text);
    line-height: 1;
    padding: var(--space-2) var(--space-3);
  }
  .drawer-close:hover {
    color: var(--accent-red);
  }
  .drawer-body {
    flex: 1;
    overflow-y: auto;
    padding: var(--space-5);
  }
  .stub {
    color: var(--fg-strong);
    font-size: var(--text);
    margin: 0 0 var(--space-3) 0;
  }

  /* Boot-failed gate — sits over the whole viewport since the rest of
   * the shell can't function without a session. */
  .boot-failed {
    position: fixed;
    inset: 0;
    background: var(--bg-deep);
    color: var(--fg);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-4);
    padding: var(--space-8);
    text-align: center;
  }
  .boot-failed h1 {
    color: var(--accent-red);
    margin: 0;
    font-size: var(--text-lg);
  }
  .boot-failed .message {
    color: var(--fg-dim);
    font-family: var(--font-mono);
    margin: 0;
    max-width: 70ch;
    word-break: break-word;
  }
  .boot-failed .hint {
    color: var(--fg-muted);
    margin: 0;
    font-size: var(--text-sm);
  }
  .boot-failed code {
    background: var(--bg-elev);
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius);
    color: var(--accent-blue);
  }
  .retry {
    margin-top: var(--space-5);
    background: var(--bg-elev);
    color: var(--accent-blue);
    border: 1px solid var(--border);
    padding: var(--space-3) var(--space-6);
    border-radius: var(--radius);
    font-size: var(--text-sm);
    transition: background var(--dur) var(--ease-out);
  }
  .retry:hover {
    background: var(--accent-subtle);
  }
</style>
