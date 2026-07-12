<script lang="ts">
  // saklas workbench shell.  The primary frame is a desktop research
  // cockpit: rail navigation, the threads (loom) column, central
  // chat/canvas, right-side inspector, and a wide drawer host for deep
  // tools.

  import { onMount, tick } from "svelte";

  import InspectorPanel from "./panels/InspectorPanel.svelte";
  import Chat from "./panels/Chat.svelte";
  import LoomSidebar from "./panels/loom/LoomSidebar.svelte";
  import CommandPalette from "./panels/CommandPalette.svelte";
  import Toaster from "./lib/Toaster.svelte";
  import {
    paletteState,
    closePalette,
    togglePalette,
  } from "./lib/stores/palette.svelte";

  import * as Drawers from "./drawers";

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
    "subspace",
    "manifolds",
    "manifold_builder",
    "system_prompt",
    "save_conversation",
    "load_conversation",
  ]);

  type BootStatus = "loading" | "ready" | "failed";
  let bootStatus: BootStatus = $state("loading");
  let bootError: string | null = $state(null);
  let drawerEl: HTMLElement | null = $state(null);

  const FOCUSABLE = [
    "button:not([disabled])",
    "[href]",
    "input:not([disabled])",
    "select:not([disabled])",
    "textarea:not([disabled])",
    '[tabindex]:not([tabindex="-1"])',
  ].join(",");

  $effect(() => {
    if (drawerState.open === null) return;
    void tick().then(() => {
      const first = drawerEl?.querySelector<HTMLElement>(FOCUSABLE);
      (first ?? drawerEl)?.focus();
    });
  });

  function onDrawerKeydown(ev: KeyboardEvent): void {
    if (ev.key !== "Tab" || !drawerEl) return;
    const focusable = [...drawerEl.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(
      (el) => el.offsetParent !== null,
    );
    if (focusable.length === 0) {
      ev.preventDefault();
      drawerEl.focus();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (ev.shiftKey && (document.activeElement === first || document.activeElement === drawerEl)) {
      ev.preventDefault();
      last.focus();
    } else if (!ev.shiftKey && document.activeElement === last) {
      ev.preventDefault();
      first.focus();
    }
  }

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
      // Palette overlays everything — its own input handler closes it when
      // focused; this catches Esc after focus wandered (backdrop click-arm,
      // devtools, etc.).
      if (paletteState.open) {
        closePalette();
        ev.preventDefault();
        return;
      }
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

    const mod = ev.ctrlKey || ev.metaKey;
    // ⌘K — the command palette; works everywhere, including pre-boot and
    // before the tree is loaded (it's pure navigation).
    if (mod && !ev.shiftKey && ev.key.toLowerCase() === "k") {
      ev.preventDefault();
      togglePalette();
      return;
    }

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
    <main
      class="layout"
      inert={paletteState.open || bootStatus !== "ready"}
      aria-busy={bootStatus === "loading"}
    >
      <section class="loom-zone" aria-label="Threads" inert={drawerState.open !== null}>
        <LoomSidebar />
      </section>

      <section class="chat-zone" aria-label="Chat" inert={drawerState.open !== null}>
        <Chat />
      </section>

      <section class="rack-zone" aria-label="Control rack" inert={drawerState.open !== null}>
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
        <div
          bind:this={drawerEl}
          class="drawer"
          class:narrow={NARROW_DRAWERS.has(drawerState.open)}
          role="dialog"
          aria-modal="true"
          aria-label="{drawerState.open} drawer"
          tabindex="-1"
          onkeydown={onDrawerKeydown}
        >
          {#if drawerState.open === "subspace"}
            <Drawers.RackDrawer
              params={{
                ...(drawerState.params as Record<string, unknown>),
                family: "subspace",
              }}
            />
          {:else if drawerState.open === "manifolds"}
            <Drawers.RackDrawer
              params={{
                ...(drawerState.params as Record<string, unknown>),
                family: "manifold",
              }}
            />
          {:else if drawerState.open === "manifold_builder"}
            <Drawers.ManifoldBuilder params={drawerState.params} />
          {:else if drawerState.open === "manifold_merge"}
            <Drawers.ManifoldMerge params={drawerState.params} />
          {:else if drawerState.open === "manifold_pack"}
            <Drawers.ManifoldPack params={drawerState.params} />
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
          {:else if drawerState.open === "token_drilldown"}
            <Drawers.TokenDrilldown params={drawerState.params} />
          {:else if drawerState.open === "correlation"}
            <Drawers.Correlation params={drawerState.params} />
          {:else if drawerState.open === "layer_norms"}
            <Drawers.LayerNorms params={drawerState.params} />
          {:else if drawerState.open === "probe_inspector"}
            <Drawers.ProbeInspector params={drawerState.params} />
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
          {:else if drawerState.open === "template_lab"}
            <Drawers.TemplateLab params={drawerState.params} />
          {:else if drawerState.open === "cast"}
            <Drawers.Cast params={drawerState.params} />
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
        </div>
      {/if}
    </main>

    {#if bootStatus === "loading"}
      <div class="boot-loading" role="status" aria-live="polite">
        <span class="boot-spinner" aria-hidden="true"></span>
        <span>loading workbench</span>
      </div>
    {/if}

    <Toaster />
    <CommandPalette />
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
    /* Keep the real frame visible for orientation while the explicit
     * readiness veil below prevents controls racing their source data. */
    color-scheme: dark;
  }
  .shell.loading .layout {
    filter: saturate(0.72) brightness(0.82);
  }
  .boot-loading {
    position: fixed;
    inset: 0;
    z-index: calc(var(--z-drawer) + 20);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-3);
    color: var(--fg-strong);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    letter-spacing: 0.04em;
    text-transform: lowercase;
    background: rgba(2, 3, 8, 0.38);
    backdrop-filter: blur(1px);
    pointer-events: none;
  }
  .boot-spinner {
    width: 12px;
    height: 12px;
    border: 1px solid color-mix(in srgb, var(--accent) 35%, transparent);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: boot-spin 0.7s linear infinite;
  }
  @keyframes boot-spin {
    to { transform: rotate(360deg); }
  }
  /* Three permanent columns: threads · chat · rack.  The threads (loom)
   * column is 376px, the rack column 432px; min-width 1280px keeps the
   * chat column usable (1280 − 376 − 432 ≈ 472px floor).  The former
   * 56px workspace rail is gone — every tool it launched lives in the
   * ⌘K command palette, hinted from the chat header. */
  .layout {
    display: grid;
    grid-template-columns: 376px minmax(0, 1fr) 432px;
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

  /* Sheet host — analysis tools float in from the right as a rounded
   * sheet inset from the frame edges (v2: no full-height wall).  Backdrop
   * blurs the bench underneath so the live data reads as "behind", not
   * "gone". */
  .drawer-backdrop {
    position: absolute;
    inset: 0;
    background: rgba(2, 3, 8, 0.5);
    backdrop-filter: blur(2px);
    z-index: var(--z-drawer);
    border: 0;
    cursor: pointer;
  }
  .drawer {
    position: absolute;
    top: var(--space-4);
    right: var(--space-4);
    bottom: var(--space-4);
    width: min(980px, 78%);
    background: var(--bg-alt);
    border: 1px solid var(--glass-line);
    border-radius: var(--radius-lg);
    z-index: calc(var(--z-drawer) + 1);
    display: flex;
    flex-direction: column;
    box-shadow:
      inset 0 1px 0 rgba(255, 255, 255, 0.04),
      var(--shadow-overlay);
    overflow: hidden;
    animation: drawer-in var(--dur-slow) var(--ease-out);
  }
  /* Forms / pickers — sized to their content rather than the wide
   * analysis panel. */
  .drawer.narrow {
    width: min(480px, 92%);
  }
  @keyframes drawer-in {
    from {
      transform: translateX(28px);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }
  .drawer-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-3) var(--space-6);
  }
  .drawer-title {
    color: var(--accent);
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
    color: var(--accent);
  }
  .retry {
    margin-top: var(--space-5);
    background: var(--bg-elev);
    color: var(--accent);
    border: 0;
    padding: var(--space-3) var(--space-6);
    border-radius: var(--radius);
    font-size: var(--text-sm);
    transition: background var(--dur) var(--ease-out);
  }
  .retry:hover {
    background: var(--accent-subtle);
  }
</style>
