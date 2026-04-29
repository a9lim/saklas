<script lang="ts">
  // saklas v1.7 layout shell.  Two-column main (chat / control rack) with
  // a topbar above and status footer below.  Drawer host slides in from
  // the right over the rack zone when ``drawerState.open !== null``.
  //
  // Wave-3 panels are stubbed with placeholder boxes — the visual frame
  // is the deliverable here, content swaps in via /panels/*.svelte
  // imports in later phases.

  import { onMount } from "svelte";

  import Topbar from "./panels/Topbar.svelte";
  import StatusFooter from "./panels/StatusFooter.svelte";
  import Chat from "./panels/Chat.svelte";
  import SamplingStrip from "./panels/SamplingStrip.svelte";
  import SteeringRack from "./panels/SteeringRack.svelte";
  import ProbeRack from "./panels/ProbeRack.svelte";
  import ReferenceCollapsibles from "./panels/ReferenceCollapsibles.svelte";

  import * as Drawers from "./drawers";
  import SweepDrawer from "./drawers/SweepDrawer.svelte";
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
  } from "./lib/stores.svelte";

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
  // Shift-R → regen (handled in Topbar via its button + this accelerator
  // duplicates the regen click).  Cmd/Ctrl-Enter is left for the chat
  // input to handle locally.
  function onWindowKey(ev: KeyboardEvent) {
    // Escape: stop in-flight gen, then close drawer if any.
    if (ev.key === "Escape") {
      if (genStatus.active) {
        sendStop();
        ev.preventDefault();
        return;
      }
      if (drawerState.open !== null) {
        closeDrawer();
        ev.preventDefault();
        return;
      }
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
    <Topbar />

    <main class="layout">
      <section class="chat-zone" aria-label="Chat">
        <Chat />
        <SamplingStrip />
      </section>

      <section class="rack-zone" aria-label="Control rack">
        <SteeringRack />
        <ProbeRack />
        <ReferenceCollapsibles />
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
        <aside class="drawer" aria-label="{drawerState.open} drawer">
          {#if drawerState.open === "extract"}
            <Drawers.Extract params={drawerState.params} />
          {:else if drawerState.open === "load"}
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
          {:else if drawerState.open === "model_info"}
            <Drawers.ModelInfo params={drawerState.params} />
          {:else if drawerState.open === "help"}
            <Drawers.Help params={drawerState.params} />
          {:else if drawerState.open === "export"}
            <Drawers.Export params={drawerState.params} />
          {:else if drawerState.open === "sweep"}
            <SweepDrawer params={drawerState.params} />
          {:else if drawerState.open === "pack"}
            <PackDrawer params={drawerState.params} />
          {:else if drawerState.open === "merge"}
            <MergeDrawer params={drawerState.params} />
          {:else if drawerState.open === "clone"}
            <CloneDrawer params={drawerState.params} />
          {:else if drawerState.open === "token_drilldown"}
            <TokenDrilldownDrawer params={drawerState.params} />
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

    <StatusFooter />
  </div>
{/if}

<style>
  .shell {
    display: grid;
    grid-template-rows: auto 1fr auto;
    height: 100vh;
    width: 100vw;
    min-width: 1280px;
    min-height: 720px;
    background: var(--bg);
    color: var(--fg);
  }
  .shell.loading {
    /* Slight desaturation so users can tell bootstrap hasn't finished
     * without us blocking the entire frame. */
    opacity: 0.85;
  }
  .layout {
    display: grid;
    grid-template-columns: 55fr 45fr;
    grid-template-rows: 1fr;
    min-height: 0; /* let children scroll inside */
    position: relative; /* drawer sits over rack-zone via absolute pos */
    background: var(--border);
    gap: 1px;
  }
  .chat-zone,
  .rack-zone {
    background: var(--bg);
    overflow: hidden;
    padding: var(--panel-pad);
    min-height: 0;
  }
  .chat-zone {
    display: flex;
    flex-direction: column;
  }
  /* Three-row grid: steering rack, probe rack, reference collapsibles
   * (auto height so they expand without fighting the racks for space).
   * No overflow at this level — each rack handles its own internal
   * scroll so its actions row stays anchored at the bottom. */
  .rack-zone {
    display: grid;
    grid-template-rows: 1fr 1fr auto;
    gap: 0.6em;
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
    width: min(640px, 60%);
    background: var(--bg-alt);
    border-left: 1px solid var(--border);
    z-index: calc(var(--z-drawer) + 1);
    display: flex;
    flex-direction: column;
    box-shadow: -8px 0 24px rgba(0, 0, 0, 0.45);
    animation: drawer-in 160ms ease-out;
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
    padding: 0.5em 1em;
    border-bottom: 1px solid var(--border);
  }
  .drawer-title {
    color: var(--accent-blue);
    font-size: 0.95em;
    text-transform: lowercase;
    letter-spacing: 0.04em;
  }
  .drawer-close {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    font-size: 1em;
    line-height: 1;
    padding: 0.25em 0.4em;
  }
  .drawer-close:hover {
    color: var(--accent-red);
  }
  .drawer-body {
    flex: 1;
    overflow-y: auto;
    padding: 1em;
  }
  .stub {
    color: var(--fg-strong);
    font-size: 0.95em;
    margin: 0 0 0.4em 0;
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
    gap: 0.6em;
    padding: 2em;
    text-align: center;
  }
  .boot-failed h1 {
    color: var(--accent-red);
    margin: 0;
    font-size: 1.4em;
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
    font-size: 0.9em;
  }
  .boot-failed code {
    background: var(--bg-elev);
    padding: 0.1em 0.35em;
    border-radius: 3px;
    color: var(--accent-blue);
  }
  .retry {
    margin-top: 1em;
    background: transparent;
    color: var(--accent-blue);
    border: 1px solid var(--accent-blue);
    padding: 0.4em 1em;
    border-radius: 3px;
    font-size: 0.9em;
  }
  .retry:hover {
    background: rgba(88, 166, 255, 0.12);
  }
</style>
