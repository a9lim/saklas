<script lang="ts">
  // ⌘K command palette — the primary navigation surface.  Flattens the
  // rail's tool registry plus instrument-tab jumps and pages into one
  // filterable list.  The rail's category fly-outs remain as the mouse
  // path; this is the keyboard path (and the only place everything is
  // reachable from one input).

  import { tick } from "svelte";
  import { paletteState, closePalette } from "../lib/stores/palette.svelte";
  import { paletteCommands, type PaletteCommand } from "../lib/commands";
  import { openDrawer, setInspectorTab } from "../lib/stores.svelte";

  const COMMANDS = paletteCommands();

  let query = $state("");
  let selected = $state(0);
  let inputEl: HTMLInputElement | null = $state(null);
  let listEl: HTMLElement | null = $state(null);
  let paletteEl: HTMLElement | null = $state(null);

  const FOCUSABLE = [
    "button:not([disabled]):not([tabindex='-1'])",
    "[href]",
    "input:not([disabled])",
    "select:not([disabled])",
    "textarea:not([disabled])",
    '[tabindex]:not([tabindex="-1"])',
  ].join(",");

  const filtered = $derived.by(() => {
    const q = query.trim().toLowerCase();
    if (!q) return COMMANDS;
    const terms = q.split(/\s+/);
    return COMMANDS.filter((c) => {
      const hay = `${c.label} ${c.group} ${c.keywords ?? ""}`.toLowerCase();
      return terms.every((t) => hay.includes(t));
    });
  });

  // Reset + focus on every open; clamp selection as the filter narrows.
  $effect(() => {
    if (paletteState.open) {
      query = "";
      selected = 0;
      void tick().then(() => inputEl?.focus());
    }
  });
  $effect(() => {
    if (selected >= filtered.length) selected = Math.max(0, filtered.length - 1);
  });
  $effect(() => {
    // Keep the keyboard selection visible while scrolling with arrows.
    void selected;
    listEl
      ?.querySelector('[data-selected="true"]')
      ?.scrollIntoView({ block: "nearest" });
  });

  function run(cmd: PaletteCommand): void {
    closePalette();
    switch (cmd.action.kind) {
      case "drawer":
        // Palette focus restores to its launcher first; opening the drawer
        // in the following microtask lets the drawer remember that same
        // launcher and return there when it closes.
        {
          const drawer = cmd.action.drawer;
          queueMicrotask(() => openDrawer(drawer));
        }
        break;
      case "tab":
        setInspectorTab(cmd.action.tab);
        break;
    }
  }

  function onKey(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      ev.preventDefault();
      ev.stopPropagation();
      closePalette();
      return;
    }
    if (ev.key === "ArrowDown") {
      ev.preventDefault();
      selected = Math.min(filtered.length - 1, selected + 1);
      return;
    }
    if (ev.key === "ArrowUp") {
      ev.preventDefault();
      selected = Math.max(0, selected - 1);
      return;
    }
    if (ev.key === "Enter") {
      ev.preventDefault();
      const cmd = filtered[selected];
      if (cmd) run(cmd);
    }
  }

  function trapFocus(ev: KeyboardEvent): void {
    if (ev.key !== "Tab" || !paletteEl) return;
    const focusable = [...paletteEl.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(
      (el) => el.offsetParent !== null,
    );
    if (focusable.length === 0) {
      ev.preventDefault();
      paletteEl.focus();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (ev.shiftKey && (document.activeElement === first || document.activeElement === paletteEl)) {
      ev.preventDefault();
      last.focus();
    } else if (!ev.shiftKey && document.activeElement === last) {
      ev.preventDefault();
      first.focus();
    }
  }

  const TAB_HUE: Record<string, string> = {
    subspace: "var(--pillar-subspace)",
    manifold: "var(--pillar-manifold)",
    sae: "var(--pillar-sae)",
    lens: "var(--pillar-lens)",
  };
  function hueFor(cmd: PaletteCommand): string | null {
    return cmd.action.kind === "tab" ? TAB_HUE[cmd.action.tab] : null;
  }
</script>

{#if paletteState.open}
  <div
    class="backdrop"
    role="button"
    tabindex="-1"
    aria-label="Close command palette"
    onclick={closePalette}
    onkeydown={(ev) => {
      if (ev.key === "Enter" || ev.key === " ") closePalette();
    }}
  ></div>
  <div
    bind:this={paletteEl}
    class="palette"
    role="dialog"
    aria-modal="true"
    aria-label="Command palette"
    tabindex="-1"
    onkeydown={trapFocus}
  >
    <div class="input-row">
      <svg viewBox="0 0 24 24" aria-hidden="true" class="glass-icon">
        <circle cx="11" cy="11" r="7"></circle>
        <path d="M21 21l-4.35-4.35"></path>
      </svg>
      <input
        bind:this={inputEl}
        bind:value={query}
        role="combobox"
        aria-autocomplete="list"
        aria-controls="command-palette-listbox"
        aria-expanded="true"
        aria-activedescendant={filtered.length > 0
          ? `command-palette-option-${selected}`
          : undefined}
        placeholder="search…"
        aria-label="Filter commands"
        onkeydown={onKey}
      />
      <kbd>esc</kbd>
    </div>
    <div
      id="command-palette-listbox"
      class="list"
      bind:this={listEl}
      role="listbox"
    >
      {#if filtered.length === 0}
        <p class="none">no matches</p>
      {:else}
        {#each filtered as cmd, i (cmd.group + cmd.label)}
          {@const hue = hueFor(cmd)}
          <button
            id={`command-palette-option-${i}`}
            type="button"
            class="row"
            role="option"
            tabindex="-1"
            aria-selected={i === selected}
            data-selected={i === selected}
            onclick={() => run(cmd)}
            onmousemove={() => (selected = i)}
          >
            {#if hue}<span class="dot" style:background={hue}></span>{/if}
            <span class="label">{cmd.label}</span>
            <span class="group">{cmd.group}</span>
          </button>
        {/each}
      {/if}
    </div>
  </div>
{/if}

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    background: var(--scrim-strong);
    backdrop-filter: blur(2px);
    z-index: calc(var(--z-modal) + 10);
    border: 0;
    cursor: default;
  }

  .palette {
    position: fixed;
    top: 16vh;
    left: 50%;
    transform: translateX(-50%);
    width: min(560px, 92vw);
    max-height: 56vh;
    display: flex;
    flex-direction: column;
    z-index: calc(var(--z-modal) + 11);
    background: color-mix(in srgb, var(--surface-hi) 92%, transparent);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-line);
    border-radius: var(--radius-lg);
    box-shadow:
      inset 0 1px 0 var(--top-light-strong),
      var(--shadow-overlay);
    overflow: hidden;
    animation: palette-in var(--dur) var(--ease-out);
  }
  @keyframes palette-in {
    from {
      opacity: 0;
      transform: translateX(-50%) translateY(-6px);
    }
    to {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }
  }

  .input-row {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    padding: var(--space-5) var(--space-6);
  }
  .glass-icon {
    width: 15px;
    height: 15px;
    flex: none;
    fill: none;
    stroke: var(--fg-muted);
    stroke-width: 2;
    stroke-linecap: round;
  }
  input {
    flex: 1;
    background: transparent;
    border: 0;
    outline: none;
    color: var(--fg);
    font-size: var(--text);
    font-family: var(--font-ui);
  }
  input::placeholder {
    color: var(--fg-muted);
  }
  kbd {
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    color: var(--fg-muted);
    border: 1px solid var(--glass-line);
    border-radius: var(--radius-sm);
    padding: 2px 6px;
  }

  .list {
    overflow-y: auto;
    padding: var(--space-3);
    display: flex;
    flex-direction: column;
    gap: 1px;
  }
  .none {
    margin: 0;
    padding: var(--space-5);
    color: var(--fg-muted);
    font-size: var(--text-sm);
    text-align: center;
  }
  .row {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    padding: var(--space-3) var(--space-5);
    background: transparent;
    border: 0;
    border-radius: var(--radius);
    text-align: left;
    color: var(--fg-dim);
    font-size: var(--text-sm);
    cursor: pointer;
  }
  .row[data-selected="true"] {
    background: var(--accent-subtle);
    color: var(--fg);
  }
  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex: none;
  }
  .label {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .group {
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--fg-muted);
    flex: none;
  }
</style>
