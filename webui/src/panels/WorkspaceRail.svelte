<script lang="ts">
  // Workspace rail — the primary navigation surface.  A slim 64px icon
  // strip: three category icons, each opening a fly-out list of that
  // category's tools.  This is where the old Topbar "tools ▾" menu's
  // ~19 drawer launchers now live, so the topbar can stay a thin
  // brand + status strip.
  //
  // Fly-outs are ``position: fixed`` (the rail-zone clips overflow) and
  // anchored off the clicked icon's bounding rect.

  import { onMount } from "svelte";
  import { openDrawer } from "../lib/stores.svelte";
  import type { DrawerName } from "../lib/types";

  interface Tool {
    label: string;
    drawer: DrawerName;
  }
  interface Category {
    key: string;
    label: string;
    /** SVG path data for the 24×24 rail glyph. */
    icon: string;
    tools: Tool[];
  }

  const CATEGORIES: Category[] = [
    {
      key: "vectors",
      label: "Steering & vectors",
      icon: "M5 19L19 5M19 5h-7M19 5v7",
      tools: [
        { label: "load vector…", drawer: "load" },
        { label: "merge vector…", drawer: "merge" },
        { label: "clone vector…", drawer: "clone" },
        { label: "packs…", drawer: "pack" },
      ],
    },
    {
      key: "analysis",
      label: "Analysis",
      icon: "M4 18l5-12 4 8 3-5 4 9",
      tools: [
        { label: "correlation matrix…", drawer: "correlation" },
        { label: "pairwise compare…", drawer: "compare" },
        { label: "layer norms…", drawer: "layer_norms" },
        { label: "activation atlas…", drawer: "activation_atlas" },
        { label: "experiment lab…", drawer: "experiment_lab" },
        { label: "recipe builder…", drawer: "recipe_builder" },
      ],
    },
    {
      key: "session",
      label: "Session & model",
      icon: "M5 21v-6M5 11V3M12 21v-9M12 8V3M19 21v-4M19 13V3M2 15h6M9 8h6M16 13h6",
      tools: [
        { label: "model health…", drawer: "health" },
        { label: "session / auth…", drawer: "session_admin" },
        { label: "help / shortcuts…", drawer: "help" },
      ],
    },
  ];

  // Which category fly-out is open, and where to anchor it.
  let openKey: string | null = $state(null);
  let flyoutTop = $state(0);
  let flyoutLeft = $state(0);

  const openCategory = $derived(
    CATEGORIES.find((c) => c.key === openKey) ?? null,
  );

  function toggleCategory(cat: Category, ev: MouseEvent): void {
    ev.stopPropagation();
    if (openKey === cat.key) {
      openKey = null;
      return;
    }
    const r = (ev.currentTarget as HTMLElement).getBoundingClientRect();
    flyoutLeft = r.right + 6;
    flyoutTop = r.top;
    openKey = cat.key;
  }

  function pickTool(drawer: DrawerName): void {
    openKey = null;
    openDrawer(drawer);
  }

  // Close the fly-out on any outside click or Escape.
  function onDocClick(ev: MouseEvent): void {
    if (openKey === null) return;
    const t = ev.target as HTMLElement | null;
    if (t && (t.closest(".rail") || t.closest(".flyout"))) return;
    openKey = null;
  }
  function onDocKey(ev: KeyboardEvent): void {
    if (ev.key === "Escape" && openKey !== null) openKey = null;
  }

  onMount(() => {
    document.addEventListener("click", onDocClick);
    document.addEventListener("keydown", onDocKey);
    return () => {
      document.removeEventListener("click", onDocClick);
      document.removeEventListener("keydown", onDocKey);
    };
  });
</script>

<nav class="rail" aria-label="saklas workspace rail">
  <div class="items">
    {#each CATEGORIES as cat (cat.key)}
      <button
        type="button"
        class="rail-btn"
        class:active={openKey === cat.key}
        title={cat.label}
        aria-label={cat.label}
        aria-haspopup="menu"
        aria-expanded={openKey === cat.key}
        onclick={(ev) => toggleCategory(cat, ev)}
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d={cat.icon}></path>
        </svg>
        <span>{cat.key}</span>
      </button>
    {/each}
  </div>
</nav>

{#if openCategory}
  <div
    class="flyout"
    role="menu"
    aria-label={openCategory.label}
    style:top="{flyoutTop}px"
    style:left="{flyoutLeft}px"
  >
    <p class="flyout-title" role="presentation">{openCategory.label}</p>
    {#each openCategory.tools as tool (tool.drawer)}
      <button
        type="button"
        role="menuitem"
        onclick={() => pickTool(tool.drawer)}
      >
        {tool.label}
      </button>
    {/each}
  </div>
{/if}

<style>
  .rail {
    display: flex;
    flex-direction: column;
    padding: var(--space-5) var(--space-4);
    background: var(--bg-deep);
    border-right: 1px solid var(--border);
    min-height: 0;
  }

  .items {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
  }

  .rail-btn {
    min-height: 3.25rem;
    width: 100%;
    display: grid;
    place-items: center;
    gap: var(--space-1);
    border: 1px solid transparent;
    border-radius: var(--radius);
    background: transparent;
    color: var(--fg-subtle);
    padding: var(--space-2) var(--space-1);
    font-family: var(--font-ui);
    cursor: pointer;
    transition:
      background var(--dur) var(--ease-out),
      border-color var(--dur) var(--ease-out),
      color var(--dur) var(--ease-out);
  }
  .rail-btn:hover:not(:disabled),
  .rail-btn.active {
    background: var(--accent-subtle);
    border-color: var(--accent);
    color: var(--fg);
  }
  .rail-btn:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }
  .rail-btn svg {
    width: 1.15rem;
    height: 1.15rem;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .rail-btn span {
    font-size: var(--text-2xs);
    line-height: 1;
    text-transform: uppercase;
    letter-spacing: 0;
  }

  /* Fly-out tool list — fixed so it escapes the rail-zone's clip. */
  .flyout {
    position: fixed;
    z-index: var(--z-modal);
    min-width: 200px;
    background: var(--surface-strong);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--space-2) 0;
    box-shadow: var(--shadow-overlay);
    display: flex;
    flex-direction: column;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    animation: flyout-in var(--dur) var(--ease-out);
  }
  @keyframes flyout-in {
    from {
      opacity: 0;
      transform: translateX(-4px);
    }
    to {
      opacity: 1;
      transform: translateX(0);
    }
  }
  .flyout-title {
    margin: 0;
    padding: var(--space-3) var(--space-5) var(--space-2);
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .flyout button {
    background: transparent;
    border: 0;
    text-align: left;
    padding: var(--space-3) var(--space-5);
    color: var(--fg-strong);
    font: inherit;
    font-family: var(--font-mono);
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease-out);
  }
  .flyout button:hover {
    background: var(--bg-elev);
    color: var(--accent-blue);
  }
</style>
