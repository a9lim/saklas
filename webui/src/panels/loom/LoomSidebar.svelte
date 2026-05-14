<script lang="ts">
  // Collapsible left-edge loom sidebar.  Mounted from App.svelte when
  // ``loomUiState.sidebarOpen`` is true.  Renders the tree as an
  // indented list — recursive component composition would be cleaner
  // visually but produces a deep DOM nesting; a flat depth-first walk
  // with per-node indentation is fast enough for the trees we expect
  // (low hundreds of nodes).
  //
  // Phase 3 scope:
  //   * Tree render — active path bolded, dead branches dimmed.
  //   * Click → navigate.
  //   * Right-click → context menu (regenerate, edit, branch, navigate,
  //     delete, star/unstar, add note).
  //   * Keyboard within the sidebar: j/k siblings, h/l up/down,
  //     Enter activates, s stars, n note, / fuzzy search.
  //   * Modals for regenerate-N, edit text, branch text, delete confirm,
  //     note text, fuzzy search.
  //
  // Phase-5 hooks (not wired):
  //   * Pin to compare pane, fan-out grid, cross-branch diff.

  import { onMount, tick } from "svelte";
  import {
    applyTreeFilter,
    autoRegenState,
    clearNodeSelection,
    clearTreeFilter,
    currentRecipeOverride,
    filterState,
    highlightState,
    loomRegenerateFromUser,
    loomTree,
    loomNavigate,
    loomEdit,
    loomBranch,
    loomDelete,
    loomStar,
    loomNote,
    loomRegenerateActive,
    loomUiState,
    nodeSelection,
    openDrawer,
    pinNodeForComparison,
    pinnedComparison,
    refreshLoomTree,
    toggleLoomSidebar,
    toggleNodeSelection,
  } from "../../lib/stores.svelte";
  import LoomNode from "./LoomNode.svelte";
  import LoomEdge from "./LoomEdge.svelte";
  import type { LoomNodeJSON } from "../../lib/types";

  // ----------------------------------------- flat tree walk + depth --

  /** Depth-first walk of the tree starting at root.  Yields one row
   *  per node with its depth, so the renderer indents linearly. */
  interface Row {
    node: LoomNodeJSON;
    depth: number;
    isActivePath: boolean;
    isDead: boolean;
    /** Phase 5 filter dim: true iff the filter is on and this node
     *  doesn't match.  Rendered at 50% opacity — distinct visual
     *  channel from the 30% dead-branch dim. */
    filteredOut: boolean;
  }

  const activePathSet = $derived(new Set(loomTree.activePath));
  const selectionSet = $derived(new Set(nodeSelection.ids));

  const rows = $derived.by<Row[]>(() => {
    const out: Row[] = [];
    if (!loomTree.root_id) return out;
    const matching = filterState.matchingIds;
    const stack: { id: string; depth: number; deadAncestor: boolean }[] = [
      { id: loomTree.root_id, depth: 0, deadAncestor: false },
    ];
    while (stack.length) {
      const { id, depth, deadAncestor } = stack.pop()!;
      const node = loomTree.nodes.get(id);
      if (!node) continue;
      const onActive = activePathSet.has(id);
      const isDead = deadAncestor || !onActive;
      // Skip the synthetic root system node from the visible list —
      // it has no text and is just an anchor.
      if (!(node.parent_id === null && node.role === "system" && !node.text)) {
        const filteredOut = matching !== null && !matching.has(id);
        out.push({
          node,
          depth,
          isActivePath: onActive,
          isDead: isDead && !onActive,
          filteredOut,
        });
      }
      // Push children in reverse so DFS visits them in order.
      const children = loomTree.children_of.get(id) ?? [];
      for (let i = children.length - 1; i >= 0; i--) {
        stack.push({
          id: children[i],
          depth: depth + (node.parent_id === null && node.role === "system" && !node.text ? 0 : 1),
          deadAncestor: deadAncestor || (!onActive && node.parent_id !== null),
        });
      }
    }
    return out;
  });

  // ----------------------------------------- ring decoration --------

  /** Per-node ring fill keyed off the currently-selected highlight
   *  probe (Decision 10 in docs/plans/loom.md).  Returns the node's
   *  aggregate reading for ``highlightState.target`` in [-1, 1], or
   *  ``null`` when no probe is selected, the node has no aggregate
   *  readings yet, or the selected probe is missing from this node's
   *  reading map.  ``LoomNode`` renders the ring only when this value
   *  is non-null. */
  function ringFor(node: LoomNodeJSON): number | null {
    const target = highlightState.target;
    if (!target) return null;
    const readings = node.aggregate_readings;
    if (!readings) return null;
    const v = readings[target];
    return typeof v === "number" ? v : null;
  }

  // ----------------------------------------- filter input ------------

  let filterInput = $state("");

  // Mirror server-applied filter into the local input so external
  // clears (Esc, clearTreeFilter) reflect immediately.
  $effect(() => {
    if (filterState.expr !== filterInput) {
      filterInput = filterState.expr;
    }
  });

  async function commitFilter(): Promise<void> {
    const trimmed = filterInput.trim();
    if (!trimmed) {
      clearTreeFilter();
      return;
    }
    await applyTreeFilter(trimmed);
  }

  function onFilterKey(ev: KeyboardEvent): void {
    if (ev.key === "Enter") {
      ev.preventDefault();
      ev.stopPropagation();
      void commitFilter();
    } else if (ev.key === "Escape") {
      ev.preventDefault();
      ev.stopPropagation();
      filterInput = "";
      clearTreeFilter();
    }
  }

  // ------------------------------------------- focus / keyboard nav --

  let focusedId: string | null = $state(null);

  // When the active node changes, focus moves to it (only when there's
  // no manual focus selected yet).
  $effect(() => {
    if (focusedId === null && loomTree.active_node_id) {
      focusedId = loomTree.active_node_id;
    }
    // If the focused node has been removed, fall back to active.
    if (focusedId !== null && !loomTree.nodes.has(focusedId)) {
      focusedId = loomTree.active_node_id;
    }
  });

  function focusedIndex(): number {
    if (!focusedId) return -1;
    return rows.findIndex((r) => r.node.id === focusedId);
  }

  function siblingIds(nodeId: string): string[] {
    const node = loomTree.nodes.get(nodeId);
    if (!node || node.parent_id === null) return [nodeId];
    return loomTree.children_of.get(node.parent_id) ?? [nodeId];
  }

  function navUpDown(dir: -1 | 1): void {
    // h (=-1) / l (=+1): move to parent / first child along the
    // active-or-focused path.
    if (!focusedId) return;
    const node = loomTree.nodes.get(focusedId);
    if (!node) return;
    if (dir === -1) {
      if (node.parent_id) focusedId = node.parent_id;
    } else {
      const kids = loomTree.children_of.get(focusedId);
      if (kids && kids.length > 0) focusedId = kids[0];
    }
  }

  function navSibling(dir: -1 | 1): void {
    // j (=+1) / k (=-1) within siblings.
    if (!focusedId) return;
    const sibs = siblingIds(focusedId);
    const idx = sibs.indexOf(focusedId);
    if (idx < 0) return;
    const next = sibs[idx + dir];
    if (next) focusedId = next;
  }

  // ----------------------------------------- context menu state --

  interface MenuState {
    open: boolean;
    x: number;
    y: number;
    nodeId: string | null;
  }
  let menu: MenuState = $state({ open: false, x: 0, y: 0, nodeId: null });

  function openMenu(ev: MouseEvent, nodeId: string): void {
    ev.preventDefault();
    ev.stopPropagation();
    menu = { open: true, x: ev.clientX, y: ev.clientY, nodeId };
    focusedId = nodeId;
  }

  function closeMenu(): void {
    menu = { open: false, x: 0, y: 0, nodeId: null };
  }

  // ----------------------------------------- external modal requests --

  /** When App.svelte fires Ctrl+R/E/B/N/D it bumps
   *  ``loomUiState.modalRequest.seq``.  Mirror that into the local
   *  modal state. */
  let _lastSeenSeq = $state(0);
  $effect(() => {
    const req = loomUiState.modalRequest;
    if (req.seq !== _lastSeenSeq && req.kind) {
      _lastSeenSeq = req.seq;
      void openModal(req.kind, req.nodeId, req.text, req.n);
    }
  });

  // ----------------------------------------- modals --

  interface ModalState {
    kind:
      | null
      | "regenerate"
      | "edit"
      | "branch"
      | "delete"
      | "note"
      | "navpicker"
      | "search"
      | "fanout"
      | "regen_mode";
    nodeId: string | null;
    text: string;
    n: number;
    /** Phase-5: vector name for the fan-out modal. */
    vector?: string;
    /** Phase-5: mode for the regen-with-modifier modal. */
    mode?: string;
    /** Inline validation error.  Non-empty means commit failed; the
     *  modal stays open with this message rendered below the input. */
    error?: string;
  }
  let modal: ModalState = $state({
    kind: null,
    nodeId: null,
    text: "",
    n: 1,
    vector: "",
    mode: "unsteered",
    error: "",
  });
  let modalInput: HTMLInputElement | HTMLTextAreaElement | null = $state(null);

  async function openModal(
    kind: ModalState["kind"],
    nodeId: string | null,
    initialText: string = "",
    initialN: number = 1,
  ): Promise<void> {
    modal = {
      kind,
      nodeId,
      text: initialText,
      n: initialN,
      vector: "",
      mode: "unsteered",
      error: "",
    };
    await tick();
    modalInput?.focus();
    modalInput?.select?.();
  }

  function closeModal(): void {
    modal = {
      kind: null,
      nodeId: null,
      text: "",
      n: 1,
      vector: "",
      mode: "unsteered",
      error: "",
    };
  }

  function setModalError(message: string): void {
    // Clear input focus so the error reads; modal stays open.
    modal = { ...modal, error: message };
  }

  async function commitModal(): Promise<void> {
    const m = modal;
    if (!m.kind || !m.nodeId) return closeModal();
    switch (m.kind) {
      case "regenerate":
        await loomRegenerateActive(Math.max(1, Math.floor(m.n)));
        break;
      case "edit":
        await loomEdit(m.nodeId, m.text);
        break;
      case "branch": {
        const newId = await loomBranch(m.nodeId, m.text);
        if (newId) await loomNavigate(newId);
        break;
      }
      case "delete":
        await loomDelete(m.nodeId);
        break;
      case "note":
        await loomNote(m.nodeId, m.text);
        break;
      case "navpicker": {
        // Resolve a node by id-prefix or full id.  Ambiguity keeps the
        // modal open with a list of matches so the user can disambiguate
        // rather than navigating to a random first match (Map insertion
        // order leaks the bug otherwise).
        const r = resolveByPrefix(m.text);
        if (r.id) {
          await loomNavigate(r.id);
          break;
        }
        if (r.matches.length === 0) {
          setModalError(`no node matches '${m.text}'`);
          return;
        }
        const preview = r.matches.slice(0, 6).map((s) => s.slice(0, 8)).join(", ");
        setModalError(
          `ambiguous: ${r.matches.length} matches (${preview}` +
            (r.matches.length > 6 ? ", …" : "") + ")",
        );
        return;
      }
      case "search": {
        const target = resolveBySearch(m.text);
        if (target) {
          focusedId = target;
          // Don't navigate — search jumps focus only.
          break;
        }
        setModalError(`no text match for '${m.text}'`);
        return;
      }
      case "fanout": {
        // Phase 5 fan-out: anchor on the user node, send one regen
        // per alpha as a sibling.  We use a sequential dispatch
        // rather than the engine's ``generate_sweep`` because the
        // active rack already carries the rest of the steering
        // context — we only need to overlay the swept vector's α
        // per call.
        const vector = (m.vector ?? "").trim();
        if (!vector) {
          setModalError("vector name required");
          return;
        }
        const alphas = parseAlphaList(m.text);
        if (alphas.length === 0) {
          setModalError(
            "couldn't parse alphas — try a comma list (0.0, 0.3, 0.7) " +
            "or linspace(-1, 1, 5)",
          );
          return;
        }
        const userId = m.nodeId;
        for (const alpha of alphas) {
          // Recipe override carries the per-row alpha for the swept
          // vector.  The engine's modifier resolver accepts a partial
          // recipe expression on the ``steering`` axis.
          await loomRegenerateFromUser(userId, {
            n: 1,
            recipe_override: `${alpha} ${vector}`,
          });
        }
        break;
      }
      case "regen_mode": {
        // Manual regen-with-modifier: anchor on the active assistant
        // (or its user-parent) and dispatch N siblings under the
        // chosen mode.  Modes are resolved engine-side.
        const node = loomTree.nodes.get(m.nodeId);
        const mode = (m.mode ?? "unsteered").trim();
        const N = Math.max(1, Math.floor(m.n));
        if (node?.role === "user") {
          await loomRegenerateFromUser(m.nodeId, {
            n: N,
            recipe_override: mode,
          });
        } else {
          // Active assistant — set it as active then call regen.
          if (loomTree.active_node_id !== m.nodeId) {
            await loomNavigate(m.nodeId);
          }
          await loomRegenerateActive(N, { recipe_override: mode });
        }
        break;
      }
    }
    closeModal();
  }

  // ----------------------------------------- alpha-list parser ------

  /** Parse a sweep alpha string (comma list / ``linspace(a, b, n)`` /
   *  ``start:stop:step``) into ``number[]``.  Mirrors the legacy
   *  SweepDrawer parser. */
  function parseAlphaList(raw: string): number[] {
    const trimmed = raw.trim();
    if (!trimmed) return [];

    const linspaceMatch = trimmed.match(
      /^linspace\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,)]+?)\s*\)\s*$/i,
    );
    if (linspaceMatch) {
      const start = Number(linspaceMatch[1]);
      const stop = Number(linspaceMatch[2]);
      const count = Number(linspaceMatch[3]);
      if (!Number.isFinite(start) || !Number.isFinite(stop)) return [];
      if (!Number.isInteger(count) || count < 1) return [];
      if (count === 1) return [start];
      const step = (stop - start) / (count - 1);
      const out: number[] = [];
      for (let i = 0; i < count; i++) out.push(start + step * i);
      return out;
    }

    if (trimmed.includes(":")) {
      const parts = trimmed.split(":").map((p) => p.trim());
      if (parts.length !== 3) return [];
      const start = Number(parts[0]);
      const stop = Number(parts[1]);
      const step = Number(parts[2]);
      if (![start, stop, step].every(Number.isFinite)) return [];
      if (step === 0) return [];
      if ((stop - start) * step < 0) return [];
      const out: number[] = [];
      const eps = Math.abs(step) * 1e-9;
      const ascending = step > 0;
      let v = start;
      let guard = 0;
      while (
        (ascending ? v <= stop + eps : v >= stop - eps) &&
        guard++ < 10000
      ) {
        out.push(Number.parseFloat(v.toPrecision(12)));
        v += step;
      }
      if (guard >= 10000) return [];
      return out;
    }

    const out: number[] = [];
    for (const part of trimmed.split(",")) {
      const t = part.trim();
      if (!t) continue;
      const v = Number(t);
      if (!Number.isFinite(v)) return [];
      out.push(v);
    }
    return out;
  }

  interface PrefixResolution {
    /** Single unambiguous match — caller can navigate immediately. */
    id: string | null;
    /** Every node id whose ulid starts with the prefix.  ``length > 1``
     *  signals ambiguity; ``length == 0`` signals no match. */
    matches: string[];
  }

  function resolveByPrefix(prefix: string): PrefixResolution {
    const p = prefix.trim();
    if (!p) return { id: null, matches: [] };
    if (p === "root") return { id: loomTree.root_id, matches: [loomTree.root_id] };
    const matches: string[] = [];
    for (const id of loomTree.nodes.keys()) {
      if (id === p) return { id, matches: [id] };  // exact match wins
      if (id.startsWith(p)) matches.push(id);
    }
    if (matches.length === 1) return { id: matches[0], matches };
    return { id: null, matches };
  }

  function resolveBySearch(query: string): string | null {
    const q = query.trim().toLowerCase();
    if (!q) return null;
    let best: string | null = null;
    for (const [id, n] of loomTree.nodes) {
      const text = (n.text ?? "").toLowerCase();
      if (text.includes(q)) {
        if (!best) best = id;
        // Prefer active-path matches.
        if (activePathSet.has(id)) return id;
      }
    }
    return best;
  }

  // --------------------------------------- click + global keys --

  function onNodeClick(node: LoomNodeJSON, ev: MouseEvent): void {
    // Ctrl/Cmd-click on an assistant node toggles its multi-select
    // membership for the cross-branch diff drawer.  Plain click
    // still navigates.
    if ((ev.ctrlKey || ev.metaKey) && node.role === "assistant") {
      ev.preventDefault();
      ev.stopPropagation();
      toggleNodeSelection(node.id);
      focusedId = node.id;
      return;
    }
    focusedId = node.id;
    void loomNavigate(node.id);
  }

  function onSidebarKey(ev: KeyboardEvent): void {
    // Only fire when the sidebar (or one of its children) is focused.
    if (modal.kind !== null || menu.open) return;
    const k = ev.key;
    if (k === "j") { ev.preventDefault(); navSibling(+1); return; }
    if (k === "k") { ev.preventDefault(); navSibling(-1); return; }
    if (k === "h") { ev.preventDefault(); navUpDown(-1); return; }
    if (k === "l") { ev.preventDefault(); navUpDown(+1); return; }
    if (k === "Enter") {
      if (focusedId) {
        ev.preventDefault();
        void loomNavigate(focusedId);
      }
      return;
    }
    if (k === "s" && focusedId) {
      ev.preventDefault();
      const node = loomTree.nodes.get(focusedId);
      void loomStar(focusedId, !node?.starred);
      return;
    }
    if (k === "n" && focusedId) {
      ev.preventDefault();
      const node = loomTree.nodes.get(focusedId);
      void openModal("note", focusedId, node?.notes ?? "");
      return;
    }
    if (k === "/") {
      ev.preventDefault();
      void openModal("search", focusedId ?? loomTree.active_node_id);
      return;
    }
    if (k === "Escape") {
      // v2.3: Esc inside the sidebar defocuses the active element /
      // search input rather than collapsing the whole panel.  Most
      // users hit Esc to back out of a focused control; auto-closing
      // the sidebar surprised people who just wanted to dismiss a
      // modal/menu/search.  The topbar's "loom" button still toggles
      // the panel; only an open menu/modal short-circuits this branch
      // (handled in ``onWindowKey``).
      ev.preventDefault();
      const active = document.activeElement as HTMLElement | null;
      active?.blur?.();
      return;
    }
  }

  // Close context menu on outside click or Escape (window-level).
  function onWindowClick(ev: MouseEvent): void {
    if (!menu.open) return;
    const t = ev.target as HTMLElement | null;
    if (t && t.closest(".loom-menu")) return;
    closeMenu();
  }
  function onWindowKey(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      if (menu.open) { closeMenu(); ev.preventDefault(); return; }
      if (modal.kind) { closeModal(); ev.preventDefault(); return; }
    }
  }

  // ---------------------------------------- menu actions --

  async function menuRegenerate(): Promise<void> {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    if (loomTree.active_node_id !== nid) await loomNavigate(nid);
    await openModal("regenerate", nid, "", 1);
  }
  async function menuEdit(): Promise<void> {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    const node = loomTree.nodes.get(nid);
    await openModal("edit", nid, node?.text ?? "");
  }
  async function menuBranch(): Promise<void> {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    const node = loomTree.nodes.get(nid);
    await openModal("branch", nid, node?.text ?? "");
  }
  async function menuNavigate(): Promise<void> {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    await loomNavigate(nid);
  }
  async function menuDelete(): Promise<void> {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    await openModal("delete", nid);
  }
  async function menuStar(): Promise<void> {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    const node = loomTree.nodes.get(nid);
    await loomStar(nid, !node?.starred);
  }
  async function menuNote(): Promise<void> {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    const node = loomTree.nodes.get(nid);
    await openModal("note", nid, node?.notes ?? "");
  }

  // ---------------------------------- phase 5 context-menu actions --

  function menuPin(): void {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    pinNodeForComparison(nid);
  }

  function menuFanOut(): void {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    const node = loomTree.nodes.get(nid);
    // Anchor the fan-out on the user node.  If the user clicked an
    // assistant node, walk up to its user parent.
    let anchorId = nid;
    if (node?.role === "assistant" && node.parent_id) {
      anchorId = node.parent_id;
    }
    void openModal("fanout", anchorId, "0.0, 0.3, 0.6", 1);
  }

  function menuCompareChildren(): void {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    const childIds = loomTree.children_of.get(nid) ?? [];
    const assistantChildren = childIds.filter((id) => {
      const c = loomTree.nodes.get(id);
      return c?.role === "assistant";
    });
    if (assistantChildren.length < 2) return;
    openDrawer("node_compare", {
      node_ids: assistantChildren,
      parent_id: nid,
    });
  }

  function menuToggleSelection(): void {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    toggleNodeSelection(nid);
  }

  function menuCompareSelected(): void {
    closeMenu();
    if (nodeSelection.ids.length < 2) return;
    openDrawer("node_compare", { node_ids: [...nodeSelection.ids] });
  }

  function menuRegenWithMode(): void {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    void openModal("regen_mode", nid, "", 1);
  }

  // ---------------------------------------- refresh / error UI --

  function fullRefresh(): void {
    // Re-fetch — useful if the user suspects drift.
    void refreshLoomTree();
  }

  // Element ref + on-mount focus so j/k/h/l work the instant the
  // user opens the sidebar — without it, the panel mounts unfocused
  // and the keyboard nav silently no-ops until they click a node.
  let asideEl: HTMLElement | null = $state(null);
  onMount(() => {
    void tick().then(() => asideEl?.focus());
  });
</script>

<svelte:window onclick={onWindowClick} onkeydown={onWindowKey} />

<!-- svelte-ignore a11y_no_noninteractive_element_to_interactive_role -->
<aside
  class="loom-sidebar"
  role="tree"
  aria-label="Loom tree"
  onkeydown={onSidebarKey}
  tabindex="-1"
  bind:this={asideEl}
>
  <header class="loom-header">
    <span class="title">loom</span>
    <span class="rev" title="server tree revision">rev {loomTree.rev}</span>
    <button
      type="button"
      class="icon-btn"
      onclick={fullRefresh}
      title="Refresh tree from server"
      aria-label="Refresh"
    >↻</button>
    <button
      type="button"
      class="icon-btn"
      onclick={toggleLoomSidebar}
      title="Close sidebar"
      aria-label="Close"
    >✕</button>
  </header>

  <div class="filter-bar">
    <input
      type="text"
      class="filter-input"
      bind:value={filterInput}
      onkeydown={onFilterKey}
      placeholder="filter: agg:angry.calm > 0.4"
      aria-label="Filter expression"
      title="Filter grammar: agg:|any:|last:<probe> <op> <threshold> — comma-AND'd"
    />
    {#if filterState.expr}
      <button
        type="button"
        class="icon-btn"
        onclick={() => { filterInput = ""; clearTreeFilter(); }}
        title="Clear filter"
        aria-label="Clear filter"
      >✕</button>
    {/if}
    {#if filterState.loading}
      <span class="filter-status">…</span>
    {:else if filterState.error}
      <span class="filter-status err" title={filterState.error}>!</span>
    {:else if filterState.matchingIds !== null}
      <span class="filter-status" title="matches">
        {filterState.matchingIds.size}
      </span>
    {/if}
  </div>

  {#if nodeSelection.ids.length > 0}
    <div class="selection-bar">
      <span>{nodeSelection.ids.length} selected</span>
      <button
        type="button"
        class="action-btn"
        onclick={menuCompareSelected}
        disabled={nodeSelection.ids.length < 2}
      >compare</button>
      <button
        type="button"
        class="action-btn"
        onclick={clearNodeSelection}
      >clear</button>
    </div>
  {/if}

  {#if loomTree.unavailable}
    <div class="empty">
      <p>Server doesn't support loom yet.</p>
      <p class="hint">Update saklas server to v2.3+ to enable the tree view.</p>
    </div>
  {:else if loomTree.error}
    <div class="empty err">
      <p>tree error: {loomTree.error}</p>
      <button type="button" onclick={fullRefresh}>retry</button>
    </div>
  {:else if rows.length === 0}
    <div class="empty">
      <p>(empty tree — start a conversation)</p>
    </div>
  {:else}
    <div class="tree-scroll">
      {#each rows as row (row.node.id)}
        <div
          class="tree-row"
          class:filtered-out={row.filteredOut}
          class:selected={selectionSet.has(row.node.id)}
          class:pinned={pinnedComparison.nodeId === row.node.id}
          style="padding-left: {row.depth * 0.9}em"
        >
          {#if row.depth > 0}
            <LoomEdge
              active={row.isActivePath}
              dead={row.isDead}
              parentId={row.node.parent_id}
              childId={row.node.id}
            />
          {/if}
          <LoomNode
            node={row.node}
            onActivePath={row.isActivePath}
            focused={focusedId === row.node.id}
            dead={row.isDead}
            streaming={loomTree.pendingNodeId === row.node.id}
            ring={ringFor(row.node)}
            onclick={(ev) => onNodeClick(row.node, ev)}
            oncontextmenu={(ev) => openMenu(ev, row.node.id)}
          />
        </div>
      {/each}
    </div>
  {/if}

  <footer class="loom-footer">
    <span class="hint">j/k siblings · h/l up/down · Enter activate · s star · n note · / search</span>
  </footer>
</aside>

{#if menu.open && menu.nodeId}
  {@const menuNode = loomTree.nodes.get(menu.nodeId)}
  {@const menuChildren = loomTree.children_of.get(menu.nodeId) ?? []}
  {@const assistantChildCount = menuChildren.filter((id) => {
    const c = loomTree.nodes.get(id);
    return c?.role === "assistant";
  }).length}
  <div
    class="loom-menu"
    style="left: {menu.x}px; top: {menu.y}px"
    role="menu"
  >
    <button type="button" role="menuitem" onclick={menuRegenerate}>regenerate…</button>
    <button type="button" role="menuitem" onclick={menuRegenWithMode}>regen N with mode…</button>
    <button type="button" role="menuitem" onclick={menuEdit}>edit…</button>
    <button type="button" role="menuitem" onclick={menuBranch}>branch…</button>
    <button type="button" role="menuitem" onclick={menuNavigate}>navigate</button>
    <hr />
    <button type="button" role="menuitem" onclick={menuStar}>
      {menuNode?.starred ? "unstar" : "star"}
    </button>
    <button type="button" role="menuitem" onclick={menuNote}>add note…</button>
    <hr />
    {#if menuNode?.role === "assistant"}
      <button type="button" role="menuitem" onclick={menuPin}>
        {pinnedComparison.nodeId === menu.nodeId
          ? "unpin from comparison"
          : "pin to comparison"}
      </button>
      <button type="button" role="menuitem" onclick={menuToggleSelection}>
        {selectionSet.has(menu.nodeId)
          ? "deselect for compare"
          : "select for compare"}
      </button>
    {/if}
    {#if menuNode?.role === "user"}
      <button
        type="button"
        role="menuitem"
        onclick={menuCompareChildren}
        disabled={assistantChildCount < 2}
        title={assistantChildCount < 2 ? "needs ≥2 assistant children" : ""}
      >compare children…</button>
    {/if}
    <button type="button" role="menuitem" onclick={menuFanOut}>fan out…</button>
    <hr />
    <button type="button" role="menuitem" onclick={menuDelete} class="danger">delete subtree…</button>
  </div>
{/if}

{#if modal.kind}
  <div
    class="loom-modal-backdrop"
    role="button"
    tabindex="-1"
    aria-label="Cancel"
    onclick={closeModal}
    onkeydown={(ev) => { if (ev.key === "Enter" || ev.key === " ") closeModal(); }}
  ></div>
  <div class="loom-modal" role="dialog" aria-modal="true" aria-label={modal.kind}>
    <header class="modal-header">
      <span>
        {#if modal.kind === "regenerate"}regenerate N siblings
        {:else if modal.kind === "edit"}edit node text
        {:else if modal.kind === "branch"}branch new sibling
        {:else if modal.kind === "delete"}delete subtree
        {:else if modal.kind === "note"}note
        {:else if modal.kind === "navpicker"}navigate to id-prefix
        {:else if modal.kind === "search"}search node text
        {:else if modal.kind === "fanout"}fan out — α grid
        {:else if modal.kind === "regen_mode"}regen N with mode
        {/if}
      </span>
      <button type="button" class="icon-btn" onclick={closeModal} aria-label="Close">✕</button>
    </header>
    <div class="modal-body">
      {#if modal.kind === "regenerate"}
        <label>
          <span>N siblings</span>
          <input
            bind:this={modalInput as HTMLInputElement}
            bind:value={modal.n}
            type="number"
            min="1"
            max="16"
            onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
          />
        </label>
        <p class="hint">Re-runs the active assistant's parent user turn with the current rack.</p>
      {:else if modal.kind === "delete"}
        <p>Delete node <code>{(modal.nodeId ?? "").slice(0, 12)}</code> and its entire subtree?</p>
        <p class="hint danger">This is destructive. Ancestors of the active node cannot be deleted — navigate away first.</p>
      {:else if modal.kind === "navpicker" || modal.kind === "search"}
        <input
          bind:this={modalInput as HTMLInputElement}
          bind:value={modal.text}
          type="text"
          placeholder={modal.kind === "navpicker" ? "node id prefix (or 'root')" : "search node text"}
          onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
        />
      {:else if modal.kind === "fanout"}
        <label>
          <span>vector name</span>
          <input
            bind:this={modalInput as HTMLInputElement}
            bind:value={modal.vector}
            type="text"
            placeholder="e.g. honest, calm, deer.wolf"
            onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
          />
        </label>
        <label>
          <span>alphas</span>
          <input
            bind:value={modal.text}
            type="text"
            placeholder="0.0, 0.3, 0.6 · linspace(-1, 1, 5) · 0:1:0.25"
            onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
          />
        </label>
        <p class="hint">One sibling per α — comma list, linspace(), or start:stop:step.</p>
      {:else if modal.kind === "regen_mode"}
        <label>
          <span>mode</span>
          <select bind:value={modal.mode}>
            <option value="unsteered">unsteered</option>
            <option value="inverted">inverted</option>
            <option value="reseed">reseed</option>
            <option value="cool">cool</option>
            <option value="hot">hot</option>
          </select>
        </label>
        <label>
          <span>N siblings</span>
          <input
            bind:this={modalInput as HTMLInputElement}
            bind:value={modal.n}
            type="number"
            min="1"
            max="16"
            onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
          />
        </label>
        <p class="hint">Recipe-override modifier overlays the parent's recipe.</p>
      {:else}
        <textarea
          bind:this={modalInput as HTMLTextAreaElement}
          bind:value={modal.text}
          rows="6"
          placeholder={modal.kind === "branch" ? "(empty = branch from blank)" : ""}
          onkeydown={(ev) => {
            if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) {
              ev.preventDefault();
              void commitModal();
            }
            // Prevent browser bold-formatting on Ctrl+B inside the
            // textarea for branch buffers.
            if (ev.key === "b" && (ev.metaKey || ev.ctrlKey)) {
              ev.preventDefault();
            }
          }}
        ></textarea>
        <p class="hint">⌘/Ctrl+Enter to commit</p>
      {/if}
      {#if modal.error}
        <p class="modal-error" role="alert">{modal.error}</p>
      {/if}
    </div>
    <footer class="modal-footer">
      <button type="button" class="cancel" onclick={closeModal}>cancel</button>
      <button
        type="button"
        class={modal.kind === "delete" ? "danger" : "primary"}
        onclick={() => void commitModal()}
      >
        {#if modal.kind === "delete"}delete{:else}commit{/if}
      </button>
    </footer>
  </div>
{/if}

<style>
  .loom-sidebar {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-alt);
    border-right: 1px solid var(--border);
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--font-size-small);
    min-width: 0;
    overflow: hidden;
  }

  .filter-bar {
    display: flex;
    gap: 0.3em;
    align-items: center;
    padding: 0.35em 0.5em;
    border-bottom: 1px solid var(--border-dim);
    background: var(--bg-deep);
  }
  .filter-input {
    flex: 1 1 auto;
    background: var(--bg-alt);
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: 0.25em 0.45em;
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--font-size-tiny);
    min-width: 0;
  }
  .filter-input:focus {
    outline: none;
    border-color: var(--accent-blue);
  }
  .filter-status {
    color: var(--accent-yellow);
    font-size: var(--font-size-tiny);
    min-width: 1.5em;
    text-align: right;
  }
  .filter-status.err {
    color: var(--accent-red);
  }

  .selection-bar {
    display: flex;
    align-items: center;
    gap: 0.5em;
    padding: 0.3em 0.5em;
    border-bottom: 1px solid var(--border-dim);
    background: rgba(88, 166, 255, 0.08);
    color: var(--accent-blue);
    font-size: var(--font-size-tiny);
  }
  .action-btn {
    background: transparent;
    color: var(--fg-strong);
    border: 1px solid var(--border);
    padding: 0.15em 0.55em;
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--font-size-tiny);
    cursor: pointer;
  }
  .action-btn:hover:not(:disabled) {
    background: var(--bg-elev);
    border-color: var(--accent-blue);
  }
  .action-btn:disabled {
    color: var(--fg-muted);
    border-color: var(--border-dim);
    cursor: not-allowed;
  }

  .loom-header {
    display: flex;
    align-items: center;
    gap: 0.5em;
    padding: 0.4em 0.6em;
    border-bottom: 1px solid var(--border-dim);
    background: var(--bg-deep);
  }
  .title {
    color: var(--accent-green);
    font-weight: bold;
    letter-spacing: 0.05em;
    text-transform: lowercase;
    flex: 0 0 auto;
  }
  .rev {
    color: var(--fg-muted);
    font-size: var(--font-size-tiny);
    flex: 1 1 auto;
  }
  .icon-btn {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    cursor: pointer;
    padding: 0.1em 0.3em;
    font: inherit;
    font-family: var(--font-mono);
  }
  .icon-btn:hover {
    color: var(--accent-blue);
  }

  .tree-scroll {
    overflow-y: auto;
    overflow-x: hidden;
    padding: 0.3em 0.2em;
    min-height: 0;
    flex: 1 1 auto;
  }
  .tree-row {
    padding-top: 1px;
    padding-bottom: 1px;
    display: flex;
    align-items: stretch;
    gap: 0;
    min-height: 1.4em;
  }
  /* The LoomNode lives inside the row's flex; let it take remaining width. */
  .tree-row :global(.node) {
    flex: 1 1 auto;
    min-width: 0;
  }

  /* Phase 5 filter dim — distinct from the 30% dead-branch dim. */
  .tree-row.filtered-out {
    opacity: 0.5;
  }
  .tree-row.filtered-out:hover {
    opacity: 0.8;
  }
  /* Multi-select highlight for cross-branch compare. */
  .tree-row.selected :global(.node) {
    box-shadow: inset 0 0 0 1px var(--accent-yellow);
    background: rgba(210, 153, 34, 0.10);
  }
  /* Pin marker — purple-ish ring keys off the comparison-pane glyph. */
  .tree-row.pinned :global(.node) {
    box-shadow: inset 0 0 0 1px var(--accent-purple);
  }

  .empty {
    padding: 1em;
    color: var(--fg-muted);
    font-size: var(--font-size-small);
    display: flex;
    flex-direction: column;
    gap: 0.4em;
  }
  .empty .hint {
    font-size: var(--font-size-tiny);
    color: var(--fg-subtle);
  }
  .empty.err {
    color: var(--accent-red);
  }
  .empty button {
    align-self: flex-start;
    background: transparent;
    border: 1px solid var(--accent-red);
    color: var(--accent-red);
    padding: 0.2em 0.6em;
    cursor: pointer;
    font: inherit;
    font-family: var(--font-mono);
  }

  .loom-footer {
    padding: 0.3em 0.6em;
    border-top: 1px solid var(--border-dim);
    background: var(--bg-deep);
  }
  .hint {
    color: var(--fg-muted);
    font-size: var(--font-size-tiny);
  }
  .modal-error {
    color: var(--err, #d83b3b);
    font-size: var(--font-size-small);
    margin-top: 0.5em;
    background: var(--err-bg, rgba(216, 59, 59, 0.08));
    border-left: 2px solid var(--err, #d83b3b);
    padding: 0.4em 0.6em;
    font-family: var(--font-mono);
  }

  .loom-menu {
    position: fixed;
    z-index: var(--z-modal);
    background: var(--bg-alt);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.25em 0;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45);
    min-width: 180px;
    display: flex;
    flex-direction: column;
    font-family: var(--font-mono);
    font-size: var(--font-size-small);
  }
  .loom-menu button {
    background: transparent;
    border: 0;
    text-align: left;
    padding: 0.35em 0.8em;
    color: var(--fg-strong);
    cursor: pointer;
    font: inherit;
    font-family: var(--font-mono);
  }
  .loom-menu button:hover:not(:disabled) {
    background: var(--bg-elev);
    color: var(--accent-blue);
  }
  .loom-menu button.danger {
    color: var(--accent-red);
  }
  .loom-menu button.danger:hover {
    background: rgba(248, 81, 73, 0.12);
  }
  .loom-menu button:disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
  }
  .loom-menu hr {
    border: 0;
    border-top: 1px solid var(--border-dim);
    margin: 0.2em 0;
  }

  .loom-modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(1, 4, 9, 0.55);
    z-index: var(--z-modal);
    border: 0;
    cursor: pointer;
  }
  .loom-modal {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: calc(var(--z-modal) + 1);
    background: var(--bg-alt);
    border: 1px solid var(--border);
    border-radius: 4px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.55);
    min-width: 420px;
    max-width: min(640px, 90vw);
    display: flex;
    flex-direction: column;
    font-family: var(--font-mono);
    font-size: var(--font-size-base);
    color: var(--fg);
  }
  .modal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6em 1em;
    border-bottom: 1px solid var(--border-dim);
    color: var(--accent-blue);
    text-transform: lowercase;
    letter-spacing: 0.04em;
  }
  .modal-body {
    padding: 1em;
    display: flex;
    flex-direction: column;
    gap: 0.6em;
  }
  .modal-body label {
    display: flex;
    flex-direction: column;
    gap: 0.3em;
  }
  .modal-body input,
  .modal-body textarea {
    background: var(--bg-deep);
    color: var(--fg);
    border: 1px solid var(--border);
    padding: 0.4em 0.6em;
    font: inherit;
    font-family: var(--font-mono);
    resize: vertical;
  }
  .modal-body input:focus,
  .modal-body textarea:focus {
    outline: none;
    border-color: var(--accent-blue);
  }
  .modal-body code {
    color: var(--accent-yellow);
    background: var(--bg-elev);
    padding: 0.1em 0.3em;
    border-radius: 2px;
  }
  .modal-body .danger {
    color: var(--accent-red);
  }
  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 0.5em;
    padding: 0.6em 1em;
    border-top: 1px solid var(--border-dim);
  }
  .modal-footer button {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--fg-strong);
    padding: 0.4em 1em;
    cursor: pointer;
    font: inherit;
    font-family: var(--font-mono);
  }
  .modal-footer .primary {
    border-color: var(--accent-green);
    color: var(--accent-green);
  }
  .modal-footer .primary:hover {
    background: rgba(126, 231, 135, 0.12);
  }
  .modal-footer .danger {
    border-color: var(--accent-red);
    color: var(--accent-red);
  }
  .modal-footer .danger:hover {
    background: rgba(248, 81, 73, 0.12);
  }
  .modal-footer .cancel:hover {
    border-color: var(--fg-muted);
  }
</style>
