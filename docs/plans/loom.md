# Loom for saklas

Plan for a tree-of-completions interface, modeled after Janus's Loom but with every branch carrying its own steering recipe so the tree visualizes both alternative outputs and the steering choices that produced them.

The reference shape: nodes are conversation turns, children are alternative continuations, the active path is what the model sees as context for the next generation. The saklas-native angle: every assistant node carries a reproducible `Recipe` (steering expression, sampling, seed, probe set), edges between siblings can be labeled with steering deltas, and "sweep" becomes the natural tree operation "fan out from this node with N alphas."

Phases below are ordered by ship order. Phase 1 lands the engine-side data model with no UI changes; phase 2 wires the server routes; phases 3 and 4 are the two surfaces; phase 5 adds the saklas-native flourishes. Phases 1 + 2 should ship as one PR (the data model is unusable without the routes). Each surface phase is independent and can ship on its own.

---

## Architectural choice: engine-side tree

The tree lives in `saklas.core.loom` and attaches to `SaklasSession` as `session.tree`. Both surfaces (TUI, webui) and the Python API read and write through session methods. The HTTP server exposes the tree over `/saklas/v1/sessions/{id}/tree/*`.

Three reasons this is the right factoring:

1. Saklas's design principle is one engine, three frontends. Putting loom in the engine preserves that. A per-surface tree would duplicate the data model, the persistence concern, and the algorithms (active-path walk, prune predicate, recipe extraction).

2. Persistence becomes a single concern. The tree serializes to one file format, loadable from any surface and from a Python script. The current webui localStorage layer becomes a cache of server state rather than the source of truth, which retires the "server-restart guard" hack in `stores.svelte.ts::loadPersistedChat`.

3. The Python API gets `session.tree.edit(node_id, text)` and `session.generate(parent_node_id=..., n=...)` as first-class research primitives. Scripted loom traversals from a notebook become a one-liner. Recipe export/import becomes a session-level feature available everywhere.

The cost is one larger commit instead of several smaller ones, plus the existing `session.history` flat-list becomes a derived view over `tree.active_path`. The compat shim (a `history` property that walks the active path) preserves the v2 Python API for callers that reach in directly.

---

## Core operations

Five primitives. Every surface verb composes from these; nothing else exists at the engine level.

| Op | What it does | Surface manifestations |
|---|---|---|
| **Regenerate** | N new assistant siblings of a parent node, with optional recipe override | `/regen`, `Ctrl+R`, `/regen N`, `/fan vector alphas`, auto-regen modifier |
| **Edit** | Mutate a node's text in place — no new node, no tree-shape change. For typo fixes and prompt tweaks where the user wants iteration, not history. The verb that doesn't grow the tree. | `/edit`, `Ctrl+E`, inline-text affordance on tree nodes |
| **Branch** | New sibling of any node with text X. Covers what used to be called "branch" (X fresh) and "fork" (X copied) — fork-without-mutate collapses into branch under the same primitive, since the only difference is whether the text buffer pre-fills with the parent's text. Always-sibling: the original is preserved. | `/branch`, `Ctrl+B`, "branch from here" context menu, `/fan vector alphas` for the steering-grid case |
| **Navigate** | Re-point active node to a given node | `/nav <id-prefix>`, `Ctrl+N` (opens picker), sidebar click, loom-screen Enter |
| **Delete subtree** | Drop a subtree; refuses to delete an ancestor of the active node (forces navigate-away first), and refuses while an in-flight generation reserves the subtree | `/del`, `Ctrl+D` (with confirm), tree-node context menu |

Plus one non-structural mutation kept explicit for completeness:

- **Annotate** — `star` / `note`. No tree-shape change; UI-decoration only.

And two engine-level workflows that compose on top of the primitives:

- **Clear** — `tree.reset()`, drops the whole tree. Destructive by name and intent — matches today's user expectation that `/clear` means wipe.
- **Rewind** — `tree.navigate(grandparent_of_active_node)`. Walks one user→assistant pair up the active path. Under loom this is **non-destructive**: the rewound pair stays in the tree as a dead branch, navigable back to via the sidebar / loom screen. Strict improvement over today's destructive rewind; no user-visible regression because nothing today depends on the dropped state being unrecoverable.

The semantic split between clear and rewind: `/clear` is "I want a fresh tree," `/rewind` is "I want to back up one turn." Two verbs, two meanings, no overloading. Users who want "go back to root without losing the tree" navigate to the root explicitly (sidebar click / `/nav root`).

### What dropped from the earlier op list

The Q2 discussion had seven ops; collapse takes it to five:

- **branch** + **fork** → one op (create-sibling-with-text-X). The only difference is whether the text buffer pre-fills with the parent's text — a UI flavor, not an engine difference. One verb at the surface: `branch`.
- **edit** stays distinct from `branch`. In-place mutation and always-sibling have different costs in the tree: edits should not grow the tree (or typo fixes produce chains of near-duplicate dead siblings); branches should (the whole point is to keep the alternates). Collapsing them was the earlier draft; the visual-collapse-in-renderer remedy was the consolation prize for a fat tree, better to not grow them in the first place.
- **continue** → not a primitive at all. It's `navigate` followed by `send` — two separate user actions, no atomicity. Drop from the op list; the workflow is "click a node, type."

---

## Phase 1 — Engine-side LoomTree

**Why first:** every later phase depends on the data model. No visible behavior change after this phase alone; the tree has exactly one path and behaves like today's linear history.

### Module: `saklas/core/loom.py`

```python
@dataclass(frozen=False)
class LoomNode:
    id: str                                    # ulid
    parent_id: str | None
    role: Literal["user", "assistant", "system"]
    text: str
    tokens: list[TokenScoreDict] | None = None
    thinking_tokens: list[TokenScoreDict] | None = None
    recipe: Recipe | None = None               # assistant nodes only
    aggregate_readings: dict[str, float] = field(default_factory=dict)
    applied_steering: str | None = None
    finish_reason: str | None = None
    starred: bool = False
    notes: str = ""
    created_at: float = field(default_factory=time.time)
    edited_at: float | None = None             # last in-place edit timestamp
    edit_count: int = 0                        # bumps on each in-place edit


@dataclass
class Recipe:
    steering: str | None                       # canonical expression
    sampling: SamplingConfig | None
    thinking: bool | None
    seed: int | None
    # Probe set captured at gen time so replay matches even if the user
    # has added/removed probes since.  ``probe_hashes`` is sha256 of each
    # probe's baked tensor bytes — transcript replay diffs these to catch
    # probe drift between save and load.
    probes: list[str] = field(default_factory=list)
    probe_hashes: dict[str, str] = field(default_factory=dict)


class LoomTree:
    """Mutation-safe tree of conversation nodes.  Owned by SaklasSession;
    callers go through session methods rather than mutating directly so
    locking and event emission happen in one place."""

    root_id: str
    nodes: dict[str, LoomNode]
    children_of: dict[str, list[str]]          # parent_id -> ordered ids
    active_node_id: str                        # may be a leaf or interior node
    rev: int                                   # monotonic tree revision (WS delta cursor)

    # Read
    def active_path(self) -> list[LoomNode]: ...
    def messages_for(self, leaf_id: str | None = None) -> list[ChatMessage]: ...
    def children(self, node_id: str) -> list[LoomNode]: ...
    def descendants(self, node_id: str) -> Iterator[LoomNode]: ...

    # Mutate (returns new node id, fires LoomMutated event)
    def add_user_turn(self, text: str, parent_id: str | None = None) -> str: ...
    def begin_assistant(self, parent_id: str, recipe: Recipe) -> str: ...
    def append_token(self, node_id: str, score: TokenScoreDict) -> None: ...
    def finalize_assistant(self, node_id: str, **kwargs) -> None: ...

    # The five core primitives (see "Core operations"):
    def edit(self, node_id: str, text: str) -> None: ...
                                               # in-place; mutates node.text
    def branch(self, node_id: str, text: str) -> str: ...
                                               # always-sibling — returns new id
    def navigate(self, node_id: str) -> None: ...
    def delete_subtree(self, node_id: str) -> int: ...
    # regenerate is exposed via session.generate(parent_node_id=..., n=...) —
    # the gen path already owns the streaming machinery.

    # Decoration:
    def star(self, node_id: str, on: bool = True) -> None: ...
    def annotate(self, node_id: str, notes: str) -> None: ...

    # Engine-level workflows:
    def reset(self) -> None: ...               # clear — fresh root, drops tree
    def rewind(self) -> None: ...              # navigate one pair up active path

    # Predicate ops (phase 5 builds the UI; phase 1 ships the engine)
    def filter(self, pred: Callable[[LoomNode], bool]) -> set[str]: ...

    # Persistence
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "LoomTree": ...
    def save(self, path: Path) -> None: ...    # atomic write via io/atomic.py
    @classmethod
    def load(cls, path: Path) -> "LoomTree": ...
```

Tree IDs are ulids (sortable + unique). The root is a synthetic system node with no text; first user turn is its child. Nodes are stored flat in `nodes`; `children_of` is the structure map. Operations that mutate the tree bump `rev` and fire a `LoomMutated` event on `session.events` so surfaces can re-render without polling.

**Edit vs branch.** `edit(id, new_text)` is in-place: mutates `node.text`, no tree-shape change, bumps `node.edit_count` and `node.edited_at`, fires `LoomMutated(op="edit")`. `branch(id, new_text)` creates a new sibling with `new_text`, makes it the active node, and leaves the original untouched. Two distinct ops because they have different costs in the tree: typo fixes should not grow the tree; deliberate alternates should. Editing an assistant node's text is allowed but the flagged metadata lets downstream consumers know the text isn't pristine from the model.

**Active node, not active leaf.** `active_node_id` may point at a leaf or an interior node. Interior selection happens when a user navigates back up the tree to send a new turn from there. Send semantics are role-aware — see "Active-node send semantics" below.

### Session integration

```python
class SaklasSession:
    tree: LoomTree                             # new attribute

    @property
    def history(self) -> list[ChatMessage]:
        """Compat shim — returns messages for the active path."""
        return self.tree.messages_for()

    def clear_history(self) -> None:
        """Resets the tree to a fresh root.  Existing callers preserved."""
        ...

    # generate() gains optional parent_node_id.  When provided, results
    # attach as a child of that node rather than the active node.  When
    # absent (default), behaves like today — but is_loom branches still
    # land in the tree, just at the current leaf.
    def generate(self, input, *, parent_node_id=None, n=1, **kwargs) -> ...

    def generate_stream(self, input, *, parent_node_id=None, n=1, **kwargs) -> ...
```

`n > 1` is the engine-side primitive that fan-out uses. Each of the N runs creates its own sibling under `parent_node_id` (or the active node's parent if input is a user turn that already exists, or a fresh user node otherwise — exact semantics in the tests).

**Deterministic seed schedule for N-way regen.** When `n > 1`, per-sibling seeds derive from the parent's seed via a BLAKE2b-8-byte-digest avalanche on the packed `(parent_seed, i)` pair, masked to 31 bits. The implementation lives in `saklas.core.loom._mix_seed`; the doc-level "fnv1a64" wording was a planning placeholder. The substitution to BLAKE2b is deliberate — FNV-1a-64 over little-endian bytes is nearly linear for small `i` (the high bytes of small integers are zero, so consecutive indices differ in one input byte and FNV-1a's output barely scrambles them, producing correlated seeds within a single fan-out). BLAKE2b's 8-byte digest avoids that small-`i` collision pathology while keeping a single, fast hashlib call. A fan-out from the same parent with the same N is reproducible across sessions and across machines — important for shared-transcript replay. When the parent recipe has no seed, the schedule resolves an entropy-derived base seed and records it in each sibling's `Recipe.seed`. Single regen (`n=1`) without an explicit seed uses fresh entropy as today.

#### Active-node send semantics

What "send" does depends on the active node:

- **Active node is a leaf assistant node** — new user turn as child + assistant turn as grandchild. The today-flow.
- **Active node is a leaf user node** — error: a user turn is already waiting for an assistant. Suggest `/regen` or pop the user turn first.
- **Active node is root (synthetic system node)** — new user turn at root + assistant grandchild.
- **Active node is an interior assistant node** — new user turn as child (appended to the end of the existing children list) + assistant grandchild. The loom-native "continue from here" workflow.
- **Active node is an interior user node** — same as leaf user node: error.

In any "create a new user turn" case, if the exact text being sent matches one of the active node's existing user-turn children, the engine treats this as a branch-from-existing-user-turn and only spawns the assistant grandchild as a new sibling. Spares users one redundant tree level for the regen-style workflow.

#### Concurrency: mutations vs in-flight generation

The invariant: **the `_gen_lock` holder owns the subtree containing its target node.** A generation in flight against node N reserves the subtree rooted at N's user-parent (its "gen reservation"). Mutations on that reservation follow these rules:

- **Star, note, annotate** — always free. Decoration-only, no race.
- **Branch on any node in the reservation** — free. Creating a sibling doesn't disturb the streaming target.
- **Edit on any node in the reservation** — refused with `MutationDuringGenerationError` (Python) / `409 Conflict` (HTTP). Even editing a non-target node in the reservation can corrupt token-score replay for downstream readers.
- **Delete subtree containing the gen target** — refused with `409 Conflict` and a "stop the generation first" hint. Delete subtree disjoint from the gen target — free.
- **Navigate away** — free. The gen-in-flight stays attached to its original target; the user sees the trait panel switch to the new active node's path but the streaming continues invisibly. `node_id`-tagged token events keep flowing on the WS; the user can navigate back at any time.
- **Stop during N-way regen** — `stop_requested` cancels the currently-streaming sibling. Remaining queued siblings are skipped, not started. Sibling boundaries are the only valid stop points; mid-sibling stop trims the current generation cleanly.
- **`tree.reset()` during in-flight gen** — refused with `409 Conflict`. User must stop the generation first.

These rules are spec'd in `tests/test_loom_concurrency.py` and enforced via `session._assert_no_conflict(node_id, op)` at the entry of every mutator on `LoomTree`.

### Backward compat

Existing callers of `session.history`, `session.clear_history`, `session.rewind`, and `session.generate(input)` work unchanged. The flat `history` is a derived property; mutations to it (which a few CLI paths do directly) are migrated to operate on the tree.

`session.rewind()` walks back one user→assistant pair on the active path by re-pointing `active_node_id` to the grandparent of the current leaf. The dropped nodes stay in the tree (they're branches now); a separate `session.tree.delete_subtree` is the destructive op.

A/B compare's `abPair`-on-turn pattern goes away in the engine. The tree replaces it: A and B become two assistant siblings of the same user node. The A/B toggle still has meaning — "always pre-spawn an unsteered sibling on send" — and that's how phase 3 and 4 implement it.

### Tests: `tests/test_loom.py`

- Basic tree ops: add user, begin assistant, append tokens, finalize, navigate.
- `messages_for(node)` returns the right path under siblings, depth-mixed branches.
- Edit-in-place: `edit(id, new_text)` mutates `node.text`, no new node, `edit_count` increments, `rev` bumps, `LoomMutated(op="edit")` fires.
- Branch-always-sibling: `branch(id, new_text)` produces a sibling, original preserved; empty `new_text` is the "branch from blank" flavor; copying parent text and mutating is the "fork-and-edit" flavor — both are the same primitive.
- Active-node send semantics: every (role, leaf?) combination dispatches per the table above; interior-assistant send creates a child sibling-aware.
- Concurrency: edit during in-flight gen on the same reservation raises `MutationDuringGenerationError`; branch on the same reservation succeeds; delete subtree containing gen target raises; delete subtree disjoint from gen target succeeds; navigate-away during gen leaves the gen attached to its original target.
- `delete_subtree` cascades and refuses to delete the active path's ancestor (forces navigate-away first).
- `reset` (clear) drops everything; `rewind` navigates up one pair without deletion.
- Regenerate via `session.generate(parent_node_id=..., n=...)` produces N siblings under the right parent; per-sibling seeds match the FNV-1a-64 schedule; recipe override (Q5 mechanism) overlays correctly.
- Save/load roundtrip preserves order of children, ulid stability, token-score fidelity, `rev`, and `model_id`.
- Compat: `session.history` matches the active path's messages exactly; `session.rewind()` and `session.clear_history()` behave per the locked semantics.

---

## Phase 2 — Server routes and WS extension

**Why second:** without server routes, neither surface can talk to the tree from a non-embedded position. WS already carries the gen lifecycle; we add tree-aware fields and new lifecycle events.

### REST routes

```
GET    /saklas/v1/sessions/{id}/tree
       Full tree as JSON (nodes + children_of + active_node_id).

GET    /saklas/v1/sessions/{id}/tree/active
       Active path only — what today's "render the chat" path needs.
       Cheaper than full tree for surfaces that don't need the structure.

POST   /saklas/v1/sessions/{id}/tree/navigate
       Body: {node_id: str}.  Re-points active_node_id.  Returns active path.

POST   /saklas/v1/sessions/{id}/tree/edit
       Body: {node_id: str, text: str}.  In-place text replacement.
       Returns the mutated node.  409 if the node is in the reservation
       of an in-flight generation.

POST   /saklas/v1/sessions/{id}/tree/branch
       Body: {node_id: str, text: str}.  Always-branch — creates a new
       sibling with the given text (empty text is the "branch from blank"
       UI flavor).  Returns the new node id.  Allowed during in-flight gen.

DELETE /saklas/v1/sessions/{id}/tree/{node_id}
       Subtree delete.  Refuses to delete an ancestor of the active node
       (409 with a "navigate first" hint).  Refuses (409) when the node
       is in the reservation of an in-flight generation.

POST   /saklas/v1/sessions/{id}/tree/star
       Body: {node_id: str, on: bool}.

POST   /saklas/v1/sessions/{id}/tree/note
       Body: {node_id: str, text: str}.

POST   /saklas/v1/sessions/{id}/tree/transcript
       Body: {node_id?: str}.  Returns a transcript YAML for the path
       ending at node_id (or active node).  Phase 5 wires the import side.
```

### WS extension

The existing `generate` request grows two optional fields:

```ts
{
  type: "generate",
  input: ...,
  steering: ...,
  sampling: ...,
  thinking: ...,
  stateless: ...,
  raw: ...,

  // New for loom:
  parent_node_id?: string | null,  // attach result here; null = active node
  n?: number,                       // number of siblings, default 1
  recipe_override?: Partial<Recipe>,  // overlays on the parent recipe; phase-5 auto-regen / fan-out
}
```

`n > 1` streams interleaved token events tagged with `node_id` so the client can route to the right sibling render. The `started` and `done` events also carry `node_id`. Existing single-node clients ignore the field and behave as today.

Two new WS events:

```ts
{ type: "tree_mutated",
  op: "edit" | "branch" | "navigate" | "delete" | "star" | "note" | "reset",
  // Delta payload — clients apply directly, no re-fetch needed.
  added?: LoomNodeJSON[],          // new nodes (branch, n-way gen prelude)
  removed?: string[],              // node_ids dropped (delete_subtree)
  updated?: { node_id: string, fields: Partial<LoomNodeJSON> }[],
                                   // edit/star/note partial-update
  active_node_id?: string,         // present on navigate (and any op that moves it)
  rev: number,                     // monotonic server-side tree revision
}
  // Clients track the last-seen `rev`. On gap (missed event, reconnect),
  // they GET /tree to resync. The delta payload is sufficient for in-place
  // application of every op above; full re-fetch is the missed-event fallback.

{ type: "node_created", node_id: string, parent_id: string, role: "...", rev: number }
  // Fired at the start of each branch in an n-way generate.  Lets the
  // client allocate render slots before token events arrive.
```

### Backward compat for non-loom HTTP clients

OpenAI `/v1/*` and Ollama `/api/*` routes don't grow tree awareness. They keep operating on the active path's history — same as today, because `session.history` continues to mean the active path.

---

## Phase 3 — Webui sidebar

**Surface: left sidebar, collapsible.** Per decision logged below — lowest-risk, ships fastest, preserves the existing 841-line `Chat.svelte` which keeps rendering the active path linearly.

### Components

- `LoomSidebar.svelte` — collapsible left-edge panel mounted by `App.svelte`. Toggleable from the topbar.
- `LoomNode.svelte` — single-node chip: role glyph, first ~40 chars of text, steering-delta-from-parent (phase 5), colored ring decoration (highlight-probe aggregate). Active path bolded; dead branches dimmed.
- `LoomEdge.svelte` — connector lines with optional steering-delta labels (phase 5).

### Store: `loomTree` slice in `stores.svelte.ts`

- Mirrors the server's tree state. Fetched on bootstrap; refreshed on WS `tree_mutated`.
- `chatLog.turns` becomes `$derived` from `loomTree.activePath` — `Chat.svelte` keeps reading `chatLog.turns` and doesn't notice the substrate change.
- `chatLog.pendingIndex` → `loomTree.pendingNodeId`. WS handlers route by node id.
- The shadow-gen path's `messages` builder repoints to walk the tree's active path. A/B becomes "fire an unsteered sibling generate alongside the steered one" — same protocol, different framing.

### Persistence migration

- `PERSIST_VERSION` bumps from 1 to 2.
- v1 loader hydrates the linear log as a single-branch tree (root → user → assistant → ... → leaf). No data loss.
- localStorage becomes a cache of server state. On bootstrap, fetch tree from server and reconcile; on conflict, server wins.

### Node operations (right-click context menu + keyboard)

The five core ops plus annotate. Edit and branch are distinct entries: edit mutates in place (no new node), branch pre-fills the input with the current node's text and commits a sibling on Enter (clear the buffer for "branch from blank").

- Regenerate (N) — N siblings of the active assistant, same recipe, deterministic seed schedule.
- Edit — in-place text replacement; mutates the node, no new sibling.
- Branch — new sibling with text pre-filled from the current node; user mutates or clears.
- Navigate — single-click on any node.
- Delete subtree — confirm dialog; refuses ancestors of active node and reservations of in-flight gens.
- Star / unstar, add note — decoration ops.
- Pin to compare pane (phase 5) — see "Branch pinning."
- Compare children (phase 5) — surfaced on user nodes with ≥2 assistant children; opens the cross-branch diff drawer.
- Fan out (phase 5) — α-grid regenerate; surfaced under the assistant-node context menu.

### Keyboard navigation

Global shortcuts (work from chat input):

- `Ctrl+R` — regenerate the active assistant (N=1, same recipe).
- `Ctrl+E` — edit the active node in place (opens inline buffer with current text; commit replaces).
- `Ctrl+B` — branch from the active node (opens inline buffer pre-filled with current text; commit creates a sibling). Browser `Ctrl+B` bold-formatting is intercepted in chat inputs via `preventDefault`.
- `Ctrl+N` — open the nav picker (text-prefix search; Enter to navigate).
- `Ctrl+D` — delete the active subtree (confirm).

Sidebar-focused navigation:

- `j`/`k` move within siblings of the focused node.
- `h`/`l` move up/down the tree.
- `Enter` makes the focused node the active node (same as `Ctrl+N` followed by selection).
- `s` star, `n` note.
- `/` opens an in-sidebar fuzzy search across node text.

---

## Phase 4 — TUI loom

**Surface: full-screen alternate mode.** The TUI's three-column layout (vectors / chat / traits) is already crowded; adding a fourth column doesn't fit on standard terminals. Loom in the TUI lives in a separate Textual `Screen` — `Ctrl+L` switches to it, `Esc` returns to the chat screen.

This is a different shape from the webui's sidebar but it's the right shape for the TUI: navigation is keyboard-native already, and a full-screen tree gives breathing room for node bodies. The chat screen stays linear (active path); the loom screen is for traversal.

### Module: `saklas/tui/loom_screen.py`

Textual `Screen` subclass. Renders a `Tree` widget on the left (node IDs + role + first-line snippet), a node-detail pane on the right (full text, recipe, probe readings, notes). Footer shows binding hints.

Bindings on the loom screen:

- `j`/`k` next/prev sibling.
- `h`/`l` (or `←`/`→`) up/down the tree.
- `Enter` make-active + return to chat (same as `Ctrl+N` selection in webui).
- `r` regen (prompts for N), `e` edit in place, `b` branch (pre-filled with current text), `d` delete (with confirm).
- `s` star, `a` add note (opens a single-line input). `n` is taken by navigate-picker in the global shortcut family, so the loom-screen note key is `a`.
- `f` fan out (prompts for vector + alpha list).
- `/` text search.
- `Esc` back to chat.

### Slash commands on the chat screen

The five core verbs plus extras for fan-out, transcript IO, tree-screen toggle, and decoration:

- `/regen [N]` — N siblings of active assistant. Default 1. Bound to `Ctrl+R`.
- `/edit` — open inline buffer with active-node text; commit replaces in place (no new sibling). Bound to `Ctrl+E`.
- `/branch` — open inline buffer pre-filled with active-node text; commit creates a sibling. Bound to `Ctrl+B`. Clear-buffer-and-commit is "branch from blank."
- `/nav <id-prefix>` — navigate by ulid prefix (git-short-sha style). Bound to `Ctrl+N` (opens picker).
- `/del` — delete the active subtree. Bound to `Ctrl+D` with a confirm.
- `/fan <vector> <alphas>` — fan-out sweep (regenerate-with-grid-override). Alphas accept comma list, `linspace(...)`, or `start:stop:step` (same parser as the old sweep drawer).
- `/tree` — open the loom screen.
- `/path` — print active-path summary inline.
- `/star`, `/note <text>` — decoration on the active node.
- `/prune <when-expr>` — set the loom screen's filter highlighting; persists until cleared.
- `/auto-regen [mode]` — toggle / configure the auto-regen modifier (see phase 5). `Ctrl+A` keeps its current meaning (toggle on/off); the slash command sets the mode.
- `/transcript export <path>` — export active path as transcript YAML.
- `/transcript load <path> [--here | --merge] [--strict]` — import a transcript. Default attaches at root; `--here` attaches at the active node; `--merge` finds the deepest matching **user-turn** prefix between the active path and the transcript and attaches the non-matching tail there. `--strict` refuses on probe-hash drift.
- `/diff <id1> <id2> [--full]` — text diff + readings delta between two assistant nodes. `--full` prints the complete readings table; default prints unified-diff form + top-5 reading deltas.
- `/diff --siblings` — same as above but across all children of the active user-parent.

### A/B in the TUI

Today: `Ctrl+A` toggles a side-by-side mode, `_TurnRow` Horizontal in `chat_panel.py` paints two columns per turn. Phase 4 keeps the visual mode but reframes the underlying data: the two columns are two assistant siblings, both rendered. Toggling A/B on with prior history fires the one-shot backfill (matches today's `toggleAb` semantics), but the backfill becomes "spawn unsteered sibling" rather than "attach abPair to turn." Identical UX, cleaner model.

### Mid-gen interrupt + branching

Today, any conflicting action stops the current gen via `_pending_action` and dispatches after `("done",)`. With `n > 1` gens, the interrupt has to handle multiple in-flight branches. Two reasonable shapes: serialize (one branch at a time, queue the rest) or parallelize (all N stream concurrently). On a single GPU, serializing is correct — parallel decode of N independent contexts doesn't share much, and the throughput regression isn't worth the implementation complexity. Phase 4 ships serial; parallel is a phase-6 idea if it earns its way.

---

## Phase 5 — Saklas-native flourishes

These are what make this not-just-loom. Each is independent; ship as separate small PRs.

### Steering-delta edge labels

When two siblings have different recipes, render the edge with the delta against the parent's effective steering. Delta is computed by parsing both expressions, subtracting alpha by alpha, and rendering the non-zero terms compactly: `+0.2 calm`, `−honest`, `+0.3 warm@after`.

Webui: edge labels rendered as small text on the connector lines in `LoomSidebar.svelte`. TUI: node-detail pane shows the delta in a `Recipe` block alongside the full expression.

### Sweep deprecation and `/fan` as the canonical sweep

Per the locked decision, sweep-as-table goes away. The existing `SweepDrawer.svelte` and the TUI's `/sweep` command both become thin wrappers over the new fan-out primitive: pick a user-anchor node, pick a vector, supply an alpha grid, fire as siblings.

The migration: keep the sweep SSE route, repoint the result handler to land siblings instead of table rows. The drawer's table UI is deleted. The `SweepRow` type stays as an internal step but no longer renders.

### Filter grammar for tree pruning

The trees can grow large; users want to filter. The grammar is **adjacent to** the steering `@when:` grammar but distinct, because the underlying scalar is different: steering gates on per-step probe readings during generation, tree filtering gates on per-node aggregates. Reusing `@when:` would silently change semantics across contexts.

The pruning grammar:

```
filter_clauses := clause ("," clause)*           # multi-clause is AND
clause         := agg_op ":" probe op threshold
agg_op         := "agg" | "any" | "last"
                  #   agg  = aggregate (default; ProbeReadings.mean)
                  #   any  = max over per-token scores
                  #   last = last-token score
op             := > | >= | < | <=
probe          := <probe name as in @when:>
threshold      := <float>
```

Examples:

- `agg:angry.calm > 0.4` — branches whose every assistant node averages above 0.4.
- `any:hallucinating.grounded > 0.7, agg:honest > 0` — every node has some token above 0.7 on hallucination AND a positive honesty average.
- `last:refusal.compliant < 0` — every node ends below 0 on refusal.

Branches where every node matches stay bright; others dim. Filter-dim is a separate visual channel from dead-branch-dim (dotted edges or 50% opacity, vs. dead branches' 30%).

`/prune <filter>` sets the filter in the TUI; the webui has a filter input above the sidebar. `/prune` with no args clears.

### Cross-branch diff

Loom's killer use case for steering research: "what does +0.2 calm actually change in the output." When two assistant siblings share a parent user node, render a diff on demand.

Two layers:

- **Text diff** — word-level Myers diff on the two assistant texts, rendered side-by-side or unified. Same primitive as `git diff --word-diff`.
- **Readings diff** — `Δreading = sibling_B.readings - sibling_A.readings` per probe, sorted by magnitude. Color the top-k deltas; expose the full table on click.

Webui: right-click a user node with ≥2 assistant children → "compare children" → opens a comparison drawer with both texts side by side (word-level diff) and a sortable readings-delta table below. Multi-select (Ctrl-click) lets the user diff more than two siblings; rendering switches to N-column.

TUI: `/diff <id1> <id2>` prints to chat — unified-diff form (cheaper on terminal width) + top-5 reading deltas. `/diff <id1> <id2> --full` adds the complete readings table; `/diff --siblings` operates on all children of the active user-parent.

**Per-token cross-branch highlight** falls out for free: the existing per-token probe-score coloring extends to "color this token by its delta against the corresponding token in the pinned sibling." Hover a token in branch A → position-aligned token in branch B lights up with its per-token reading delta. Position-alignment uses byte offsets, not token offsets, because different siblings can tokenize the same prefix differently; we re-tokenize against a common reference at diff time.

### Branch pinning to the comparison pane (webui)

Today's A/B pane is "active path + its unsteered counterpart" via the auto-regen toggle. The natural generalization: pin any sibling to the right column.

Right-click any assistant node → "pin to comparison pane." Pinning replaces whatever was there before. The chat view's right column renders the pinned sibling's full subtree path; auto-regen still writes into "the right column" but the right column now means "whatever's pinned." Toggling auto-regen off while a pin is active keeps the pinned view; turning auto-regen back on overwrites the pin with the auto-regen output. A small "unpin" affordance on the right column header clears it.

Strict generalization of today's A/B. Drops out of the cross-branch-diff infrastructure with one extra context-menu entry plus a `pinned_node_id` field in the webui store.

### Transcript export/import

A **transcript** is a saved path through the tree: system prompt + every user turn + every assistant turn's `Recipe` (steering, sampling, seed, probe set, per-probe content hash) + final aggregate readings. Serializes to YAML. The per-node thing remains `Recipe`; the file/export concept is `Transcript` so the doc and CLI stop overloading.

```yaml
saklas_transcript: 1
model_id: google/gemma-3-4b-it
system_prompt: "You are a helpful assistant."
probes:
  - name: angry.calm
    sha256: 8a4f...
  - name: honest.deceptive
    sha256: c019...
turns:
  - role: user
    text: "What makes a good day?"
  - role: assistant
    text: "..."
    recipe:
      steering: "0.3 honest"
      sampling: {temperature: 0.7, max_tokens: 256, seed: 42}
      thinking: false
      probe_hashes: {angry.calm: 8a4f..., honest.deceptive: c019...}
    readings: {angry.calm: -0.12, honest.deceptive: 0.41, ...}
  - role: user
    ...
```

`saklas transcript run <path>` is the CLI verb — loads, replays, and reports. Inside a loom session, `/transcript load` imports as a new branch.

**Three import modes:**

- **Default** (no flag) — attaches as a new top-level branch off the root. Clean import; no interaction with the active path.
- **`--here`** — attaches as a child of the active node. Useful for splicing a transcript-as-prompt into an existing conversation.
- **`--merge`** — walks the active path from root, finds the deepest matching **user-turn** prefix between the active path and the transcript (matching = same user-turn text in the same order; assistant outputs are advisory only and may differ), attaches the transcript's non-matching tail as a sibling branch at that point. Falls back to root-attach when no user-turn prefix matches.

User-turn-only matching is the load-bearing choice. Assistant outputs are a function of seed × steering × model × probe-state, so byte-equal assistant text across replay is the rare case, not the common case. Matching on assistant text would make `--merge` fall through to root-attach for nearly every real load. The user-turn shape captures the genuine "your conversation and mine started the same way" intuition users have.

**Guards:**

- **Transcript's model differs from current session's** → warn; refuse `--merge` (semantic mismatch); allow `default` and `--here` with a banner on the imported branch.
- **Transcript's system prompt differs from current session's** → warn but proceed; the imported branch carries a note with the original prompt.
- **Transcript's probe set differs from current session's** (any name missing) → warn; readings are recorded as-imported for display but won't update on regen unless the user loads the missing probes.
- **Probe content drift** (probe name present but `sha256` doesn't match) → warn loudly with the diff (which probes drifted); replay proceeds but the user sees that downstream readings may not reproduce. `--strict` refuses the load on any hash mismatch.

The webui surfaces the three modes as radio buttons in the load drawer; the strict flag is a checkbox. The TUI uses the flag form on `/transcript load`.

This closes the loop: find an interesting steering combo → save the path → share the YAML → another saklas user replays exactly and continues from where you left off. The reproducible-research move.

### Auto-regen as a regen modifier

The fusion: regenerate takes a `recipe_override` parameter (a partial Recipe that overlays the parent's), and the existing A/B toggle becomes "fire one regen with this configured override after every primary gen." Today's A/B is the special case where the override is `{steering: ""}` (unsteered).

**Modes that fall out naturally:**

- **Unsteered** — `{steering: ""}`. Today's A/B behavior; the default mode so existing users see no regression.
- **Inverted** — flips every term's α sign. "What would the opposite steering produce."
- **Reseed** — `{seed: <new>}`. "Another sample of the exact same recipe."
- **Cool** / **Hot** — `{sampling: {temperature: 0.3}}` or `{...: 1.2}`. "What if I sampled differently."
- **Custom** — user-typed partial recipe. The fully general form.

**Webui:** the topbar A/B checkbox becomes an "auto-regen" toggle with a small ⚙ next to it. ⚙ opens a popover with the mode radio + a custom-recipe input field. The two-column chat view stays; the right column now reflects whatever override mode is active.

**TUI:** `Ctrl+A` keeps its meaning (toggle auto-regen on/off). `/auto-regen <mode>` sets the mode (`unsteered`, `inverted`, `reseed`, `cool`, `hot`, or any quoted partial recipe for custom). `/auto-regen` with no args reports the current mode.

**Manual regen with overrides:** the same `recipe_override` parameter is exposed in the manual op. Webui: a "regen N" right-click action opens a modal with the override picker (defaults to "same recipe"). TUI: `/regen` is the plain form (N=1, no override); `/regen N <mode>` runs the variant. Manual and automatic share one engine primitive.

**Migration:** existing A/B users land on `auto-regen: unsteered` the first time they toggle on. Identical behavior to today; configurability is additive.

---

## Persistence

> **v2.3 status: minimal.** Single-file persistence shipped:
> `~/.saklas/sessions/<session_id>/tree.json` via atomic write, plus a
> persistent anonymous default session id at
> `~/.saklas/sessions/.default` so the tree survives process restarts
> without an explicit `--session` name. `SaklasSession(session_id=...)`
> threads the id through `__init__` / `from_pretrained`; a debounced
> 1s save fires on every `LoomMutated` event and `session.close()` /
> `__exit__` flush synchronously. The rest of this section is deferred
> to v2.4: per-node token blobs split into `tokens/<node_id>.json`,
> named sessions via `--session <name>`, `saklas session ls/resume/rm`
> CLI verbs, and 30-day auto-prune of anonymous unstarred sessions.

### Session identity

Saklas today has no `session_id` concept — the TUI is one-shot, the Python API is ephemeral, the server has many concurrent sessions but no naming. Loom needs persistence, so:

- **Auto-ulid sessions.** Every fresh session gets a `session_id = ulid()` on first mutation (anything that touches the tree). Persists to `~/.saklas/sessions/<session_id>/`.
- **Named sessions** — `saklas tui --session <name>` and `saklas serve --session <name>` open or create a named session. Names are slugs (`[a-z][a-z0-9._-]*`); collisions resume the existing session.
- **`saklas session ls / resume / rm`** — listing, resuming (re-attaches to an existing tree by id or name), and explicit deletion verbs.
- **Auto-prune.** Sessions older than 30 days **with no name and no starred nodes** are GC'd at process startup. Named sessions and sessions containing stars persist indefinitely.
- **Tree carries `model_id`.** Saved trees record the model they were generated against. Loading a tree against a different model warns and refuses to attach new gens unless `--force` is passed (steering tensors don't transfer). Same guard the transcript-import section uses.
- **Header carries `saklas_version`** — for future migrations, atomic-write versioning following the pack-format pattern.

### File format

`~/.saklas/sessions/<session_id>/tree.json` — atomic write via `saklas.io.atomic.write_json_atomic`. Schema versioned (`tree_format: 1`); future migrations follow the pack-format pattern. Carries `model_id`, `saklas_version`, `session_id`, `name` (when named), and `rev`.

`~/.saklas/sessions/<session_id>/tokens/<node_id>.json` — per-node token-score blobs split out so the main tree file stays small. Lazy-loaded on demand by the surfaces. Streamed-in tokens write through to this file at finalize. Includes `tokens`, `thinking_tokens`, and per-token logprobs when present; everything else (text, recipe, aggregate readings) stays in `tree.json` so navigation doesn't pay the token-blob cost.

The tree file is small (metadata + structure); the token blobs are the bulk. Split lets the tree file land within a single atomic write while the larger token data is per-node and append-only at gen time.

### Size management

Webui localStorage gets a per-tree size budget (~5MB default). Crossing it surfaces a toast suggesting recipe export + tree clear; doesn't hard-stop. Engine-side `tree.json` has no client-side budget but `tree.prune_dead_branches(min_age_days=30)` is provided as a manual operation; not automatic.

Per-node token blobs are the persistence bulk; the sidebar's "decoration ring" reads from `aggregate_readings` only, not per-token data, so navigating a huge tree doesn't need to load token blobs.

### Engine-side cache, surface-side render

The webui's localStorage stops being authoritative after phase 3. On bootstrap: fetch tree from server, hydrate the store, render. localStorage caches the last fetched tree for instant first-paint, but reconciles against server on mount. This retires the server-restart-guard hack — the server is the truth.

---

## Migration

### From v2.2 (current) to v2.3 (loom)

1. `session.history` keeps working as a property derived from `tree.active_path`. Existing Python callers don't break.
2. `session.clear_history()` becomes "reset tree to fresh root." Existing semantics preserved (destructive — matches today's user expectation of `/clear`).
3. `session.rewind()` re-points the active node one user→assistant pair back. Under loom this is **non-destructive**: dropped turns stay in the tree as a now-dead branch; user can navigate back via `/nav <prefix>` (TUI) or the sidebar (webui). Strict improvement; nothing today depends on rewind being unrecoverable.
4. Webui `PERSIST_VERSION` 1 → 2 with auto-migration on load (single-branch tree from the linear log).
5. TUI's `_TurnRow` Horizontal stays; the data behind it becomes "two siblings" rather than "turn + abPair." UX identical.
6. `Ctrl+A` keeps its meaning in both surfaces (toggle the auto-regen modifier). Default mode is `unsteered` — bit-identical to today's A/B for existing users.

### Hard breaks

- `abPair` on `ChatTurn` is removed from the Python type (compat shim possible but loom siblings express it more cleanly). Webui keeps a one-release deprecation period.
- `SweepDrawer.svelte` and `saklas/tui/commands.py`'s `/sweep` are repointed to fan-out; the table view is deleted from the webui. **Landed in v2.3 (not v2.4 as originally scoped)** — the surface migration was small enough to bundle with the loom PR rather than carve out its own. `/sweep` survives as a deprecation alias that routes to `/fan` and prints a one-line "use /fan" banner so muscle memory still works.
- `session.config` for sampling defaults still works; per-turn recipe carries the per-call sampling and the session default is the fallback. No change visible to callers.
- "Recipe" the file format renames to "Transcript" in CLI verbs, slash commands, and YAML preamble (`saklas_transcript: 1`). The per-node `Recipe` dataclass keeps its name. One-release alias on `/recipe load` → `/transcript load` for migration.

---

## Decisions logged

These are the forks resolved during planning. New forks belong in this section as they're decided.

1. **Tree lives engine-side, not per-surface.** Reasons in "Architectural choice." Engine-side is bigger scope per commit but the right factoring.

2. **Webui tree UI: left sidebar, collapsible.** Lowest-risk; the existing `Chat.svelte` is preserved. Full-graph view is deferred.

3. **TUI tree UI: full-screen alternate Screen, `Ctrl+L` toggle.** Three-column layout is too crowded for a fourth column; full-screen gives breathing room and the TUI is already keyboard-native.

4. **Edit and branch are distinct primitives.** In-place edit mutates `node.text`; branch creates a sibling. Two ops because typo fixes and deliberate alternates have different costs in the tree: edit shouldn't grow the tree, branch should. Earlier draft collapsed them under always-branch + a phase-5 visual collapse for consecutive edits; that was wrong — visual collapse is the consolation prize for a fat tree of dead siblings, better to not grow them in the first place.

5. **Sweep results: loom siblings, table view deprecated.** Strongest unification. The drawer becomes a thin form over fan-out.

6. **Active path replaces server-side history at the protocol level.** OpenAI / Ollama / native HTTP clients keep operating on the active path's serialized history. No protocol change for non-loom clients.

7. **N-way gen is serial in v1.** Single GPU; parallel decode of N independent contexts doesn't share the cache enough to be worth the implementation complexity. Phase 6 idea if it earns its way.

8. **Op set collapses to five primitives.** `regenerate`, `edit` (in-place), `branch` (always-sibling), `navigate`, `delete subtree`. The earlier seven-op list (separate `branch`, `fork`, `continue`) had two flavors of the new `branch` (branch/fork = always-sibling with different starting buffer state) plus `continue` which is `navigate` + `send` (not atomic, not a primitive).

9. **Surface verb naming.** Slash commands: `/regen`, `/edit`, `/branch`, `/nav`, `/del`. Shortcuts: `Ctrl+R`, `Ctrl+E`, `Ctrl+B`, `Ctrl+N`, `Ctrl+D`. `/edit` is in-place — mutates the active node's text. `/branch` is always-sibling — pre-fills the buffer with the active node's text by default; clear the buffer for "branch from blank." Browser `Ctrl+B` is intercepted via `preventDefault` in the webui chat input to avoid the bold-formatting collision.

10. **Tree decoration ring follows single-probe selection per surface.** Webui uses `highlightState.target`; TUI uses the trait-panel's selected probe. No ring when nothing is selected.

11. **Transcript import: three modes, user-turn-only `--merge`.** Default attaches at root. `--here` attaches at active node. `--merge` walks for the deepest matching **user-turn** prefix and attaches the non-matching tail there. Assistant-text matching is too brittle to be the merge anchor (seed × steering drift breaks it). Guards on model mismatch (refuses `--merge`), system-prompt mismatch (warn + proceed with a note), missing probes (warn + display-only readings), probe content drift (warn with diff; `--strict` refuses).

12. **Per-node token blobs always split.** Uniform shape; no per-node size threshold. Modern storage budgets make the split overhead trivial.

13. **Auto-regen fuses into regen as a recipe-override modifier.** `Ctrl+A` toggles auto-regen on/off; mode is configurable (unsteered, inverted, reseed, cool, hot, custom). Manual regen takes the same `recipe_override` parameter. Today's A/B becomes the default mode (`unsteered`); no user-visible regression. The auto-regen toggle stays indefinitely — it's the auto-fire affordance over the manual op, not a deprecated legacy. **Custom-mode shape (v2.3 implementation):** `Recipe.compose_modifier(mode)` accepts both `str` (the five built-ins) and `Recipe` (the custom path). Surfaces parse `custom: <steering expression>` into a `Recipe(steering=<canonical form>)` partial via `saklas.core.steering_expr.parse_expr` and pass the typed partial through `regen_with_modifier`; the engine's `compose_modifier` is a no-op passthrough for Recipe instances. v2.3 covers steering only on the custom axis — sampling / thinking / seed overrides ride the named modes (reseed/cool/hot) or programmatic `Recipe` construction.

14. **Rewind becomes non-destructive.** `/rewind` re-points the active node one pair up; the dropped pair stays in the tree as a dead branch, navigable. `/clear` remains the destructive op for "wipe the tree." Two verbs, two meanings.

15. **Active node, not active leaf.** `active_node_id` may point at a leaf or an interior node. Interior selection happens when a user navigates back up the tree to send a new turn from there. Send semantics are role-aware: from an assistant node (leaf or interior), a new user turn attaches as a child; from a user node, send errors (suggest `/regen` or `/edit`). Enforced engine-side in `SaklasSession._check_user_send_target(parent_node_id)`, called from `_generate_core` before `tree.add_user_turn`; surfaces raise `InvalidNodeOperationError` (HTTP 400). Regen surfaces pass `parent_node_id=<user.parent_id>` so the dedup at `add_user_turn` re-uses the existing user sibling rather than tripping the guard.

16. **Concurrency: gen reserves its subtree.** The `_gen_lock` holder owns the subtree rooted at the user-parent of its target node. Decoration ops (star, note) and branches are always free; edits and deletes on that reservation refuse with 409. Navigate-away is free — gen continues invisibly; user can navigate back at any time. N-way stop cancels the current sibling and skips the queued remainder; mid-sibling stop trims cleanly.

17. **`Recipe` is per-node; `Transcript` is the export.** The dataclass per-node stays `Recipe` (steering + sampling + seed + probe set + content hashes). The file/export concept renames to `Transcript`. CLI verb: `saklas transcript run`. Slash command: `/transcript load`. Frees `Recipe` of the doubled meaning.

18. **Filter grammar is distinct from `@when:`.** Tree-pruning uses `agg:`/`any:`/`last:` prefixes on per-node probe readings. Steering's `@when:` gates on per-step readings during generation. Same probes, different scalars, different evaluators. One grammar would silently change semantics across contexts.

19. **Probe content hashes ride in `Recipe`.** Every entry in `Recipe.probes` is paired with an sha256 in `Recipe.probe_hashes`. Transcript replay catches probe drift via hash comparison; `--strict` refuses on mismatch.

20. **Deterministic N-way seed schedule.** When `n > 1`, sibling seeds derive via a BLAKE2b-8-byte-digest avalanche over the packed `(parent_seed, i)` pair, masked to 31 bits — see `saklas.core.loom._mix_seed`. The doc-level "fnv1a64" wording was a planning placeholder; FNV-1a-64 was rejected during implementation because over little-endian bytes it's nearly linear for small `i` (the high bytes of small integers are zero, so consecutive indices barely scramble), producing correlated within-fan-out seeds. BLAKE2b avoids that. Same parent + same N → same N siblings, byte-equal, across machines. Reproducibility for shared transcripts. Without a parent seed, the schedule resolves an entropy-derived base and records it.

21. **WS `tree_mutated` carries delta payloads, not just event types.** Each event includes the added / removed / updated node lists plus a monotonic `rev`. Clients apply in-place; full re-fetch is the missed-event fallback (detected via `rev` gap).

22. **Session identity is auto-ulid'd, optionally named.** Fresh sessions get `ulid()` on first tree mutation; `--session <name>` opens or creates a named one. `saklas session ls / resume / rm` are the verbs. Anonymous sessions older than 30 days with no stars are auto-pruned at startup; named or starred sessions persist indefinitely. Trees carry `model_id` and refuse to attach new gens on a mismatched session unless `--force`.
