# tui/

Textual frontend over `SaklasSession`. Three panels (left=vectors, center=chat, right=traits). ~15 FPS poll.

## app.py

Thin frontend. Owns local alpha/enabled/thinking state per panel, passes through per call as `SamplingConfig` + `Steering`. Thinking defaults ON for models that support it (`Ctrl+T` toggles).

**The `_generate` worker is a thin wrapper over `session.generate_stream`** — no direct reach into `_apply_steering` / `_begin_capture` / `_end_capture` / `_clear_steering` / `generate_steered`. Builds `SamplingConfig` + `Steering` from current UI state, calls `session.generate_stream(input, steering=..., sampling=..., on_token=...)`, and forwards tokens through an **app-local `_ui_token_queue: SimpleQueue`** with tagged messages `("tok", text, thinking)` / `("finalize", widget)` / `("error", msg)` / `("done",)`. `_poll_generation` drains the app-local queue; the worker's `finally` emits `("done",)` after the session-side teardown has already run.

## Slash commands

- **Steering**: `/steer <expression>` takes one full steering expression using the shared grammar from `saklas.core.steering_expr` — `/steer 0.5 honest`, `/steer 0.3 warm@after`, `/steer 0.5 honest:sae`. Each plain term extracts + registers + sets the local alpha. A bare term with no coefficient uses `DEFAULT_COEFF` (0.5). `/alpha <val> <name>` (adjust existing only — value-first to match the expression grammar; errors if unregistered), `/unsteer <name>`. Projection terms (`a|b`, `a~b`) and triggers (`@after`, etc.) are accepted by the parser; projections route through session materialization, triggers land on the local alphas state.
- **Probes**: `/probe <name>` (also seeds `_highlight_probe` and flips `_highlight_on = True`), `/unprobe <name>`, `/extract <pos> <neg>` or `/extract <name>` (to disk without wiring — and the sole path for creating new bipolar concepts from `<pos> <neg>`).
- **Namespace bulk** (`/steer <ns>/`, `/probe <ns>/`, `/unsteer <ns>/`, `/unprobe <ns>/`): trailing-slash form detected by `_detect_namespace_selector` before the parser runs. Add-side enumerates `_all_concepts()` filtered to the namespace and routes each through `session._try_autoload_vector` (cache-hit only — no PCA, no scenario gen). `/steer ns/` registers loaded concepts at `α=DEFAULT_ALPHA` with `enabled=False` so users flip them on individually from the left panel; `/probe ns/` seeds `_highlight_probe` to the lexicographically last loaded probe and turns highlight on (matches single-`/probe` UX). Skipped concepts (no tensor on disk for the active model) get listed in a one-line summary with a `pack refresh <ns> -m <model>` hint. Remove-side filters `_alphas`/`monitor.probe_names` on the `<ns>/` prefix and removes the matching subset; `/unprobe ns/` clears the highlight only when the seed sat inside the namespace. Add-side defers via `_pending_action` on in-flight gens (same shape as single-concept paths); remove-side mirrors single-concept policy and runs immediately.
- **Session**: `/clear`, `/rewind`, `/regen`, `/sys <prompt>`, `/temp`, `/top-p`, `/max`, `/seed <n>`, `/save <name>`, `/load <name>`, `/export <path>`.
- **Analysis**: `/compare <a> [b]` (1-arg: ranked cosine vs all loaded profiles; 2-arg: pairwise score).
- **Info**: `/model` (arch/device/layers/thinking/active state), `/help`.

**`/steer` parser**: delegates to `saklas.core.steering_expr.parse_expr`. Bare names route through `resolve_pole` (same as every other surface) and inherit canonical + sign flip for installed bipolar poles. Variant suffixes (`:sae`, `:sae-<release>`) are grammar-native, so the old `--sae` preamble is gone — type the variant directly into the term.

## Loom destructive ops confirm

`/del` requires `/del yes` to actually delete the active subtree — bare `/del` prints the confirm hint. **`Ctrl+D` routes through the same guard** (`action_delete_subtree` calls `_handle_del("")`) so users see the hint and have to type the slash form to confirm; no instant-delete on the keybinding. The loom screen's `d` binding overlays a typed-`yes` modal (`loom_screen.py:610`). Three surfaces, one consistent confirm shape.

`Ctrl+E` / `Ctrl+B` / `Ctrl+N` / `Ctrl+D` from the chat screen all open the loom screen for tree navigation; the inline slash forms (`/edit <text>`, `/branch [text]`, `/nav <prefix>`, `/del yes`) are how you mutate without leaving chat. `/help` documents this.

## Mid-gen interruption

Any conflicting action (Ctrl+R, new message, any modifying slash command) stops current gen and defers via `_pending_action`; `_poll_generation` consumes the `("done",)` sentinel and calls `_dispatch_pending_action` (single-site try/except that resets state and surfaces errors via `add_system_message`). Panel focus uses index constants `_LEFT`/`_CHAT`/`_TRAIT`.

## Per-token highlighting

Default-on when a probe is explicitly selected via `/probe`. `Ctrl+Y` toggles the visual overlay silently (no system message — user can see whether it's on); trait-panel selection updates the seed live while highlighting is on.

**Live highlighting**: TUI worker forwards `event.scores` from each `TokenEvent` through `_ui_token_queue`; `_poll_generation` calls `widget.append_token_score(scores, is_thinking)` per emit, which appends to the per-probe score list, clears the markup cache, and re-renders. `_build_highlight_markup` tolerates `len(scores) < len(token_strs)` (unscored tokens render plain) and skips leading-whitespace tokens on the response side. At finalize the canonical projected scores from `session.last_per_token_scores` overwrite live ones via `_finalize_widget_highlight`.

`Ctrl+A` toggles persistent A/B side-by-side mode (mirrors the webui's auto-regen toggle since v2.3 — the standalone A/B button is gone). When on, the chat log paints two columns per turn — primary on the left, modifier output on the right — driven by the `_TurnRow` Horizontal in `chat_panel.py` whose shadow column is gated `display: none` until the panel carries `.ab-on`. The right-column gen runs after each primary `done` and routes by `_loom_auto_regen_mode`: `unsteered` → `_start_shadow_generation(row)` rebuilds the conversation as a messages list via `_build_shadow_messages`, calls `session.generate_stream(messages, steering=None, stateless=True, …)`, and streams tokens into the shadow column through the same `_ui_token_queue` (items tagged `is_shadow=True` so gen-stat counters and the steered-`done` follow-up don't double-fire). Any other mode (inverted / reseed / cool / hot / custom) → `_fire_auto_regen(row=row)` calls `session.generate_stream(..., recipe_override=mode, parent_node_id=user_parent_id)` and streams into the same shadow column under the same `is_shadow=True` tag — live tokens in the right column for the modifier sibling, not a one-line system message. Toggling AB on with prior history fires a one-shot backfill for the most recent assistant turn (matches the webui's retroactive fire). `Ctrl+S` cycles trait sort — both documented in `/help`.

## Trait-panel WHY footer

Bottom-third split below the traits list. `#why-header` is always the literal `LAYERS` (no probe name — the trait panel above already shows what's selected). `#why-scroll` body shows `||baked||` for the trait-panel-selected probe as a horizontal histogram in layer order, bucketed into `HIST_BUCKETS = 16` evenly-sized groups (mean norm per bucket; bar scaled to the largest bucket) so the whole profile fits the 16-row box regardless of layer count. Models with ≤16 layers render one-bar-per-layer. Bucket labels are `LXX` for single-layer buckets, `LXX-YY` otherwise, zero-padded to the model's max-index width. No token list — per-token highlighting in the chat already surfaces which tokens the probe lights up on. Driven by `_refresh_trait_why()` from app.py; fired on trait nav, probe add/remove, finalize, clear, and on_mount — **not** per streamed emit, since layer norms are static for the duration of a gen. No `/why` chat command — selection alone drives the readout. `HIST_BUCKETS` lives in `saklas.core.histogram` alongside `bucketize()` — shared with `cli vector why`.

## Alpha keybindings

`←`/`→` nudge the selected vector's alpha by `_ALPHA_STEP_FINE = 0.01`; `shift+←`/`shift+→` use `_ALPHA_STEP_COARSE = 0.1`. Both clamp via `MAX_ALPHA` inside `_adjust_alpha`. The shift variants live in `on_key` (alongside the plain arrows) rather than the `BINDINGS` table because arrow handling is already context-sensitive — input-focused vs panel-focused — and the table-driven path doesn't pass through the same gate. `↑`/`↓` stay panel-nav (or input-history, see below) across both shift states.

## Input history (↑/↓ in chat input)

Shell-style recall on the chat input. Every line submitted through `ChatPanel.UserSubmitted` (chat messages and slash commands alike) lands in `_input_history` via `_push_input_history` — readline-flavored: ping-pong A→B→A records both A's, but A→A→A collapses to one. `_INPUT_HISTORY_MAX = 200` caps the ring; overflow drops oldest in a single slice op. ↑ calls `_history_navigate(-1)`, ↓ calls `_history_navigate(+1)`. First ↑ from the live slot stashes the in-progress draft (`_history_stash`); ↓ past the newest entry restores it and clears the cursor. Past-oldest pins to entry 0 (no wrap). Implementation lives in `on_key`'s input-focused branch, gated on `getattr(self.focused, "id", None) == "chat-input"` so the panel-side `↑`/`↓` (trait-panel nav) keeps its meaning when the input doesn't have focus. Textual's `Input` is single-line and doesn't bind ↑/↓ natively, so no override conflict. In-memory only — process-scoped, dies on TUI exit. The webui recall path lives in `webui/src/lib/stores.svelte.ts::inputHistory` (in-memory, edge-only multi-line policy in `Chat.svelte::shouldRecall{Up,Down}`).

## utils.py

`build_bar(value, max, width)` renders filled/empty bar pairs. `BAR_WIDTH = 24` used by every bar in the UI (footer token progress, vector alpha, gen-config temp + top-p, trait-panel probes) — one knob controls them all. Vector panel's `RIGHT_W` derives from `BAR_WIDTH` so gen-config right-edge glyph alignment stays correct through any width change.

## Loom phase 5 (v2.3)

Phase 5 fills in the surface for engine APIs that landed alongside (see `saklas.core.tree_filter`, `saklas.core.loom_diff`, `saklas.core.transcript`, `session.regen_with_modifier`, `session.generate_sweep(..., return_node_ids=True)`):

- **`/prune <filter-expr>`** parses through `saklas.core.tree_filter.parse_filter` (catches `FilterParseError` to chat), stashes on `app._loom_prune_expr`. The loom screen's `_rebuild_tree` calls `LoomTree.filter_by_expr` and dims every node outside the matching set via Rich `[dim]` markup on the tree label. Loom screen footer surfaces the active expression via `Screen.sub_title`; the chat-screen status footer adds `· filter:<expr>` via the `prune_expr` kwarg on `chat_panel.update_status` so users see filter state without leaving the chat surface. Empty arg clears. The filter grammar's `agg:` prefix is now optional — bare `angry.calm > 0.4` defaults to `agg:` (the per-node aggregate reading).
- **`/diff <id1> <id2> [--full]`** resolves ids by ulid prefix (`resolve_node_prefix`), calls `session.diff_nodes`, prints unified-diff word-level text plus signed-colored top-5 (or full) reading deltas via Rich markup. **`/diff --siblings`** walks the active user-parent's assistant children: two siblings → pairwise diff; ≥3 → a per-pair top-1 reading-delta matrix.
- **`/transcript load <path> [--here|--merge] [--strict]`** routes through `Transcript.load` + `Transcript.import_into`. `--here` attaches at active node, `--merge` walks the deepest user-turn prefix, `--strict` raises on probe-hash drift. Guards (model/system-prompt mismatch, missing probes, content drift) print to chat with a `⚠` sigil via `warnings.catch_warnings`. Active node navigates to the imported leaf on success.
- **`/transcript export <path>`** emits YAML via `Transcript.dump()` — round-trips through `Transcript.load`. Uses pyyaml's `safe_dump` when available, falls back to `_emit_yaml_minimal` in `saklas.core.transcript`.
- **`/auto-regen [on|off|<mode>]`** stores both an enabled flag (`_loom_auto_regen_on`) and a mode (`_loom_auto_regen_mode`). Valid modes: `unsteered` / `inverted` / `reseed` / `cool` / `hot`, or `custom: <partial recipe>`. **`Ctrl+A`** (`action_ab_compare`) flips both `_ab_mode` and `_loom_auto_regen_on` together (decision 13: Ctrl+A is the on/off toggle, the visible side-by-side layout is the consequence). The chat-screen status footer surfaces `· auto:<mode>` via `chat_panel.update_status`'s `auto_regen_mode` kwarg so users see the active mode without checking. After every primary steered `done`, `_poll_generation` consults the mode: `unsteered` → existing `_start_shadow_generation` path (bit-identical to today's A/B); anything else → `_fire_auto_regen(row=row)` which streams `session.generate_stream(..., recipe_override=mode, parent_node_id=user_parent_id)` tokens into the right column under `is_shadow=True` — same column the unsteered path uses, so users see live modifier tokens rather than a one-line system message.
- **`/regen [N] [mode]`** extends the phase-4 form: a trailing mode token routes through `session.regen_with_modifier(user_parent_id, mode, n=N)` instead of the plain N-way regen path. `/regen 4 inverted` means "4 siblings, inverted steering."
- **`/fan`** is the canonical sweep. The handler routes through `session.generate_sweep(prompt, {vector: alphas}, return_node_ids=True, parent_node_id=...)` so every sibling lands under one shared user-turn anchor in the loom tree. The legacy per-α `session.generate` loop is gone. **`/sweep`** is preserved as a deprecation alias that prints `"/sweep is deprecated — use /fan"` and routes to the same handler.
- **Steering-delta in node-detail pane**: `format_node_detail` now emits a `--- Recipe ---` block below the per-node fields whenever the assistant node has assistant cousins under the same user-parent. Each sibling line renders `Δ from sibling N (<id>): <steering_delta output>` via `saklas.core.loom_diff.steering_delta`. Refreshes on navigation through the existing `_refresh_detail` call.

CLI side: `saklas transcript run <path>` (in `cli/parsers.py` + `cli/runners.py`) was already wired for phase 5 — loads via `Transcript.load`, imports via `import_into(mode="default")`, then replays each user turn with the recorded recipe and prints reading-drift deltas. Falls back to `transcript.model_id` when `--model` isn't given; respects `--strict` for probe-hash drift.

## Status footer

`chat_panel.update_status`: one-line footer showing dot + `gen_tokens/max_new_tokens` + progress bar (green while generating, dim between runs — persists post-gen so the bar doesn't appear/disappear each turn; collapses to `○ idle` before the first generation or when `max_tokens` is unknown) · tok/s · elapsed · `ppl <mean>`. The count sits left of the bar. VRAM lives on the left panel already; context info isn't rendered here.

Perplexity is the geometric mean of per-step `TokenEvent.perplexity` values — `exp(sum(log(ppl)) / count)` across scored steps in the current gen. `_log_ppl_sum` / `_ppl_count` reset at `_start_generation`; `_last_gen_state` dedupe tuple tracks `_ppl_count` so the footer refreshes when the aggregate moves. Computed in `generate_steered` as `exp` of full-vocab fp32 Shannon entropy on pre-temperature post-steering logits; one extra softmax + one `.item()` sync per step, inside the steered-throughput headroom.

## Panels

**Trait-panel row layout**: two lines per probe — `> <name> [dim]<sparkline>[/]` on line 1 (bold wraps name only when selected), `  <bar> <val:+.2f><arrow>` on line 2 (two-space indent, color from sign of val). Full untruncated names — no fixed column width. Category headers are single lines. `trait_panel._nav_items` is `list[str]`.

**`chat_panel` is Markdown-free**: user + assistant messages are plain `Static` widgets with Rich-escaped text. `_AssistantMessage` holds exactly two content `Static`s (`#thinking-view`, `#response-view`) + a `Collapsible` wrapper for thinking. Tokens escaped once on append into `_escaped_chat_text`/`_escaped_thinking_text` (O(n) total). Per-token probe scores appended live via `append_token_score`; each append clears the markup cache and re-renders if highlighting is on. `set_token_data` runs at finalize to overwrite with canonical projected scores.

Saturation fixed at `_HIGHLIGHT_SAT = 0.5` mapped to full `rgb(0..255, 0, 0)` / `rgb(0, 0..255, 0)`; zero scores get no background span (neutral text stays terminal-default).

**Response-view leading-whitespace strip**: models often emit `\n\n` after `</think>` which would otherwise render as blank rows below the collapsed thinking title; `_render_response` lstrips the plain-text branch and `_build_highlight_markup` skips leading whitespace-only tokens.

Thinking-block CSS overrides Textual's `Collapsible` defaults (`padding-bottom: 1`, `padding-left: 1`, `border-top: hkey`) — set to zero, with `#thinking-block.-collapsed { height: 1 }` so the collapsed block is exactly the title row. Transitions via idempotent `ensure_thinking_collapsed()` (no `_in_thinking`, no `finalize()`, no `end_thinking()`).
