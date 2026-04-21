# tui/

Textual frontend over `SaklasSession`. Three panels (left=vectors, center=chat, right=traits). ~15 FPS poll.

## app.py

Thin frontend. Owns local alpha/enabled/thinking state per panel, passes through per call as `SamplingConfig` + `Steering`. Thinking defaults ON for models that support it (`Ctrl+T` toggles).

**The `_generate` worker is a thin wrapper over `session.generate_stream`** — no direct reach into `_apply_steering` / `_begin_capture` / `_end_capture` / `_clear_steering` / `generate_steered`. Builds `SamplingConfig` + `Steering` from current UI state, calls `session.generate_stream(input, steering=..., sampling=..., on_token=...)`, and forwards tokens through an **app-local `_ui_token_queue: SimpleQueue`** with tagged messages `("tok", text, thinking)` / `("finalize", widget)` / `("error", msg)` / `("done",)`. `_poll_generation` drains the app-local queue; the worker's `finally` emits `("done",)` after the session-side teardown has already run.

## Slash commands

- **Steering**: `/steer <expression>` takes one full steering expression using the shared grammar from `saklas.core.steering_expr` — `/steer 0.5 honest`, `/steer 0.3 warm@after`, `/steer 0.5 honest:sae`. Each plain term extracts + registers + sets the local alpha. A bare term with no coefficient uses `DEFAULT_COEFF` (0.5). `/alpha <val> <name>` (adjust existing only — value-first to match the expression grammar; errors if unregistered), `/unsteer <name>`. Projection terms (`a|b`, `a~b`) and triggers (`@after`, etc.) are accepted by the parser; projections route through session materialization, triggers land on the local alphas state.
- **Probes**: `/probe <name>` (also seeds `_highlight_probe` and flips `_highlight_on = True`), `/unprobe <name>`, `/extract <pos> <neg>` or `/extract <name>` (to disk without wiring — and the sole path for creating new bipolar concepts from `<pos> <neg>`).
- **Session**: `/clear`, `/rewind`, `/regen`, `/sys <prompt>`, `/temp`, `/top-p`, `/max`, `/seed <n>`, `/save <name>`, `/load <name>`, `/export <path>`.
- **Analysis**: `/compare <a> [b]` (1-arg: ranked cosine vs all loaded profiles; 2-arg: pairwise score).
- **Info**: `/model` (arch/device/layers/thinking/active state), `/help`.

**`/steer` parser**: delegates to `saklas.core.steering_expr.parse_expr`. Bare names route through `resolve_pole` (same as every other surface) and inherit canonical + sign flip for installed bipolar poles. Variant suffixes (`:sae`, `:sae-<release>`) are grammar-native, so the old `--sae` preamble is gone — type the variant directly into the term.

## Mid-gen interruption

Any conflicting action (Ctrl+R, new message, any modifying slash command) stops current gen and defers via `_pending_action`; `_poll_generation` consumes the `("done",)` sentinel and calls `_dispatch_pending_action` (single-site try/except that resets state and surfaces errors via `add_system_message`). Panel focus uses index constants `_LEFT`/`_CHAT`/`_TRAIT`.

## Per-token highlighting

Default-on when a probe is explicitly selected via `/probe`. `Ctrl+Y` toggles the visual overlay silently (no system message — user can see whether it's on); trait-panel selection updates the seed live while highlighting is on.

**Live highlighting**: TUI worker forwards `event.scores` from each `TokenEvent` through `_ui_token_queue`; `_poll_generation` calls `widget.append_token_score(scores, is_thinking)` per emit, which appends to the per-probe score list, clears the markup cache, and re-renders. `_build_highlight_markup` tolerates `len(scores) < len(token_strs)` (unscored tokens render plain) and skips leading-whitespace tokens on the response side. At finalize the canonical projected scores from `session.last_per_token_scores` overwrite live ones via `_finalize_widget_highlight`.

`Ctrl+A` A/B compare and `Ctrl+S` cycle trait sort — documented in `/help`.

## Trait-panel WHY footer

Bottom-third split below the traits list. `#why-header` is always the literal `LAYERS` (no probe name — the trait panel above already shows what's selected). `#why-scroll` body shows `||baked||` for the trait-panel-selected probe as a horizontal histogram in layer order, bucketed into `HIST_BUCKETS = 16` evenly-sized groups (mean norm per bucket; bar scaled to the largest bucket) so the whole profile fits the 16-row box regardless of layer count. Models with ≤16 layers render one-bar-per-layer. Bucket labels are `LXX` for single-layer buckets, `LXX-YY` otherwise, zero-padded to the model's max-index width. No token list — per-token highlighting in the chat already surfaces which tokens the probe lights up on. Driven by `_refresh_trait_why()` from app.py; fired on trait nav, probe add/remove, finalize, clear, and on_mount — **not** per streamed emit, since layer norms are static for the duration of a gen. No `/why` chat command — selection alone drives the readout. `HIST_BUCKETS` lives in `saklas.core.histogram` alongside `bucketize()` — shared with `cli vector why`.

## utils.py

`build_bar(value, max, width)` renders filled/empty bar pairs. `BAR_WIDTH = 24` used by every bar in the UI (footer token progress, vector alpha, gen-config temp + top-p, trait-panel probes) — one knob controls them all. Vector panel's `RIGHT_W` derives from `BAR_WIDTH` so gen-config right-edge glyph alignment stays correct through any width change.

## Status footer

`chat_panel.update_status`: one-line footer showing dot + `gen_tokens/max_new_tokens` + progress bar (green while generating, dim between runs — persists post-gen so the bar doesn't appear/disappear each turn; collapses to `○ idle` before the first generation or when `max_tokens` is unknown) · tok/s · elapsed · `ppl <mean>`. The count sits left of the bar. VRAM lives on the left panel already; context info isn't rendered here.

Perplexity is the geometric mean of per-step `TokenEvent.perplexity` values — `exp(sum(log(ppl)) / count)` across scored steps in the current gen. `_log_ppl_sum` / `_ppl_count` reset at `_start_generation`; `_last_gen_state` dedupe tuple tracks `_ppl_count` so the footer refreshes when the aggregate moves. Computed in `generate_steered` as `exp` of full-vocab fp32 Shannon entropy on pre-temperature post-steering logits; one extra softmax + one `.item()` sync per step, inside the steered-throughput headroom.

## Panels

**Trait-panel row layout**: two lines per probe — `> <name> [dim]<sparkline>[/]` on line 1 (bold wraps name only when selected), `  <bar> <val:+.2f><arrow>` on line 2 (two-space indent, color from sign of val). Full untruncated names — no fixed column width. Category headers are single lines. `trait_panel._nav_items` is `list[str]`.

**`chat_panel` is Markdown-free**: user + assistant messages are plain `Static` widgets with Rich-escaped text. `_AssistantMessage` holds exactly two content `Static`s (`#thinking-view`, `#response-view`) + a `Collapsible` wrapper for thinking. Tokens escaped once on append into `_escaped_chat_text`/`_escaped_thinking_text` (O(n) total). Per-token probe scores appended live via `append_token_score`; each append clears the markup cache and re-renders if highlighting is on. `set_token_data` runs at finalize to overwrite with canonical projected scores.

Saturation fixed at `_HIGHLIGHT_SAT = 0.5` mapped to full `rgb(0..255, 0, 0)` / `rgb(0, 0..255, 0)`; zero scores get no background span (neutral text stays terminal-default).

**Response-view leading-whitespace strip**: models often emit `\n\n` after `</think>` which would otherwise render as blank rows below the collapsed thinking title; `_render_response` lstrips the plain-text branch and `_build_highlight_markup` skips leading whitespace-only tokens.

Thinking-block CSS overrides Textual's `Collapsible` defaults (`padding-bottom: 1`, `padding-left: 1`, `border-top: hkey`) — set to zero, with `#thinking-block.-collapsed { height: 1 }` so the collapsed block is exactly the title row. Transitions via idempotent `ensure_thinking_collapsed()` (no `_in_thinking`, no `finalize()`, no `end_thinking()`).
