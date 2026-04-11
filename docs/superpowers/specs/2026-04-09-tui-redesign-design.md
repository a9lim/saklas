# TUI Redesign Spec

## Problem

The current TUI is functional but sparse. It wastes screen space, hides useful data behind commands, lacks keyboard navigation within panels, and doesn't show generation stats, live memory, probe statistics, or generation config controls. The "selected vector controls" panel is a separate section that only shows one vector's details at a time.

## Goal

Redesign the TUI to be responsive, information-dense, keyboard-navigable, and show all available data at all times. Every piece of information the backend can provide should be visible without commands.

---

## Layout

Three-column layout, all panels always visible:

```
┌──────────────┬───────────────────────────┬────────────────────┐
│ LEFT (24%)   │ CENTER (50%)              │ RIGHT (26%)        │
│              │                           │                    │
│ Model info   │ Chat log                  │ Trait monitor      │
│ Vectors      │ (scrollable)              │ (scrollable)       │
│  (inline     │                           │                    │
│   controls)  │                           │ Probes with inline │
│ Gen config   │                           │ sparklines + stats │
│ Key ref      │ Status bar                │                    │
│              │ Input field               │                    │
└──────────────┴───────────────────────────┴────────────────────┘
```

Column widths: left `1fr`, center `2fr`, right `1fr`. The center gets twice the space. Borders between panels use the accent color.

---

## Left Panel

Four stacked sections, top to bottom:

### MODEL section
Static info, set on startup:
- Model ID (e.g., `gemma-2-9b-it`)
- Layer count × hidden dim (e.g., `42L × 3584d`)
- dtype, device, VRAM, param count

### VECTORS section
Header line: `VECTORS` + count summary (e.g., `3 total, 2 active`) + ortho state (`ortho: OFF`).

Every vector is displayed with full inline controls — no separate "selected" section. Each vector row shows:
- Selection marker (`>` for focused) and enable dot (`●` green / `○` dim)
- Name, method (`actadd`/`caa`)
- Alpha bar: `α ████████░░░░░░ +0.8`
- Layer indicator: `L 18/42`

The **focused** vector (marked with `>`) additionally shows:
- Layer position visualizer bar (a mini bar with a cursor showing where in the stack the vector sits)
- Keybinding hints for adjustment (`←/→` alpha, `↑/↓` layer)

Disabled vectors render dimmed (name, bar, and value all in gray).

Footer hint line: `Ctrl+N add · Ctrl+D rm · Enter toggle · Ctrl+O ortho`

### GENERATION section
Shows all generation config with visual bars where applicable:
- Temperature: value + visual bar + keybinding (`[` / `]`)
- Top-p: value + visual bar + keybinding (`{` / `}`)
- Max tokens: value + keybinding (`/max` command)
- System prompt: value or `(none)` + keybinding (`/sys` command)

### KEYS section
Compact keybinding quick reference, always visible:
- `Tab focus · j/k nav · Esc stop`
- `Ctrl+R regen · Ctrl+A A/B`
- `Ctrl+Q quit · /help cmds`

---

## Center Panel (Chat)

Three vertical sections:

### Chat log
Scrollable message history. Messages styled by role:
- User: bold cyan label
- Assistant: bold green label
- System: dimmed text

Scrollable via `PageUp`/`PageDown` when the chat panel is focused.

### Status bar
Single line between chat log and input. Shows data not visible elsewhere:
- **Left side**: Generation indicator (green `●` when generating, dim `○` when idle), token progress (`42/512 tok`), speed (`18.3 tok/s`), elapsed time (`2.3s`)
- **Right side**: Prompt token count (`prompt: 128 tok`), live VRAM usage (`VRAM: 18.4 GB`)

Generation stats update in real-time during generation (driven by the existing poll timer). VRAM updates on each poll tick. When not generating, the left side shows the stats from the last generation (or blank if none yet).

### Input field
Text input at bottom. Placeholder text: `▸ Type a message...`

Supports existing slash commands (`/steer`, `/clear`, `/sys`, `/temp`, `/max`, `/help`, `/probes`) plus new ones:
- `/max <n>` — set max tokens
- `/top-p <n>` — set top-p (alternative to `{`/`}` keys)
- `/help` — show all commands and keybindings

---

## Right Panel (Trait Monitor)

Header: `TRAIT MONITOR` + sort mode indicator + `Ctrl+S` hint.

### Probe list
Scrollable list of probes grouped by category. Each category has a collapsible header:
- `▾ Emotion (8)` when expanded, `▸ Emotion (8)` when collapsed
- Default: Emotion and Personality expanded, Safety/Cultural/Gender collapsed

Each probe row (one line) shows:
- Selection marker (`>` for focused probe)
- Name (9 chars, left-aligned)
- Magnitude bar (10 chars, `████░░░░░░`)
  - Green for positive values, red for negative, gray for near-zero
- Value (`+.72` format, 4 chars)
- Direction arrow (`↑`/`↓`/space)
- Mini sparkline (8 chars, last 8 tokens of history)

The **focused** probe expands a detail stats row below it:
- `μ=+.58 σ=.12 lo=+.31 hi=+.79 Δ=+.04/tok`
- These are computed from `monitor.history[name]`: mean, standard deviation, min, max, and average per-token change

Footer hint line: `j/k nav · Enter select/collapse · Ctrl+S sort`

### Sort modes
Cycle with `Ctrl+S`: name → magnitude → change (same as current, unchanged).

---

## Keyboard Navigation

### Panel focus
- `Tab` / `Shift+Tab` cycle focus: vectors → chat → traits → vectors
- Focused panel gets a highlighted border (accent color instead of secondary)
- On startup, chat panel is focused (input field active)

### Vector panel (when focused)
| Key | Action |
|-----|--------|
| `j` / `Down` | Move selection down |
| `k` / `Up` | Move selection up |
| `Left` | Decrease alpha by 0.1 (min -3.0) |
| `Right` | Increase alpha by 0.1 (max +3.0) |
| `Shift+Up` | Increase layer by 1 |
| `Shift+Down` | Decrease layer by 1 |
| `Enter` | Toggle enabled/disabled |

### Trait panel (when focused)
| Key | Action |
|-----|--------|
| `j` / `Down` | Move selection down (through probes and category headers) |
| `k` / `Up` | Move selection up |
| `Enter` on category | Toggle collapse/expand |
| `Enter` on probe | Select for detail stats |

### Chat panel (when focused)
| Key | Action |
|-----|--------|
| Normal typing | Goes to input field |
| `PageUp` / `PageDown` | Scroll chat history |

### Global keybindings (always work regardless of focus)
| Key | Action |
|-----|--------|
| `Ctrl+Q` | Quit |
| `Escape` | Stop generation |
| `Ctrl+N` | Add vector (shows `/steer` usage) |
| `Ctrl+D` | Remove focused vector |
| `Ctrl+T` | Toggle focused vector |
| `Ctrl+O` | Toggle orthogonalization |
| `Ctrl+R` | Regenerate last prompt |
| `Ctrl+A` | A/B compare |
| `Ctrl+S` | Cycle sort mode |
| `[` / `]` | Decrease/increase temperature by 0.05 |
| `{` / `}` | Decrease/increase top-p by 0.05 |

---

## Data Flow Changes

### Generation stats
`generate_steered` already returns token IDs and the TUI already has a poll timer. Track:
- `_gen_start_time`: set in `_start_generation`, cleared on completion
- `_gen_token_count`: incremented in the poll loop as tokens are consumed
- `_prompt_token_count`: set from `len(input_ids[0])` before generation starts

Speed = `_gen_token_count / (now - _gen_start_time)`. Updated each poll tick.

### Live VRAM
Call `model.py:_get_memory_gb(self._device)` in the poll loop. This is a cheap call (no GPU sync, just reads allocator state).

### Probe statistics
Computed in `_poll_generation` from `monitor.history[name]`:
- `mean`, `std`, `min`, `max`: standard Python/list operations on the float list
- `Δ/tok`: `(history[-1] - history[0]) / len(history)` if len > 1, else 0

Only computed for the selected probe (not all probes every tick) to keep it cheap.

### Top-p control
Add `top_p` adjustment to `GenerationConfig`. Wire `{`/`}` keybindings to `_adjust_top_p(delta)` which modifies `self._gen_config.top_p` and updates the left panel display. Clamp to [0.0, 1.0].

### Max tokens command
Add `/max <n>` command handler that sets `self._gen_config.max_new_tokens`.

---

## File Changes

### Files to rewrite
- `steer/tui/app.py` — New layout composition, panel focus system, new keybindings, generation stats tracking, VRAM polling, top-p/temp key handlers
- `steer/tui/vector_panel.py` — Merged vector+controls widget with inline alpha bars, layer visualizer, per-vector detail rendering
- `steer/tui/trait_panel.py` — Inline sparklines per probe, expandable stats row, keyboard navigation within probe list, category collapse via Enter
- `steer/tui/chat_panel.py` — Status bar widget with generation stats, prompt token count, VRAM display
- `steer/tui/styles.tcss` — New three-column layout, panel focus highlighting, section styling

### Files with minor changes
- `steer/generation.py` — No changes needed (stats tracked in app.py from existing data)
- `steer/model.py` — `_get_memory_gb` already exists, no changes needed

### New data needed from backend
- Param count in model info: add to `get_model_info()` in `model.py` via `sum(p.numel() for p in model.parameters())`
- Everything else (generation stats, probe stats, VRAM) is computed from existing data in `app.py`

---

## Verification

1. Launch with `steer google/gemma-2-2b-it` (or any small model) and verify three-column layout renders correctly
2. Verify all four left panel sections display correctly with model info, vectors, gen config, and key reference
3. Add vectors via `/steer happy`, `/steer creative` and verify they all show inline controls
4. Press `Tab` to cycle focus — verify border highlight changes
5. With vector panel focused, use `j/k` to navigate, `←/→` to adjust alpha, verify visual updates
6. With trait panel focused, use `j/k` to navigate probes, `Enter` to collapse/expand categories, verify stats row appears for selected probe
7. Send a message and verify status bar shows token count, speed, and elapsed time updating in real-time
8. Verify `[`/`]` and `{`/`}` adjust temperature and top-p with visual feedback
9. Verify VRAM display updates during generation
10. Resize terminal and verify layout scales (panels should flex proportionally)
