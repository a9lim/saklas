# TUI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the TUI to a three-column, information-dense, fully keyboard-navigable interface showing all available data at all times.

**Architecture:** Replace the existing two-column layout (chat + right sidebar) with a three-column layout (vectors/controls | chat | trait monitor). Each panel is a self-contained widget. A panel focus system routes arrow/j/k keys to the focused panel. Generation stats, live VRAM, and probe statistics are computed in the poll loop from existing backend data.

**Tech Stack:** Python 3.11+, Textual >=0.50, torch (for device memory queries)

---

### Task 1: Add param count to model info

**Files:**
- Modify: `steer/model.py:243-257`

- [ ] **Step 1: Add param_count to get_model_info**

In `steer/model.py`, add a `param_count` field to the dict returned by `get_model_info`:

```python
def get_model_info(model, tokenizer) -> dict:
    """Summary dict: model_type, num_layers, hidden_dim, device, dtype, vram_used_gb."""
    layers = get_layers(model)
    first_param = next(model.parameters())
    model_id = getattr(model.config, "_name_or_path", "unknown")
    param_count = sum(p.numel() for p in model.parameters())
    return {
        "model_id": model_id,
        "model_type": model.config.model_type,
        "num_layers": len(layers),
        "hidden_dim": _text_config(model).hidden_size,
        "device": str(first_param.device),
        "dtype": str(first_param.dtype),
        "vram_used_gb": _get_memory_gb(str(first_param.device)),
        "param_count": param_count,
    }
```

- [ ] **Step 2: Verify it works**

Run: `python3 -c "from steer.model import load_model, get_model_info; m,t = load_model('gpt2', device='cpu', no_compile=True); print(get_model_info(m,t))"`

Expected: Dict with `param_count` key showing ~124M for GPT-2.

- [ ] **Step 3: Commit**

```bash
git add steer/model.py
git commit -m "feat: add param_count to get_model_info"
```

---

### Task 2: Rewrite styles.tcss for three-column layout

**Files:**
- Rewrite: `steer/tui/styles.tcss`

- [ ] **Step 1: Write the new stylesheet**

Replace `steer/tui/styles.tcss` entirely:

```css
Screen {
    layout: vertical;
}

#main-area {
    height: 1fr;
    layout: horizontal;
}

/* Three-column layout: left (1fr) | center (2fr) | right (1fr) */
#left-panel {
    width: 1fr;
    height: 100%;
    border-right: solid $accent;
    padding: 0 1;
    layout: vertical;
}

#chat-panel {
    width: 2fr;
    height: 100%;
    layout: vertical;
}

#trait-panel {
    width: 1fr;
    height: 100%;
    border-left: solid $accent;
    padding: 0 1;
    layout: vertical;
}

/* Left panel focused */
#left-panel.focused {
    border-right: solid $accent;
}

/* Chat panel focused */
#chat-panel.focused {
    border: solid $accent;
}

/* Trait panel focused */
#trait-panel.focused {
    border-left: solid $accent;
}

/* Left panel sections */
.section-header {
    color: $accent;
    text-style: bold;
    height: 1;
}

.section-body {
    height: auto;
    padding: 0;
}

#model-section {
    height: auto;
}

#vectors-section {
    height: 1fr;
    min-height: 5;
}

#vector-scroll {
    height: 1fr;
    overflow-y: auto;
}

#gen-section {
    height: auto;
}

#keys-section {
    height: auto;
}

.vector-row {
    height: auto;
}

.vector-row-selected {
    height: auto;
    background: $surface;
}

/* Chat panel internals */
#chat-log {
    height: 1fr;
    overflow-y: auto;
    padding: 0 1;
}

#status-bar {
    height: 1;
    background: $surface;
    color: $text-muted;
    padding: 0 1;
}

#chat-input {
    height: 3;
    dock: bottom;
    border-top: solid $accent;
}

.user-message {
    color: $text;
    margin: 1 0 0 0;
}

.assistant-message {
    color: $text;
    margin: 0 0 1 0;
}

/* Trait panel internals */
#trait-header {
    height: 1;
}

#trait-scroll {
    height: 1fr;
    overflow-y: auto;
}

.trait-category {
    text-style: bold;
    color: $accent;
    height: 1;
}

.trait-row {
    height: 1;
}

.trait-row-selected {
    height: auto;
    background: $surface;
}

#trait-hints {
    height: 1;
    color: $text-muted;
}

/* Footer */
Footer {
    height: 1;
}
```

- [ ] **Step 2: Commit**

```bash
git add steer/tui/styles.tcss
git commit -m "feat: rewrite stylesheet for three-column layout"
```

---

### Task 3: Rewrite vector_panel.py — merged inline controls

**Files:**
- Rewrite: `steer/tui/vector_panel.py`

- [ ] **Step 1: Write the new VectorPanel**

Replace `steer/tui/vector_panel.py` entirely. This removes `ControlsPanel` (merged into VectorPanel) and renders every vector with its alpha bar and layer inline. The selected vector gets expanded detail.

```python
"""Left panel: model info, steering vectors with inline controls, generation config, key reference."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static
from textual.widget import Widget
from textual.message import Message


class LeftPanel(Widget):
    """Entire left column: model, vectors, gen config, keys."""

    class VectorSelected(Message):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

    def __init__(self, model_info: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model_info = model_info
        self._vectors: list[dict] = []
        self._selected_idx: int = 0
        self._orthogonalize: bool = False
        self._temperature: float = 0.7
        self._top_p: float = 0.9
        self._max_tokens: int = 512
        self._system_prompt: str | None = None

    def compose(self) -> ComposeResult:
        info = self._model_info
        # Model section
        yield Static("[bold]MODEL[/]", classes="section-header")
        model_id = info.get("model_id", "unknown")
        # Truncate long model IDs
        if len(model_id) > 28:
            model_id = "..." + model_id[-25:]
        params = info.get("param_count", 0)
        param_str = f"{params / 1e9:.1f}B" if params >= 1e9 else f"{params / 1e6:.0f}M"
        yield Static(
            f"{model_id}\n"
            f"{info['num_layers']}L × {info['hidden_dim']}d · {info.get('dtype', '?')}\n"
            f"{info.get('device', '?')} · {info.get('vram_used_gb', 0):.1f} GB · {param_str}",
            id="model-info",
        )
        # Vectors section
        yield Static("[bold]VECTORS[/] [dim]0 total, 0 active · ortho: OFF[/]",
                      id="vectors-header", classes="section-header")
        yield VerticalScroll(id="vector-scroll")
        yield Static("[dim]Ctrl+N add · Ctrl+D rm · Enter toggle · Ctrl+O ortho[/]",
                      id="vector-hints")
        # Generation section
        yield Static("[bold]GENERATION[/]", classes="section-header")
        yield Static("", id="gen-config")
        # Keys section
        yield Static("[bold]KEYS[/]", classes="section-header")
        yield Static(
            "[dim]Tab focus · j/k nav · Esc stop\n"
            "Ctrl+R regen · Ctrl+A A/B\n"
            "Ctrl+Q quit · /help cmds[/]",
            id="key-ref",
        )

    def update_vectors(self, vectors: list[dict], orthogonalize: bool = False) -> None:
        self._vectors = vectors
        self._orthogonalize = orthogonalize
        if self._vectors:
            self._selected_idx = min(self._selected_idx, len(self._vectors) - 1)
        else:
            self._selected_idx = 0
        self._render_vectors()

    def update_gen_config(self, temperature: float, top_p: float,
                          max_tokens: int, system_prompt: str | None) -> None:
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._render_gen_config()

    def select_next(self) -> None:
        if self._vectors:
            self._selected_idx = (self._selected_idx + 1) % len(self._vectors)
            self._render_vectors()
            self._post_selection()

    def select_prev(self) -> None:
        if self._vectors:
            self._selected_idx = (self._selected_idx - 1) % len(self._vectors)
            self._render_vectors()
            self._post_selection()

    def get_selected(self) -> dict | None:
        if self._vectors and 0 <= self._selected_idx < len(self._vectors):
            return self._vectors[self._selected_idx]
        return None

    def _post_selection(self) -> None:
        sel = self.get_selected()
        if sel:
            self.post_message(self.VectorSelected(sel["name"]))

    def _render_vectors(self) -> None:
        # Update header
        active = sum(1 for v in self._vectors if v.get("enabled", True))
        total = len(self._vectors)
        ortho_str = "ON" if self._orthogonalize else "OFF"
        header = self.query_one("#vectors-header", Static)
        header.update(
            f"[bold]VECTORS[/] [dim]{total} total, {active} active · ortho: {ortho_str}[/]"
        )

        vscroll = self.query_one("#vector-scroll", VerticalScroll)
        vscroll.remove_children()

        num_layers = self._model_info["num_layers"]
        for i, v in enumerate(self._vectors):
            is_selected = i == self._selected_idx
            enabled = v.get("enabled", True)
            name = v["name"]
            alpha = v["alpha"]
            layer = v["layer_idx"]
            method = v.get("method", "actadd")

            # Alpha bar
            bar_width = 14
            filled = int(abs(alpha) / 3.0 * bar_width)
            filled = min(filled, bar_width)
            bar_full = "█" * filled
            bar_empty = "░" * (bar_width - filled)
            color = "green" if alpha >= 0 else "red"
            dim = "" if enabled else "dim "

            if is_selected:
                # Expanded view for selected vector
                marker = ">"
                dot = "[green]●[/]" if enabled else "[dim]○[/]"

                # Layer position visualizer
                if num_layers > 0:
                    lbar_width = min(20, num_layers)
                    lpos = int(layer / max(num_layers - 1, 1) * (lbar_width - 1))
                    lbar = "▁" * lpos + "█" + "▁" * (lbar_width - lpos - 1)
                else:
                    lbar = "█"

                text = (
                    f"{marker} {dot} [{dim}bold]{name}[/] [{dim}]{method}[/]\n"
                    f"  α [{dim}{color}]{bar_full}[/][dim]{bar_empty}[/] "
                    f"[{dim}{color}]{alpha:+.1f}[/] [dim]←/→[/]\n"
                    f"  L [dim]{lbar}[/] {layer}/{num_layers} [dim]S-↑/↓[/]"
                )
                vscroll.mount(Static(text, classes="vector-row-selected"))
            else:
                # Compact view for non-selected vectors
                marker = " "
                dot = "[green]●[/]" if enabled else "[dim]○[/]"
                text = (
                    f"{marker} {dot} [{dim}]{name}[/] [{dim}]{method}[/]\n"
                    f"  α [{dim}{color}]{bar_full}[/][dim]{bar_empty}[/] "
                    f"[{dim}]{alpha:+.1f}[/]  L{layer}"
                )
                vscroll.mount(Static(text, classes="vector-row"))

    def _render_gen_config(self) -> None:
        gen = self.query_one("#gen-config", Static)
        # Temperature bar
        t_bar_w = 20
        t_filled = int(self._temperature / 2.0 * t_bar_w)
        t_filled = min(t_filled, t_bar_w)
        t_bar = "█" * t_filled + "░" * (t_bar_w - t_filled)
        # Top-p bar
        p_bar_w = 20
        p_filled = int(self._top_p * p_bar_w)
        p_filled = min(p_filled, p_bar_w)
        p_bar = "█" * p_filled + "░" * (p_bar_w - p_filled)

        sys_str = self._system_prompt[:15] + "..." if self._system_prompt and len(self._system_prompt) > 15 else (self._system_prompt or "(none)")

        gen.update(
            f"Temp  {self._temperature:.2f} [dim]{t_bar}[/] [dim]\\[/][/]\n"
            f"Top-p {self._top_p:.2f} [dim]{p_bar}[/] [dim]{{/}}[/]\n"
            f"Max   {self._max_tokens} tok       [dim]/max[/]\n"
            f"Sys   [dim]{sys_str}[/]    [dim]/sys[/]"
        )
```

- [ ] **Step 2: Commit**

```bash
git add steer/tui/vector_panel.py
git commit -m "feat: rewrite vector panel with inline controls, remove ControlsPanel"
```

---

### Task 4: Rewrite chat_panel.py — add status bar

**Files:**
- Rewrite: `steer/tui/chat_panel.py`

- [ ] **Step 1: Write the new ChatPanel with status bar**

Replace `steer/tui/chat_panel.py`:

```python
"""Chat panel: message display, status bar, and input."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Input
from textual.widget import Widget
from textual.message import Message


class ChatPanel(Widget):

    class UserSubmitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-log")
        yield Static("", id="status-bar")
        yield Input(placeholder="Type a message...", id="chat-input")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        self.add_user_message(text)
        self.post_message(self.UserSubmitted(text))

    def add_user_message(self, text: str) -> None:
        log = self.query_one("#chat-log", VerticalScroll)
        log.mount(Static(f"[bold cyan]User:[/] {text}", classes="user-message"))
        log.scroll_end(animate=False)

    def start_assistant_message(self) -> Static:
        log = self.query_one("#chat-log", VerticalScroll)
        widget = Static("[bold green]Assistant:[/] ", classes="assistant-message")
        widget._chat_text = "[bold green]Assistant:[/] "
        log.mount(widget)
        return widget

    def append_to_assistant(self, widget: Static, token: str) -> None:
        widget._chat_text += token
        widget.update(widget._chat_text)
        log = self.query_one("#chat-log", VerticalScroll)
        log.scroll_end(animate=False)

    def add_system_message(self, text: str) -> None:
        log = self.query_one("#chat-log", VerticalScroll)
        log.mount(Static(f"[dim]{text}[/]"))
        log.scroll_end(animate=False)

    def update_status(
        self,
        generating: bool = False,
        gen_tokens: int = 0,
        max_tokens: int = 0,
        tok_per_sec: float = 0.0,
        elapsed: float = 0.0,
        prompt_tokens: int = 0,
        vram_gb: float = 0.0,
    ) -> None:
        """Update the status bar with generation stats."""
        bar = self.query_one("#status-bar", Static)
        dot = "[green]●[/]" if generating else "[dim]○[/]"
        if generating:
            left = f"{dot} {gen_tokens}/{max_tokens} tok · {tok_per_sec:.1f} tok/s · {elapsed:.1f}s"
        elif gen_tokens > 0:
            left = f"{dot} {gen_tokens} tok · {tok_per_sec:.1f} tok/s · {elapsed:.1f}s"
        else:
            left = f"{dot} idle"
        right = ""
        if prompt_tokens > 0:
            right += f"prompt: {prompt_tokens} tok"
        if vram_gb > 0:
            if right:
                right += " · "
            right += f"VRAM: {vram_gb:.1f} GB"
        bar.update(f"{left}{'':>4}{right}")
```

- [ ] **Step 2: Commit**

```bash
git add steer/tui/chat_panel.py
git commit -m "feat: rewrite chat panel with status bar for gen stats and VRAM"
```

---

### Task 5: Rewrite trait_panel.py — inline sparklines and probe stats

**Files:**
- Rewrite: `steer/tui/trait_panel.py`

- [ ] **Step 1: Write the new TraitPanel with inline sparklines and expandable stats**

Replace `steer/tui/trait_panel.py`:

```python
"""Live trait monitor panel with inline sparklines, expandable stats, category collapsing."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static
from textual.widget import Widget

# Probe categories and their default members
PROBE_CATEGORIES: dict[str, list[str]] = {
    "Emotion": [
        "happy", "sad", "angry", "fearful", "surprised",
        "disgusted", "calm", "excited",
    ],
    "Personality": [
        "sycophantic", "honest", "creative", "formal", "casual",
        "verbose", "concise", "authoritative", "uncertain", "confident",
    ],
    "Safety": [
        "refusal", "compliance", "deceptive", "hallucinating",
    ],
    "Cultural": [
        "western-individualist", "eastern-collectivist",
        "formal-hierarchical", "casual-egalitarian",
        "direct-communication", "indirect-communication",
        "high-context", "low-context",
        "religious", "secular",
        "traditional", "progressive",
    ],
    "Gender": [
        "masculine-coded", "feminine-coded",
        "agentic", "communal",
        "paternal", "maternal",
    ],
}

DEFAULT_COLLAPSED = {"Cultural", "Gender", "Safety"}


class TraitPanel(Widget):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._collapsed: set[str] = set(DEFAULT_COLLAPSED)
        self._current_values: dict[str, float] = {}
        self._previous_values: dict[str, float] = {}
        self._sparklines: dict[str, str] = {}
        self._histories: dict[str, list[float]] = {}
        self._selected_probe: str | None = None
        self._active_probes: set[str] = set()
        self._sort_mode: str = "name"
        # Navigation: flat list of (type, name) for j/k movement
        self._nav_items: list[tuple[str, str]] = []  # ("category", name) or ("probe", name)
        self._nav_idx: int = 0

    def compose(self) -> ComposeResult:
        yield Static(
            "[bold]TRAIT MONITOR[/] [dim]sort: name · Ctrl+S[/]",
            id="trait-header",
        )
        yield VerticalScroll(id="trait-scroll")
        yield Static("[dim]j/k nav · Enter select/collapse · Ctrl+S sort[/]",
                      id="trait-hints")

    def set_active_probes(self, probe_names: set[str]) -> None:
        self._active_probes = probe_names

    def update_values(
        self,
        current: dict[str, float],
        previous: dict[str, float],
        sparklines: dict[str, str],
        histories: dict[str, list[float]] | None = None,
    ) -> None:
        self._current_values = current
        self._previous_values = previous
        self._sparklines = sparklines
        if histories is not None:
            self._histories = histories
        self._render_probes()

    def toggle_category(self, category: str) -> None:
        if category in self._collapsed:
            self._collapsed.discard(category)
        else:
            self._collapsed.add(category)
        self._render_probes()

    def cycle_sort(self) -> None:
        modes = ["name", "magnitude", "change"]
        idx = modes.index(self._sort_mode)
        self._sort_mode = modes[(idx + 1) % len(modes)]
        # Update header
        header = self.query_one("#trait-header", Static)
        header.update(
            f"[bold]TRAIT MONITOR[/] [dim]sort: {self._sort_mode[:3]} · Ctrl+S[/]"
        )
        self._render_probes()

    def select_probe(self, name: str) -> None:
        self._selected_probe = name
        self._render_probes()

    def nav_down(self) -> None:
        if self._nav_items and self._nav_idx < len(self._nav_items) - 1:
            self._nav_idx += 1
            self._apply_nav_selection()

    def nav_up(self) -> None:
        if self._nav_items and self._nav_idx > 0:
            self._nav_idx -= 1
            self._apply_nav_selection()

    def nav_enter(self) -> None:
        """Enter on category toggles collapse, on probe selects it for stats."""
        if not self._nav_items:
            return
        item_type, name = self._nav_items[self._nav_idx]
        if item_type == "category":
            self.toggle_category(name)
        else:
            self._selected_probe = name
            self._render_probes()

    def _apply_nav_selection(self) -> None:
        if not self._nav_items:
            return
        item_type, name = self._nav_items[self._nav_idx]
        if item_type == "probe":
            self._selected_probe = name
        self._render_probes()

    def _render_probes(self) -> None:
        tscroll = self.query_one("#trait-scroll", VerticalScroll)
        tscroll.remove_children()
        self._nav_items = []

        nav_idx_counter = 0
        for category, members in PROBE_CATEGORIES.items():
            active_members = [m for m in members if m in self._active_probes]
            if not active_members:
                continue

            collapsed = category in self._collapsed
            arrow = "▸" if collapsed else "▾"
            count = len(active_members)

            is_nav_selected = nav_idx_counter == self._nav_idx
            cat_marker = ">" if is_nav_selected else " "
            self._nav_items.append(("category", category))
            nav_idx_counter += 1

            tscroll.mount(Static(
                f"{cat_marker}[bold]{arrow} {category}[/] [dim]({count})[/]",
                classes="trait-category",
            ))

            if collapsed:
                continue

            sorted_members = self._sort_probes(active_members)
            for name in sorted_members:
                is_nav_selected = nav_idx_counter == self._nav_idx
                self._nav_items.append(("probe", name))
                nav_idx_counter += 1

                val = self._current_values.get(name, 0.0)
                prev = self._previous_values.get(name, 0.0)
                if val != val:
                    val = 0.0
                if prev != prev:
                    prev = 0.0
                delta = val - prev

                # Direction arrow
                if abs(delta) < 0.01:
                    arrow_ch = " "
                elif delta > 0:
                    arrow_ch = "↑"
                else:
                    arrow_ch = "↓"

                # Bar (10 chars)
                bar_width = 10
                filled = int(abs(val) * bar_width)
                filled = min(filled, bar_width)
                bar_full = "█" * filled
                bar_empty = "░" * (bar_width - filled)
                color = "green" if val >= 0 else "red"

                # Mini sparkline (8 chars)
                spark = self._sparklines.get(name, "")
                mini_spark = spark[-8:] if spark else ""

                sel = ">" if is_nav_selected else " "
                display_name = name[:9].ljust(9)

                line = (
                    f"{sel} {display_name}[{color}]{bar_full}[/][dim]{bar_empty}[/] "
                    f"{val:+.2f}{arrow_ch} [dim]{mini_spark}[/]"
                )

                is_detail_probe = name == self._selected_probe
                if is_detail_probe:
                    # Compute stats from history
                    hist = self._histories.get(name, [])
                    stats_line = self._compute_stats_line(hist)
                    tscroll.mount(Static(
                        f"{line}\n  [dim]{stats_line}[/]",
                        classes="trait-row-selected",
                    ))
                else:
                    tscroll.mount(Static(line, classes="trait-row"))

    def _compute_stats_line(self, hist: list[float]) -> str:
        if not hist:
            return "no data"
        n = len(hist)
        mean = sum(hist) / n
        lo = min(hist)
        hi = max(hist)
        if n > 1:
            variance = sum((x - mean) ** 2 for x in hist) / n
            std = variance ** 0.5
            delta_per_tok = (hist[-1] - hist[0]) / (n - 1)
        else:
            std = 0.0
            delta_per_tok = 0.0
        return (
            f"μ={mean:+.2f} σ={std:.2f} "
            f"lo={lo:+.2f} hi={hi:+.2f} "
            f"Δ={delta_per_tok:+.2f}/tok"
        )

    def _sort_probes(self, names: list[str]) -> list[str]:
        if self._sort_mode == "magnitude":
            return sorted(names, key=lambda n: abs(self._current_values.get(n, 0.0)), reverse=True)
        elif self._sort_mode == "change":
            return sorted(names, key=lambda n: abs(
                self._current_values.get(n, 0.0) - self._previous_values.get(n, 0.0)
            ), reverse=True)
        return sorted(names)
```

- [ ] **Step 2: Commit**

```bash
git add steer/tui/trait_panel.py
git commit -m "feat: rewrite trait panel with inline sparklines, nav, expandable stats"
```

---

### Task 6: Rewrite app.py — three-column layout, panel focus, generation stats

**Files:**
- Rewrite: `steer/tui/app.py`

This is the largest change. The new app.py must:
1. Use three-column layout with LeftPanel, ChatPanel, TraitPanel
2. Implement panel focus system (Tab/Shift+Tab)
3. Route j/k/arrow/Enter to focused panel
4. Track generation stats (token count, speed, elapsed, prompt tokens)
5. Poll live VRAM
6. Add temperature/top-p key adjustments
7. Add /max, /sys, /top-p, /help commands
8. Pass probe histories to trait panel for stats computation

- [ ] **Step 1: Write the new app.py**

Replace `steer/tui/app.py`:

```python
"""Main Textual application for steer."""

from __future__ import annotations

import queue
import time

import torch
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer
from textual.timer import Timer

from steer.generation import GenerationConfig, GenerationState, build_chat_input, generate_steered
from steer.hooks import SteeringManager
from steer.model import _get_memory_gb
from steer.monitor import TraitMonitor
from steer.tui.chat_panel import ChatPanel
from steer.tui.vector_panel import LeftPanel
from steer.tui.trait_panel import TraitPanel

PANELS = ["left-panel", "chat-panel", "trait-panel"]


class SteerApp(App):
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+n", "new_vector", "New Vector"),
        Binding("ctrl+d", "remove_vector", "Remove"),
        Binding("ctrl+a", "ab_compare", "A/B"),
        Binding("escape", "stop_generation", "Stop", show=False),
        Binding("ctrl+r", "regenerate", "Regen"),
        Binding("ctrl+t", "toggle_vector", "Toggle", show=False),
        Binding("ctrl+o", "toggle_ortho", "Ortho", show=False),
        Binding("ctrl+s", "cycle_sort", "Sort", show=False),
        Binding("tab", "focus_next_panel", "Focus→", show=False),
        Binding("shift+tab", "focus_prev_panel", "←Focus", show=False),
        # j/k navigation (routed to focused panel)
        Binding("j", "nav_down", show=False),
        Binding("k", "nav_up", show=False),
        Binding("down", "nav_down", show=False),
        Binding("up", "nav_up", show=False),
        Binding("left", "nav_left", show=False),
        Binding("right", "nav_right", show=False),
        Binding("shift+up", "layer_up", show=False),
        Binding("shift+down", "layer_down", show=False),
        Binding("enter", "nav_enter", show=False),
        # Temperature / top-p
        Binding("[", "temp_down", show=False),
        Binding("]", "temp_up", show=False),
        Binding("{", "top_p_down", show=False),  # Shift+[
        Binding("}", "top_p_up", show=False),  # Shift+]
    ]

    def __init__(
        self,
        model,
        tokenizer,
        layers,
        model_info: dict,
        probes: dict[str, torch.Tensor],
        system_prompt: str | None = None,
        max_tokens: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model = model
        self._tokenizer = tokenizer
        self._layers = layers
        self._model_info = model_info
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens

        # Chat history
        self._messages: list[dict[str, str]] = []

        # Steering
        self._steering = SteeringManager()
        self._orthogonalize = False

        # Generation state
        self._gen_state = GenerationState()
        self._gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        # Cache device/dtype
        first_param = next(self._model.parameters())
        self._device = first_param.device
        self._dtype = first_param.dtype
        self._device_str = str(self._device)

        # Monitor
        monitor_layer = self._model_info["num_layers"] - 2
        self._monitor = TraitMonitor(probes, monitor_layer) if probes else None
        if self._monitor:
            self._monitor.attach(self._layers, self._device, self._dtype)

        # TUI state
        self._current_assistant_widget = None
        self._poll_timer: Timer | None = None
        self._last_prompt: str | None = None

        # Panel focus
        self._focused_panel_idx: int = 1  # Start with chat focused

        # Generation stats
        self._gen_start_time: float = 0.0
        self._gen_token_count: int = 0
        self._prompt_token_count: int = 0
        self._last_tok_per_sec: float = 0.0
        self._last_elapsed: float = 0.0

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-area"):
            yield LeftPanel(self._model_info, id="left-panel")
            yield ChatPanel(id="chat-panel")
            yield TraitPanel(id="trait-panel")
        yield Footer()

    def on_mount(self) -> None:
        # Initialize left panel gen config
        lp = self.query_one("#left-panel", LeftPanel)
        lp.update_gen_config(
            self._gen_config.temperature,
            self._gen_config.top_p,
            self._gen_config.max_new_tokens,
            self._gen_config.system_prompt,
        )

        # Set up trait panel
        if self._monitor:
            trait_panel = self.query_one("#trait-panel", TraitPanel)
            trait_panel.set_active_probes(set(self._monitor.probe_names))
            if self._monitor.probe_names:
                trait_panel.select_probe(self._monitor.probe_names[0])

        # Start poll timer (~15 FPS)
        self._poll_timer = self.set_interval(1 / 15, self._poll_generation)

        # Set initial panel focus
        self._update_panel_focus()

        # Welcome
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.add_system_message(
            f"Model loaded: {self._model_info.get('model_id', 'unknown')}. "
            f"Type a message to chat. Ctrl+N to add steering vectors. Tab to switch panels."
        )

    # -- Panel Focus --

    def _update_panel_focus(self) -> None:
        for i, panel_id in enumerate(PANELS):
            panel = self.query_one(f"#{panel_id}")
            if i == self._focused_panel_idx:
                panel.add_class("focused")
            else:
                panel.remove_class("focused")
        # If chat is focused, focus the input
        if PANELS[self._focused_panel_idx] == "chat-panel":
            self.query_one("#chat-input").focus()

    def action_focus_next_panel(self) -> None:
        self._focused_panel_idx = (self._focused_panel_idx + 1) % len(PANELS)
        self._update_panel_focus()

    def action_focus_prev_panel(self) -> None:
        self._focused_panel_idx = (self._focused_panel_idx - 1) % len(PANELS)
        self._update_panel_focus()

    # -- Navigation (routed to focused panel) --

    def action_nav_down(self) -> None:
        panel = PANELS[self._focused_panel_idx]
        if panel == "left-panel":
            lp = self.query_one("#left-panel", LeftPanel)
            lp.select_next()
            self._refresh_left_panel()
        elif panel == "trait-panel":
            tp = self.query_one("#trait-panel", TraitPanel)
            tp.nav_down()

    def action_nav_up(self) -> None:
        panel = PANELS[self._focused_panel_idx]
        if panel == "left-panel":
            lp = self.query_one("#left-panel", LeftPanel)
            lp.select_prev()
            self._refresh_left_panel()
        elif panel == "trait-panel":
            tp = self.query_one("#trait-panel", TraitPanel)
            tp.nav_up()

    def action_nav_left(self) -> None:
        if PANELS[self._focused_panel_idx] == "left-panel":
            self._adjust_alpha(-0.1)

    def action_nav_right(self) -> None:
        if PANELS[self._focused_panel_idx] == "left-panel":
            self._adjust_alpha(0.1)

    def action_nav_enter(self) -> None:
        panel = PANELS[self._focused_panel_idx]
        if panel == "left-panel":
            self.action_toggle_vector()
        elif panel == "trait-panel":
            tp = self.query_one("#trait-panel", TraitPanel)
            tp.nav_enter()

    def action_layer_up(self) -> None:
        self._adjust_layer(1)

    def action_layer_down(self) -> None:
        self._adjust_layer(-1)

    # -- Chat --

    def on_chat_panel_user_submitted(self, event: ChatPanel.UserSubmitted) -> None:
        text = event.text
        if text.startswith("/"):
            self._handle_command(text)
            return
        self._last_prompt = text
        self._messages.append({"role": "user", "content": text})
        self._start_generation()

    def _handle_command(self, text: str) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == "/steer":
            if len(parts) < 2:
                chat.add_system_message("Usage: /steer <concept> [alpha] [layer]")
                return
            self._add_vector_from_text(parts[1])
        elif cmd == "/probes":
            self._show_probe_info()
        elif cmd == "/clear":
            self._messages.clear()
            chat.add_system_message("Chat history cleared.")
        elif cmd in ("/system", "/sys"):
            if len(parts) < 2:
                chat.add_system_message(f"System prompt: {self._system_prompt or '(none)'}")
            else:
                self._system_prompt = parts[1]
                self._gen_config.system_prompt = parts[1]
                chat.add_system_message("System prompt set.")
                self._refresh_gen_config()
        elif cmd == "/temp":
            if len(parts) < 2:
                chat.add_system_message(f"Temperature: {self._gen_config.temperature}")
            else:
                try:
                    self._gen_config.temperature = max(0.0, float(parts[1]))
                    chat.add_system_message(f"Temperature set to {self._gen_config.temperature}")
                    self._refresh_gen_config()
                except ValueError:
                    chat.add_system_message("Invalid temperature value")
        elif cmd == "/top-p":
            if len(parts) < 2:
                chat.add_system_message(f"Top-p: {self._gen_config.top_p}")
            else:
                try:
                    self._gen_config.top_p = max(0.0, min(1.0, float(parts[1])))
                    chat.add_system_message(f"Top-p set to {self._gen_config.top_p}")
                    self._refresh_gen_config()
                except ValueError:
                    chat.add_system_message("Invalid top-p value")
        elif cmd == "/max":
            if len(parts) < 2:
                chat.add_system_message(f"Max tokens: {self._gen_config.max_new_tokens}")
            else:
                try:
                    self._gen_config.max_new_tokens = max(1, int(parts[1]))
                    chat.add_system_message(f"Max tokens set to {self._gen_config.max_new_tokens}")
                    self._refresh_gen_config()
                except ValueError:
                    chat.add_system_message("Invalid max tokens value")
        elif cmd == "/help":
            chat.add_system_message(
                "Commands: /steer <concept> [alpha] [layer], /clear, /sys [prompt], "
                "/temp [val], /top-p [val], /max [n], /probes, /help\n"
                "Keys: Tab focus · j/k nav · ←/→ alpha · S-↑/↓ layer · Enter toggle\n"
                "Ctrl+N add · Ctrl+D rm · Ctrl+O ortho · Ctrl+R regen · Ctrl+A A/B\n"
                "[ ] temp · { } top-p · Ctrl+S sort · Esc stop · Ctrl+Q quit"
            )
        else:
            chat.add_system_message(f"Unknown command: {cmd}. Type /help for commands.")

    # -- Vector Management --

    def _add_vector_from_text(self, text: str) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        parts = text.split()
        concept = parts[0]
        alpha = float(parts[1]) if len(parts) > 1 else 1.0
        layer_idx = int(parts[2]) if len(parts) > 2 else self._model_info["num_layers"] // 2

        chat.add_system_message(f"Extracting '{concept}'...")

        def _extract():
            from steer.vectors import extract_actadd
            vec = extract_actadd(self._model, self._tokenizer, concept, layer_idx, layers=self._layers)
            self._steering.add_vector(concept, vec, alpha, layer_idx)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )
            self.call_from_thread(self._on_vector_extracted, concept, alpha, layer_idx)

        self.run_worker(_extract, thread=True)

    def _on_vector_extracted(self, concept: str, alpha: float, layer_idx: int) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.add_system_message(
            f"Vector '{concept}' active (α={alpha:+.1f}, L{layer_idx})"
        )
        self._refresh_left_panel()

    def _refresh_left_panel(self) -> None:
        lp = self.query_one("#left-panel", LeftPanel)
        vectors = self._steering.get_active_vectors()
        lp.update_vectors(vectors, orthogonalize=self._orthogonalize)

    def _refresh_gen_config(self) -> None:
        lp = self.query_one("#left-panel", LeftPanel)
        lp.update_gen_config(
            self._gen_config.temperature,
            self._gen_config.top_p,
            self._gen_config.max_new_tokens,
            self._gen_config.system_prompt,
        )

    # -- Generation --

    def action_stop_generation(self) -> None:
        if self._gen_state.is_generating.is_set():
            self._gen_state.request_stop()

    def _start_generation(self) -> None:
        if self._gen_state.is_generating.is_set():
            self._gen_state.request_stop()
            chat = self.query_one("#chat-panel", ChatPanel)
            chat.add_system_message("Stopping current generation. Please resubmit.")
            return

        self._gen_state.reset()
        if self._monitor:
            self._monitor.reset_history()

        self._gen_token_count = 0
        self._gen_start_time = time.monotonic()

        chat = self.query_one("#chat-panel", ChatPanel)
        self._current_assistant_widget = chat.start_assistant_message()

        def _generate():
            input_ids = build_chat_input(
                self._tokenizer, self._messages, self._gen_config.system_prompt,
            )
            input_ids = input_ids.to(self._device)
            self._prompt_token_count = input_ids.shape[-1]

            def on_token(tok: str):
                self._gen_state.token_queue.put(tok)

            generated = generate_steered(
                self._model, self._tokenizer, input_ids,
                self._gen_config, self._gen_state,
                on_token=on_token,
            )

            full_text = self._tokenizer.decode(generated, skip_special_tokens=True)
            if full_text.strip():
                self._messages.append({"role": "assistant", "content": full_text})

        self.run_worker(_generate, thread=True)

    def _poll_generation(self) -> None:
        """~15 FPS poll: drain tokens, update stats, update monitor."""
        chat = self.query_one("#chat-panel", ChatPanel)
        tokens_consumed = 0
        generating = self._gen_state.is_generating.is_set()

        while tokens_consumed < 20:
            try:
                token = self._gen_state.token_queue.get_nowait()
            except queue.Empty:
                break
            if token is None:
                self._current_assistant_widget = None
                generating = False
                break
            if self._current_assistant_widget:
                chat.append_to_assistant(self._current_assistant_widget, token)
            self._gen_token_count += 1
            tokens_consumed += 1

        # Update status bar
        elapsed = time.monotonic() - self._gen_start_time if self._gen_start_time > 0 else 0.0
        tok_per_sec = self._gen_token_count / elapsed if elapsed > 0.1 else 0.0
        if generating or self._gen_token_count > 0:
            self._last_tok_per_sec = tok_per_sec
            self._last_elapsed = elapsed

        vram_gb = _get_memory_gb(self._device_str)

        chat.update_status(
            generating=generating,
            gen_tokens=self._gen_token_count,
            max_tokens=self._gen_config.max_new_tokens,
            tok_per_sec=self._last_tok_per_sec,
            elapsed=self._last_elapsed,
            prompt_tokens=self._prompt_token_count,
            vram_gb=vram_gb,
        )

        # Update trait monitor
        if self._monitor:
            self._monitor.flush_to_cpu()
            current = self._monitor.get_current()
            previous = self._monitor.get_previous()
            if any(self._monitor.history[n] for n in self._monitor.probe_names):
                sparklines = {name: self._monitor.get_sparkline(name, width=64)
                              for name in self._monitor.probe_names}
                trait_panel = self.query_one("#trait-panel", TraitPanel)
                trait_panel.update_values(
                    current, previous, sparklines,
                    histories=self._monitor.history,
                )

    # -- Actions --

    def action_new_vector(self) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.add_system_message(
            "Type: /steer <concept> [alpha] [layer]  (e.g. /steer happy 0.8 18)"
        )

    def action_remove_vector(self) -> None:
        lp = self.query_one("#left-panel", LeftPanel)
        sel = lp.get_selected()
        if sel:
            self._steering.remove_vector(sel["name"])
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )
            self._refresh_left_panel()

    def action_toggle_vector(self) -> None:
        lp = self.query_one("#left-panel", LeftPanel)
        sel = lp.get_selected()
        if sel:
            self._steering.toggle_vector(sel["name"])
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )
            self._refresh_left_panel()

    def _adjust_alpha(self, delta: float) -> None:
        lp = self.query_one("#left-panel", LeftPanel)
        sel = lp.get_selected()
        if sel:
            new_alpha = max(-3.0, min(3.0, sel["alpha"] + delta))
            self._steering.set_alpha(sel["name"], new_alpha)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )
            self._refresh_left_panel()

    def _adjust_layer(self, delta: int) -> None:
        lp = self.query_one("#left-panel", LeftPanel)
        sel = lp.get_selected()
        if sel:
            new_layer = max(0, min(len(self._layers) - 1, sel["layer_idx"] + delta))
            self._steering.set_layer(sel["name"], new_layer, self._layers)
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )
            self._refresh_left_panel()

    def action_toggle_ortho(self) -> None:
        self._orthogonalize = not self._orthogonalize
        self._steering.apply_to_model(
            self._layers, self._device, self._dtype,
            orthogonalize=self._orthogonalize,
        )
        self._refresh_left_panel()

    def action_temp_down(self) -> None:
        self._gen_config.temperature = max(0.0, round(self._gen_config.temperature - 0.05, 2))
        self._refresh_gen_config()

    def action_temp_up(self) -> None:
        self._gen_config.temperature = round(self._gen_config.temperature + 0.05, 2)
        self._refresh_gen_config()

    def action_top_p_down(self) -> None:
        self._gen_config.top_p = max(0.0, round(self._gen_config.top_p - 0.05, 2))
        self._refresh_gen_config()

    def action_top_p_up(self) -> None:
        self._gen_config.top_p = min(1.0, round(self._gen_config.top_p + 0.05, 2))
        self._refresh_gen_config()

    def action_regenerate(self) -> None:
        if not self._messages:
            return
        if self._messages and self._messages[-1]["role"] == "assistant":
            self._messages.pop()
        self._start_generation()

    def action_ab_compare(self) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        if self._gen_state.is_generating.is_set():
            chat.add_system_message("Cannot A/B compare while generating. Stop generation first.")
            return
        if not self._last_prompt:
            chat.add_system_message("No previous prompt to compare.")
            return
        chat.add_system_message("A/B comparison: generating unsteered response...")

        def _ab_generate():
            msgs = [{"role": "user", "content": self._last_prompt}]
            input_ids = build_chat_input(
                self._tokenizer, msgs, self._gen_config.system_prompt,
            ).to(self._device)

            saved_vectors = self._steering.get_active_vectors()
            self._steering.clear_all()

            ab_state = GenerationState()
            generated = generate_steered(
                self._model, self._tokenizer, input_ids,
                self._gen_config, ab_state,
            )
            unsteered = self._tokenizer.decode(generated, skip_special_tokens=True)

            for v in saved_vectors:
                self._steering.add_vector(v["name"], v["vector"], v["alpha"], v["layer_idx"])
                if not v.get("enabled", True):
                    self._steering.toggle_vector(v["name"])
            self._steering.apply_to_model(
                self._layers, self._device, self._dtype,
                orthogonalize=self._orthogonalize,
            )

            self.call_from_thread(self._show_ab_result, unsteered)

        self.run_worker(_ab_generate, thread=True)

    def _show_ab_result(self, unsteered: str) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        chat.add_system_message(f"[Unsteered]: {unsteered}")

    def _show_probe_info(self) -> None:
        chat = self.query_one("#chat-panel", ChatPanel)
        if not self._monitor:
            chat.add_system_message("No probes loaded.")
            return
        chat.add_system_message(
            f"Active probes ({len(self._monitor.probe_names)}): "
            + ", ".join(self._monitor.probe_names)
        )

    def action_cycle_sort(self) -> None:
        trait_panel = self.query_one("#trait-panel", TraitPanel)
        trait_panel.cycle_sort()
```

- [ ] **Step 2: Commit**

```bash
git add steer/tui/app.py
git commit -m "feat: rewrite app with three-column layout, panel focus, gen stats"
```

---

### Task 7: Update __init__.py to export new names

**Files:**
- Modify: `steer/tui/__init__.py`

- [ ] **Step 1: Ensure __init__.py is clean**

The file should be empty or just a docstring. The old import of `ControlsPanel` from `vector_panel` is gone. Verify nothing imports `ControlsPanel` or `VectorPanel` from outside the TUI:

Run: `grep -r "ControlsPanel\|from steer.tui.vector_panel import" steer/ --include="*.py"`

If anything references the old names, update those imports. The main consumer is `app.py` which now imports `LeftPanel`.

- [ ] **Step 2: Commit if changes were needed**

```bash
git add steer/tui/__init__.py
git commit -m "chore: clean up tui __init__ imports"
```

---

### Task 8: Add top_p to GenerationConfig

**Files:**
- Modify: `steer/generation.py:15-31`

- [ ] **Step 1: Verify top_p already exists on GenerationConfig**

Check `steer/generation.py`. The `GenerationConfig` class already has `top_p` in `__slots__` and `__init__`. No change needed — it's already there and wired into the generation loop.

- [ ] **Step 2: Verify generation.py uses self._gen_config.top_p**

The generation loop at line 146 uses `config.top_p`. This is already correct. No changes to generation.py needed.

---

### Task 9: Smoke test the full TUI

- [ ] **Step 1: Verify syntax**

Run: `python3 -m py_compile steer/tui/app.py && python3 -m py_compile steer/tui/chat_panel.py && python3 -m py_compile steer/tui/vector_panel.py && python3 -m py_compile steer/tui/trait_panel.py && echo "OK"`

Expected: `OK`

- [ ] **Step 2: Verify imports resolve**

Run: `python3 -c "from steer.tui.app import SteerApp; print('imports OK')"`

Expected: `imports OK`

- [ ] **Step 3: Run existing tests**

Run: `python3 -m pytest tests/test_smoke.py -v` (if CUDA available)

Or for a quick non-GPU check: `python3 -c "from steer.tui.vector_panel import LeftPanel; from steer.tui.trait_panel import TraitPanel; from steer.tui.chat_panel import ChatPanel; print('all widgets import OK')"`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete TUI redesign — three-column, info-dense, keyboard-navigable"
```
