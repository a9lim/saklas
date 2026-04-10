"""Left panel: model info, steering vectors with inline controls, generation config, key reference."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
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
        yield VerticalScroll(Static("", id="vector-content"), id="vector-scroll")
        yield Static("[dim]Ctrl+N add · Ctrl+D rm · Enter toggle · Ctrl+O ortho[/]",
                      id="vector-hints")
        # Generation section
        yield Static("[bold]GENERATION[/]", classes="section-header")
        yield Static("", id="gen-config")
        # Keys section
        yield Static("[bold]KEYS[/]", classes="section-header")
        yield Static(
            "[dim]Tab focus panels · Esc stop gen\n"
            "Ctrl+N add vec · Ctrl+D rm vec\n"
            "Ctrl+R regen · Ctrl+A A/B\n"
            "Ctrl+T toggle vec · Ctrl+O ortho\n"
            "Ctrl+S sort probes · Ctrl+Q quit\n"
            "── Tab to side panel first ──\n"
            "↑/↓ navigate · Enter select\n"
            "h/l alpha · j/k layer\n"
            "[ ] temp · { } top-p[/]",
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
        active = sum(1 for v in self._vectors if v.get("enabled", True))
        total = len(self._vectors)
        ortho_str = "ON" if self._orthogonalize else "OFF"
        header = self.query_one("#vectors-header", Static)
        header.update(
            f"[bold]VECTORS[/] [dim]{total} total, {active} active · ortho: {ortho_str}[/]"
        )

        lines: list[str] = []
        num_layers = self._model_info["num_layers"]
        for i, v in enumerate(self._vectors):
            is_selected = i == self._selected_idx
            enabled = v.get("enabled", True)
            name = v["name"]
            alpha = v["alpha"]
            layer = v["layer_idx"]
            method = v.get("method", "actadd")

            bar_width = 14
            filled = int(abs(alpha) / 3.0 * bar_width)
            filled = min(filled, bar_width)
            bar_full = "█" * filled
            bar_empty = "░" * (bar_width - filled)
            color = "green" if alpha >= 0 else "red"

            if is_selected:
                marker = ">"
                dot = "[green]●[/]" if enabled else "[dim]○[/]"
                if num_layers > 0:
                    lbar_width = min(20, num_layers)
                    lpos = int(layer / max(num_layers - 1, 1) * (lbar_width - 1))
                    lbar = "▁" * lpos + "█" + "▁" * (lbar_width - lpos - 1)
                else:
                    lbar = "█"
                if enabled:
                    text = (
                        f"{marker} {dot} [bold]{name}[/] {method}\n"
                        f"  α [{color}]{bar_full}[/][dim]{bar_empty}[/] "
                        f"[{color}]{alpha:+.1f}[/] [dim]h/l[/]\n"
                        f"  L [dim]{lbar}[/] {layer}/{num_layers} [dim]j/k[/]"
                    )
                else:
                    text = (
                        f"{marker} {dot} [dim bold]{name}[/] [dim]{method}[/]\n"
                        f"  α [dim {color}]{bar_full}[/][dim]{bar_empty}[/] "
                        f"[dim {color}]{alpha:+.1f}[/] [dim]h/l[/]\n"
                        f"  L [dim]{lbar}[/] {layer}/{num_layers} [dim]j/k[/]"
                    )
            else:
                marker = " "
                dot = "[green]●[/]" if enabled else "[dim]○[/]"
                if enabled:
                    text = (
                        f"{marker} {dot} {name} {method}\n"
                        f"  α [{color}]{bar_full}[/][dim]{bar_empty}[/] "
                        f"{alpha:+.1f}  L{layer}"
                    )
                else:
                    text = (
                        f"{marker} {dot} [dim]{name}[/] [dim]{method}[/]\n"
                        f"  α [dim {color}]{bar_full}[/][dim]{bar_empty}[/] "
                        f"[dim]{alpha:+.1f}[/]  L{layer}"
                    )
            lines.append(text)

        content = self.query_one("#vector-content", Static)
        content.update("\n".join(lines))

    def _render_gen_config(self) -> None:
        gen = self.query_one("#gen-config", Static)
        t_bar_w = 20
        t_filled = int(self._temperature / 2.0 * t_bar_w)
        t_filled = min(t_filled, t_bar_w)
        t_bar = "█" * t_filled + "░" * (t_bar_w - t_filled)
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
