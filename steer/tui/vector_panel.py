"""Left panel: model info, steering vectors with inline controls, generation config, key reference."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static
from textual.widget import Widget
from textual.message import Message


def _build_bar(value: float, max_value: float, width: int) -> tuple[str, str]:
    filled = min(int(abs(value) / max_value * width), width)
    return "█" * filled, "░" * (width - filled)


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
        self._temperature: float = 1.0
        self._top_p: float = 0.9
        self._max_tokens: int = 1024
        self._system_prompt: str | None = None

    def on_mount(self) -> None:
        self._vectors_header = self.query_one("#vectors-header", Static)
        self._vector_content = self.query_one("#vector-content", Static)
        self._gen_config_widget = self.query_one("#gen-config", Static)

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
        yield Static("[dim]⌫ remove · ↩ toggle · ⌃O ortho[/]",
                      id="vector-hints")
        # Generation section
        yield Static("[bold]GENERATION[/]", classes="section-header")
        yield Static("", id="gen-config")
        # Keys section
        yield Static("[bold]KEYS[/]", classes="section-header")
        yield Static(
            "[dim]⇥ focus panels · ⎋ stop gen\n"
            "⌃R regen · ⌃A A/B\n"
            "⌃Q quit\n"
            "── ⇥ to side panel first ──\n"
            "↑/↓ navigate · ↩ select\n"
            "←/→ alpha\n"
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
        header = self._vectors_header
        header.update(
            f"[bold]VECTORS[/] [dim]{total} total, {active} active · ortho: {ortho_str}[/]"
        )

        lines: list[str] = []
        for i, v in enumerate(self._vectors):
            is_selected = i == self._selected_idx
            enabled = v.get("enabled", True)
            name = v["name"]
            alpha = v["alpha"]
            profile = v["profile"]
            peak = max(profile, key=lambda k: profile[k][1])
            n_active = len(profile)
            layer_tag = f"{n_active}L pk{peak}"

            bar_full, bar_empty = _build_bar(alpha, 5.0, 16)
            if alpha > 0:
                color = "ansi_green"
            elif alpha < 0:
                color = "ansi_red"
            else:
                color = "ansi_default"

            if is_selected:
                marker = ">"
                dot = "[ansi_green]●[/]" if enabled else "[dim]○[/]"
                if enabled:
                    text = (
                        f"{marker} {dot} [bold]{name}[/] [dim]{layer_tag}[/]\n"
                        f"  α [{color}]{bar_full}[/][dim]{bar_empty}[/] "
                        f"[{color}]{alpha:+.1f}[/] [dim]←/→[/]"
                    )
                else:
                    text = (
                        f"{marker} {dot} [dim bold]{name}[/] [dim]{layer_tag}[/]\n"
                        f"  α [dim {color}]{bar_full}[/][dim]{bar_empty}[/] "
                        f"[dim {color}]{alpha:+.1f}[/] [dim]←/→[/]"
                    )
            else:
                marker = " "
                dot = "[ansi_green]●[/]" if enabled else "[dim]○[/]"
                if enabled:
                    text = (
                        f"{marker} {dot} {name} [dim]{layer_tag}[/]\n"
                        f"  α [{color}]{bar_full}[/][dim]{bar_empty}[/] "
                        f"[{color}]{alpha:+.1f}[/]"
                    )
                else:
                    text = (
                        f"{marker} {dot} [dim]{name} {layer_tag}[/]\n"
                        f"  α [dim {color}]{bar_full}[/][dim]{bar_empty}[/] "
                        f"[dim {color}]{alpha:+.1f}[/]"
                    )
            lines.append(text)

        content = self._vector_content
        content.update("\n".join(lines))

    def _render_gen_config(self) -> None:
        gen = self._gen_config_widget
        t_full, t_empty = _build_bar(self._temperature, 2.0, 20)
        t_bar = t_full + t_empty
        p_full, p_empty = _build_bar(self._top_p, 1.0, 20)
        p_bar = p_full + p_empty

        sys_str = self._system_prompt[:15] + "..." if self._system_prompt and len(self._system_prompt) > 15 else (self._system_prompt or "(none)")

        gen.update(
            f"Temp  {self._temperature:.2f} [dim]{t_bar}[/] [dim]\\[/][/]\n"
            f"Top-p {self._top_p:.2f} [dim]{p_bar}[/] [dim]{{/}}[/]\n"
            f"Max   {self._max_tokens} tok       [dim]/max[/]\n"
            f"Sys   [dim]{sys_str}[/]    [dim]/sys[/]\n"
            f"[dim]type /help for commands[/]"
        )
