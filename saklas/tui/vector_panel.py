"""Left panel: model info, steering vectors with inline controls, generation config, key reference."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static
from textual.widget import Widget

from saklas.tui.utils import BAR_WIDTH, build_bar

MAX_ALPHA = 1.0


class LeftPanel(Widget):
    """Entire left column: model, vectors, gen config, keys."""

    def __init__(self, model_info: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model_info = model_info
        self._vectors: list[dict[str, Any]] = []
        self._selected_idx: int = 0
        self._thinking: bool | None = None  # None = model doesn't support it
        # True when the model thinks unconditionally (gpt-oss / Mistral-3
        # Reasoning / Qwen3-Thinking) — the toggle is shown locked so
        # the user knows ``thinking=False`` would be a no-op.
        self._thinking_forced: bool = False
        self._temperature: float = 1.0
        self._top_p: float = 0.9
        self._max_tokens: int = 1024
        self._system_prompt: str | None = None
        # Current per-token highlight mode: "off" / "surprise" / a probe
        # name. Surfaced as the persistent HL line in the GENERATION
        # block so the Ctrl+Y cycle has an always-visible readout.
        self._highlight_mode: str = "off"

    def on_mount(self) -> None:
        self._vectors_header = self.query_one("#vectors-header", Static)
        self._vector_content = self.query_one("#vector-content", Static)
        self._gen_config_widget = self.query_one("#gen-config", Static)

    def compose(self) -> ComposeResult:
        info = self._model_info
        # Model section
        yield Static("[bold]MODEL[/]", classes="section-header")
        model_id = info.get("model_id", "unknown")
        if "/" in model_id:
            model_id = model_id.split("/", 1)[1]
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
        yield Static("[bold]STEERING VECTORS[/] [dim]0 total, 0 active[/]",
                      id="vectors-header", classes="section-header")
        yield VerticalScroll(Static("", id="vector-content"), id="vector-scroll")
        yield Static(
            "[dim]⌫ remove · ⏎ toggle · ←/→ alpha · ⇧←/→ coarse[/]",
            id="vector-hints",
        )
        # Generation section
        yield Static("[bold]GENERATION[/]", classes="section-header")
        yield Static("", id="gen-config")
        # Keys section
        yield Static("[bold]KEYS[/]", classes="section-header")
        yield Static(
            "[dim]⇥ focus panels · ⎋ stop gen\n"
            "⌃R regen · ⌃A A/B · ⌃T think\n"
            "⌃Y highlight · ⌃L loom\n"
            "⌃B branch · ⌃N nav · ⌃D delete\n"
            "/fan grid · /diff compare · /prune filter\n"
            "⌃Q quit\n"
            "── ⇥ to side panel first ──\n"
            "↑/↓ navigate\n"
            "[ ] temp · { } top-p[/]",
            id="key-ref",
        )

    def update_vectors(self, vectors: list[dict[str, Any]]) -> None:
        self._vectors = vectors
        if self._vectors:
            self._selected_idx = min(self._selected_idx, len(self._vectors) - 1)
        else:
            self._selected_idx = 0
        self._render_vectors()

    def update_gen_config(self, temperature: float, top_p: float,
                          max_tokens: int, system_prompt: str | None,
                          thinking: bool | None = None,
                          thinking_forced: bool = False) -> None:
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._thinking = thinking
        self._thinking_forced = thinking_forced
        self._render_gen_config()

    def update_highlight(self, mode: str) -> None:
        """Set the highlight-mode readout and re-render the GENERATION
        block.  ``mode`` is ``"off"`` / ``"surprise"`` / a probe name.
        Guarded against pre-mount calls — ``_gen_config_widget`` only
        exists after ``on_mount``."""
        self._highlight_mode = mode
        if hasattr(self, "_gen_config_widget"):
            self._render_gen_config()

    def select_next(self) -> None:
        if self._vectors:
            self._selected_idx = (self._selected_idx + 1) % len(self._vectors)
            self._render_vectors()

    def select_prev(self) -> None:
        if self._vectors:
            self._selected_idx = (self._selected_idx - 1) % len(self._vectors)
            self._render_vectors()

    def get_selected(self) -> dict[str, Any] | None:
        if self._vectors and 0 <= self._selected_idx < len(self._vectors):
            return self._vectors[self._selected_idx]
        return None

    def _render_vectors(self) -> None:
        active = sum(1 for v in self._vectors if v.get("enabled", True))
        total = len(self._vectors)
        header = self._vectors_header
        header.update(
            f"[bold]STEERING VECTORS[/] [dim]{total} total, {active} active[/]"
        )

        lines: list[str] = []
        for i, v in enumerate(self._vectors):
            is_selected = i == self._selected_idx
            enabled = v.get("enabled", True)
            name = v["name"]

            if v.get("kind") == "manifold":
                lines.append(self._render_manifold_row(v, is_selected, enabled))
                continue

            alpha = v["alpha"]
            peak = v["peak"]
            n_active = v["n_active"]
            layer_tag = f"{n_active}L pk{peak}"

            bar_full, bar_empty = build_bar(alpha, MAX_ALPHA, BAR_WIDTH)
            if alpha > 0:
                color = "ansi_green"
            elif alpha < 0:
                color = "ansi_red"
            else:
                color = "ansi_default"

            marker = ">" if is_selected else " "
            dot = "[ansi_green]●[/]" if enabled else "[dim]○[/]"
            hint = " [dim]←/→[/]" if is_selected else ""

            if is_selected and enabled:
                name_str = f"[bold]{name}[/]"
            elif is_selected and not enabled:
                name_str = f"[dim bold]{name}[/]"
            elif not enabled:
                name_str = f"[dim]{name}[/]"
            else:
                name_str = name

            dim_prefix = "dim " if not enabled else ""
            text = (
                f"{marker} {dot} {name_str} [dim]{layer_tag}[/]\n"
                f"  α [{dim_prefix}{color}]{bar_full}[/][dim]{bar_empty}[/] "
                f"[{dim_prefix}{color}]{alpha:+.2f}[/]{hint}"
            )
            lines.append(text)

        content = self._vector_content
        content.update("\n".join(lines))

    def _render_manifold_row(
        self, v: dict[str, Any], is_selected: bool, enabled: bool,
    ) -> str:
        """Render one manifold row.

        A manifold term has a fixed position on a fitted surface, not a
        scalar alpha — so there's no alpha bar.  The second line shows
        the authoring position (``manifold % 0.3,0.8``) and the blend
        coefficient instead.
        """
        manifold = v.get("manifold", v["name"])
        coords = v.get("coords", "")
        blend = v.get("blend", 0.0)

        marker = ">" if is_selected else " "
        dot = "[ansi_green]●[/]" if enabled else "[dim]○[/]"

        if is_selected and enabled:
            name_str = f"[bold]{manifold}[/]"
        elif is_selected and not enabled:
            name_str = f"[dim bold]{manifold}[/]"
        elif not enabled:
            name_str = f"[dim]{manifold}[/]"
        else:
            name_str = manifold

        dim_prefix = "dim " if not enabled else ""
        # Phase C.3: manifold rows render in purple
        # (``ansi_magenta``) — visually distinct from vector rows
        # (red/green per the alpha sign) so a bare-name resolution
        # that lands on a manifold-label is immediately legible in
        # the rack.  The GUI ``ManifoldStrip`` uses the same accent.
        return (
            f"{marker} {dot} {name_str} [dim]manifold[/]\n"
            f"  [{dim_prefix}ansi_magenta]% {coords}[/] "
            f"[dim]blend[/] [{dim_prefix}ansi_magenta]{blend:.2f}[/]"
        )

    def _render_gen_config(self) -> None:
        gen = self._gen_config_widget
        t_full, t_empty = build_bar(self._temperature, 2.0, BAR_WIDTH)
        t_bar = t_full + t_empty
        p_full, p_empty = build_bar(self._top_p, 1.0, BAR_WIDTH)
        p_bar = p_full + p_empty

        sys_str = self._system_prompt[:15] + "..." if self._system_prompt and len(self._system_prompt) > 15 else (self._system_prompt or "(none)")

        # Right-edge column for the hint glyphs. Bar lines render as
        # "Temp  1.00 " (11) + bar (BAR_WIDTH) + " [/]" (4), so anything
        # else padded to the same width lines up the right edges.
        RIGHT_W = 11 + BAR_WIDTH + 4

        def _pad(left_visible: str, right_text: str) -> str:
            return " " * max(1, RIGHT_W - len(left_visible) - len(right_text))

        max_prefix = f"Max   {self._max_tokens} tok"
        think_str = "ON" if self._thinking else "OFF"
        think_prefix = f"Think {think_str}"
        sys_prefix = f"Sys   {sys_str}"

        lines = [
            f"Temp  {self._temperature:.2f} [dim]{t_bar}[/] [dim]\\[/][/]",
            f"Top-p {self._top_p:.2f} [dim]{p_bar}[/] [dim]{{/}}[/]",
            f"{max_prefix}{_pad(max_prefix, '/max')}[dim]/max[/]",
        ]
        if self._thinking is not None:
            if self._thinking_forced:
                # Toggle has no prompt-level effect: dim the line and drop
                # the ⌃T hint so the user knows pressing it is a no-op.
                forced_prefix = "Think ON"
                lines.append(
                    f"[dim]{forced_prefix}{_pad(forced_prefix, 'forced')}forced[/]"
                )
            else:
                lines.append(
                    f"{think_prefix}{_pad(think_prefix, '⌃T')}[dim]⌃T[/]"
                )
        lines.append(
            f"Sys   [dim]{sys_str}[/]{_pad(sys_prefix, '/sys')}[dim]/sys[/]"
        )

        # HL: current per-token highlight mode (Ctrl+Y cycles it). Probe
        # names can run long (e.g. high_context.low_context) — truncate
        # to the same 15-char budget as the Sys prompt preview.
        hl = self._highlight_mode
        hl_str = hl[:15] + "..." if len(hl) > 15 else hl
        hl_prefix = f"HL    {hl_str}"
        hl_body = f"[dim]{hl_str}[/]" if hl == "off" else hl_str
        lines.append(
            f"HL    {hl_body}{_pad(hl_prefix, '⌃Y')}[dim]⌃Y[/]"
        )

        lines.append("[dim]type /help for commands[/]")

        gen.update("\n".join(lines))
