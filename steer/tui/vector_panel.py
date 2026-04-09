"""Steering vector list + controls panel."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static
from textual.widget import Widget
from textual.message import Message


class VectorPanel(Widget):

    class VectorSelected(Message):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._vectors: list[dict] = []
        self._selected_idx: int = 0

    def compose(self) -> ComposeResult:
        yield Static("[bold]STEERING VECTORS[/]", id="vector-header")
        yield Vertical(id="vector-list")
        yield Static("[dim]\\[+] Ctrl+N add  [x] Ctrl+D remove  [S] save  [L] load[/]",
                      id="vector-actions")

    def update_vectors(self, vectors: list[dict]) -> None:
        self._vectors = vectors
        vlist = self.query_one("#vector-list", Vertical)
        vlist.remove_children()
        for i, v in enumerate(vectors):
            marker = ">" if i == self._selected_idx else " "
            dot = "[green]●[/]" if v.get("enabled", True) else "[dim]○[/]"
            name = v["name"][:10].ljust(10)
            alpha = f"a={v['alpha']:+.1f}"
            layer = f"L{v['layer_idx']}"
            method = v.get("method", "")[:6]
            line = f"{marker} {dot} {name} {alpha:>8}  {layer:>4}  {method}"
            vlist.mount(Static(line, classes="vector-row"))

    def select_next(self) -> None:
        if self._vectors:
            self._selected_idx = (self._selected_idx + 1) % len(self._vectors)
            self.update_vectors(self._vectors)
            self._post_selection()

    def select_prev(self) -> None:
        if self._vectors:
            self._selected_idx = (self._selected_idx - 1) % len(self._vectors)
            self.update_vectors(self._vectors)
            self._post_selection()

    def get_selected(self) -> dict | None:
        if self._vectors and 0 <= self._selected_idx < len(self._vectors):
            return self._vectors[self._selected_idx]
        return None

    def _post_selection(self) -> None:
        sel = self.get_selected()
        if sel:
            self.post_message(self.VectorSelected(sel["name"]))


class ControlsPanel(Widget):

    def compose(self) -> ComposeResult:
        yield Static("[bold]CONTROLS[/]", id="controls-header")
        yield Static("No vector selected", id="alpha-display")
        yield Static("", id="layer-display")
        yield Static("", id="ortho-display")

    def update_for_vector(self, vec: dict | None, num_layers: int = 0) -> None:
        if vec is None:
            self.query_one("#alpha-display").update("No vector selected")
            self.query_one("#layer-display").update("")
            self.query_one("#ortho-display").update("")
            return

        alpha = vec["alpha"]
        bar_width = 30
        filled = int(abs(alpha) / 3.0 * bar_width)
        bar = ("█" * filled).ljust(bar_width, "░")
        sign = "+" if alpha >= 0 else "-"
        color = "green" if alpha >= 0 else "red"
        self.query_one("#alpha-display").update(
            f"Alpha [{color}]{bar}[/] {sign}{abs(alpha):.1f}  [dim]\\[←/→ adjust][/]"
        )

        layer = vec["layer_idx"]
        self.query_one("#layer-display").update(
            f"Layer  {layer} / {num_layers}  [dim]\\[↑/↓ change][/]"
        )

        ortho = vec.get("orthogonalize", False)
        ortho_str = "[green]ON[/]" if ortho else "[dim]OFF[/]"
        self.query_one("#ortho-display").update(f"Orthogonalize  {ortho_str}  [dim]\\[O toggle][/]")
