"""Live trait monitor panel with inline sparklines and always-visible stats."""

from __future__ import annotations

import math

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static
from textual.widget import Widget

from saklas.tui.utils import build_bar



class TraitPanel(Widget):

    def __init__(self, categories: dict[str, list[str]] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._categories: dict[str, list[str]] = dict(categories) if categories else {}
        self._current_values: dict[str, float] = {}
        self._previous_values: dict[str, float] = {}
        self._sparklines: dict[str, str] = {}
        self._active_probes: set[str] = set()
        self._sort_mode: str = "name"
        self._nav_items: list[str] = []
        self._nav_idx: int = 0
        self._cached_render_text: str = ""

    def compose(self) -> ComposeResult:
        yield Static(
            "[bold]TRAIT MONITOR[/] [dim]sort: name[/]",
            id="trait-header", classes="section-header",
        )
        yield VerticalScroll(Static("", id="trait-content"), id="trait-scroll")
        yield Static("[dim]⌫ remove · ⌃S sort[/]",
                      id="trait-hints")

    def on_mount(self) -> None:
        self._trait_header = self.query_one("#trait-header", Static)
        self._trait_content = self.query_one("#trait-content", Static)

    def set_active_probes(self, probe_names: set[str]) -> None:
        self._active_probes = probe_names
        # Collect probes not in any known category into "custom"
        categorized = {m for members in self._categories.values() for m in members}
        custom = sorted(probe_names - categorized)
        if custom:
            self._categories["custom"] = custom
        elif "custom" in self._categories:
            del self._categories["custom"]
        self._render_probes()

    def update_values(
        self,
        current: dict[str, float],
        previous: dict[str, float],
        sparklines: dict[str, str],
    ) -> None:
        if (current == self._current_values
                and previous == self._previous_values
                and sparklines == self._sparklines):
            return
        self._current_values = current
        self._previous_values = previous
        self._sparklines = sparklines
        self._render_probes()

    def cycle_sort(self) -> None:
        modes = ["name", "value", "change"]
        idx = modes.index(self._sort_mode)
        self._sort_mode = modes[(idx + 1) % len(modes)]
        header = self._trait_header
        header.update(
            f"[bold]TRAIT MONITOR[/] [dim]sort: {self._sort_mode}[/]"
        )
        self._render_probes()

    def get_selected_probe(self) -> str | None:
        """Return the name of the currently nav-selected probe, or None."""
        if not self._nav_items:
            return None
        if self._nav_idx >= len(self._nav_items):
            return None
        return self._nav_items[self._nav_idx]

    def nav_down(self) -> None:
        if self._nav_items and self._nav_idx < len(self._nav_items) - 1:
            self._nav_idx += 1
            self._render_probes()

    def nav_up(self) -> None:
        if self._nav_items and self._nav_idx > 0:
            self._nav_idx -= 1
            self._render_probes()

    def _render_probes(self) -> None:
        self._nav_items = []
        lines: list[str] = []

        for category, members in self._categories.items():
            active_members = [m for m in members if m in self._active_probes]
            if not active_members:
                continue

            count = len(active_members)
            lines.append(
                f" [bold]{category}[/] [dim]({count})[/]"
            )

            sorted_members = self._sort_probes(active_members)
            for name in sorted_members:
                is_nav_selected = len(self._nav_items) == self._nav_idx
                self._nav_items.append(name)

                val = self._current_values.get(name, 0.0)
                prev = self._previous_values.get(name, 0.0)
                if math.isnan(val):
                    val = 0.0
                if math.isnan(prev):
                    prev = 0.0
                delta = val - prev

                if abs(delta) < 0.01:
                    arrow_ch = " "
                elif delta > 0:
                    arrow_ch = "↑"
                else:
                    arrow_ch = "↓"

                bar_full, bar_empty = build_bar(val, 1.0, 16)
                if val > 0:
                    color = "ansi_green"
                elif val < 0:
                    color = "ansi_red"
                else:
                    color = "ansi_default"

                mini_spark = self._sparklines.get(name, "")

                sel = ">" if is_nav_selected else " "
                display_name = name[:16].ljust(16)

                line = (
                    f"{sel} {display_name}[{color}]{bar_full}[/][dim]{bar_empty}[/] "
                    f"[{color}]{val:+.2f}{arrow_ch}[/] [dim]{mini_spark}[/]"
                )

                lines.append(line)

        text = "\n".join(lines)
        if text != self._cached_render_text:
            self._cached_render_text = text
            self._trait_content.update(text)

    def _sort_probes(self, names: list[str]) -> list[str]:
        if self._sort_mode == "value":
            return sorted(names, key=lambda n: self._current_values.get(n, 0.0), reverse=True)
        elif self._sort_mode == "change":
            return sorted(names, key=lambda n: abs(
                self._current_values.get(n, 0.0) - self._previous_values.get(n, 0.0)
            ), reverse=True)
        return sorted(names)
