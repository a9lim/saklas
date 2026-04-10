"""Live trait monitor panel with inline sparklines and always-visible stats."""

from __future__ import annotations

import math

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static
from textual.widget import Widget

from steer.tui.vector_panel import _build_bar



class TraitPanel(Widget):

    def __init__(self, categories: dict[str, list[str]] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._categories: dict[str, list[str]] = categories or {}
        self._current_values: dict[str, float] = {}
        self._previous_values: dict[str, float] = {}
        self._sparklines: dict[str, str] = {}
        self._probe_stats: dict[str, dict] = {}
        self._active_probes: set[str] = set()
        self._sort_mode: str = "name"
        self._nav_items: list[tuple[str, str]] = []
        self._nav_idx: int = 0
        self._cached_stats_lines: dict[str, tuple[dict, str]] = {}
        self._cached_sort: tuple[str, tuple, tuple, list] | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            "[bold]TRAIT MONITOR[/] [dim]sort: name · Ctrl+S[/]",
            id="trait-header",
        )
        yield VerticalScroll(Static("", id="trait-content"), id="trait-scroll")
        yield Static("[dim]↑/↓ nav · Ctrl+D remove · Ctrl+S sort[/]",
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
        stats: dict[str, dict] | None = None,
    ) -> None:
        self._current_values = current
        self._previous_values = previous
        self._sparklines = sparklines
        if stats is not None:
            self._probe_stats = stats
        self._render_probes()

    def cycle_sort(self) -> None:
        modes = ["name", "magnitude", "change"]
        idx = modes.index(self._sort_mode)
        self._sort_mode = modes[(idx + 1) % len(modes)]
        header = self._trait_header
        header.update(
            f"[bold]TRAIT MONITOR[/] [dim]sort: {self._sort_mode[:3]} · Ctrl+S[/]"
        )
        self._render_probes()

    def get_selected_probe(self) -> str | None:
        """Return the name of the currently nav-selected probe, or None."""
        if not self._nav_items:
            return None
        if self._nav_idx >= len(self._nav_items):
            return None
        item_type, name = self._nav_items[self._nav_idx]
        if item_type == "probe":
            return name
        return None

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

        nav_idx_counter = 0
        for category, members in self._categories.items():
            active_members = [m for m in members if m in self._active_probes]
            if not active_members:
                continue

            count = len(active_members)
            lines.append(
                f" [bold]▾ {category}[/] [dim]({count})[/]"
            )

            sorted_members = self._sort_probes(active_members)
            for name in sorted_members:
                is_nav_selected = nav_idx_counter == self._nav_idx
                self._nav_items.append(("probe", name))
                nav_idx_counter += 1

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

                bar_full, bar_empty = _build_bar(val, 1.0, 10)
                color = "green" if val >= 0 else "red"

                spark = self._sparklines.get(name, "")
                mini_spark = spark[-8:] if spark else ""

                sel = ">" if is_nav_selected else " "
                display_name = name[:9].ljust(9)

                line = (
                    f"{sel} {display_name}[{color}]{bar_full}[/][dim]{bar_empty}[/] "
                    f"{val:+.2f}{arrow_ch} [dim]{mini_spark}[/]"
                )

                stats = self._probe_stats.get(name, {})
                cached = self._cached_stats_lines.get(name)
                if cached and cached[0] is stats:
                    stats_line = cached[1]
                else:
                    stats_line = self._compute_stats_line(stats)
                    self._cached_stats_lines[name] = (stats, stats_line)
                lines.append(f"{line}\n  [dim]{stats_line}[/]")

        content = self._trait_content
        content.update("\n".join(lines))

    def _compute_stats_line(self, stats: dict) -> str:
        n = stats.get("count", 0)
        if n == 0:
            return "no data"
        mean = stats["sum"] / n
        lo = stats["min"]
        hi = stats["max"]
        if n > 1:
            variance = max(0.0, stats["sum_sq"] / n - mean ** 2)
            std = variance ** 0.5
            delta_per_tok = (stats["last"] - stats["first"]) / (n - 1)
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
            vals = tuple(self._current_values.get(n, 0.0) for n in names)
            key = (self._sort_mode, tuple(names), vals)
            if self._cached_sort and self._cached_sort[:3] == key:
                return self._cached_sort[3]
            result = sorted(names, key=lambda n: abs(self._current_values.get(n, 0.0)), reverse=True)
            self._cached_sort = (*key, result)
            return result
        elif self._sort_mode == "change":
            vals = tuple((self._current_values.get(n, 0.0), self._previous_values.get(n, 0.0)) for n in names)
            key = (self._sort_mode, tuple(names), vals)
            if self._cached_sort and self._cached_sort[:3] == key:
                return self._cached_sort[3]
            result = sorted(names, key=lambda n: abs(
                self._current_values.get(n, 0.0) - self._previous_values.get(n, 0.0)
            ), reverse=True)
            self._cached_sort = (*key, result)
            return result
        self._cached_sort = None
        return sorted(names)
