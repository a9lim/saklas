"""Live trait monitor panel with bars, sparklines, category collapsing."""

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

# Categories collapsed by default
DEFAULT_COLLAPSED = {"Cultural", "Gender", "Safety"}


class TraitPanel(Widget):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._collapsed: set[str] = set(DEFAULT_COLLAPSED)
        self._current_values: dict[str, float] = {}
        self._previous_values: dict[str, float] = {}
        self._sparklines: dict[str, str] = {}
        self._selected_probe: str | None = None
        self._active_probes: set[str] = set()
        self._sort_mode: str = "name"  # "name", "magnitude", "change"

    def compose(self) -> ComposeResult:
        yield Static("[bold]TRAIT MONITOR[/]  [dim]\\[P]robes  \\[S]ort[/]",
                      id="trait-header")
        yield VerticalScroll(id="trait-list")
        yield Static("", id="sparkline-display")

    def set_active_probes(self, probe_names: set[str]) -> None:
        self._active_probes = probe_names

    def update_values(
        self,
        current: dict[str, float],
        previous: dict[str, float],
        sparklines: dict[str, str],
    ) -> None:
        self._current_values = current
        self._previous_values = previous
        self._sparklines = sparklines
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
        self._render_probes()

    def select_probe(self, name: str) -> None:
        self._selected_probe = name
        self._update_sparkline()

    def _render_probes(self) -> None:
        tlist = self.query_one("#trait-list", VerticalScroll)
        tlist.remove_children()

        for category, members in PROBE_CATEGORIES.items():
            active_members = [m for m in members if m in self._active_probes]
            if not active_members:
                continue

            collapsed = category in self._collapsed
            arrow = "▾" if collapsed else "▸"
            count = len(active_members)
            tlist.mount(Static(
                f"[bold]{arrow} {category}[/]  [dim]({count})[/]"
                + ("  [dim](collapsed)[/]" if collapsed else ""),
                classes="trait-category",
            ))

            if collapsed:
                continue

            sorted_members = self._sort_probes(active_members)
            for name in sorted_members:
                val = self._current_values.get(name, 0.0)
                prev = self._previous_values.get(name, 0.0)
                delta = val - prev

                # Direction arrow
                if abs(delta) < 0.01:
                    arrow_ch = " "
                elif delta > 0:
                    arrow_ch = "↑"
                else:
                    arrow_ch = "↓"

                # Bar
                bar_width = 18
                filled = int(abs(val) * bar_width)
                filled = min(filled, bar_width)
                bar = "█" * filled + "░" * (bar_width - filled)

                color = "green" if val >= 0 else "red"
                sel = ">" if name == self._selected_probe else " "
                display_name = name[:10].ljust(10)

                tlist.mount(Static(
                    f"{sel} {display_name} [{color}]{bar}[/] {val:+.2f} {arrow_ch}",
                    classes="trait-row",
                ))

        self._update_sparkline()

    def _sort_probes(self, names: list[str]) -> list[str]:
        if self._sort_mode == "magnitude":
            return sorted(names, key=lambda n: abs(self._current_values.get(n, 0.0)), reverse=True)
        elif self._sort_mode == "change":
            return sorted(names, key=lambda n: abs(
                self._current_values.get(n, 0.0) - self._previous_values.get(n, 0.0)
            ), reverse=True)
        return sorted(names)

    def _update_sparkline(self) -> None:
        display = self.query_one("#sparkline-display", Static)
        if self._selected_probe and self._selected_probe in self._sparklines:
            spark = self._sparklines[self._selected_probe]
            display.update(f"[bold]{spark}[/] {self._selected_probe} (last 64 tok)")
        else:
            display.update("")
