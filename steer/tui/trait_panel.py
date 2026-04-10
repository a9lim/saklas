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
        self._nav_items: list[tuple[str, str]] = []
        self._nav_idx: int = 0

    def compose(self) -> ComposeResult:
        yield Static(
            "[bold]TRAIT MONITOR[/] [dim]sort: name · Ctrl+S[/]",
            id="trait-header",
        )
        yield VerticalScroll(Static("", id="trait-content"), id="trait-scroll")
        yield Static("[dim]↑/↓ nav · Enter select/collapse · Ctrl+S sort[/]",
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
        self._nav_items = []
        lines: list[str] = []

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

            lines.append(
                f"{cat_marker}[bold]{arrow} {category}[/] [dim]({count})[/]"
            )

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

                if abs(delta) < 0.01:
                    arrow_ch = " "
                elif delta > 0:
                    arrow_ch = "↑"
                else:
                    arrow_ch = "↓"

                bar_width = 10
                filled = int(abs(val) * bar_width)
                filled = min(filled, bar_width)
                bar_full = "█" * filled
                bar_empty = "░" * (bar_width - filled)
                color = "green" if val >= 0 else "red"

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
                    hist = self._histories.get(name, [])
                    stats_line = self._compute_stats_line(hist)
                    lines.append(f"{line}\n  [dim]{stats_line}[/]")
                else:
                    lines.append(line)

        content = self.query_one("#trait-content", Static)
        content.update("\n".join(lines))

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
