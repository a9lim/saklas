"""Live trait monitor panel with inline sparklines and always-visible stats."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static
from textual.widget import Widget

from saklas.core.histogram import HIST_BUCKETS, bucketize
from saklas.tui.utils import BAR_WIDTH, build_bar

if TYPE_CHECKING:
    from saklas.core.results import ManifoldAggregate, ManifoldTokenReading


# Mini-map dimensions for 2-D BoxDomain manifolds (the canonical case
# is the bundled ``pad``).  Fits comfortably within the trait
# panel's column width — `MINIMAP_W = 17` matches BAR_WIDTH's visual
# weight without overflowing on narrow terminals; `MINIMAP_H = 7` gives
# enough vertical room to discriminate the four quadrants and the origin.
MINIMAP_W = 17
MINIMAP_H = 7


class TraitPanel(Widget):

    def __init__(self, categories: dict[str, list[str]] | None = None, **kwargs: Any) -> None:
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
        # Manifold-probe state.  ``_manifold_probes`` maps probe-name →
        # the attached ``Manifold`` artifact (needed for domain/axis
        # introspection at render time — bounds, intrinsic_dim).
        # ``_manifold_readings`` carries the latest per-token reading
        # streamed off the engine; ``_manifold_aggregates`` carries the
        # end-of-gen aggregate populated from
        # ``GenerationResult.manifold_readings`` at finalize.  Either
        # may be empty; rendering degrades cleanly.
        self._manifold_probes: dict[str, Any] = {}
        self._manifold_readings: dict[str, "ManifoldTokenReading"] = {}
        self._manifold_aggregates: dict[str, "ManifoldAggregate"] = {}
        self._cached_manifold_text: str = ""

    def compose(self) -> ComposeResult:
        yield Static(
            "[bold]MONITOR PROBES[/] [dim]sort: name[/]",
            id="trait-header", classes="section-header",
        )
        yield VerticalScroll(Static("", id="trait-content"), id="trait-scroll")
        yield Static(
            "", id="manifold-header", classes="section-header",
        )
        yield VerticalScroll(
            Static("", id="manifold-content"), id="manifold-scroll",
        )
        yield Static("[dim]⌫ remove · ⌃S sort · ⌃Y highlight[/]",
                      id="trait-hints")
        yield Static("[bold]WHY[/]", id="why-header", classes="section-header")
        yield VerticalScroll(Static("", id="why-content"), id="why-scroll")

    def on_mount(self) -> None:
        self._trait_header = self.query_one("#trait-header", Static)
        self._trait_content = self.query_one("#trait-content", Static)
        self._manifold_header = self.query_one("#manifold-header", Static)
        self._manifold_content = self.query_one("#manifold-content", Static)
        self._why_header = self.query_one("#why-header", Static)
        self._why_content = self.query_one("#why-content", Static)
        # Initial render so empty-state surfaces don't print None.
        self._render_manifold_probes()

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
            f"[bold]MONITOR PROBES[/] [dim]sort: {self._sort_mode}[/]"
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
        nav_idx = self._nav_idx
        cur = self._current_values
        prv = self._previous_values
        sparks = self._sparklines

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
                is_nav_selected = len(self._nav_items) == nav_idx
                self._nav_items.append(name)

                val = cur.get(name, 0.0)
                prev = prv.get(name, 0.0)
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

                bar_full, bar_empty = build_bar(val, 1.0, BAR_WIDTH)
                if val > 0:
                    color = "ansi_green"
                elif val < 0:
                    color = "ansi_red"
                else:
                    color = "ansi_default"

                mini_spark = sparks.get(name, "")

                sel = ">" if is_nav_selected else " "
                name_str = f"[bold]{name}[/]" if is_nav_selected else name
                spark_tail = f" [dim]{mini_spark}[/]" if mini_spark else ""

                line = (
                    f"{sel} {name_str}\n"
                    f"  [{color}]{bar_full}[/][dim]{bar_empty}[/] "
                    f"[{color}]{val:+.2f}{arrow_ch}[/]{spark_tail}"
                )

                lines.append(line)

        text = "\n".join(lines)
        if text != self._cached_render_text:
            self._cached_render_text = text
            self._trait_content.update(text)

    def update_why(
        self,
        probe: str | None,
        layer_norms: list[tuple[int, float]],
    ) -> None:
        if probe is None:
            self._why_header.update("")
            self._why_content.update("")
            return
        self._why_header.update("[bold]LAYERS[/]")
        lines: list[str] = []
        if layer_norms:
            buckets = bucketize(layer_norms, HIST_BUCKETS)
            max_norm = max(v for _, _, v in buckets) or 1.0
            label_w = max(2, len(str(max(hi for _, hi, _ in buckets))))
            has_range = any(lo != hi for lo, hi, _ in buckets)
            label_col = (2 * label_w + 2) if has_range else (label_w + 1)
            for lo, hi, norm in buckets:
                full, empty = build_bar(norm, max_norm, BAR_WIDTH)
                label = f"L{lo:0{label_w}}" if lo == hi else f"L{lo:0{label_w}}-{hi:0{label_w}}"
                lines.append(f"  {label:<{label_col}} [dim]{full}[/][dim]{empty}[/]")
        self._why_content.update("\n".join(lines))

    def _sort_probes(self, names: list[str]) -> list[str]:
        if self._sort_mode == "value":
            return sorted(names, key=lambda n: self._current_values.get(n, 0.0), reverse=True)
        elif self._sort_mode == "change":
            return sorted(names, key=lambda n: abs(
                self._current_values.get(n, 0.0) - self._previous_values.get(n, 0.0)
            ), reverse=True)
        return sorted(names)

    # ------------------------------------------------- manifold probes ---

    def set_active_manifold_probes(self, probes: dict[str, Any]) -> None:
        """Replace the attached-manifold-probe registry.

        ``probes`` maps probe name (the key the session registered) to the
        loaded :class:`~saklas.core.manifold.Manifold` artifact.  The panel
        introspects the domain at render time — ``intrinsic_dim`` to pick
        between mini-map (2-D) and label-only layout, plus per-axis bounds
        for the 2-D mini-map scaling.  Calling with an empty dict hides
        the section.
        """
        self._manifold_probes = dict(probes)
        # Drop stale readings whose probes are no longer attached so a
        # remove + re-add cycle doesn't surface ghost data from the old
        # registration.
        self._manifold_readings = {
            k: v for k, v in self._manifold_readings.items() if k in probes
        }
        self._manifold_aggregates = {
            k: v for k, v in self._manifold_aggregates.items() if k in probes
        }
        self._render_manifold_probes()

    def update_manifold_readings(
        self,
        per_token: dict[str, "ManifoldTokenReading"] | None = None,
        aggregates: dict[str, "ManifoldAggregate"] | None = None,
    ) -> None:
        """Push fresh manifold-probe data to the panel.

        ``per_token`` is the latest per-token reading streamed off the
        engine; mid-gen renders the live fraction bar + top-N nearest
        labels + (for 2-D BoxDomain manifolds) the coords mini-map with
        the running coord marker.  ``aggregates`` is the end-of-gen
        :class:`ManifoldAggregate` map populated from
        ``GenerationResult.manifold_readings`` at finalize; aggregates
        override per-token entries for the bar / nearest list and supply
        the inferred ``coords`` for the mini-map dot.  Either may be
        ``None`` — pass only what changed.
        """
        if per_token is not None:
            self._manifold_readings = dict(per_token)
        if aggregates is not None:
            self._manifold_aggregates = dict(aggregates)
        self._render_manifold_probes()

    def clear_manifold_readings(self) -> None:
        """Reset per-token + aggregate readings (e.g. on /clear)."""
        self._manifold_readings = {}
        self._manifold_aggregates = {}
        self._render_manifold_probes()

    def _render_manifold_probes(self) -> None:
        if not hasattr(self, "_manifold_header"):
            # Compose hasn't run yet — initial render after on_mount.
            return
        if not self._manifold_probes:
            self._manifold_header.update("")
            self._manifold_content.update("")
            self._cached_manifold_text = ""
            return
        self._manifold_header.update("[bold]MANIFOLD PROBES[/]")
        lines: list[str] = []
        for name in sorted(self._manifold_probes):
            manifold = self._manifold_probes[name]
            agg = self._manifold_aggregates.get(name)
            live = self._manifold_readings.get(name)
            # Aggregate beats live for the bar / nearest — it's the
            # EV-weighted mean over the generated tokens, more stable
            # than the last-token reading.
            fraction = (
                agg.fraction_mean if agg is not None
                else (live.fraction if live is not None else 0.0)
            )
            nearest = (
                agg.nearest if agg is not None
                else (live.nearest if live is not None else [])
            )
            if math.isnan(fraction):
                fraction = 0.0
            bar_full, bar_empty = build_bar(fraction, 1.0, BAR_WIDTH)
            lines.append(f" [bold]{name}[/]")
            lines.append(
                f"  [ansi_cyan]{bar_full}[/][dim]{bar_empty}[/] "
                f"[ansi_cyan]{fraction:.2f}[/]"
            )
            if nearest:
                # Top-3 labels with their distances.  Names can be long;
                # tab/space delimiters keep them aligned without an
                # alignment pass that'd cost more than it saves on the
                # short label lists this widget typically gets.
                pieces = [
                    f"{label} [dim]({dist:.2f})[/]"
                    for label, dist in nearest[:3]
                ]
                lines.append("  " + "  ".join(pieces))
            # 2-D BoxDomain → mini-map.  Higher-dim manifolds skip it —
            # ASCII visual loses meaning past two intrinsic axes and the
            # nearest-label list already covers the discrete-token case.
            mini = self._render_minimap(manifold, agg, live)
            if mini:
                lines.append(mini)
        text = "\n".join(lines)
        if text != self._cached_manifold_text:
            self._cached_manifold_text = text
            self._manifold_content.update(text)

    def _render_minimap(
        self,
        manifold: Any,
        agg: "ManifoldAggregate | None",
        live: "ManifoldTokenReading | None",
    ) -> str:
        """ASCII mini-map of a 2-D BoxDomain manifold's coord plane.

        Returns an empty string for any other domain shape.  The map
        plots node positions as ``·`` markers and the inferred coord
        (from the aggregate, falling back to the live reading's
        ``nearest[0]`` node coord) as ``●`` in ``ansi_cyan``.  Box
        axes carry per-axis ``(lo, hi)`` bounds — the coord is mapped
        through those bounds onto the ``MINIMAP_W x MINIMAP_H`` grid.
        ``CustomDomain`` / ``SphereDomain`` map skipped (no meaningful
        rectangular layout).
        """
        domain = getattr(manifold, "domain", None)
        if domain is None:
            return ""
        if getattr(domain, "intrinsic_dim", 0) != 2:
            return ""
        axes = getattr(domain, "axes", None)
        if axes is None or len(axes) != 2:
            return ""
        # Resolve per-axis (lo, hi).  Periodic axes use [0, period].
        def _axis_bounds(ax: Any) -> tuple[float, float]:
            if getattr(ax, "periodic", False):
                return 0.0, float(getattr(ax, "period", 1.0))
            return float(getattr(ax, "lo", 0.0)), float(getattr(ax, "hi", 1.0))
        x_lo, x_hi = _axis_bounds(axes[0])
        y_lo, y_hi = _axis_bounds(axes[1])
        if not (math.isfinite(x_lo) and math.isfinite(x_hi)
                and math.isfinite(y_lo) and math.isfinite(y_hi)):
            return ""
        if x_hi <= x_lo or y_hi <= y_lo:
            return ""

        def _project(x: float, y: float) -> tuple[int, int]:
            """Map (x, y) in axis bounds → (col, row) in the grid.

            Row 0 is the top of the printed map (highest y), so the
            row axis is flipped.  Out-of-range coords clamp to the
            edge.
            """
            cx = (x - x_lo) / (x_hi - x_lo)
            cy = (y - y_lo) / (y_hi - y_lo)
            col = max(0, min(MINIMAP_W - 1, int(round(cx * (MINIMAP_W - 1)))))
            row = max(0, min(MINIMAP_H - 1, int(round((1.0 - cy) * (MINIMAP_H - 1)))))
            return col, row

        # Build the grid with node markers.
        grid = [[" " for _ in range(MINIMAP_W)] for _ in range(MINIMAP_H)]
        coords = getattr(manifold, "node_coords", None)
        if coords is not None:
            try:
                rows = coords.tolist()
            except AttributeError:
                rows = list(coords)
            for row in rows:
                if len(row) < 2:
                    continue
                c, r = _project(float(row[0]), float(row[1]))
                grid[r][c] = "·"

        # Inferred coord — aggregate first, else look up the live
        # nearest node's authoring coords.
        coord_dot: tuple[int, int] | None = None
        if agg is not None and agg.coords and len(agg.coords) >= 2:
            coord_dot = _project(float(agg.coords[0]), float(agg.coords[1]))
        elif live is not None and live.nearest:
            label = live.nearest[0][0]
            labels = getattr(manifold, "node_labels", None) or []
            try:
                idx = labels.index(label)
            except ValueError:
                idx = -1
            if idx >= 0 and coords is not None:
                try:
                    row = coords[idx].tolist()
                except AttributeError:
                    row = list(coords[idx])
                if len(row) >= 2:
                    coord_dot = _project(float(row[0]), float(row[1]))

        out_lines: list[str] = []
        for r, row in enumerate(grid):
            if coord_dot is not None and r == coord_dot[1]:
                cells: list[str] = []
                for c, ch in enumerate(row):
                    if c == coord_dot[0]:
                        cells.append("[ansi_cyan]●[/]")
                    else:
                        cells.append(ch)
                out_lines.append("  " + "".join(cells))
            else:
                out_lines.append("  " + "".join(row))
        return "\n".join(out_lines)
