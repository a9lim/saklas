"""Plotly figure builders for saklas data structures.

Each function returns a ``plotly.graph_objects.Figure`` rendered live in
Jupyter, exportable to standalone HTML via ``.write_html(...)`` or PNG
via ``.write_image(...)`` (PNG export needs the optional ``kaleido``
package, which the ``[notebook]`` extra pulls in).

Plotly is imported lazily — bare ``import saklas.notebook`` succeeds even
without the extra installed; the import error surfaces only when one of
the public functions is actually called.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from saklas.core.errors import SaklasError

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go

    from saklas.core.profile import Profile
    from saklas.core.results import ProbeReadings, ResultCollector


class NotebookExtraNotInstalled(ImportError, SaklasError):
    """Raised when a notebook helper is called without the ``[notebook]`` extra.

    ``message`` carries the missing dependency name; the install hint is
    appended automatically so users see one consistent remediation line
    across plotly / pandas / kaleido.
    """

    def __init__(self, missing: str) -> None:
        super().__init__(
            f"saklas.notebook needs '{missing}' but it is not installed. "
            f"Install the extra: pip install 'saklas[notebook]'"
        )

    def user_message(self) -> tuple[int, str]:
        return (500, str(self))


# ---------------------------------------------------------------------------
# Lazy dependency loaders.
# ---------------------------------------------------------------------------


def _require_plotly() -> Any:
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError:
        raise NotebookExtraNotInstalled("plotly") from None
    return go


# ---------------------------------------------------------------------------
# plot_alpha_sweep
# ---------------------------------------------------------------------------


def _detect_alpha_column(df: "pd.DataFrame") -> str:
    """Find the unique ``vector_<name>_alpha`` column on a sweep DataFrame.

    ``ResultCollector`` writes one ``vector_<name>_alpha`` column per
    registered vector.  When a sweep moves a single alpha and holds the
    others fixed, the moving one is the unique column with >1 distinct
    value — a robust auto-detection signal that's resilient to users
    adding extra registered vectors at fixed alpha.

    Falls back to "the only ``vector_*_alpha`` column" when the variance
    heuristic ties or finds none (e.g. a one-row DataFrame), and raises
    when no candidate exists.
    """
    candidates = [c for c in df.columns if c.startswith("vector_") and c.endswith("_alpha")]
    if not candidates:
        raise ValueError(
            "plot_alpha_sweep: no 'vector_<name>_alpha' column on the "
            "DataFrame. Ensure each row was added through ResultCollector."
        )
    if len(candidates) == 1:
        return candidates[0]
    # Pick the column whose values vary across rows.  ``nunique`` handles
    # NaN gracefully (counted as a single class) and is well-defined on
    # empty frames (returns 0).
    moving = [c for c in candidates if df[c].nunique(dropna=False) > 1]
    if len(moving) == 1:
        return moving[0]
    if len(moving) > 1:
        raise ValueError(
            f"plot_alpha_sweep: multiple alpha columns vary across rows "
            f"({', '.join(moving)}); pass alpha_column=... explicitly."
        )
    # Nothing varies — single-row DataFrame, or every alpha was fixed.
    # Pick the first candidate; the resulting figure has one bar per
    # probe and one secondary point, which is still informative.
    return candidates[0]


def plot_alpha_sweep(
    source: "ResultCollector | list[Any] | pd.DataFrame",
    *,
    alpha_column: str | None = None,
    metric: str = "tok_per_sec",
    title: str | None = None,
) -> "go.Figure":
    """Dual-axis line chart: alpha → probe means + a secondary metric.

    ``source`` accepts the same shapes as :func:`saklas.notebook.to_dataframe`.
    The alpha column is auto-detected when not specified — see
    :func:`_detect_alpha_column` for the rule.

    Probe traces (left axis) are drawn for every ``probe_<name>_mean``
    column on the DataFrame; one line per probe, sorted by name.  The
    secondary metric (right axis) defaults to ``tok_per_sec``; pass any
    other column name on your DataFrame (e.g. ``perplexity`` if you
    tagged each row with one through ``ResultCollector.add(...,
    perplexity=...)``).

    Returns a :class:`plotly.graph_objects.Figure` with no rendering side
    effects — call ``.show()`` / ``.write_html(...)`` / ``.write_image(...)``
    on the result.
    """
    go = _require_plotly()
    from saklas.notebook.data import to_dataframe

    df = to_dataframe(source).copy()
    if df.empty:
        raise ValueError("plot_alpha_sweep: empty DataFrame")

    alpha_col = alpha_column or _detect_alpha_column(df)
    df = df.sort_values(alpha_col).reset_index(drop=True)

    probe_cols = sorted(c for c in df.columns if c.startswith("probe_") and c.endswith("_mean"))

    fig = go.Figure()
    for col in probe_cols:
        # "probe_<name>_mean" → "<name>" — strip the prefix/suffix exactly.
        probe_name = col[len("probe_"):-len("_mean")]
        fig.add_trace(go.Scatter(
            x=df[alpha_col],
            y=df[col],
            mode="lines+markers",
            name=probe_name,
            yaxis="y",
        ))

    if metric in df.columns:
        fig.add_trace(go.Scatter(
            x=df[alpha_col],
            y=df[metric],
            mode="lines+markers",
            name=metric,
            yaxis="y2",
            line={"dash": "dash"},
        ))

    fig.update_layout(
        title=title or f"Alpha sweep: {alpha_col}",
        xaxis={"title": alpha_col},
        yaxis={"title": "probe mean"},
        yaxis2={
            "title": metric,
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


# ---------------------------------------------------------------------------
# plot_probe_correlation
# ---------------------------------------------------------------------------


def plot_probe_correlation(
    profiles: "dict[str, Profile]",
    *,
    title: str = "Probe correlation (magnitude-weighted cosine)",
    color_scale: str = "RdBu",
) -> "go.Figure":
    """N×N magnitude-weighted cosine heatmap across profiles.

    Diagonal is 1.0; off-diagonals are ``Profile.cosine_similarity(...)``
    aggregated across shared layers.  Profiles with no shared layers
    render as ``NaN`` cells (plotly handles missing values cleanly).

    Names are rendered in dict-iteration order along both axes so callers
    can pre-sort by category (e.g. affect/epistemic/...).
    """
    go = _require_plotly()
    if not profiles:
        raise ValueError("plot_probe_correlation: at least one profile required")

    names = list(profiles.keys())
    n = len(names)
    matrix: list[list[float]] = [[float("nan")] * n for _ in range(n)]
    for i, a_name in enumerate(names):
        for j, b_name in enumerate(names):
            if j < i:
                # Symmetric — copy from upper triangle.
                matrix[i][j] = matrix[j][i]
                continue
            if i == j:
                matrix[i][j] = 1.0
                continue
            try:
                # ``cosine_similarity`` without ``per_layer=`` returns
                # the magnitude-weighted aggregate ``float`` — narrow
                # explicitly because the method's union return type is
                # ``float | dict[int, float]``.
                agg = profiles[a_name].cosine_similarity(profiles[b_name])
                cos = float(agg) if isinstance(agg, (int, float)) else float("nan")
            except Exception:
                cos = float("nan")
            matrix[i][j] = cos

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=names,
        y=names,
        zmin=-1.0,
        zmax=1.0,
        colorscale=color_scale,
        colorbar={"title": "cosine"},
        hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis={"side": "bottom", "tickangle": -45},
        yaxis={"autorange": "reversed"},  # row 0 at top, conventional matrix layout
        # Square aspect at small N; plotly handles autosize otherwise.
        height=max(400, 24 * n + 100),
        width=max(400, 24 * n + 200),
    )
    return fig


# ---------------------------------------------------------------------------
# plot_layer_norms
# ---------------------------------------------------------------------------


def plot_layer_norms(
    profile: "Profile",
    *,
    title: str | None = None,
    color: str = "steelblue",
) -> "go.Figure":
    """Per-layer ``||baked||`` bar chart.

    Full resolution — one bar per retained layer.  This is the
    notebook-side counterpart to the TUI's 16-bucket histogram footer
    and the ``saklas vector why`` output, surfaced here without bucketing
    so users can see the exact distribution.
    """
    go = _require_plotly()

    layers = sorted(profile.keys())
    if not layers:
        raise ValueError("plot_layer_norms: profile has no layers")

    norms = [float(profile[layer].norm().item()) for layer in layers]

    fig = go.Figure(data=go.Bar(
        x=[f"L{layer}" for layer in layers],
        y=norms,
        marker_color=color,
        hovertemplate="layer %{x}: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title or f"Layer norms ({len(layers)} retained layers)",
        xaxis={"title": "layer"},
        yaxis={"title": "||baked||"},
    )
    return fig


# ---------------------------------------------------------------------------
# plot_trait_history
# ---------------------------------------------------------------------------


def plot_trait_history(
    readings: "dict[str, ProbeReadings]",
    *,
    title: str = "Trait history",
    show_mean: bool = True,
) -> "go.Figure":
    """Per-probe ``per_generation`` timeline as overlaid lines.

    One trace per probe; x-axis is the generation index (0..N-1 within
    the readings list), y-axis is the cosine reading.  When
    ``show_mean=True`` the per-probe mean lands as a horizontal dashed
    line in the same color so users can eyeball drift against baseline.

    Click a legend entry to isolate a single probe.
    """
    go = _require_plotly()

    if not readings:
        raise ValueError("plot_trait_history: readings dict is empty")

    fig = go.Figure()
    for probe_name, r in readings.items():
        series = list(r.per_generation)
        x = list(range(len(series)))
        fig.add_trace(go.Scatter(
            x=x,
            y=series,
            mode="lines+markers",
            name=probe_name,
            hovertemplate=f"{probe_name} step %{{x}}: %{{y:.3f}}<extra></extra>",
        ))
        if show_mean and series:
            fig.add_trace(go.Scatter(
                x=[0, len(series) - 1] if len(series) > 1 else [0, 0],
                y=[r.mean, r.mean],
                mode="lines",
                name=f"{probe_name} mean",
                line={"dash": "dash"},
                showlegend=False,
                hoverinfo="skip",
                opacity=0.4,
            ))

    fig.update_layout(
        title=title,
        xaxis={"title": "generation step"},
        yaxis={"title": "probe cosine"},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


