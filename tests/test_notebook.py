"""saklas.notebook plot helpers + DataFrame coercion.

CPU-only, skipped wholesale when plotly/pandas aren't installed (the
``[notebook]`` extra is opt-in).  Each test asserts shape + the trace /
column structure that downstream rendering depends on, not the visual
appearance — plotly figures don't render in pytest anyway.
"""
from __future__ import annotations

import pytest
import torch

# Skip module-wide if the [notebook] extra isn't installed locally.
plotly = pytest.importorskip("plotly")
pd = pytest.importorskip("pandas")

# Import notebook surface AFTER the extras-check so pytest reports the
# skip cleanly when the deps are missing.
from saklas.notebook import (  # noqa: E402
    NotebookExtraNotInstalled,
    plot_alpha_sweep,
    plot_layer_norms,
    plot_probe_correlation,
    plot_trait_history,
    to_dataframe,
)
from saklas.core.profile import Profile  # noqa: E402
from saklas.core.results import (  # noqa: E402
    GenerationResult,
    ProbeReadings,
    ResultCollector,
)


# ---------------------------------------------------------------------------
# Test fixtures.
# ---------------------------------------------------------------------------


def _make_result(
    *,
    text: str = "ok",
    alpha: float = 0.0,
    probe_mean: float = 0.0,
    tok_per_sec: float = 50.0,
) -> GenerationResult:
    """Build a minimal ``GenerationResult`` with one probe reading + one vector."""
    return GenerationResult(
        text=text,
        tokens=[1, 2, 3],
        token_count=3,
        tok_per_sec=tok_per_sec,
        elapsed=0.06,
        readings={
            "honest": ProbeReadings(
                per_generation=[probe_mean - 0.05, probe_mean, probe_mean + 0.05],
                mean=probe_mean,
                std=0.05,
                min=probe_mean - 0.05,
                max=probe_mean + 0.05,
                delta_per_gen=0.0,
            ),
        },
        vectors={"honest.deceptive": alpha},
    )


def _make_profile(layers: dict[int, list[float]]) -> Profile:
    """Build a Profile from a per-layer values dict (cheap: no torch.norm)."""
    return Profile(
        {layer: torch.tensor(values, dtype=torch.float32) for layer, values in layers.items()},
        metadata={"method": "contrastive_pca"},
    )


# ---------------------------------------------------------------------------
# to_dataframe — accepted shapes.
# ---------------------------------------------------------------------------


class TestToDataFrame:
    def test_from_result_collector(self) -> None:
        rc = ResultCollector()
        rc.add(_make_result(alpha=0.0, probe_mean=0.1), alpha=0.0)
        rc.add(_make_result(alpha=0.3, probe_mean=0.4), alpha=0.3)

        df = to_dataframe(rc)
        assert "vector_honest.deceptive_alpha" in df.columns
        assert "probe_honest_mean" in df.columns
        assert len(df) == 2

    def test_from_list_of_results(self) -> None:
        results = [
            _make_result(alpha=0.0, probe_mean=0.1),
            _make_result(alpha=0.5, probe_mean=0.6),
        ]
        df = to_dataframe(results)
        assert "vector_honest.deceptive_alpha" in df.columns
        assert "probe_honest_mean" in df.columns
        assert len(df) == 2

    def test_from_list_of_dicts(self) -> None:
        rows = [
            {"alpha": 0.0, "probe_honest_mean": 0.1},
            {"alpha": 0.3, "probe_honest_mean": 0.4},
        ]
        df = to_dataframe(rows)
        assert list(df.columns) == ["alpha", "probe_honest_mean"]
        assert len(df) == 2

    def test_dataframe_passes_through(self) -> None:
        original = pd.DataFrame({"a": [1, 2, 3]})
        df = to_dataframe(original)
        assert df is original

    def test_empty_list_yields_empty_frame(self) -> None:
        df = to_dataframe([])
        assert df.empty

    def test_unsupported_source_raises(self) -> None:
        with pytest.raises(TypeError, match="unsupported source type"):
            to_dataframe("not a collector")  # type: ignore[arg-type]

    def test_mixed_list_raises(self) -> None:
        results = [_make_result(), {"text": "raw dict"}]
        with pytest.raises(TypeError, match="mixed types"):
            to_dataframe(results)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# plot_alpha_sweep — auto-detect, dual-axis trace count.
# ---------------------------------------------------------------------------


class TestPlotAlphaSweep:
    def test_basic_sweep_has_probe_and_metric_traces(self) -> None:
        rc = ResultCollector()
        for alpha, probe in [(0.0, 0.1), (0.3, 0.4), (0.6, 0.7)]:
            rc.add(_make_result(alpha=alpha, probe_mean=probe, tok_per_sec=45.0))

        fig = plot_alpha_sweep(rc)

        # One probe + tok_per_sec metric → 2 traces
        names = [t.name for t in fig.data]
        assert "honest" in names
        assert "tok_per_sec" in names
        # Dual-axis configured
        assert fig.layout.yaxis2.title.text == "tok_per_sec"

    def test_alpha_column_auto_detect_with_multiple_vectors(self) -> None:
        # Two vectors registered: honest swept, sycophantic fixed.
        rows = [
            {
                "vector_honest_alpha": 0.0,
                "vector_sycophantic_alpha": 0.5,
                "probe_honest_mean": 0.1,
                "tok_per_sec": 50.0,
            },
            {
                "vector_honest_alpha": 0.3,
                "vector_sycophantic_alpha": 0.5,
                "probe_honest_mean": 0.4,
                "tok_per_sec": 48.0,
            },
            {
                "vector_honest_alpha": 0.6,
                "vector_sycophantic_alpha": 0.5,
                "probe_honest_mean": 0.7,
                "tok_per_sec": 47.0,
            },
        ]
        fig = plot_alpha_sweep(pd.DataFrame(rows))
        # X-axis points at the swept column
        assert fig.layout.xaxis.title.text == "vector_honest_alpha"

    def test_explicit_alpha_column_overrides_detection(self) -> None:
        rows = [
            {"alpha_a": 0.0, "alpha_b": 0.0, "probe_x_mean": 0.1, "tok_per_sec": 50.0},
            {"alpha_a": 0.0, "alpha_b": 0.5, "probe_x_mean": 0.4, "tok_per_sec": 48.0},
        ]
        fig = plot_alpha_sweep(pd.DataFrame(rows), alpha_column="alpha_b")
        assert fig.layout.xaxis.title.text == "alpha_b"

    def test_empty_collector_raises(self) -> None:
        with pytest.raises(ValueError, match="empty DataFrame"):
            plot_alpha_sweep(ResultCollector())

    def test_no_alpha_column_raises(self) -> None:
        df = pd.DataFrame({"probe_x_mean": [0.1, 0.2], "tok_per_sec": [50.0, 48.0]})
        with pytest.raises(ValueError, match="no 'vector_<name>_alpha'"):
            plot_alpha_sweep(df)

    def test_ambiguous_alpha_columns_raise(self) -> None:
        rows = [
            {"vector_a_alpha": 0.0, "vector_b_alpha": 0.0, "probe_x_mean": 0.1, "tok_per_sec": 50.0},
            {"vector_a_alpha": 0.3, "vector_b_alpha": 0.5, "probe_x_mean": 0.4, "tok_per_sec": 48.0},
        ]
        with pytest.raises(ValueError, match="multiple alpha columns vary"):
            plot_alpha_sweep(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# plot_probe_correlation — N×N heatmap shape.
# ---------------------------------------------------------------------------


class TestPlotProbeCorrelation:
    def test_diagonal_is_one(self) -> None:
        profiles = {
            "p1": _make_profile({0: [1.0, 0.0], 1: [0.0, 1.0]}),
            "p2": _make_profile({0: [0.0, 1.0], 1: [1.0, 0.0]}),
        }
        fig = plot_probe_correlation(profiles)
        # Heatmap z-matrix: diagonal entries should be 1.0.
        z = fig.data[0].z
        assert z[0][0] == pytest.approx(1.0)
        assert z[1][1] == pytest.approx(1.0)

    def test_axis_labels_match_dict_order(self) -> None:
        profiles = {
            "alpha": _make_profile({0: [1.0, 0.0]}),
            "beta": _make_profile({0: [0.0, 1.0]}),
            "gamma": _make_profile({0: [1.0, 1.0]}),
        }
        fig = plot_probe_correlation(profiles)
        assert list(fig.data[0].x) == ["alpha", "beta", "gamma"]
        assert list(fig.data[0].y) == ["alpha", "beta", "gamma"]

    def test_symmetric_off_diagonal(self) -> None:
        profiles = {
            "a": _make_profile({0: [1.0, 0.0]}),
            "b": _make_profile({0: [0.5, 0.5]}),
        }
        fig = plot_probe_correlation(profiles)
        z = fig.data[0].z
        assert z[0][1] == pytest.approx(z[1][0])

    def test_empty_profiles_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one profile"):
            plot_probe_correlation({})


# ---------------------------------------------------------------------------
# plot_layer_norms.
# ---------------------------------------------------------------------------


class TestPlotLayerNorms:
    def test_one_bar_per_layer(self) -> None:
        profile = _make_profile({0: [1.0, 0.0], 5: [0.0, 1.0], 12: [1.0, 1.0]})
        fig = plot_layer_norms(profile)

        bar = fig.data[0]
        assert list(bar.x) == ["L0", "L5", "L12"]
        assert len(bar.y) == 3
        # Magnitudes computed correctly (values are unit-norm vectors).
        assert bar.y[0] == pytest.approx(1.0)
        assert bar.y[2] == pytest.approx(2.0**0.5)

    def test_empty_profile_constructor_rejects(self) -> None:
        # Profile itself rejects empty dicts; this confirms our plot
        # function never receives one in practice.
        from saklas.core.profile import ProfileError

        with pytest.raises(ProfileError):
            Profile({}, metadata={})


# ---------------------------------------------------------------------------
# plot_trait_history.
# ---------------------------------------------------------------------------


class TestPlotTraitHistory:
    def test_one_trace_per_probe(self) -> None:
        readings = {
            "honest": ProbeReadings(
                per_generation=[0.1, 0.2, 0.3],
                mean=0.2, std=0.1, min=0.1, max=0.3, delta_per_gen=0.1,
            ),
            "warm": ProbeReadings(
                per_generation=[0.5, 0.4, 0.3],
                mean=0.4, std=0.1, min=0.3, max=0.5, delta_per_gen=-0.1,
            ),
        }
        fig = plot_trait_history(readings)

        trace_names = [t.name for t in fig.data]
        # show_mean=True adds a dashed mean trace per probe (showlegend=False
        # but still in fig.data); the user-facing legend names land first.
        assert "honest" in trace_names
        assert "warm" in trace_names

    def test_show_mean_off_yields_one_trace_per_probe(self) -> None:
        readings = {
            "honest": ProbeReadings(
                per_generation=[0.1, 0.2, 0.3],
                mean=0.2, std=0.1, min=0.1, max=0.3, delta_per_gen=0.1,
            ),
        }
        fig = plot_trait_history(readings, show_mean=False)
        assert len(fig.data) == 1

    def test_empty_readings_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            plot_trait_history({})


# ---------------------------------------------------------------------------
# Missing-extra error class — surfaces correctly when import fails.
# ---------------------------------------------------------------------------


class TestNotebookExtraNotInstalled:
    def test_user_message_format(self) -> None:
        e = NotebookExtraNotInstalled("plotly")
        assert "saklas[notebook]" in str(e)
        assert "plotly" in str(e)
        # SaklasError MRO surface for the server-side error mapper.
        code, msg = e.user_message()
        assert code == 500
        assert "plotly" in msg

    def test_inherits_importerror(self) -> None:
        # ``except ImportError`` at user sites still catches the SaklasError.
        e = NotebookExtraNotInstalled("plotly")
        assert isinstance(e, ImportError)
