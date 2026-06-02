"""4.0 step 6c-2 — the manifold-fold fallback for the disk-side inspection verbs.

Bundled & user concepts ship as 2-node ``pca`` manifolds, so ``subspace
compare`` / ``subspace why`` (the verbs that read baked tensors off disk
without a model) fall back to folding a **fitted** manifold tensor when no
``vectors/`` tensor resolves.  These tests synthesize a fitted 2-node manifold
on disk (no model) and exercise the CLI runners end-to-end.
"""
from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
import torch

from saklas import cli
from saklas.cli.runners import _run_compare, _run_why
from saklas.core.manifold import save_manifold
from saklas.core.vectors import _fold_centroids_to_affine_manifold
from saklas.io.paths import manifold_dir, tensor_filename

_MODEL = "test/model"


@pytest.fixture(autouse=True)
def _isolated_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Generator[None, None, None]:
    from saklas.io import selectors as _sel
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _sel.invalidate()
    yield
    _sel.invalidate()


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


def _write_fitted_manifold(
    ns: str, name: str, *, seed: int = 0, d: int = 8,
) -> Path:
    """Synthesize a fitted 2-node affine manifold tensor on disk (no model)."""
    g = torch.Generator().manual_seed(seed)
    d0, d1 = _unit(torch.randn(d, generator=g)), _unit(torch.randn(d, generator=g))
    pos = {2: 5.0 + 1.1 * d0, 5: 3.0 + 0.6 * d1}
    neg = {2: 5.0 - 1.1 * d0, 5: 3.0 - 0.6 * d1}
    neutral = {2: 5.0 + 0.2 * d0, 5: 3.0 - 0.1 * d1}
    pos_label, neg_label = (
        name.split(".", 1) if "." in name else (name, f"{name}_neg")
    )
    mfld = _fold_centroids_to_affine_manifold(
        name, pos, neg, pos_label=pos_label, neg_label=neg_label,
        layer_means=neutral,
    )
    folder = manifold_dir(ns, name)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / tensor_filename(_MODEL, release=None)
    save_manifold(mfld, path, {"method": "folded_vector",
                               "share_metric": mfld.metadata["share_metric"]})
    return path


# ---------------------------------------------------------------------------
# the io-level fold helper
# ---------------------------------------------------------------------------

class TestFoldHelper:
    def test_fold_returns_profile_for_fitted_manifold(self) -> None:
        from saklas.cli.runners import _fold_manifold_to_profile

        _write_fitted_manifold("default", "happy.sad")
        prof = _fold_manifold_to_profile("happy.sad", _MODEL, None)
        assert prof is not None
        assert sorted(prof.layers) == [2, 5]

    def test_fold_returns_none_when_unfitted(self) -> None:
        from saklas.cli.runners import _fold_manifold_to_profile
        # No tensor on disk for this model → miss (caller nudges to fit).
        assert _fold_manifold_to_profile("happy.sad", _MODEL, None) is None

    def test_fold_bare_name_collision_raises(self) -> None:
        from saklas.cli.runners import _fold_manifold_to_profile
        from saklas.io.selectors import AmbiguousSelectorError

        _write_fitted_manifold("default", "happy.sad")
        _write_fitted_manifold("alice", "happy.sad")
        with pytest.raises(AmbiguousSelectorError):
            _fold_manifold_to_profile("happy.sad", _MODEL, None)
        # Namespace-qualified resolves cleanly.
        assert _fold_manifold_to_profile("alice/happy.sad", _MODEL, None) is not None

    def test_fold_all_fitted_excludes_target(self) -> None:
        from saklas.cli.runners import _fold_all_fitted_manifolds

        _write_fitted_manifold("default", "happy.sad", seed=1)
        _write_fitted_manifold("default", "warm.clinical", seed=2)
        pool = _fold_all_fitted_manifolds(_MODEL, exclude="happy.sad")
        assert "warm.clinical" in pool
        assert "happy.sad" not in pool


# ---------------------------------------------------------------------------
# subspace why — folds a fitted manifold
# ---------------------------------------------------------------------------

class TestWhyFold:
    def test_why_on_fitted_manifold(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_fitted_manifold("default", "happy.sad")
        args = cli.parse_args(["subspace", "why", "happy.sad", "-m", _MODEL])
        _run_why(args)
        out = capsys.readouterr().out
        assert "happy.sad" in out

    def test_why_unfitted_manifold_nudges_to_fit(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Manifold folder exists but no fitted tensor for the model.
        manifold_dir("default", "happy.sad").mkdir(parents=True, exist_ok=True)
        args = cli.parse_args(["subspace", "why", "happy.sad", "-m", _MODEL])
        with pytest.raises(SystemExit):
            _run_why(args)
        err = capsys.readouterr().err
        assert "fit" in err

    def test_why_json_on_fitted_manifold(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        import json
        _write_fitted_manifold("default", "happy.sad")
        args = cli.parse_args(["subspace", "why", "happy.sad", "-m", _MODEL, "-j"])
        _run_why(args)
        payload = json.loads(capsys.readouterr().out)
        assert payload["concept"] == "happy.sad"
        assert {row["layer"] for row in payload["layers"]} == {2, 5}


# ---------------------------------------------------------------------------
# subspace compare — folds fitted manifolds (named + 1-arg rank-all)
# ---------------------------------------------------------------------------

class TestCompareFold:
    def test_compare_two_fitted_manifolds(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_fitted_manifold("default", "happy.sad", seed=1)
        _write_fitted_manifold("default", "warm.clinical", seed=2)
        args = cli.parse_args([
            "subspace", "compare", "happy.sad", "warm.clinical",
            "-m", _MODEL, "--metric", "euclidean",
        ])
        _run_compare(args)
        out = capsys.readouterr().out
        assert "happy.sad" in out
        assert "warm.clinical" in out

    def test_compare_one_arg_ranks_fitted_manifolds(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_fitted_manifold("default", "happy.sad", seed=1)
        _write_fitted_manifold("default", "warm.clinical", seed=2)
        _write_fitted_manifold("default", "angry.calm", seed=3)
        args = cli.parse_args([
            "subspace", "compare", "happy.sad",
            "-m", _MODEL, "--metric", "euclidean",
        ])
        _run_compare(args)
        out = capsys.readouterr().out
        # 1-arg mode ranks the others (folded manifolds) against the target.
        assert "warm.clinical" in out
        assert "angry.calm" in out
