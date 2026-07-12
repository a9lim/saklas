"""4.0 step 6c-2 — the manifold-fold fallback for the disk-side inspection verbs.

Bundled & user concepts ship as 2-node ``pca`` manifolds, so ``manifold
compare`` / ``manifold why`` (the verbs that read baked tensors off disk
without a model) fall back to folding a **fitted** manifold tensor when no
``vectors/`` tensor resolves.  These tests synthesize a fitted 2-node manifold
on disk (no model) and exercise the CLI runners end-to-end.

The legacy bipolar-centroid fold (``_fold_centroids_to_affine_manifold``) was
retired in the Mahalanobis-only collapse; the fitted manifold here is built via
the production ``fit_affine_subspace`` primitive.  ``manifold compare`` is now
Mahalanobis-only, so the fixture seeds a per-model neutral cache on disk for the
runner's ``LayerWhitener.from_cache`` build.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Generator, cast

import pytest
import torch

from saklas import cli
from saklas.cli.runners import _run_manifold_compare, _run_manifold_why
from saklas.core.manifold import (
    MANIFOLD_FIT_POLICY_VERSION, CustomDomain, Manifold,
    fit_affine_subspace, save_manifold, subspace_share,
)
from saklas.io.paths import manifold_dir, model_dir, tensor_filename

_MODEL = "test/model"
_LAYERS = (2, 5)
_DIM = 8


@pytest.fixture(autouse=True)
def _isolated_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Generator[None, None, None]:
    from saklas.io import selectors as _sel
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    _sel.invalidate()
    _seed_neutral_cache(_MODEL)
    yield
    _sel.invalidate()


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


def _seed_neutral_cache(model_id: str, *, n: int = 64, seed: int = 5) -> None:
    """Write a per-model neutral-activation disk cache.

    ``manifold compare`` builds its (mandatory) whitener via
    ``LayerWhitener.from_cache(model_id)`` — there is no Euclidean path — so the
    sharded disk cache must exist for the layers the folded manifolds occupy. The
    probe-centering mean is derived from these activations (``X.mean(0)``), so
    there is no separate ``layer_means`` cache to seed.
    """
    import json

    from safetensors.torch import save_file
    from saklas.io.packs import hash_file

    md = model_dir(model_id)
    md.mkdir(parents=True, exist_ok=True)
    acts: dict[str, torch.Tensor] = {}
    for L in _LAYERS:
        g = torch.Generator().manual_seed(seed * 13 + L)
        scale = 0.5 + torch.rand(_DIM, generator=g, dtype=torch.float32)
        mu = torch.randn(_DIM, generator=g, dtype=torch.float32) * 0.1
        X = torch.randn(n, _DIM, generator=g, dtype=torch.float32) * scale + mu
        acts[f"layer_{L}"] = X
    tensor_files: dict[str, str] = {}
    tensor_sha256: dict[str, str] = {}
    for layer in _LAYERS:
        tensor_path = md / f"neutral_activations.layer-{layer}.gen-test.safetensors"
        save_file({f"layer_{layer}": acts[f"layer_{layer}"]}, str(tensor_path))
        tensor_files[str(layer)] = tensor_path.name
        tensor_sha256[str(layer)] = hash_file(tensor_path)
    (md / "neutral_activations.json").write_text(json.dumps({
        "method": "neutral_activations",
        "format_version": 3,
        "capture_version": 1,
        "capture_sha256": "test-capture",
        "model_fingerprint": "test-fingerprint",
        "model_source_fingerprint": "test-source",
        "tensor_sha256": tensor_sha256,
        "tensor_files": tensor_files,
        "layers": list(_LAYERS),
        "tensor_schema": {
            str(layer): {
                "shape": list(acts[f"layer_{layer}"].shape),
                "dtype": "torch.float32",
            }
            for layer in _LAYERS
        },
        "n_prompts": n,
        "n_layers": len(_LAYERS),
    }))


def _write_fitted_manifold(
    ns: str, name: str, *, seed: int = 0, d: int = _DIM,
) -> Path:
    """Synthesize a fitted 2-node affine manifold tensor on disk (no model).

    Built via the production ``fit_affine_subspace`` per layer — node 0 = pos,
    node 1 = neg — exactly as ``ManifoldExtractionPipeline`` does for a 2-node
    ``pca`` fit, so the on-disk shape matches a real fit.
    """
    g = torch.Generator().manual_seed(seed)
    d0, d1 = _unit(torch.randn(d, generator=g)), _unit(torch.randn(d, generator=g))
    pos = {2: 5.0 + 1.1 * d0, 5: 3.0 + 0.6 * d1}
    neg = {2: 5.0 - 1.1 * d0, 5: 3.0 - 0.6 * d1}
    neutral = {2: 5.0 + 0.2 * d0, 5: 3.0 - 0.1 * d1}
    pos_label, neg_label = (
        name.split(".", 1) if "." in name else (name, f"{name}_neg")
    )
    layers = {}
    share = {}
    for L in _LAYERS:
        cent = torch.stack([pos[L].float(), neg[L].float()])  # (2, D), node 0 = pos
        sub, mu_coords, _ev = fit_affine_subspace(
            cent, neutral_mean=neutral[L].float(), orient_to=0,
        )
        layers[L] = sub
        share[L] = subspace_share(mu_coords, sub.basis)  # Euclidean spread (no whitener)
    mfld = Manifold(
        name=name,
        domain=CustomDomain(1),
        node_labels=[pos_label, neg_label],
        node_coords=torch.tensor([[1.0], [-1.0]]),
        layers=layers,
        mahalanobis_share=share,
    )
    folder = manifold_dir(ns, name)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / tensor_filename(_MODEL, release=None)
    metadata: dict[str, object] = {
        "method": "folded_vector", "share_metric": "euclidean",
        "model_fingerprint": "test-fingerprint",
    }
    manifest = folder / "manifold.json"
    if manifest.exists():
        from saklas.io.manifolds import ManifoldFolder

        metadata["nodes_sha256"] = ManifoldFolder.load(
            folder, verify_manifest=False,
        ).nodes_sha256()
        metadata["fit_policy_version"] = MANIFOLD_FIT_POLICY_VERSION
    save_manifold(mfld, path, metadata)
    return path


# ---------------------------------------------------------------------------
# the io-level fold helper
# ---------------------------------------------------------------------------

class TestFoldHelper:
    def test_fold_returns_profile_for_fitted_manifold(self) -> None:
        from saklas.cli.runners import _fold_manifold_to_profile_with_identity

        _write_fitted_manifold("default", "happy.sad")
        folded = _fold_manifold_to_profile_with_identity("happy.sad", _MODEL, None)
        assert folded is not None
        prof, ns, bare = folded
        assert sorted(prof.layers) == [2, 5]
        assert (ns, bare) == ("default", "happy.sad")

    def test_fold_returns_none_when_unfitted(self) -> None:
        from saklas.cli.runners import _fold_manifold_to_profile_with_identity
        # No tensor on disk for this model → miss (caller nudges to fit).
        assert (
            _fold_manifold_to_profile_with_identity("happy.sad", _MODEL, None)
            is None
        )

    def test_fold_bare_name_collision_raises(self) -> None:
        from saklas.cli.runners import _fold_manifold_to_profile_with_identity
        from saklas.io.selectors import AmbiguousSelectorError

        _write_fitted_manifold("default", "happy.sad")
        _write_fitted_manifold("alice", "happy.sad")
        with pytest.raises(AmbiguousSelectorError):
            _fold_manifold_to_profile_with_identity("happy.sad", _MODEL, None)
        # Namespace-qualified resolves cleanly.
        folded = _fold_manifold_to_profile_with_identity(
            "alice/happy.sad", _MODEL, None,
        )
        assert folded is not None
        assert folded[1:] == ("alice", "happy.sad")

    def test_fold_all_fitted_excludes_target(self) -> None:
        from saklas.cli.runners import _fold_all_fitted_manifolds

        _write_fitted_manifold("default", "happy.sad", seed=1)
        _write_fitted_manifold("default", "warm.clinical", seed=2)
        pool = _fold_all_fitted_manifolds(
            _MODEL, exclude_identity=("default", "happy.sad"),
        )
        assert "default/warm.clinical" in pool
        assert "default/happy.sad" not in pool

    def test_fold_all_fitted_preserves_namespace_collisions(self) -> None:
        from saklas.cli.runners import _fold_all_fitted_manifolds

        _write_fitted_manifold("default", "happy.sad", seed=1)
        _write_fitted_manifold("alice", "happy.sad", seed=2)
        pool = _fold_all_fitted_manifolds(_MODEL)
        assert set(pool) == {"alice/happy.sad", "default/happy.sad"}


# ---------------------------------------------------------------------------
# manifold why — folds a fitted manifold
# ---------------------------------------------------------------------------

class TestWhyFold:
    def test_why_on_fitted_manifold(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_fitted_manifold("default", "happy.sad")
        args = cli.parse_args(["manifold", "why", "happy.sad", "-m", _MODEL])
        _run_manifold_why(args)
        out = capsys.readouterr().out
        assert "happy.sad" in out

    def test_why_unfitted_manifold_nudges_to_fit(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Manifold folder exists but no fitted tensor for the model.
        manifold_dir("default", "happy.sad").mkdir(parents=True, exist_ok=True)
        args = cli.parse_args(["manifold", "why", "happy.sad", "-m", _MODEL])
        with pytest.raises(SystemExit):
            _run_manifold_why(args)
        err = capsys.readouterr().err
        assert "fit" in err

    def test_why_json_on_fitted_manifold(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        import json
        _write_fitted_manifold("default", "happy.sad")
        args = cli.parse_args(["manifold", "why", "happy.sad", "-m", _MODEL, "-j"])
        _run_manifold_why(args)
        payload = json.loads(capsys.readouterr().out)
        assert payload["concept"] == "happy.sad"
        assert {row["layer"] for row in payload["layers"]} == {2, 5}

    def test_why_json_keeps_explicit_namespace(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        import json
        _write_fitted_manifold("alice", "happy.sad")
        args = cli.parse_args([
            "manifold", "why", "alice/happy.sad", "-m", _MODEL, "-j",
        ])
        _run_manifold_why(args)
        payload = json.loads(capsys.readouterr().out)
        assert payload["concept"] == "alice/happy.sad"


# ---------------------------------------------------------------------------
# manifold compare — folds fitted manifolds (named + 1-arg rank-all).
# Mahalanobis-only: the fixture seeds the per-model neutral cache.
# ---------------------------------------------------------------------------

class TestCompareFold:
    def test_compare_two_fitted_manifolds(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_fitted_manifold("default", "happy.sad", seed=1)
        _write_fitted_manifold("default", "warm.clinical", seed=2)
        args = cli.parse_args([
            "manifold", "compare", "happy.sad", "warm.clinical", "-m", _MODEL,
        ])
        _run_manifold_compare(args)
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
            "manifold", "compare", "happy.sad", "-m", _MODEL,
        ])
        _run_manifold_compare(args)
        out = capsys.readouterr().out
        # 1-arg mode ranks the others (folded manifolds) against the target.
        assert "warm.clinical" in out
        assert "angry.calm" in out

    def test_compare_explicit_namespace_collision_keeps_both(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_fitted_manifold("default", "happy.sad", seed=1)
        _write_fitted_manifold("alice", "happy.sad", seed=2)
        args = cli.parse_args([
            "manifold", "compare", "default/happy.sad", "alice/happy.sad",
            "-m", _MODEL,
        ])
        _run_manifold_compare(args)
        out = capsys.readouterr().out
        assert "default/happy.sad" in out
        assert "alice/happy.sad" in out

    def test_compare_one_arg_rank_all_keeps_namespace_collision(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_fitted_manifold("default", "happy.sad", seed=1)
        _write_fitted_manifold("alice", "warm.clinical", seed=2)
        _write_fitted_manifold("default", "warm.clinical", seed=3)
        args = cli.parse_args([
            "manifold", "compare", "happy.sad", "-m", _MODEL,
        ])
        _run_manifold_compare(args)
        out = capsys.readouterr().out
        assert "alice/warm.clinical" in out
        assert "default/warm.clinical" in out


def _make_full_manifold(ns: str, name: str, *, seed: int = 0) -> Path:
    """A complete manifold folder (manifold.json + nodes) plus a fitted tensor."""
    from saklas.io.manifolds import (
        create_discover_manifold_folder, ManifoldFolder, hash_manifold_files,
    )
    pos_label, neg_label = (
        name.split(".", 1) if "." in name else (name, f"{name}_neg")
    )
    create_discover_manifold_folder(
        ns, name, f"test {name}", fit_mode="pca",
        node_corpora={pos_label: ["a", "b"], neg_label: ["c", "d"]},
        hyperparams={"max_dim": 1, "var_threshold": 0.70},
    )
    folder = _write_fitted_manifold(ns, name, seed=seed).parent
    # Refresh the manifest so the fitted tensor passes integrity on load.
    ManifoldFolder.load(folder).write_metadata(files=hash_manifold_files(folder))
    return folder


# ---------------------------------------------------------------------------
# manifold bake — folds fitted manifold components
# ---------------------------------------------------------------------------

class TestMergeFold:
    def test_merge_folds_manifold_components(self) -> None:
        from saklas.io.merge import merge_into_manifold
        from saklas.io.paths import safe_model_id

        _write_fitted_manifold("default", "happy.sad", seed=1)
        _write_fitted_manifold("default", "warm.clinical", seed=2)
        dst = merge_into_manifold(
            "merged",
            "0.5 default/happy.sad + 0.5 default/warm.clinical",
            model=_MODEL,
        )
        assert (dst / f"{safe_model_id(_MODEL)}.safetensors").exists()

    def test_merge_missing_component_errors(self) -> None:
        from saklas.io.merge import merge_into_manifold, MergeError

        _write_fitted_manifold("default", "happy.sad", seed=1)
        with pytest.raises(MergeError):
            merge_into_manifold(
                "merged",
                "0.5 default/happy.sad + 0.5 default/nonexistent",
                model=_MODEL,
            )


# ---------------------------------------------------------------------------
# pack export gguf — folds a fitted manifold
# ---------------------------------------------------------------------------

class TestGgufFold:
    def test_export_gguf_folds_manifold(self, tmp_path: Path) -> None:
        pytest.importorskip("gguf")  # writing the GGUF needs the optional extra
        from saklas.io.cache_ops import export_gguf_manifold

        _make_full_manifold("default", "happy.sad")
        out = tmp_path / "happy.gguf"
        written = export_gguf_manifold(
            "default", "happy.sad",
            model_scope=_MODEL, output=str(out), model_hint="llama",
        )
        assert written == [out]
        assert out.is_file()

    def test_export_gguf_unfitted_errors(self, tmp_path: Path) -> None:
        from saklas.io.cache_ops import export_gguf_manifold

        # manifold.json but no fitted tensor for the model.
        from saklas.io.manifolds import create_discover_manifold_folder
        create_discover_manifold_folder(
            "default", "happy.sad", "x", fit_mode="pca",
            node_corpora={"happy": ["a"], "sad": ["b"]},
            hyperparams={"max_dim": 1},
        )
        with pytest.raises(RuntimeError, match="no fitted manifold"):
            export_gguf_manifold(
                "default", "happy.sad",
                model_scope=_MODEL, output=str(tmp_path / "x.gguf"),
                model_hint="llama",
            )

    def test_export_preflight_skips_unrelated_variant_hashes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from saklas.io import gguf_io, packs
        from saklas.io.cache_ops import export_gguf_manifold
        from saklas.io.manifolds import ManifoldFolder

        folder = _make_full_manifold("default", "happy.sad")
        raw = folder / tensor_filename(_MODEL)
        unrelated = folder / tensor_filename(_MODEL, release="unrelated")
        unrelated.write_bytes(raw.read_bytes())
        unrelated.with_suffix(".json").write_bytes(
            raw.with_suffix(".json").read_bytes()
        )
        ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(
            unrelated, unrelated.with_suffix(".json"),
        )
        packs._FINGERPRINT_CACHE.clear()
        real_hash = packs.hash_file
        hashed: list[str] = []

        def track_hash(path: Path) -> str:
            hashed.append(Path(path).name)
            return real_hash(Path(path))

        def fake_write(_profile: object, path: Path, **_kwargs: object) -> None:
            Path(path).write_bytes(b"gguf")

        monkeypatch.setattr(packs, "hash_file", track_hash)
        monkeypatch.setattr(gguf_io, "write_gguf_profile", fake_write)
        export_gguf_manifold(
            "default", "happy.sad", model_scope=_MODEL,
            output=str(tmp_path / "out.gguf"), model_hint="llama",
        )

        assert raw.name in hashed and raw.with_suffix(".json").name in hashed
        assert unrelated.name not in hashed
        assert unrelated.with_suffix(".json").name not in hashed


def test_default_probe_preflight_skips_unrelated_variant_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.core.manifold import load_manifold
    from saklas.core.session import SaklasSession
    from saklas.io import packs
    from saklas.io.manifolds import ManifoldFolder
    import saklas.io.manifolds as manifolds_module
    import saklas.io.probes_bootstrap as probes_module

    folder = _make_full_manifold("default", "probe")
    raw = folder / tensor_filename(_MODEL)
    unrelated = folder / tensor_filename(_MODEL, release="unrelated")
    unrelated.write_bytes(raw.read_bytes())
    unrelated.with_suffix(".json").write_bytes(raw.with_suffix(".json").read_bytes())
    ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(
        unrelated, unrelated.with_suffix(".json"),
    )
    packs._FINGERPRINT_CACHE.clear()
    real_hash = packs.hash_file
    hashed: list[str] = []

    def track_hash(path: Path) -> str:
        hashed.append(Path(path).name)
        return real_hash(Path(path))

    class StubSession:
        model_id = _MODEL
        _manifolds: dict[str, Manifold] = {}

        def ensure_manifold_loaded(self, key: str) -> None:
            self._manifolds[key] = load_manifold(raw)

    monkeypatch.setattr(packs, "hash_file", track_hash)
    monkeypatch.setattr(probes_module, "load_default_manifolds", lambda: {})
    monkeypatch.setattr(manifolds_module, "bundled_manifold_names", lambda: ["probe"])
    session = StubSession()
    probes = SaklasSession._bootstrap_manifold_probes(
        cast(Any, session), [], include_fitted_defaults=True,
    )

    assert "default/probe" in probes
    assert raw.name in hashed and raw.with_suffix(".json").name in hashed
    assert unrelated.name not in hashed
    assert unrelated.with_suffix(".json").name not in hashed
