"""Monopolar (concept-vs-neutral) extraction — CPU only, synthetic encoder.

A monopolar concept (``session.extract(c)`` with ``baseline=None``) authors a
**genuinely 1-node** ``pca`` manifold.  The pipeline recognizes the single-node
shape and folds ``concept − ν`` (ν = the model's neutral activation mean,
``layer_means``) into a 1-node neutral-anchored ray — neutral is the implicit
negative pole, sourced per-model at fit, never a stored corpus.

These tests exercise the pipeline on CPU with a deterministic stub encoder; the
end-to-end ``session.extract`` authoring is covered by the GPU smoke.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from saklas.core import vectors as V
from saklas.core.events import EventBus
from saklas.core.extraction import ManifoldExtractionPipeline
from saklas.core.vectors import folded_vector_directions
from saklas.io.manifolds import ManifoldFolder, create_discover_manifold_folder
from saklas.io.paths import manifold_dir
from tests._whitener import synthetic_whitener

_DIM = 8
_N_LAYERS = 4
# Per-layer neutral mean ν.  The concept centroid is ν + a push along dim 0, so
# the folded direction (concept − ν) is a clean unit vector along dim 0.
_NU = {L: torch.full((_DIM,), float(L + 1)) for L in range(_N_LAYERS)}


def _concept_encoder(
    model: Any, tokenizer: Any, prompt: str, response: str, layers: Any,
    device: Any, **_kwargs: Any,
) -> dict[int, torch.Tensor]:
    """Deterministic concept centroid: ν + a +2 push along dim 0, per layer."""
    out: dict[int, torch.Tensor] = {}
    for L in range(len(layers)):
        v = _NU[L].clone()
        v[0] += 2.0
        out[L] = v
    return out


class _Handle:
    """Minimal ModelHandle stub carrying ``layer_means`` (= ν, the neutral pole)."""

    def __init__(self) -> None:
        self.model_id = "stub-model"
        self.model: torch.nn.Module = torch.nn.Linear(1, 1)
        self.tokenizer: Any = object()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.layers: Any = [object()] * _N_LAYERS
        self.layer_means: dict[int, torch.Tensor] = {
            L: _NU[L].clone() for L in range(_N_LAYERS)
        }
        # The monopolar share is whitened (mandatory post-4.0); synthesize a
        # covering whitener over the same neutral means.
        self.whitener = synthetic_whitener(
            range(_N_LAYERS), _DIM, means=self.layer_means,
        )

    def _run_generator(
        self, system_msg: str, prompt: str, max_new_tokens: int,
    ) -> str:
        raise NotImplementedError("stub: not called")

    def generate_responses(self, *a: Any, **k: Any) -> dict[str, list[str]]:
        raise NotImplementedError("stub: not called")


@pytest.fixture(autouse=True)
def _stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    torch.manual_seed(0)
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path))
    monkeypatch.setattr(V, "_encode_and_capture_all", _concept_encoder)
    # Single baseline prompt → any corpus length is a multiple of k=1.
    monkeypatch.setattr(V, "_load_baseline_prompts", lambda: ["baseline prompt"])


def _author() -> Path:
    return create_discover_manifold_folder(
        "local", "agentic", "Monopolar axis: agentic (+) vs neutral (-).",
        fit_mode="pca",
        node_corpora={"agentic": [f"agentic statement {i}" for i in range(2)]},
        hyperparams={"max_dim": 1, "var_threshold": 0.7},
        node_kinds={"agentic": "abstract"},
    )


def test_monopolar_folder_is_genuinely_one_node() -> None:
    _author()
    mf = ManifoldFolder.load(manifold_dir("local", "agentic"))
    assert mf.node_labels == ["agentic"]
    assert mf.fit_mode == "pca"


def test_monopolar_fits_one_node_ray() -> None:
    folder = _author()
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # the fold path runs no DLS
        manifold = pipe.fit(folder)

    # Genuinely one node — the concept; neutral is implicit (ν).
    assert manifold.node_labels == ["agentic"]
    assert manifold.metadata.get("method") == "manifold_monopolar"
    assert manifold.metadata.get("monopolar") is True
    # Every layer is an affine rank-1 ray, so it folds to a steering vector.
    dirs = folded_vector_directions(manifold)
    assert sorted(dirs) == list(range(_N_LAYERS))
    for vec in dirs.values():
        assert torch.isfinite(vec).all()
        # δ̂ = unit(concept − ν) points +along dim 0 (the only displaced axis).
        assert float(vec[0]) > 0.0
        assert torch.allclose(vec[1:], torch.zeros(_DIM - 1), atol=1e-5)


def test_monopolar_requires_layer_means() -> None:
    # ν is the implicit negative pole — without it there is nothing to fold
    # against, so the fit fails fast with an actionable message.
    folder = _author()
    handle = _Handle()
    handle.layer_means = {}
    pipe = ManifoldExtractionPipeline(handle, EventBus())
    with pytest.raises(ValueError, match="layer_means"):
        pipe.fit(folder)


def test_monopolar_cache_hit_reloads_ray() -> None:
    folder = _author()
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    first = pipe.fit(folder)
    # Second fit cache-hits on the unchanged 1-node corpus and reloads the ray.
    second = pipe.fit(folder)
    assert second.node_labels == first.node_labels == ["agentic"]
    d1 = folded_vector_directions(first)
    d2 = folded_vector_directions(second)
    assert sorted(d1) == sorted(d2)
    for L in d1:
        assert torch.allclose(d1[L], d2[L], atol=1e-5)
