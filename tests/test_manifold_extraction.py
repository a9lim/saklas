"""ManifoldExtractionPipeline tests — CPU only, synthetic encoder.

Mirrors the stub-encoder pattern in :mod:`tests.test_dim_extraction`:
monkeypatch :func:`saklas.core.vectors._encode_and_capture_all` so no
real model is needed.
"""
from __future__ import annotations

import json
import math

import pytest
import torch

from saklas.core import vectors as V
from saklas.core.events import EventBus, ManifoldExtracted
from saklas.core.extraction import ManifoldExtractionPipeline
from saklas.core.sae import MockSaeBackend
from saklas.io.manifolds import ManifoldFolder
from saklas.io.packs import PACK_FORMAT_VERSION

_LABELS = ["calm", "uneasy", "afraid", "frantic", "numb"]
_DIM = 8
_N_LAYERS = 4


def _stub_encoder(model, tokenizer, text, layers, device):
    """Synthetic per-layer activations: each node sits on a circle."""
    label = text.split()[0]
    i = _LABELS.index(label)
    theta = 2.0 * math.pi * i / len(_LABELS)
    out: dict[int, torch.Tensor] = {}
    for layer in range(len(layers)):
        v = torch.zeros(_DIM)
        v[0] = 2.0 * math.cos(theta) + 0.7
        v[1] = 2.0 * math.sin(theta) + 0.7
        v[2] = 0.5 * layer            # layer-varying offset
        out[layer] = v + 0.01 * torch.randn(_DIM)
    return out


class _Handle:
    """Minimal ModelHandle stub."""

    def __init__(self):
        self.model_id = "stub-model"
        self.model = object()
        self.tokenizer = object()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.layers = [object()] * _N_LAYERS


def _author_manifold(root, *, cyclic=True):
    folder = root / "mood"
    (folder / "nodes").mkdir(parents=True)
    for idx, label in enumerate(_LABELS):
        (folder / "nodes" / f"{idx:02d}_{label}.json").write_text(
            json.dumps([f"{label} statement {i}" for i in range(3)])
        )
    (folder / "manifold.json").write_text(json.dumps({
        "format_version": PACK_FORMAT_VERSION,
        "name": "mood",
        "description": "moods",
        "cyclic": cyclic,
        "nodes": _LABELS,
        "files": {},
    }))
    return folder


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(V, "_encode_and_capture_all", _stub_encoder)


def test_fit_produces_manifold(tmp_path):
    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    manifold = pipe.fit(folder)

    assert manifold.name == "mood"
    assert manifold.cyclic is True
    assert manifold.node_labels == _LABELS
    assert sorted(manifold.layers) == list(range(_N_LAYERS))
    assert manifold.feature_space == "raw"
    # The circle is planar; with measurement noise the PCA subspace is
    # rank >= 2, capped at K-1.
    for sub in manifold.layers.values():
        assert 2 <= sub.rank <= len(_LABELS) - 1


def test_fit_writes_tensor_and_manifest(tmp_path):
    folder = _author_manifold(tmp_path)
    ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    assert (folder / "stub-model.safetensors").exists()
    assert (folder / "stub-model.json").exists()
    # write_metadata back-filled the integrity manifest.
    mf = ManifoldFolder.load(folder)
    assert "stub-model.safetensors" in mf.files
    assert "stub-model.json" in mf.files


def test_fit_emits_event(tmp_path):
    folder = _author_manifold(tmp_path)
    events = EventBus()
    seen = []
    events.subscribe(lambda e: seen.append(e)
                     if isinstance(e, ManifoldExtracted) else None)
    ManifoldExtractionPipeline(_Handle(), events).fit(folder)
    assert len(seen) == 1
    assert seen[0].name == "mood"


def test_fit_cache_hit_skips_forward_passes(tmp_path, monkeypatch):
    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)

    calls = {"n": 0}
    real = _stub_encoder

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all", _counting)
    manifold = pipe.fit(folder)  # second call — corpus unchanged
    assert calls["n"] == 0       # cache hit, no pooling
    assert manifold.name == "mood"


def test_fit_cache_miss_on_corpus_change(tmp_path, monkeypatch):
    folder = _author_manifold(tmp_path)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)

    # Mutate a node corpus -> nodes_sha256 changes -> re-fit.
    (folder / "nodes" / "00_calm.json").write_text(
        json.dumps(["calm statement 0", "calm statement 1"])
    )

    calls = {"n": 0}
    real = _stub_encoder

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all", _counting)
    pipe.fit(folder)
    assert calls["n"] > 0  # corpus changed -> forward passes re-run


def test_fit_cache_miss_on_cyclic_flip(tmp_path, monkeypatch):
    folder = _author_manifold(tmp_path, cyclic=True)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)

    # Flip the cyclic flag — it selects the spline system, so the cached
    # tensor is stale even though the node corpus is byte-identical.
    meta = json.loads((folder / "manifold.json").read_text())
    meta["cyclic"] = False
    (folder / "manifold.json").write_text(json.dumps(meta))

    calls = {"n": 0}
    real = _stub_encoder

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all", _counting)
    manifold = pipe.fit(folder)
    assert calls["n"] > 0  # cyclic flip -> re-fit
    assert manifold.cyclic is False


def test_fit_sae_variant(tmp_path):
    folder = _author_manifold(tmp_path)
    sae = MockSaeBackend(
        layers=frozenset(range(_N_LAYERS)), d_model=_DIM,
        release="mock-rel",
    )
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(
        folder, sae=sae,
    )
    assert manifold.feature_space == "sae-mock-rel"
    assert sorted(manifold.layers) == list(range(_N_LAYERS))
    # SAE variant lands at its own filename.
    assert (folder / "stub-model_sae-mock-rel.safetensors").exists()


def test_fit_natural_manifold(tmp_path):
    folder = _author_manifold(tmp_path, cyclic=False)
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    assert manifold.cyclic is False
    # Natural spline stores K knots (no wrap row).
    for sub in manifold.layers.values():
        assert sub.t_knots.shape[0] == len(_LABELS)
