"""ManifoldExtractionPipeline tests — CPU only, synthetic encoder.

Mirrors the stub-encoder pattern in :mod:`tests.test_dim_extraction`:
monkeypatch :func:`saklas.core.vectors._encode_and_capture_all` so no
real model is needed.
"""
from __future__ import annotations

import json

import pytest
import torch

from saklas.core import vectors as V
from saklas.core.events import EventBus, ManifoldExtracted
from saklas.core.extraction import ManifoldExtractionPipeline
from saklas.core.sae import MockSaeBackend
from saklas.io.manifolds import MANIFOLD_FORMAT_VERSION, ManifoldFolder

_LABELS = ["calm", "uneasy", "afraid", "frantic", "numb"]
_DIM = 8
_N_LAYERS = 4


def _stub_encoder(model, tokenizer, text, layers, device, **_kwargs):
    """Synthetic per-layer activations, deterministic per node label."""
    label = text.split()[0]
    seed = abs(hash(label)) % 100_000
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(_DIM, generator=g)
    out: dict[int, torch.Tensor] = {}
    for layer in range(len(layers)):
        v = base.clone()
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


def _box1d_domain(periodic: bool, k: int) -> dict:
    axis = (
        {"name": "t", "periodic": True, "period": 1.0}
        if periodic
        else {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}
    )
    return {"type": "box", "axes": [axis]}


def _author_manifold(root, *, periodic=True, labels=None,
                     domain=None, coords=None):
    labels = labels or _LABELS
    folder = root / "mood"
    (folder / "nodes").mkdir(parents=True)
    for idx, label in enumerate(labels):
        (folder / "nodes" / f"{idx:02d}_{label}.json").write_text(
            json.dumps([f"{label} statement {i}" for i in range(3)])
        )
    k = len(labels)
    if domain is None:
        domain = _box1d_domain(periodic, k)
    if coords is None:
        if periodic:
            coords = [[i / k] for i in range(k)]
        else:
            coords = [[i / (k - 1)] for i in range(k)]
    (folder / "manifold.json").write_text(json.dumps({
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": "mood",
        "description": "moods",
        "domain": domain,
        "nodes": [
            {"label": label, "coords": coords[i]}
            for i, label in enumerate(labels)
        ],
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
    assert manifold.domain.intrinsic_dim == 1
    assert manifold.node_labels == _LABELS
    assert sorted(manifold.layers) == list(range(_N_LAYERS))
    assert manifold.feature_space == "raw"


def test_fit_writes_tensor_and_manifest(tmp_path):
    folder = _author_manifold(tmp_path)
    ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    assert (folder / "stub-model.safetensors").exists()
    assert (folder / "stub-model.json").exists()
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


def test_fit_cache_miss_on_domain_change(tmp_path, monkeypatch):
    folder = _author_manifold(tmp_path, periodic=True)
    pipe = ManifoldExtractionPipeline(_Handle(), EventBus())
    pipe.fit(folder)

    # Move a node coordinate — the geometry changes, so the cached
    # tensor is stale even though the node corpus is byte-identical.
    meta = json.loads((folder / "manifold.json").read_text())
    meta["nodes"][1]["coords"] = [0.123]
    (folder / "manifold.json").write_text(json.dumps(meta))

    calls = {"n": 0}
    real = _stub_encoder

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(V, "_encode_and_capture_all", _counting)
    pipe.fit(folder)
    assert calls["n"] > 0  # geometry changed -> re-fit


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
    assert (folder / "stub-model_sae-mock-rel.safetensors").exists()


def test_fit_natural_manifold(tmp_path):
    folder = _author_manifold(tmp_path, periodic=False)
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    assert manifold.domain.intrinsic_dim == 1
    assert manifold.domain.axes[0].periodic is False
    for sub in manifold.layers.values():
        assert sub.node_params.shape[0] == len(_LABELS)


def test_fit_n2_box_manifold(tmp_path):
    labels = [f"n{i}" for i in range(9)]
    domain = {
        "type": "box",
        "axes": [
            {"name": "u", "periodic": False, "lo": 0.0, "hi": 1.0},
            {"name": "v", "periodic": False, "lo": 0.0, "hi": 1.0},
        ],
    }
    coords = [[x, y] for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)]
    folder = _author_manifold(
        tmp_path, labels=labels, domain=domain, coords=coords,
    )
    manifold = ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
    assert manifold.domain.intrinsic_dim == 2
    pt = manifold.manifold_point(0, (0.3, 0.7))
    assert pt.shape == (_DIM,)


def test_fit_rejects_poisedness_failure(tmp_path):
    # Five nodes on a 2-D domain, all collinear -> the affine term is
    # underdetermined and the RBF fit raises.
    labels = [f"n{i}" for i in range(5)]
    domain = {
        "type": "box",
        "axes": [
            {"name": "u", "periodic": False, "lo": 0.0, "hi": 1.0},
            {"name": "v", "periodic": False, "lo": 0.0, "hi": 1.0},
        ],
    }
    coords = [[t, t] for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    with pytest.warns(UserWarning, match="poised"):
        folder = _author_manifold(
            tmp_path, labels=labels, domain=domain, coords=coords,
        )
        ManifoldFolder.load(folder)
    with pytest.warns(UserWarning, match="poised"), \
            pytest.raises(ValueError, match="poisedness"):
        ManifoldExtractionPipeline(_Handle(), EventBus()).fit(folder)
