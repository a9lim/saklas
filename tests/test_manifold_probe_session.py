"""ManifoldMonitor wiring into SaklasSession.

CPU-only: a full session load needs a GPU + 8GB model download, so
these tests build a stand-in session by binding the real
``SaklasSession`` methods to a minimal stub.  The goal is to verify
wiring shape — capture widening, score-callback merging, finalize
populates ``GenerationResult.manifold_readings``, stream populates
``TokenEvent.manifold_readings`` — without paying the model-load cost.
"""
from __future__ import annotations

import types
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from saklas.core.manifold import BoxAxis, BoxDomain, LayerSubspace, Manifold
from saklas.core.manifold import (
    fit_layer_subspace as _fit_layer_subspace_with_ev,
)
from saklas.core.monitor import ManifoldMonitor, TraitMonitor
from saklas.core.results import (
    GenerationResult,
    ManifoldAggregate,
    ManifoldTokenReading,
    TokenEvent,
)
from saklas.core.session import SaklasSession


def fit_layer_subspace(*args: Any, **kwargs: Any) -> Any:
    sub, _ev = _fit_layer_subspace_with_ev(*args, **kwargs)
    return sub


def _toy_manifold(*, dim: int = 8, n_layers: int = 2) -> Manifold:
    """1-D BoxDomain manifold with 3 nodes — same shape as the
    test_manifold_monitor helper, copied to keep test files independent."""
    torch.manual_seed(0)
    domain = BoxDomain([BoxAxis("u", periodic=False, lo=-1.0, hi=1.0)])
    coords = torch.tensor([[-1.0], [0.0], [1.0]])
    layers: dict[int, LayerSubspace] = {}
    ev: dict[int, float] = {}
    e1 = torch.zeros(dim)
    e1[0] = 1.0
    e2 = torch.zeros(dim)
    e2[1] = 1.0
    for layer_idx in range(n_layers):
        scale = 1.0 + 0.5 * layer_idx
        centroids = torch.stack([
            -scale * e1,
            torch.zeros(dim),
            scale * e1,
        ]) + 0.01 * torch.stack([-e2, torch.zeros(dim), e2])
        sub, ev_ratio = _fit_layer_subspace_with_ev(
            centroids, domain.embed(coords),
        )
        layers[layer_idx] = sub
        ev[layer_idx] = ev_ratio
    return Manifold(
        name="toy",
        domain=domain,
        node_labels=["a", "b", "c"],
        node_coords=coords,
        layers=layers,
        explained_variance=ev,
    )


def _stub_session() -> SaklasSession:
    """Build a session-like object with the live methods + minimal state.

    Only the surface the manifold-probe wiring touches needs to be real:
    ``_manifold_monitor``, ``_monitor``, ``_capture._per_layer``,
    ``_manifolds``, ``_ensure_manifold_loaded``, ``_invalidate_prefix_cache``,
    ``_layers``.  Everything else stays as ``MagicMock`` defaults.
    """
    # Manifold reads are Mahalanobis-only now: the monitor needs a covering
    # whitener before ``add_probe`` builds the per-probe whitened factors.  An
    # isotropic one over the toy layers (dim 8, up to 3 layers) suffices.
    from tests._whitener import isotropic_whitener
    _whit = isotropic_whitener([0, 1, 2, 3], 8)

    session = MagicMock(spec=SaklasSession)
    session._manifold_monitor = ManifoldMonitor(whitener=_whit)
    session._monitor = TraitMonitor({}, {})
    session._manifolds = {}
    # Capture has a single ``_per_layer`` dict we read for streaming.
    session._capture = types.SimpleNamespace(_per_layer={})
    session._layers = []
    # Minimal prefix-cache invalidator + tree spy.
    session._prefix_cache = None

    def _invalidate():
        session._prefix_cache = None
    session._invalidate_prefix_cache = _invalidate

    # Bind the real methods we want to exercise.
    _add_probe_fn: Any = SaklasSession.add_manifold_probe
    session.add_manifold_probe = types.MethodType(
        _add_probe_fn.__wrapped__
        if hasattr(_add_probe_fn, "__wrapped__")
        else _add_probe_fn,
        session,
    )
    session.remove_manifold_probe = types.MethodType(
        SaklasSession.remove_manifold_probe, session,
    )
    session.manifold_monitor = session._manifold_monitor
    return session


# ==================================================== add / remove probe ===

def test_add_manifold_probe_registers_via_ensure_loaded():
    """``add_manifold_probe`` rides ``_ensure_manifold_loaded`` and lands
    the artifact on the ``ManifoldMonitor``."""
    session = _stub_session()
    m = _toy_manifold()

    # ``_ensure_manifold_loaded`` is the resolution path; stub it to
    # populate ``_manifolds`` instead of hitting disk.
    def _fake_ensure(key: str):
        session._manifolds[key] = m
    session._ensure_manifold_loaded = _fake_ensure

    name = session.add_manifold_probe("toy")
    assert name == "toy"
    assert "toy" in session._manifold_monitor.probe_names


def test_add_manifold_probe_as_name_override():
    session = _stub_session()
    m = _toy_manifold()
    session._ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )

    name = session.add_manifold_probe("toy", as_name="affect")
    assert name == "affect"
    assert "affect" in session._manifold_monitor.probe_names


def test_remove_manifold_probe():
    session = _stub_session()
    m = _toy_manifold()
    session._ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )

    session.add_manifold_probe("toy")
    session.remove_manifold_probe("toy")
    assert session._manifold_monitor.probe_names == []


def test_add_probe_invalidates_prefix_cache():
    session = _stub_session()
    m = _toy_manifold()
    session._ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )
    session._prefix_cache = ("dummy",)  # pyright: ignore[reportAttributeAccessIssue]  # test stub: wrong-shaped sentinel to verify invalidation
    session.add_manifold_probe("toy")
    assert session._prefix_cache is None


# ==================================================== capture widening ===

def test_begin_capture_widens_to_manifold_layers():
    """``_begin_capture`` must widen the capture-layer set to the union
    of vector-probe layers and manifold-probe layers."""
    session = _stub_session()
    m = _toy_manifold(n_layers=3)
    session._ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )
    session.add_manifold_probe("toy")

    # Track what layers _capture.attach was called with.
    attached_layers: list[int] = []

    def _attach(layers: Any, layer_indices: list[int]) -> None:
        attached_layers.extend(layer_indices)
    session._capture.attach = _attach
    session._capture.clear = lambda: None
    # Real layer list of length 4 (just need len() — the stub doesn't
    # forward through to anything that touches the layers).
    session._layers = [None] * 4  # pyright: ignore[reportAttributeAccessIssue]  # test stub: list[None] satisfies len() contract

    # Bind the real _begin_capture and run it.
    ok = SaklasSession._begin_capture(session, widen=False)
    assert ok
    # Manifold covers layers 0, 1, 2 — capture must attach to all of them.
    assert set(attached_layers) == {0, 1, 2}


def test_begin_capture_no_probes_returns_false():
    """No vector probes, no manifold probes → ``_begin_capture`` returns
    False (the v1 behavior)."""
    session = _stub_session()
    session._layers = [None] * 4  # pyright: ignore[reportAttributeAccessIssue]  # test stub: list[None] satisfies len() contract
    session._capture.attach = lambda *args, **kw: None
    session._capture.clear = lambda: None
    ok = SaklasSession._begin_capture(session, widen=False)
    assert ok is False


# ===================================================== gating callback ===

def test_gating_callback_merges_vector_and_manifold_scalars():
    """The closure built by ``_build_gating_score_callback`` must merge
    vector-probe scalars + manifold flat scalars when both are
    attached."""
    session = _stub_session()
    m = _toy_manifold()
    session._ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )
    session.add_manifold_probe("toy")

    # Plant a vector probe — single layer, single direction.  Mahalanobis is
    # mandatory, so the TraitMonitor gets an isotropic covering whitener.
    from tests._whitener import isotropic_whitener
    profile = {0: torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
    session._monitor = TraitMonitor(
        {"vec_probe": profile}, {0: torch.zeros(8)},
        whitener=isotropic_whitener([0], 8),
    )

    # Latest-per-layer captures: place activation inside the manifold's
    # subspace at layer 0 + 1, plus the same vector for the vector probe.
    latest: dict[int, torch.Tensor] = {}
    for layer_idx, sub in m.layers.items():
        latest[layer_idx] = sub.mean + sub.basis[0]
    session._capture._per_layer = {
        idx: [tensor] for idx, tensor in latest.items()
    }

    # latest_per_layer reads bucket[-1] per layer.
    session._capture.latest_per_layer = lambda: {
        idx: bucket[-1]
        for idx, bucket in session._capture._per_layer.items()
        if bucket
    }

    cb = SaklasSession._build_gating_score_callback(session)
    flat = cb()
    # Manifold keys are present: fraction + per-node distance.
    assert "toy:fraction" in flat
    assert "toy@a" in flat
    # Vector probe still flows through.
    assert "vec_probe" in flat


def test_gating_callback_empty_capture_returns_empty():
    session = _stub_session()
    m = _toy_manifold()
    session._ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )
    session.add_manifold_probe("toy")
    session._capture.latest_per_layer = lambda: {}

    cb = SaklasSession._build_gating_score_callback(session)
    assert cb() == {}


# ============================================================ aggregate ===

def test_score_aggregate_called_with_correct_captures():
    """Verify ``score_aggregate`` is called on the manifold monitor when
    probes are attached and gen produced tokens."""
    from tests._whitener import isotropic_whitener
    m = _toy_manifold()
    mon = ManifoldMonitor(whitener=isotropic_whitener(list(m.layers), 8))
    mon.add_probe("toy", m)

    # Build a captured stack at node 1's centroid (so coords come back
    # near 0.0).  Recompute the world-space node activation from the RBF
    # (the former ``node_values_world`` cache was removed — only the
    # reduced cache feeds scoring).
    embedded = m.domain.embed(m.domain.clamp_position(m.node_coords.float()))
    captured: dict[int, torch.Tensor] = {}
    for layer_idx, sub in m.layers.items():
        v_world = sub.eval_at(embedded)[1]
        captured[layer_idx] = v_world.unsqueeze(0).repeat(5, 1)

    agg = mon.score_aggregate(captured)
    assert isinstance(agg["toy"], ManifoldAggregate)
    assert agg["toy"].coords[0] == pytest.approx(0.0, abs=0.1)


# ============================================ TokenEvent / GenerationResult ===

def test_token_event_carries_manifold_readings_field():
    """``TokenEvent`` accepts ``manifold_readings`` and defaults to None."""
    ev = TokenEvent(text="x", token_id=42, index=0)
    assert ev.manifold_readings is None
    reading = ManifoldTokenReading(fraction=0.5, nearest=[("a", 0.1)])
    ev2 = TokenEvent(
        text="y", token_id=43, index=1,
        manifold_readings={"toy": reading},
    )
    assert ev2.manifold_readings == {"toy": reading}


def test_generation_result_carries_manifold_readings_field():
    """``GenerationResult`` accepts ``manifold_readings`` and ``to_dict``
    serializes it."""
    agg = ManifoldAggregate(
        fraction_mean=0.6,
        fraction_per_layer={0: 0.5, 1: 0.7},
        nearest=[("a", 0.2)],
        coords=(0.3,),
        coords_per_layer={0: (0.2,), 1: (0.4,)},
        residual_mean=0.05,
        residual_per_layer={0: 0.04, 1: 0.06},
    )
    result = GenerationResult(
        text="hello", tokens=[1, 2], token_count=2,
        tok_per_sec=10.0, elapsed=0.2,
        manifold_readings={"toy": agg},
    )
    d = result.to_dict()
    assert "manifold_readings" in d
    assert "toy" in d["manifold_readings"]
    assert d["manifold_readings"]["toy"]["fraction_mean"] == pytest.approx(0.6)
    assert d["manifold_readings"]["toy"]["coords"] == [0.3]


def test_generation_result_default_manifold_readings_is_empty_dict():
    result = GenerationResult(
        text="x", tokens=[], token_count=0,
        tok_per_sec=0.0, elapsed=0.0,
    )
    assert result.manifold_readings == {}
    assert result.to_dict()["manifold_readings"] == {}
