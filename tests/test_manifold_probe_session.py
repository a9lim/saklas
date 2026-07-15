"""Unified probe wiring into SaklasSession.

CPU-only: a full session load needs a GPU + 8GB model download, so
these tests build a stand-in session by binding the real
``SaklasSession`` methods to a minimal stub.  The goal is to verify
wiring shape — capture widening, score-callback merging, finalize
populates ``GenerationResult.probe_readings``, stream populates
``TokenEvent.probe_readings`` — without paying the model-load cost.

4.0 collapse: the former ``TraitMonitor`` + ``ManifoldMonitor`` are one
``Monitor`` and the former ``add_manifold_probe`` / ``remove_manifold_probe``
session surface is the unified ``add_probe`` / ``remove_probe`` over the single
``session._monitor``.  A vector probe is the rank-1 case of the same subspace
readout, so there is no separate vector-monitor to wire — the toy manifold below
covers both shapes.
"""
from __future__ import annotations

import threading
import types
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch

from saklas.core.manifold import BoxAxis, BoxDomain, LayerSubspace, Manifold
from saklas.core.manifold import (
    fit_layer_subspace as _fit_layer_subspace_with_ev,
)
from saklas.core.monitor import Monitor
from saklas.core.results import (
    GenerationResult,
    ProbeReading,
    TokenEvent,
)
from saklas.core.session import CaptureMode, CaptureState, SaklasSession


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
    share: dict[int, float] = {}
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
        assert sub.node_params is not None
        sub.sigma_rbf_weights = torch.zeros((len(coords), 1))
        sub.sigma_poly_coeffs = torch.zeros((sub.node_params.shape[1] + 1, 1))
        sub.sigma_poly_coeffs[0, 0] = -20.0
        layers[layer_idx] = sub
        share[layer_idx] = ev_ratio
    return Manifold(
        name="toy",
        domain=domain,
        node_labels=["a", "b", "c"],
        node_coords=coords,
        layers=layers,
        node_roles=[None, None, None],
        node_kinds=[None, None, None],
        mahalanobis_share=share,
        origin={layer: torch.zeros(1) for layer in layers},
    )


def _stub_session() -> SaklasSession:
    """Build a session-like object with the live methods + minimal state.

    Only the surface the unified-probe wiring touches needs to be real:
    ``_monitor`` (one :class:`Monitor`), ``_capture._per_layer``,
    ``_manifolds``, ``_profiles``, ``ensure_manifold_loaded``,
    ``_invalidate_prefix_cache``, ``_probe_hash_cache``, ``_layers``.
    Everything else stays as ``MagicMock`` defaults.
    """
    # Manifold reads are Mahalanobis-only now: the monitor needs a covering
    # whitener before ``add_probe`` builds the per-probe whitened factors.  An
    # isotropic one over the toy layers (dim 8, up to 3 layers) suffices.
    from tests._whitener import isotropic_whitener
    _whit = isotropic_whitener([0, 1, 2, 3], 8)

    session = MagicMock(spec=SaklasSession)
    session._monitor = Monitor(whitener=_whit)
    session._manifolds = {}
    session._profiles = {}
    session._probe_hash_cache = {}
    # ``add_probe`` holds the exclusive-GPU lock (``_gen_lock``) around its
    # device-touching factor build and clears the read-side analytics cache.
    session._gen_lock = threading.RLock()
    session._analytics_cpu_cache = {}
    session._invalidate_analytics_cache = types.MethodType(
        SaklasSession._invalidate_analytics_cache, session,
    )
    # Capture has a single ``_per_layer`` dict we read for streaming.
    session._capture = types.SimpleNamespace(
        _per_layer={},
        set_incremental=lambda _sink: None,
    )
    session._capture_state = CaptureState()
    session._jlens = None
    session._jlens_identity = None
    session._generation_jlens = None
    session._generation_jlens_active = False
    session._live_lens = None
    session._lens_probes = {}
    session._live_sae = None
    session._sae_probes = {}
    session._lens_step_stash = None
    session._live_lens_active_for_generation = True
    session._incremental_readings = []
    session._incremental_gate_scores = []
    session._compiled_clean_eligible = False
    session._layers = []
    # Minimal prefix-cache invalidator + tree spy.
    session._prefix_cache = None

    def _invalidate():
        session._prefix_cache = None
    session._invalidate_prefix_cache = _invalidate

    # Bind the real unified-probe methods we want to exercise.  ``add_probe``
    # rides ``_resolve_probe_manifold`` (which consults ``_profiles`` then
    # ``ensure_manifold_loaded`` → ``_manifolds``), so bind that too; tests
    # stub the public loader to populate ``_manifolds`` in place.
    session.add_probe = types.MethodType(SaklasSession.add_probe, session)
    session.remove_probe = types.MethodType(SaklasSession.remove_probe, session)
    session._resolve_probe_manifold = types.MethodType(
        SaklasSession._resolve_probe_manifold, session,
    )
    # The steering collaborator owns the gating-score-callback builder; wire a
    # real one explicitly.
    from saklas.core.steering_composer import SteeringComposer
    session._steering_composer = SteeringComposer(session)
    return session


# ==================================================== add / remove probe ===

def test_add_probe_registers_via_ensure_loaded():
    """``add_probe`` rides ``ensure_manifold_loaded`` and lands the
    artifact on the unified ``Monitor``."""
    session = _stub_session()
    m = _toy_manifold()

    # ``ensure_manifold_loaded`` is the resolution path; stub it to
    # populate ``_manifolds`` instead of hitting disk.
    def _fake_ensure(key: str):
        session._manifolds[key] = m
    session.ensure_manifold_loaded = _fake_ensure

    name = session.add_probe("toy")
    assert name == "toy"
    assert "toy" in session._monitor.probe_names


def test_add_probe_as_name_override():
    session = _stub_session()
    m = _toy_manifold()
    session.ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )

    name = session.add_probe("toy", as_name="affect")
    assert name == "affect"
    assert "affect" in session._monitor.probe_names


def test_remove_probe():
    session = _stub_session()
    m = _toy_manifold()
    session.ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )

    session.add_probe("toy")
    session.remove_probe("toy")
    assert session._monitor.probe_names == []


def test_add_probe_invalidates_prefix_cache():
    session = _stub_session()
    m = _toy_manifold()
    session.ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )
    session._prefix_cache = ("dummy",)  # pyright: ignore[reportAttributeAccessIssue]  # test stub: wrong-shaped sentinel to verify invalidation
    session.add_probe("toy")
    assert session._prefix_cache is None


# ==================================================== capture widening ===

def test_begin_capture_widens_to_manifold_layers():
    """``_begin_capture`` must widen the capture-layer set to the union
    of every attached probe's fit layers."""
    session = _stub_session()
    m = _toy_manifold(n_layers=3)
    session.ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )
    session.add_probe("toy")

    # Track what layers _capture.attach was called with.
    attached_layers: list[int] = []
    incremental_sinks: list[Any] = []

    def _attach(layers: Any, layer_indices: list[int]) -> None:
        attached_layers.extend(layer_indices)
    session._capture.attach = _attach
    session._capture.clear = lambda: None
    session._capture.set_incremental = (
        lambda step_sink: incremental_sinks.append(step_sink)
    )
    # Real layer list of length 4 (just need len() — the stub doesn't
    # forward through to anything that touches the layers).
    session._layers = [None] * 4  # pyright: ignore[reportAttributeAccessIssue]  # test stub: list[None] satisfies len() contract
    session._incremental_readings = []
    session._incremental_gate_scores = []

    # Bind the real _begin_capture and run it.
    ok = SaklasSession._begin_capture(session, widen=False)
    assert ok
    # Manifold covers layers 0, 1, 2 — capture must attach to all of them.
    assert set(attached_layers) == {0, 1, 2}
    assert session._capture_state.mode is CaptureMode.INCREMENTAL
    assert len(incremental_sinks) == 1


def test_begin_capture_no_probes_returns_false():
    """No probes attached → ``_begin_capture`` returns False (the v1
    behavior)."""
    session = _stub_session()
    session._layers = [None] * 4  # pyright: ignore[reportAttributeAccessIssue]  # test stub: list[None] satisfies len() contract
    session._capture.attach = lambda *args, **kw: None
    session._capture.clear = lambda: None
    session._incremental_readings = []
    ok = SaklasSession._begin_capture(session, widen=False)
    assert ok is False


def test_begin_capture_live_lens_uses_persistent_capture_when_available():
    """Live J-lens layers ride compile-clean persistent capture buffers."""
    session = _stub_session()
    session._layers = [None] * 4  # pyright: ignore[reportAttributeAccessIssue]  # test stub: list[None] satisfies len() contract
    session._live_lens = {"layers": [1, 3]}
    session._compiled_clean_eligible = True
    session._steering_uses_compiled_offsets = False
    session._capture_buffers = {
        idx: torch.zeros(8) for idx in range(4)
    }
    cast(Any, session)._steering = types.SimpleNamespace(
        all_fast_path=lambda: True,
    )

    persistent_layers: list[int] = []
    transient_layers: list[int] = []
    tail_depths: list[int] = []

    def _attach_persistent(layer_indices: list[int], buffers: dict[int, torch.Tensor]) -> None:
        assert buffers is session._capture_buffers
        persistent_layers.extend(layer_indices)

    def _attach(_layers: Any, layer_indices: list[int]) -> None:
        transient_layers.extend(layer_indices)

    session._capture.attach_persistent = _attach_persistent
    cast(Any, session._capture).attach = _attach
    session._capture.clear = lambda: None
    session._capture.set_aggregate_tail = lambda depth: tail_depths.append(depth)

    ok = SaklasSession._begin_capture(session, widen=False)

    assert ok is True
    assert set(persistent_layers) == {1, 3}
    assert transient_layers == []
    assert tail_depths
    assert session._capture_state.persistent is True


def test_begin_capture_live_lens_ignored_without_consumer():
    """A globally enabled live lens should not widen capture for a generation
    that cannot surface ``TokenEvent.lens_readout``."""
    session = _stub_session()
    session._layers = [None] * 4  # pyright: ignore[reportAttributeAccessIssue]
    session._live_lens = {"layers": [1, 3]}
    session._capture.attach = lambda *args, **kw: None
    session._capture.clear = lambda: None
    session._incremental_readings = []

    ok = SaklasSession._begin_capture(
        session, widen=False, live_lens_active=False,
    )

    assert ok is False


# ===================================================== gating callback ===

def test_gating_callback_emits_probe_scalars():
    """The composer-built gating callback must score the latest captures
    through the unified monitor and emit gate scalars — fraction + per-node
    distance + coordinate axis — for every attached probe (a vector probe is
    the rank-1 case of the same readout)."""
    session = _stub_session()
    m = _toy_manifold()
    session.ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )
    session.add_probe("toy")

    # Latest-per-layer captures: place activation inside the manifold's
    # subspace at every fit layer.
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

    cb = session._steering_composer.build_gating_score_callback()
    flat = cb()
    # Manifold keys are present: fraction + per-node distance + coord axis 0.
    assert "toy:fraction" in flat
    # Per-node distance channels emit; the activation sits near the frame
    # origin, so the virtual neutral candidate competes in the nearest ranking
    # (it can displace a farther node from the default top-3) and exposes the
    # uniform ``toy@neutral`` gate channel.
    assert "toy@neutral" in flat
    assert any(f"toy@{label}" in flat for label in m.node_labels)
    assert "toy" in flat  # bare name aliases coordinate axis 0


def test_gating_callback_empty_capture_returns_empty():
    session = _stub_session()
    m = _toy_manifold()
    session.ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )
    session.add_probe("toy")
    session._capture.latest_per_layer = lambda: {}

    cb = session._steering_composer.build_gating_score_callback()
    assert cb() == {}


# ============================================================ aggregate ===

def test_score_aggregate_called_with_correct_captures():
    """Verify ``score_aggregate`` is called on the unified monitor when
    probes are attached and gen produced tokens."""
    from tests._whitener import isotropic_whitener
    m = _toy_manifold()
    mon = Monitor(whitener=isotropic_whitener(list(m.layers), 8))
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
    assert isinstance(agg["toy"], ProbeReading)
    assert agg["toy"].coords[0] == pytest.approx(0.0, abs=0.1)


# ============================================ TokenEvent / GenerationResult ===

def test_token_event_carries_probe_readings_field():
    """``TokenEvent`` accepts ``probe_readings`` and defaults to None."""
    ev = TokenEvent(text="x", token_id=42, index=0)
    assert ev.probe_readings is None
    reading = ProbeReading(fraction=0.5, nearest=[("a", 0.1)])
    ev2 = TokenEvent(
        text="y", token_id=43, index=1,
        probe_readings={"toy": reading},
    )
    assert ev2.probe_readings == {"toy": reading}


def test_generation_result_carries_probe_readings_field():
    """``GenerationResult`` accepts ``probe_readings`` and ``to_dict``
    serializes it."""
    agg = ProbeReading(
        fraction=0.6,
        fraction_per_layer={0: 0.5, 1: 0.7},
        nearest=[("a", 0.2)],
        coords=(0.3,),
        coords_per_layer={0: (0.2,), 1: (0.4,)},
        residual=0.05,
        residual_per_layer={0: 0.04, 1: 0.06},
    )
    result = GenerationResult(
        text="hello", tokens=[1, 2], token_count=2,
        tok_per_sec=10.0, elapsed=0.2,
        probe_readings={"toy": agg},
    )
    d = result.to_dict()
    assert "probe_readings" in d
    assert "toy" in d["probe_readings"]
    assert d["probe_readings"]["toy"]["fraction"] == pytest.approx(0.6)
    assert d["probe_readings"]["toy"]["coords"] == [0.3]


def test_generation_result_default_probe_readings_is_empty_dict():
    result = GenerationResult(
        text="x", tokens=[], token_count=0,
        tok_per_sec=0.0, elapsed=0.0,
    )
    assert result.probe_readings == {}
    assert result.to_dict()["probe_readings"] == {}
