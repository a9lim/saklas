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
    # All three families route through real instruments now: geometry for
    # the add_probe flow (exclusive section + whitener touch + resolve +
    # monitor.add_probe), lens/SAE because ``_begin_capture`` consumes
    # their ``plan()`` demand — a MagicMock plan would poison the capture
    # layer union.  Lens/SAE state (live config, probe registry) lives ON
    # the instruments; tests set ``session._lens_instrument.live`` etc.
    from saklas.core.instruments.geometry import GeometryInstrument
    from saklas.core.instruments.lens import LensInstrument
    from saklas.core.instruments.sae import SaeInstrument
    session._geometry_instrument = GeometryInstrument(session)
    session._lens_instrument = LensInstrument(session)
    session._sae_instrument = SaeInstrument(session)
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
    # Live-lens state lives on the instrument; ``_begin_capture`` reads it
    # through the lens family's declared plan.
    session._lens_instrument.live = {"layers": [1, 3]}
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
    session._lens_instrument.live = {"layers": [1, 3]}
    session._capture.attach = lambda *args, **kw: None
    session._capture.clear = lambda: None
    session._incremental_readings = []

    ok = SaklasSession._begin_capture(
        session, widen=False, live_lens_active=False,
    )

    assert ok is False


def test_geometry_detach_rejects_while_generation_holds_the_lock():
    """Geometry has no per-generation roster snapshot (Monitor scoring
    walks the live dict), so a detach racing a generation must reject with
    retry-shortly semantics instead of mutating the roster mid-flight —
    the same exclusive section ``attach`` takes."""
    import threading as _threading

    from saklas.core.session import ConcurrentExtractionError

    session = _stub_session()
    # The MagicMock default no-ops the exclusive section; this test is
    # about exactly that guard, so bind the real one (it needs only
    # ``_gen_lock``, which the fixture provides).
    session._model_exclusive = types.MethodType(
        SaklasSession._model_exclusive, session,
    )
    m = _toy_manifold()
    session.ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m}
    )
    session.add_probe("toy")
    assert "toy" in session._monitor.probe_names

    # Simulate an in-flight generation: another thread holds the gen lock.
    held = _threading.Event()
    release = _threading.Event()

    def _hold() -> None:
        with session._gen_lock:
            held.set()
            release.wait(timeout=5)

    holder = _threading.Thread(target=_hold)
    holder.start()
    try:
        assert held.wait(timeout=5)
        with pytest.raises(ConcurrentExtractionError, match="retry shortly"):
            session.remove_probe("toy")
        # The roster is untouched — the reject protected the in-flight read.
        assert "toy" in session._monitor.probe_names
    finally:
        release.set()
        holder.join(timeout=5)

    # Idle again: the detach applies normally.
    session.remove_probe("toy")
    assert "toy" not in session._monitor.probe_names


# ================================================== geometry state lock ===

def test_geometry_state_lock_serializes_detach_and_reads():
    """The geometry-state boundary (the Monitor sibling of the lens
    round-5 fix): the session's ``remove_probe`` geometry branch and the
    coherent read surfaces (``specs``) hold the instrument's state lock —
    an idle detach can no longer land inside a concurrent registry
    iteration or tear it."""
    session = _stub_session()
    m = _toy_manifold()
    session.ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )
    session.add_probe("toy")
    inst = session._geometry_instrument

    detach_entered = threading.Event()
    read_entered = threading.Event()
    read_out: list[Any] = []

    def _detach() -> None:
        detach_entered.set()
        session.remove_probe("toy")

    def _read_specs() -> None:
        read_entered.set()
        read_out.append(inst.specs())

    with inst.state_lock:
        detacher = threading.Thread(target=_detach)
        reader = threading.Thread(target=_read_specs)
        detacher.start()
        reader.start()
        assert detach_entered.wait(timeout=5.0)
        assert read_entered.wait(timeout=5.0)
        detacher.join(timeout=0.2)
        reader.join(timeout=0.2)
        assert detacher.is_alive()  # both blocked on the geometry-state lock
        assert reader.is_alive()
        assert "toy" in session._monitor.probe_names  # nothing mutated mid-hold
    detacher.join(timeout=5.0)
    reader.join(timeout=5.0)
    assert not detacher.is_alive() and not reader.is_alive()
    assert "toy" not in session._monitor.probe_names
    assert read_out and isinstance(read_out[0], dict)


def test_geometry_idle_run_reads_hold_the_state_lock():
    """Idle-passthrough reads (an unbound ``GeometryRun.observe``) pair
    with the roster under the state lock — the round-6 idle-coherence
    contract, geometry edition.  The bound per-token path stays
    lock-free (mid-generation mutation is excluded by the detach
    reject contract, not by this lock)."""
    session = _stub_session()
    m = _toy_manifold()
    session.ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )
    session.add_probe("toy")
    inst = session._geometry_instrument
    assert inst.current_run.bound is False
    hidden = {
        layer_idx: sub.mean + sub.basis[0]
        for layer_idx, sub in m.layers.items()
    }

    entered = threading.Event()
    out: list[Any] = []

    def _idle_read() -> None:
        entered.set()
        out.append(inst.current_run.observe(0, hidden))

    with inst.state_lock:
        reader = threading.Thread(target=_idle_read)
        reader.start()
        assert entered.wait(timeout=5.0)
        reader.join(timeout=0.2)
        assert reader.is_alive()  # blocked on the geometry-state lock
    reader.join(timeout=5.0)
    assert not reader.is_alive()
    assert out and "toy" in out[0]


def test_geometry_roster_reads_do_not_tear_under_concurrent_detach():
    """The tear repro: ``plan()``'s roster walk (``probe_names`` +
    ``probe_layers``) iterates live ``Monitor._probes`` state and raised
    ``RuntimeError: dictionary changed size during iteration`` under a
    concurrent idle attach/detach before the state lock existed (the
    exact class the lens round-5 fix closed).  Hammer both sides; any
    exception on either thread fails."""
    import time as _time

    from saklas.core.instruments.types import ReadRequest

    session = _stub_session()
    manifolds = {f"toy{i}": _toy_manifold() for i in range(6)}

    def _load(key: str) -> None:
        session._manifolds.update({key: manifolds[key]})

    session.ensure_manifold_loaded = _load
    for name in manifolds:
        session.add_probe(name)
    inst = session._geometry_instrument

    stop = threading.Event()
    errors: list[BaseException] = []

    def _mutate() -> None:
        try:
            i = 0
            while not stop.is_set():
                name = f"toy{i % 6}"
                session.remove_probe(name)
                session.add_probe(name)
                i += 1
        except BaseException as exc:  # noqa: BLE001 — the assertion payload
            errors.append(exc)

    def _read() -> None:
        try:
            while not stop.is_set():
                prep = inst.prepare(ReadRequest(final_aggregate=True))
                inst.plan(prep)
                inst.specs()
                inst.manifolds()
                _ = inst.names
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=_mutate),
        threading.Thread(target=_read),
    ]
    for t in threads:
        t.start()
    _time.sleep(0.5)
    stop.set()
    for t in threads:
        t.join(timeout=10.0)
    assert not any(t.is_alive() for t in threads)
    assert not errors, errors


class _RecordingLock:
    """RLock shim recording hold depth (every production use is ``with``)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.held = 0

    def __enter__(self) -> "_RecordingLock":
        self._lock.acquire()
        self.held += 1
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.held -= 1
        self._lock.release()
        return False


def _record_roster_mutations(
    session: Any, shim: _RecordingLock,
) -> list[tuple[str, bool]]:
    """Wrap the stub monitor's roster mutators to record whether the
    geometry-state shim was held at each call."""
    monitor = session._monitor
    real_remove, real_add = monitor.remove_probe, monitor.add_probe
    held_at_call: list[tuple[str, bool]] = []

    def _rec_remove(name: str) -> None:
        held_at_call.append(("remove", shim.held > 0))
        real_remove(name)

    def _rec_add(name: str, manifold: Any, *, top_n: int = 3) -> None:
        held_at_call.append(("add", shim.held > 0))
        real_add(name, manifold, top_n=top_n)

    monitor.remove_probe = _rec_remove
    monitor.add_probe = _rec_add
    return held_at_call


def test_manifold_promotion_walk_holds_geometry_state_lock():
    """``_adopt_fitted_manifold``'s probe-promotion walk mutates the
    Monitor roster outside the instrument's attach/detach — the
    remove+add pair must land under the geometry-state lock so a
    coherent reader can't observe a half-applied promotion."""
    session = _stub_session()
    m_old = _toy_manifold()
    session.ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m_old},
    )
    session.add_probe("toy")
    session._manifolds = {"toy": m_old}
    session._device = torch.device("cpu")
    session._profiles = {}
    inst = session._geometry_instrument
    shim = _RecordingLock()
    inst.state_lock = shim  # type: ignore[assignment]
    held_at_call = _record_roster_mutations(session, shim)

    m_new = _toy_manifold()
    SaklasSession._adopt_fitted_manifold(session, "default/toy", m_new)

    assert ("remove", True) in held_at_call
    assert ("add", True) in held_at_call
    assert all(held for _op, held in held_at_call)


def test_failed_override_eviction_holds_geometry_state_lock():
    """``_evict_failed_manifold_override``'s probe eviction is the same
    out-of-instrument roster mutation — it must hold the geometry-state
    lock too."""
    session = _stub_session()
    m_old = _toy_manifold()
    session.ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m_old},
    )
    session.add_probe("toy")
    session._manifolds = {"toy": m_old}
    session._profiles = {}
    inst = session._geometry_instrument
    shim = _RecordingLock()
    inst.state_lock = shim  # type: ignore[assignment]
    held_at_call = _record_roster_mutations(session, shim)

    SaklasSession._evict_failed_manifold_override(
        session, "default/toy", sae=None,
    )

    assert ("remove", True) in held_at_call
    assert "toy" not in session._monitor.probe_names


# ============================================= observe memo priming ===

def test_geometry_run_prime_observation_contract():
    """``prime_observation`` fills the bound run's step memo so a later
    ``observe(step_id, …)`` for the same forward returns the primed
    readings without rescoring; a different step recomputes; an idle run
    never primes (it persists indefinitely)."""
    session = _stub_session()
    m = _toy_manifold()
    session.ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )
    session.add_probe("toy")
    inst = session._geometry_instrument

    # Idle run: priming is a no-op.
    idle = inst.current_run
    assert idle.bound is False
    idle.prime_observation(5, {"toy": object()})
    assert idle._memo_step is None

    from saklas.core.instruments.types import ReadRequest
    prep = inst.prepare(ReadRequest(final_aggregate=True))
    run = inst.bind(inst.plan(prep), prep)

    primed = {"toy": object()}
    run.prime_observation(5, primed)
    # Memo hit: same step returns the primed dict without touching the
    # monitor (empty hidden would otherwise score to an empty dict).
    assert run.observe(5, {}) is primed
    # A different step rescores from the given hidden states.
    latest = {
        layer_idx: sub.mean + sub.basis[0]
        for layer_idx, sub in m.layers.items()
    }
    fresh = run.observe(6, latest)
    assert fresh is not primed and "toy" in fresh
    inst.close_run()


def _wire_begin_capture(session: Any) -> dict[str, Any]:
    """Minimal extra wiring so the real ``_begin_capture`` runs on the
    fixture stub, capturing whichever step sink the mode branch installs."""
    holder: dict[str, Any] = {}
    session._layers = [None] * 4
    # The capture transaction closes ALL THREE families' runs up front —
    # bind the real method (the MagicMock default no-ops it, leaving the
    # lens/SAE runs bound and poisoning a second transaction).
    session._close_instrument_runs = types.MethodType(
        SaklasSession._close_instrument_runs, session,
    )
    session._capture.attach = lambda *_a, **_k: None
    session._capture.clear = lambda: None
    session._capture.set_incremental = (
        lambda sink: holder.__setitem__("sink", sink)
    )
    session._capture.set_tail_with_sink = (
        lambda _depth, sink, **_kw: holder.__setitem__("sink", sink)
    )
    session._incremental_readings = []
    session._incremental_gate_scores = []
    return holder


def test_full_incremental_sink_primes_geometry_observe_memo():
    """The FULL-incremental sink's rows ARE complete per-probe readings,
    so it primes the geometry run's observe memo — the gate callback and
    the token tap consult ``observe(step)`` for the same forward and hit
    it instead of rescoring."""
    session = _stub_session()
    m = _toy_manifold()
    session.ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )
    session.add_probe("toy")
    holder = _wire_begin_capture(session)

    ok = SaklasSession._begin_capture(session, widen=False, need_per_token=True)
    assert ok is True
    assert session._capture_state.mode is CaptureMode.INCREMENTAL
    run = session._geometry_instrument.current_run
    assert run.bound is True

    latest = {
        layer_idx: sub.mean + sub.basis[0]
        for layer_idx, sub in m.layers.items()
    }
    holder["sink"](3, latest)

    assert session._incremental_readings and (
        "toy" in session._incremental_readings[-1]
    )
    # The memo holds the sink's exact rows for step 3 — observe() is a hit
    # (identity, not a rescore).
    assert run.observe(3, {}) is session._incremental_readings[-1]
    session._geometry_instrument.close_run()


def test_lean_and_gating_sinks_never_prime_observe_memo():
    """LEAN rows are ``coords_only`` (no nearest / assignment / per-layer
    trace) and gating rows are scalar subsets — priming either as the full
    ``observe`` reading is the completeness trap, so neither sink primes."""
    session = _stub_session()
    m = _toy_manifold()
    session.ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )
    session.add_probe("toy")
    holder = _wire_begin_capture(session)

    ok = SaklasSession._begin_capture(
        session, widen=False, need_per_token=True, lean_per_token=True,
    )
    assert ok is True
    assert session._capture_state.mode is CaptureMode.LEAN_INCREMENTAL
    run = session._geometry_instrument.current_run

    latest = {
        layer_idx: sub.mean + sub.basis[0]
        for layer_idx, sub in m.layers.items()
    }
    holder["sink"](2, latest)
    assert session._incremental_readings  # the lean row landed
    assert run._memo_step is None  # …but never primed the memo
    session._geometry_instrument.close_run()

    # Gating-subset sink: same contract.
    session._incremental_readings = []
    session._incremental_gate_scores = []
    ok = SaklasSession._begin_capture(
        session,
        widen=False,
        need_per_token=True,
        gating_only_probes={"toy"},
        gating_probe_keys={"toy"},
        final_probe_aggregate=False,
    )
    assert ok is True
    assert session._capture_state.mode is CaptureMode.GATING_SUBSET
    run = session._geometry_instrument.current_run
    holder["sink"](7, latest)
    assert session._incremental_gate_scores  # the scalar row landed
    assert run._memo_step is None
    session._geometry_instrument.close_run()


def test_negative_step_never_caches_or_primes():
    """``step_id < 0`` is the no-identity sentinel: repeated negative
    observations rescore every time (a -1 memo would serve one stale read
    to every later sentinel call — sol's round-1 P2), and a negative
    prime is a no-op."""
    session = _stub_session()
    m = _toy_manifold()
    session.ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )
    session.add_probe("toy")
    inst = session._geometry_instrument
    from saklas.core.instruments.types import ReadRequest
    prep = inst.prepare(ReadRequest(final_aggregate=True))
    run = inst.bind(inst.plan(prep), prep)

    run.prime_observation(-1, {"toy": object()})
    assert run._memo_step is None  # negative prime is a no-op

    latest = {
        layer_idx: sub.mean + sub.basis[0]
        for layer_idx, sub in m.layers.items()
    }
    first = run.observe(-1, latest)
    assert run._memo_step is None  # negative observe never caches
    second = run.observe(-1, {})
    # A fresh scoring, not the first result served stale.
    assert second is not first and second == {}
    inst.close_run()


def test_gate_callback_consumes_the_full_incremental_memo():
    """The composer's FULL-incremental branch reads this forward's row
    through ``run.observe(step, {})`` — a memo hit on the sink's exact
    object, no rescore (the monitor is poisoned after the sink to prove
    it)."""
    session = _stub_session()
    m = _toy_manifold()
    session.ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )
    session.add_probe("toy")
    holder = _wire_begin_capture(session)
    ok = SaklasSession._begin_capture(session, widen=False, need_per_token=True)
    assert ok is True

    latest = {
        layer_idx: sub.mean + sub.basis[0]
        for layer_idx, sub in m.layers.items()
    }
    holder["sink"](0, latest)

    def _poisoned(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise AssertionError(
            "the gate callback must consume the primed memo, not rescore"
        )

    session._monitor.score_single_token = _poisoned  # type: ignore[method-assign]
    cb = session._steering_composer.build_gating_score_callback()
    flat = cb(0)
    assert "toy" in flat and "toy:fraction" in flat
    session._geometry_instrument.close_run()


def test_token_payload_consumes_the_full_incremental_memo():
    """``build_token_probe_payload``'s FULL-incremental branch is a memo
    hit too — the payload's geometry readings ARE the sink's row."""
    from saklas.core.token_payloads import build_token_probe_payload

    session = _stub_session()
    m = _toy_manifold()
    session.ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )
    session.add_probe("toy")
    holder = _wire_begin_capture(session)
    ok = SaklasSession._begin_capture(session, widen=False, need_per_token=True)
    assert ok is True

    latest = {
        layer_idx: sub.mean + sub.basis[0]
        for layer_idx, sub in m.layers.items()
    }
    holder["sink"](1, latest)

    def _poisoned(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise AssertionError(
            "the token payload must consume the primed memo, not rescore"
        )

    session._monitor.score_single_token = _poisoned  # type: ignore[method-assign]
    payload = build_token_probe_payload(
        monitor=session._monitor,
        capture=session._capture,
        capture_state=session._capture_state,
        incremental_readings=session._incremental_readings,
        geometry_run=session._geometry_instrument.current_run,
        step_id=1,
        needs_scores=True,
        persists_layer_scores=False,
        assistant_node_id=None,
    )
    assert payload.geometry_readings is session._incremental_readings[-1]
    session._geometry_instrument.close_run()


def test_geometry_run_observe_aggregate_matches_live_read():
    """The finalize aggregate routed through ``GeometryRun.observe_
    aggregate`` is bit-identical to a live ``score_single_token`` read at
    the pooled slice — the protocol claim, pinned on a CURVED probe (the
    foot-solve path, the hard case)."""
    session = _stub_session()
    m = _toy_manifold()
    session.ensure_manifold_loaded = lambda key: session._manifolds.update(
        {key: m},
    )
    session.add_probe("toy")
    pooled = {
        layer_idx: sub.mean + sub.basis[0]
        for layer_idx, sub in m.layers.items()
    }

    via_run = session._geometry_instrument.current_run.observe_aggregate(
        pooled,
    )
    live = session._monitor.score_single_token(pooled)

    assert set(via_run) == set(live) == {"toy"}
    assert via_run["toy"].coords == live["toy"].coords
    assert via_run["toy"].fraction == live["toy"].fraction
    assert via_run["toy"].residual == live["toy"].residual
    assert via_run["toy"].nearest == live["toy"].nearest


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
    flat = cb(0)
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
    assert cb(0) == {}


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
