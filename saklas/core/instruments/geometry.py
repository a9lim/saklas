"""The geometry instrument: a thin adapter over the unified ``Monitor``.

The geometry family's engine — whitened subspace reads, the four
conditional capture modes' scoring entry points, flat-batched and curved
foot-solve paths — stays in ``core/monitor.py``/``core/monitor_attach.py``
untouched: the capture modes are session/HiddenCapture state and the
Monitor is an established engine, so folding it into the instrument
abstraction would combine two independent risks (the orchestration
extraction and an engine rewrite) for no architectural reward.  This
adapter gives the family the same face as the lens/SAE instruments — the
:class:`~saklas.core.instruments.protocol.Instrument` contract: run
lifecycle, attach/detach/specs, gate-channel validation, probe-hash
identity, the live toggle, the active source, and the token-readout replay
— so the session exposes one ``instruments`` registry and the HTTP layer
dispatches over it instead of hand-writing a branch per family.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, TYPE_CHECKING

import torch

from saklas.core.instruments.types import (
    GateRef,
    GeometryLiveState,
    InstrumentBinding,
    InstrumentFamily,
    InstrumentPlan,
    InstrumentPrep,
    ReadRequest,
    next_prep_token,
    parse_gate_ref,
)

if TYPE_CHECKING:
    from saklas.core.session import SaklasSession


class GeometryRun:
    """Per-generation measurement executor for the geometry family.

    Deliberately thin: the Monitor engine owns the whitened reads and its
    own generation-scoped warm state (curved feet, warm-start flags reset
    by the capture planner), and the four capture modes' per-token scoring
    stays session/HiddenCapture wiring.  The run contributes what its
    siblings do — the immutable per-generation :class:`InstrumentBinding`
    spec snapshot, the ``observe`` step memo, and the uniform protocol
    face over the Monitor's scoring entry points.
    """

    def __init__(
        self,
        instrument: "GeometryInstrument",
        binding: InstrumentBinding,
        *,
        bound: bool = False,
    ) -> None:
        self._instrument = instrument
        self.binding = binding
        self.bound = bound
        self._memo_step: int | None = None
        self._memo_readings: dict[str, Any] | None = None

    # ------------------------------------------------------------ protocol

    def observe(
        self, step_id: int, hidden: dict[int, torch.Tensor],
    ) -> dict[str, Any]:
        """Full whitened readings for the roster at this step, memoized by
        ``step_id``.  An idle run never memoizes (it persists
        indefinitely; a repeated ``step_id`` with different hidden states
        must not read stale).  The idle read holds the geometry-state
        lock — the Monitor's roster walk tears under a concurrent idle
        attach/detach; the bound path stays lock-free (per-token hot
        path — mid-generation mutation is excluded by ``detach``'s
        reject-during-generation contract, not by a lock)."""
        if not self.bound:
            with self._instrument.state_lock:
                return self._instrument._session._monitor.score_single_token(
                    hidden,
                )
        if (
            step_id >= 0
            and self._memo_step == step_id
            and self._memo_readings is not None
        ):
            return self._memo_readings
        readings = self._instrument._session._monitor.score_single_token(hidden)
        if step_id >= 0:
            # A negative step is the "no step identity" sentinel — caching
            # under it would serve one stale read to every later
            # sentinel-stepped call.
            self._memo_step = step_id
            self._memo_readings = readings
        return readings

    def prime_observation(
        self, step_id: int, readings: dict[str, Any],
    ) -> None:
        """Prime the step memo with FULL per-probe readings a hot-path
        scorer already computed this forward (the FULL-incremental sink) —
        a later ``observe(step_id, …)`` for the same forward returns them
        without a second scoring pass.  Bound runs only (an idle run never
        memoizes); a negative step (the no-identity sentinel) never primes.
        Callers MUST pass complete-roster readings: a gating
        scalar subset or a ``coords_only`` lean row primed here would be
        served as the full reading (the completeness trap), so those sinks
        never prime."""
        if not self.bound or step_id < 0:
            return
        self._memo_step = step_id
        self._memo_readings = readings

    def observe_aggregate(
        self, pooled: dict[int, torch.Tensor],
    ) -> dict[str, Any]:
        """One full-roster read at the pooled last-content slice —
        bit-identical to a live read at that token (``Monitor.score_
        aggregate`` routes through the same full scorer as
        ``score_single_token``; it additionally fp32-casts and row-selects
        ``[T, D]`` stacks, so it accepts every pooled shape the finalize
        paths produce).  The session's ``_score_aggregate_only`` reaches
        the roster through this method.  Idle reads hold the
        state lock; the bound finalize read runs under ``_gen_lock``,
        which every roster mutation also takes (``_model_exclusive``)."""
        if not self.bound:
            with self._instrument.state_lock:
                return self._instrument._session._monitor.score_aggregate(
                    pooled,
                )
        return self._instrument._session._monitor.score_aggregate(pooled)

    def close(self) -> None:
        self._memo_step = None
        self._memo_readings = None


class GeometryInstrument:
    """Session-lifetime handle for the whitened-geometry read family."""

    family: InstrumentFamily = "geometry"

    # No gate-channel capability list: the whitened reading produces EVERY
    # channel in the key family (axes, fraction, membership, label distance,
    # soft assignment), so there is nothing for a composition preflight to
    # reject.  The lens/SAE families carry one because their single strength
    # axis genuinely cannot answer a geometry channel.

    def __init__(self, session: "SaklasSession") -> None:
        self._session = session
        # The CAA live toggle — whether per-token monitor scoring feeds live
        # consumers.  Owned here (not on the session) so the three families
        # answer ``live_state``/``set_live`` from one place.
        self._live = True
        # THE geometry-state boundary — the roster-coherence sibling of the
        # lens/SAE ``state_lock``s.  One reentrant leaf lock serializing
        # every Monitor roster mutation (attach/detach, the session's
        # fit-promotion and failed-override eviction walks, the whitener
        # rebuild) against the coherent read surfaces (``names``/``specs``/
        # ``manifolds``/``probe_hash``, the ``plan``/``bind`` roster reads,
        # and the idle-passthrough run reads) — an un-locked reader
        # iterating ``Monitor._probes`` RuntimeErrors under a concurrent
        # idle detach — the same tear class the lens boundary closes.
        # A leaf lock: nothing acquires ``_gen_lock``/``_model_exclusive``
        # while holding it (callers take those first), and it is NEVER
        # taken on the bound per-token scoring path — mid-generation
        # mutation is excluded by ``detach``'s reject-during-generation
        # contract (``_model_exclusive``), not by this lock.
        self.state_lock = threading.RLock()
        # The current per-generation run (idle passthrough until bind()).
        self.current_run = GeometryRun(
            self, InstrumentBinding(family=self.family),
        )

    # ------------------------------------------------------------ run lifecycle

    def prepare(self, request: ReadRequest) -> InstrumentPrep:
        """Generation-boundary prep — geometry has no source lifecycle
        (nothing to refresh or pin) and no run-level live channel, so
        the prep only carries the request forward for ``plan``, keeping
        the session's capture transaction uniform across families."""
        if self.current_run.bound:
            raise RuntimeError(
                "GeometryInstrument.prepare() on a bound run: close the "
                "prior generation's run (_close_instrument_runs) first"
            )
        return InstrumentPrep(
            family=self.family,
            request=request,
            token=next_prep_token(),
        )

    def bind(self, plan: InstrumentPlan, prep: InstrumentPrep) -> GeometryRun:
        """Bind an immutable per-generation run.

        The binding carries the probe-name roster only: geometry specs
        cannot be mutated by any un-locked path (attach/detach hold
        ``_model_exclusive``), so unlike the SAE family there is no
        mid-generation mutation to freeze against, and the full spec walk
        (which touches Monitor internals) stays off the per-generation
        path.
        """
        if prep.family != self.family:
            raise TypeError(
                "GeometryInstrument.bind takes the InstrumentPrep its own "
                f"prepare() returned, got family={prep.family!r}"
            )
        if plan.family != self.family:
            raise ValueError(
                f"GeometryInstrument.bind: plan family {plan.family!r} is "
                f"not {self.family!r}"
            )
        if plan.prep_token != prep.token:
            raise ValueError(
                "GeometryInstrument.bind: the plan was not derived from "
                "this prep (prep_token mismatch) — derive the plan from "
                "the same prepare() call"
            )
        with self.state_lock:
            specs = {
                str(name): {}
                for name in self._session._monitor.probe_names
            }
        run = GeometryRun(
            self,
            InstrumentBinding(family=self.family, specs=specs),
            bound=True,
        )
        self.current_run = run
        return run

    def close_run(self) -> None:
        """Close the current run and restore the idle passthrough run."""
        self.current_run.close()
        self.current_run = GeometryRun(
            self, InstrumentBinding(family=self.family),
        )

    # -------------------------------------------------------------- registry

    def attach(
        self,
        selector: str,
        *,
        as_name: str | None = None,
        top_n: int = 3,
    ) -> str:
        """Attach a manifold probe (any shape — a 2-node concept axis is
        the rank-1 case) to the unified Monitor.

        Probe attach loads the manifold onto the model device and builds
        device-resident whitened factors — GPU work that must not run
        concurrently with another model op; held under the exclusive
        ``_model_exclusive`` section like the other families' attaches.
        Cache invalidation stays at the session's ``add_probe`` boundary.
        """
        session = self._session
        name = as_name if as_name is not None else selector
        with session._model_exclusive(
            "add_probe called while another model operation is in "
            "flight; retry shortly"
        ):
            # Manifold reads are Mahalanobis-only. Build the neutral artifact
            # under this same exclusive section so a concurrent generation
            # cannot race it.
            _ = session.whitener
            manifold = session._resolve_probe_manifold(selector)
            # Geometry-state boundary: the roster write lands atomically
            # against the un-locked coherent readers (consistency against
            # eviction/promotion comes from ``_model_exclusive``, which
            # those walks' callers hold).
            with self.state_lock:
                session._monitor.add_probe(name, manifold, top_n=top_n)
        return name

    def detach(self, name: str) -> None:
        """Detach under the exclusive section, like ``attach``.

        Monitor scoring walks the live roster (the geometry family has no
        per-generation roster snapshot yet — unlike lens/SAE, whose frozen
        bindings make mid-generation detach harmless), so a removal racing
        an in-flight generation would change what that generation measures
        and can race the Monitor's cache rebuilds.  A detach during a
        generation therefore rejects with retry-shortly semantics instead
        of racing.  The roster write itself additionally holds the
        geometry-state lock so an *idle* coherent reader (``specs``,
        ``plan``, the session-info route) can't tear mid-removal.
        """
        with self._session._model_exclusive(
            "remove_probe called while another model operation is in "
            "flight; retry shortly"
        ):
            with self.state_lock:
                self._session._monitor.remove_probe(name)

    def try_detach(self, name: str) -> bool:
        """Detach ``name`` if this family owns it; ``False`` when it doesn't.

        The uniform registry-removal face (``Instrument.try_detach``) the
        session's ``remove_probe`` walks across families.  Membership test
        and removal are one atomic state-lock hold, like the lens/SAE
        siblings — but the removal itself additionally takes
        ``_model_exclusive`` (see :meth:`detach`), so the check runs first
        and cheaply.
        """
        with self.state_lock:
            if name not in self._session._monitor.probe_names:
                return False
        self.detach(name)
        return True

    def validate_gate(self, ref: GateRef) -> None:
        """Accept every channel.

        The whitened reading produces the entire key family (axes, fraction,
        membership, label distance, soft assignment), so there is nothing a
        composition preflight could reject.  Present so the composer can
        validate uniformly across ``session.instruments`` instead of
        special-casing the family that happens to answer everything.
        """
        return None

    @property
    def active_source(self) -> str | None:
        """Always ``None`` — geometry has no source lifecycle (Monitor
        probes attach directly; there is nothing to fetch or switch)."""
        return None

    @property
    def is_live(self) -> bool:
        """Whether per-token monitor scoring feeds live consumers.

        When False, generations run aggregate-only capture — probes still
        report the end-of-gen aggregate, but no per-token stream, loom token
        rows, or trait events are produced.  Probe gates are unaffected: a
        gate forces the per-token subset it needs.
        """
        return self._live

    @property
    def live_state(self) -> GeometryLiveState:
        return GeometryLiveState(enabled=self._live)

    def set_live(self, enabled: bool, **kwargs: Any) -> GeometryLiveState:
        """Toggle live per-token monitor scoring (all-or-nothing).

        Takes no family extras: per-token geometry scoring has no layer or
        width dial (the roster's own fitted layers drive capture), so a
        stray ``layers=``/``top_k=`` is a caller error, not a silent no-op.
        """
        if kwargs:
            raise TypeError(
                "geometry live takes no extras (per-token monitor scoring "
                f"is all-or-nothing), got {sorted(kwargs)}"
            )
        self._live = bool(enabled)
        return self.live_state

    @property
    def names(self) -> list[str]:
        with self.state_lock:
            return list(self._session._monitor.probe_names)

    def specs(self) -> dict[str, dict[str, Any]]:
        """Attached probe spec snapshots — the geometry analogue of the
        lens/SAE spec dicts (manifold identity + shape flags; the full
        wire info shape stays in ``server/probe_routes``).  One coherent
        state-lock read — a concurrent detach cannot tear the iteration."""
        with self.state_lock:
            out: dict[str, dict[str, Any]] = {}
            for name, probe in (
                self._session._monitor.attached_probes().items()
            ):
                out[name] = {
                    "manifold": probe.manifold.name,
                    "top_n": int(probe.top_n),
                    "is_affine": bool(probe.is_affine),
                    "layers": sorted(probe.manifold.layers),
                }
            return out

    def manifolds(self) -> "dict[str, Any]":
        """Locked snapshot of the attached probes' manifolds (name →
        :class:`Manifold`) — the geometry read behind ``session.probes``
        and the analytics roster; a raw ``Monitor.manifolds`` comprehension
        tears under a concurrent detach."""
        with self.state_lock:
            return self._session._monitor.manifolds

    # ---------------------------------------------------------------- planning

    def plan(self, prep: InstrumentPrep) -> InstrumentPlan:
        """Declare the monitor roster's capture demand for one generation.

        Demand, not mechanics (``protocol.py``): which layers must be
        captured for the roster to read, whether anything reads per step,
        and which gate scalar keys belong to this family.  The session
        planner unions plans across families and picks physical retention.
        The prep carries the request; roster mutations hold
        ``_model_exclusive``, so there is no snapshot to consume.

        When probe gates are the family's *sole* per-token consumer and
        the caller disabled final probe readings, demand narrows to the
        gated probes' layer union — a dormant pinned probe must not keep
        capture alive for a layer nothing this generation reads.
        """
        if prep.family != self.family:
            raise TypeError(
                "GeometryInstrument.plan takes the InstrumentPrep its own "
                f"prepare() returned, got family={prep.family!r}"
            )
        request = prep.request
        monitor = self._session._monitor
        # One state-lock hold for the roster reads: the name set and the
        # layer union must come from the same roster (an un-locked
        # ``probe_layers`` iterates a live generator over ``_probes`` and
        # RuntimeErrors under a concurrent idle detach; the two-read
        # TOCTOU could also pair a name set with a mutated layer union).
        with self.state_lock:
            names = set(monitor.probe_names)
            if not names:
                return InstrumentPlan(
                    family=self.family, prep_token=prep.token,
                )
            gate_keys = frozenset(
                key for key in request.gate_keys
                if parse_gate_ref(key).probe in names
            )
            per_token = bool(gate_keys or request.per_token_consumers)
            if not (per_token or request.final_aggregate):
                # Dormant roster: probes attached, but nothing this
                # generation consumes a reading (no gate, no per-token
                # consumer, final readings disabled).
                return InstrumentPlan(
                    family=self.family,
                    gate_keys=gate_keys,
                    prep_token=prep.token,
                )
            narrow_to_gated = bool(
                gate_keys
                and not request.per_token_consumers
                and not request.final_aggregate
            )
            if narrow_to_gated:
                gated_names = {
                    parse_gate_ref(key).probe for key in gate_keys
                }
                probe_layers = monitor.probe_layers(gated_names)
            else:
                # Bare call — the full-roster union (also what duck-typed
                # monitor stubs without the subset parameter implement).
                probe_layers = monitor.probe_layers()
        latest = frozenset(int(layer) for layer in probe_layers)
        return InstrumentPlan(
            family=self.family,
            latest_layers=latest,
            gate_keys=gate_keys,
            final_aggregate=bool(request.final_aggregate),
            prep_token=prep.token,
        )

    # -------------------------------------------------------------- replay

    def token_readout(
        self,
        node_id: str,
        raw_index: int,
        *,
        top_k: int | None = None,
        layers: "list[int] | str | None" = None,
        apply_steering: bool = True,
        raw: bool = False,
    ) -> dict[str, Any]:
        """The Monitor-roster replay, as the finished ``scope="replay"``
        measurement envelope.

        ``top_k`` and ``layers`` are **rejected**, not ignored: the roster's
        own fitted layers drive the capture and a whitened reading has no
        top-k width, so a caller passing either is asking for something this
        family cannot do.  (The route used to drop both silently, which is
        how a client learns the wrong thing about what it asked for.)
        """
        from saklas.core.measurements import build_measurements

        if top_k is not None:
            raise ValueError(
                "geometry token-readout takes no top_k (a whitened probe "
                "reading has no top-k width)"
            )
        if layers is not None:
            raise ValueError(
                "geometry token-readout takes no layers (the attached "
                "roster's own fitted layers drive the capture)"
            )
        out = self._session.geometry_token_readout(
            node_id, raw_index, apply_steering=apply_steering, raw=raw,
        )
        measurements = build_measurements(
            scope="replay",
            provenance="replayed",
            geometry_readings=out.get("readings"),
            # The shared binding wire shape carries both keys; geometry has
            # no source lifecycle, so source is always null.
            geometry_binding={
                "source": self.active_source,
                "steering": (out.get("steering") if apply_steering else None),
            },
        )
        return {"measurements": measurements}

    def probe_hash(self, name: str) -> str | None:
        """sha256 of the probe's baked tensor bytes (deterministic across
        machines/devices; fp32-normalized).  A 2-node concept hashes its
        folded baked-direction view for continuity with the pre-coords
        scalar monitor's drift check; a multi-node / curved probe hashes
        the per-layer subspace geometry directly."""
        session = self._session
        # Fetch under the state lock (the ``manifolds`` comprehension
        # iterates the roster); hash outside it — the manifold object is
        # immutable after attach (promotion swaps in a NEW object), so a
        # reference fetched here stays valid for the tensor walk below.
        with self.state_lock:
            manifold = session._monitor.manifolds.get(name)
        if manifold is None:
            return None
        from saklas.core.capture import folded_directions

        h = hashlib.sha256()
        try:
            profile = folded_directions(manifold)
            per_layer: dict[int, list[torch.Tensor]] = {
                L: [profile[L]] for L in profile
            }
        except ValueError:
            per_layer = {}
            for layer_idx, sub in manifold.layers.items():
                tensors = [sub.mean, sub.basis]
                if sub.node_coords is not None:
                    tensors.append(sub.node_coords)
                per_layer[layer_idx] = tensors
        for layer_idx in sorted(per_layer.keys()):
            for tensor in per_layer[layer_idx]:
                arr = tensor.detach().to("cpu").to(torch.float32).contiguous()
                h.update(arr.numpy().tobytes())
        return h.hexdigest()


__all__ = ["GeometryInstrument"]
