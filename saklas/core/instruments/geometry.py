"""The geometry instrument: a thin adapter over the unified ``Monitor``.

The geometry family's engine — whitened subspace reads, the four
conditional capture modes' scoring entry points, flat-batched and curved
foot-solve paths — stays in ``core/monitor.py``/``core/monitor_attach.py``
untouched: the capture modes are session/HiddenCapture state and the
Monitor is an established engine, so folding it into the instrument
abstraction would combine two independent risks (the orchestration
extraction and an engine rewrite) for no architectural reward.  This
adapter gives the family the same face as the lens/SAE instruments —
attach/detach/specs, gate-channel capabilities, probe-hash identity — so
the session can expose one ``instruments`` registry and the HTTP layer
can enumerate families uniformly.
"""

from __future__ import annotations

import hashlib
from typing import Any, TYPE_CHECKING

import torch

from saklas.core.instruments.types import (
    Assignment,
    Axis,
    Distance,
    Fraction,
    GateRef,
    InstrumentBinding,
    InstrumentPlan,
    InstrumentPrep,
    Membership,
    ReadRequest,
    parse_gate_ref,
    validate_gate_channels,
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
        must not read stale)."""
        if not self.bound:
            return self._instrument._session._monitor.score_single_token(hidden)
        if self._memo_step == step_id and self._memo_readings is not None:
            return self._memo_readings
        readings = self._instrument._session._monitor.score_single_token(hidden)
        self._memo_step = step_id
        self._memo_readings = readings
        return readings

    def gate_scalars(
        self,
        step_id: int,
        hidden: dict[int, torch.Tensor],
        gate_keys: frozenset[str] | set[str],
    ) -> dict[str, float]:
        """The gated subset's scalar channels at this step."""
        del step_id
        monitor = self._instrument._session._monitor
        plan = monitor.plan_gate_scalars(set(gate_keys))
        if not plan:
            return {}
        return monitor.score_planned_gate_scalars(hidden, plan)

    def observe_aggregate(
        self, pooled: dict[int, torch.Tensor],
    ) -> dict[str, Any]:
        """One full-roster read at the pooled last-content slice —
        bit-identical to a live read at that token."""
        return self._instrument._session._monitor.score_single_token(pooled)

    def observe_many(
        self, pooled_rows: "list[dict[int, torch.Tensor]]",
    ) -> list[dict[str, Any]]:
        return [self.observe_aggregate(rows) for rows in pooled_rows]

    def close(self) -> None:
        self._memo_step = None
        self._memo_readings = None


class GeometryInstrument:
    """Session-lifetime handle for the whitened-geometry read family."""

    family = "geometry"

    #: Every gate channel: axes, fraction, membership, label distance,
    #: soft assignment — the full whitened-reading key family.
    _GATE_CHANNELS: tuple[type, ...] = (
        Axis, Fraction, Membership, Distance, Assignment,
    )

    def __init__(self, session: "SaklasSession") -> None:
        self._session = session
        # The current per-generation run (idle passthrough until bind()).
        self.current_run = GeometryRun(
            self, InstrumentBinding(family=self.family),
        )

    # ------------------------------------------------------------ run lifecycle

    def prepare(self, request: ReadRequest) -> InstrumentPrep:
        """Generation-boundary prep — geometry has no source lifecycle
        (nothing to refresh or pin) and no run-level live channel, so
        the bare prep is returned purely to keep the session's capture
        transaction uniform across families."""
        if self.current_run.bound:
            raise RuntimeError(
                "GeometryInstrument.prepare() on a bound run: close the "
                "prior generation's run (_close_instrument_runs) first"
            )
        del request
        return InstrumentPrep(family=self.family)

    def bind(
        self, plan: InstrumentPlan, prep: InstrumentPrep | None = None,
    ) -> GeometryRun:
        """Bind an immutable per-generation run.

        The binding carries the probe-name roster only: geometry specs
        cannot be mutated by any un-locked path (attach/detach hold
        ``_model_exclusive``), so unlike the SAE family there is no
        mid-generation mutation to freeze against, and the full spec walk
        (which touches Monitor internals) stays off the per-generation
        path.  The prep is threaded for protocol uniformity only.
        """
        del plan, prep  # demand already consumed by the capture planner
        run = GeometryRun(
            self,
            InstrumentBinding(
                family=self.family,
                specs={
                    str(name): {}
                    for name in self._session._monitor.probe_names
                },
            ),
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
        of racing.
        """
        with self._session._model_exclusive(
            "remove_probe called while another model operation is in "
            "flight; retry shortly"
        ):
            self._session._monitor.remove_probe(name)

    @property
    def names(self) -> list[str]:
        return list(self._session._monitor.probe_names)

    def specs(self) -> dict[str, dict[str, Any]]:
        """Attached probe spec snapshots — the geometry analogue of the
        lens/SAE spec dicts (manifold identity + shape flags; the full
        wire info shape stays in ``server/probe_routes``)."""
        out: dict[str, dict[str, Any]] = {}
        for name, probe in self._session._monitor.attached_probes().items():
            out[name] = {
                "manifold": probe.manifold.name,
                "top_n": int(probe.top_n),
                "is_affine": bool(probe.is_affine),
                "layers": sorted(probe.manifold.layers),
            }
        return out

    def validate_gate(self, ref: GateRef) -> None:
        validate_gate_channels(ref, self._GATE_CHANNELS, family=self.family)

    # ---------------------------------------------------------------- planning

    def plan(self, request: ReadRequest) -> InstrumentPlan:
        """Declare the monitor roster's capture demand for one generation.

        Demand, not mechanics (``protocol.py``): which layers must be
        captured for the roster to read, whether anything reads per step,
        and which gate scalar keys belong to this family.  The session
        planner unions plans across families and picks physical retention.

        When probe gates are the family's *sole* per-token consumer and
        the caller disabled final probe readings, demand narrows to the
        gated probes' layer union — dormant pinned probes must not keep
        capture alive (FIX #4's layer-union half).
        """
        monitor = self._session._monitor
        names = set(monitor.probe_names)
        if not names:
            return InstrumentPlan(family=self.family)
        gate_keys = frozenset(
            key for key in request.gate_keys
            if parse_gate_ref(key).probe in names
        )
        per_token = bool(gate_keys or request.per_token_consumers)
        if not (per_token or request.final_aggregate):
            # Dormant roster: probes attached, but nothing this generation
            # consumes a reading (no gate, no per-token consumer, final
            # readings disabled).
            return InstrumentPlan(family=self.family, gate_keys=gate_keys)
        narrow_to_gated = bool(
            gate_keys
            and not request.per_token_consumers
            and not request.final_aggregate
        )
        if narrow_to_gated:
            gated_names = {parse_gate_ref(key).probe for key in gate_keys}
            probe_layers = monitor.probe_layers(gated_names)
        else:
            # Bare call — the full-roster union (also what duck-typed
            # monitor stubs without the subset parameter implement).
            probe_layers = monitor.probe_layers()
        latest = frozenset(int(layer) for layer in probe_layers)
        return InstrumentPlan(
            family=self.family,
            latest_layers=latest,
            per_step=per_token,
            gate_keys=gate_keys,
            final_aggregate=bool(request.final_aggregate),
            batch_aggregate=bool(request.batch and request.final_aggregate),
        )

    def probe_hash(self, name: str) -> str | None:
        """sha256 of the probe's baked tensor bytes (deterministic across
        machines/devices; fp32-normalized).  A 2-node concept hashes its
        folded baked-direction view for continuity with the pre-coords
        scalar monitor's drift check; a multi-node / curved probe hashes
        the per-layer subspace geometry directly."""
        session = self._session
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
