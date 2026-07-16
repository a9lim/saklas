"""The SAE instrument: feature probes, gates, live discovery.

Owns everything SAE-probe-shaped that used to live inline in
``SaklasSession`` (the ~560-line SAE region): the probe registry, the
live-discovery config, the per-forward stash, the per-generation active
flag, and the read surfaces (attach / per-step scoring / gate scalars /
finalize aggregate / live display step / authored-prefill computation).
The session re-exposes the state fields under their historical private
names via delegating properties.

Backend *residency* stays session-side (``_sae_backend``/``_sae_layer``/
``_sae_width``, ``_require_sae``, ``_encode_sae_hidden``, the Neuronpedia
metadata cache + fetchers, train/load/unload lifecycle): residency is
runtime state shared with steering atoms and the offline token replay,
not probe intent — the same RuntimeState/LiveConfig split the lens
family keeps.

Deliberate semantics change riding this extraction (the 5.x clean
break): **the fake gate channels die.**  SAE gate scalars historically
emitted ``<name>:fraction = 0.0`` and ``<name>:membership = 1.0`` —
constants masquerading as measurements, an artifact of pretending a
feature activation is a geometry reading.  An SAE probe now emits only
its one real channel (``<name>`` / ``<name>[0]``, the normalized
strength); a gate on a channel the family can never produce is a
composition-preflight error (``validate_gate``), not a silent constant.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Mapping, TYPE_CHECKING

import torch

from saklas.core.instruments.types import (
    AGG_TAIL_DEPTH,
    Axis,
    GateRef,
    InstrumentBinding,
    InstrumentPlan,
    InstrumentPrep,
    ReadRequest,
    next_prep_token,
    parse_gate_ref,
    validate_gate_channels,
)

if TYPE_CHECKING:
    from saklas.core.results import ProbeReading
    from saklas.core.session import SaklasSession


class SaeRun:
    """Per-generation measurement executor for the SAE family.

    Owns the generation-scoped state (``protocol.py``): the immutable
    :class:`InstrumentBinding` — probe specs frozen at bind with the
    normalization unit (``max_act``) **resolved** into the snapshot — the
    live-active flag, the per-forward encode stash, and the ``observe``
    memo.  The frozen specs are the fix for the metadata-backfill race:
    ``fetch_sae_feature_meta`` mutates attached specs and the meta cache
    without the generation lock, so a running generation must measure
    against its bind-time snapshot, never the live registry.  An idle run
    (``bound=False``) reads the live registry — between generations a
    Neuronpedia refresh still changes the unit immediately.
    """

    def __init__(
        self,
        instrument: "SaeInstrument",
        binding: InstrumentBinding,
        *,
        active: bool = True,
        bound: bool = False,
    ) -> None:
        self._instrument = instrument
        self.binding = binding
        self.active = active
        self.bound = bound
        self.step_stash: dict[str, Any] | None = None
        self.last_step_readings: dict[str, "ProbeReading"] | None = None
        self._memo_step: int | None = None
        self._memo_readings: dict[str, "ProbeReading"] | None = None

    # ------------------------------------------------------------ protocol

    def observe(
        self, step_id: int, hidden: dict[int, torch.Tensor],
    ) -> dict[str, "ProbeReading"]:
        """Readings for every attached probe at this step, memoized by
        ``step_id`` while bound.  The workers' full-roster reads prime
        this memo (``prime_observation``); the encode-level gate→display
        reuse stays the step-keyed worker stash.  An idle run never
        memoizes (it persists indefinitely; a repeated ``step_id`` with
        different hidden states must not read stale)."""
        if not self.bound:
            return self._instrument.score_probes(hidden)
        if (
            step_id >= 0
            and self._memo_step == step_id
            and self._memo_readings is not None
        ):
            return self._memo_readings
        readings = self._instrument.score_probes(hidden)
        if step_id >= 0:
            # Negative steps are the no-identity sentinel — never cacheable
            # (a -1 memo would serve stale reads to every later -1 call).
            self._memo_step = step_id
            self._memo_readings = readings
        return readings

    def gate_scalars(
        self,
        step_id: int,
        hidden: dict[int, Any] | None,
        gate_keys: frozenset[str] | set[str] | None,
    ) -> dict[str, float]:
        """The gate channels' scalars for this step (the encode stash the
        display step reuses lives on this run, keyed by ``step_id``).
        ``gate_keys=None`` scores the full roster (the session forwarder's
        bare-call shape)."""
        del hidden  # worker reads the capture's latest slices
        return self._instrument.gate_scalars(
            set(gate_keys) if gate_keys is not None else None,
            step_id=step_id,
        )

    def prime_observation(
        self, step_id: int, readings: dict[str, "ProbeReading"],
    ) -> None:
        """Prime the step memo with FULL-roster readings a hot-path worker
        already computed this forward — a later ``observe(step_id, …)``
        returns them without recomputing.  Bound runs only; callers must
        never prime an ``only=``-subset read (the completeness trap)."""
        if not self.bound or step_id < 0:
            return
        self._memo_step = step_id
        self._memo_readings = readings

    def observe_aggregate(
        self, pooled: dict[int, Any],
    ) -> dict[str, "ProbeReading"]:
        """End-of-generation aggregate at the pooled last-content slice."""
        return self._instrument.score_probes(pooled)

    def observe_many(
        self, pooled_rows: "list[dict[int, Any]]",
    ) -> list[dict[str, "ProbeReading"]]:
        """Batch-generation aggregates: one reading set per row."""
        return [self.observe_aggregate(rows) for rows in pooled_rows]

    def close(self) -> None:
        """Release generation-scoped state (stash, memo)."""
        self.step_stash = None
        self.last_step_readings = None
        self._memo_step = None
        self._memo_readings = None
        self.active = True


class SaeInstrument:
    """Session-lifetime handle for the SAE read family."""

    family = "sae"

    #: Gate channels an SAE probe can produce: the one activation axis.
    _GATE_CHANNELS: tuple[type, ...] = (Axis,)

    def __init__(self, session: "SaklasSession") -> None:
        self._session = session
        # Pinned SAE feature probes: name -> {feature_id, layer, label,
        # max_act}.  NOT monitor probes — they read the encoder channel.
        self.probes: dict[str, dict[str, Any]] = {}
        # Live feature discovery: {layer, top_k, source}, or None when off.
        self.live: dict[str, Any] | None = None
        # The SAE registry boundary (round-6 P2, the lens state_lock's
        # sibling): one reentrant leaf lock covering registry mutation
        # (attach/detach, the session's metadata backfill and load/unload
        # probe eviction) and the coherent snapshots (specs/names/
        # probe_hash, the idle _measurement_specs copy).  Never held on
        # per-token paths — bound runs read their frozen binding.
        self.state_lock = threading.RLock()
        # The current per-generation run (idle passthrough until bind()).
        self.current_run = SaeRun(
            self, InstrumentBinding(family=self.family),
        )

    # ------------------------------------------------------------- run state

    @property
    def step_stash(self) -> dict[str, Any] | None:
        return self.current_run.step_stash

    @step_stash.setter
    def step_stash(self, value: "dict[str, Any] | None") -> None:
        self.current_run.step_stash = value

    @property
    def last_step_readings(self) -> "dict[str, ProbeReading] | None":
        return self.current_run.last_step_readings

    @last_step_readings.setter
    def last_step_readings(
        self, value: "dict[str, ProbeReading] | None",
    ) -> None:
        self.current_run.last_step_readings = value

    @property
    def active_for_generation(self) -> bool:
        return self.current_run.active

    @active_for_generation.setter
    def active_for_generation(self, value: bool) -> None:
        self.current_run.active = bool(value)

    # ------------------------------------------------------------ run lifecycle

    def prepare(self, request: ReadRequest) -> InstrumentPrep:
        """Generation-boundary prep — no disk-backed source to refresh
        (backend residency is explicit lifecycle, ``load_sae``/
        ``train_sae``) and no lens-style identity coupling between the
        registry and the source, so the prep only carries the request
        forward for ``plan``/``bind``.  The bound-run guard keeps the
        transaction contract uniform across families."""
        if self.current_run.bound:
            raise RuntimeError(
                "SaeInstrument.prepare() on a bound run: close the prior "
                "generation's run (_close_instrument_runs) first"
            )
        return InstrumentPrep(
            family=self.family,
            request=request,
            token=next_prep_token(),
        )

    def bind(self, plan: InstrumentPlan, prep: InstrumentPrep) -> SaeRun:
        """Bind an immutable per-generation run from a declared plan.

        The binding freezes each attached probe's spec **with its
        normalization unit resolved**: ``max_act`` falls back to the
        session metadata cache at bind time, so a mid-generation
        Neuronpedia backfill (which mutates specs and cache without the
        generation lock) cannot change what a running generation measures.
        """
        if not isinstance(prep, InstrumentPrep) or prep.family != self.family:
            raise TypeError(
                "SaeInstrument.bind takes the InstrumentPrep its own "
                f"prepare() returned, got {type(prep).__name__} "
                f"(family={getattr(prep, 'family', None)!r})"
            )
        if plan.family != self.family:
            raise ValueError(
                f"SaeInstrument.bind: plan family {plan.family!r} is not "
                f"{self.family!r}"
            )
        if plan.prep_token != prep.token:
            raise ValueError(
                "SaeInstrument.bind: the plan was not derived from this "
                "prep (prep_token mismatch) — derive the plan from the same "
                "prepare() call"
            )
        live_active = prep.request.live
        session = self._session
        # Duck-typed narrow stubs may lack the metadata cache; a real
        # session always carries it (set in ``__init__``).
        has_meta_cache = getattr(session, "_sae_feature_meta", None) is not None
        specs: dict[str, dict[str, Any]] = {}
        with self.state_lock:
            probe_items = list(self.probes.items())
        for name, spec in probe_items:
            frozen = dict(spec)
            max_act = frozen.get("max_act")
            if not (isinstance(max_act, (int, float)) and float(max_act) > 0):
                frozen["max_act"] = (
                    session._sae_max_act(int(frozen["feature_id"]))
                    if has_meta_cache else None
                )
            specs[name] = frozen
        backend = getattr(session, "_sae_backend", None)
        run = SaeRun(
            self,
            InstrumentBinding(
                family=self.family,
                source=getattr(backend, "release", None),
                fingerprint=getattr(backend, "revision", None),
                specs=specs,
            ),
            active=live_active,
            bound=True,
        )
        self.current_run = run
        return run

    def close_run(self) -> None:
        """Close the current run and restore the idle passthrough run."""
        self.current_run.close()
        self.current_run = SaeRun(
            self, InstrumentBinding(family=self.family),
        )

    def _measurement_specs(self) -> "Mapping[str, Mapping[str, Any]]":
        """The spec source for measurement: a bound run's frozen binding
        (race-immune), else a per-call coherent snapshot of the live
        registry (idle passthrough) — handing out the live dict let one
        idle read tear mid-iteration under the un-locked detach, and a
        metadata backfill could rewrite a unit mid-read (round-6 P2)."""
        run = self.current_run
        if run.bound:
            return run.binding.specs
        with self.state_lock:
            return {
                name: dict(spec) for name, spec in self.probes.items()
            }

    # -------------------------------------------------------------- registry

    def attach(self, selector: str, *, as_name: str | None = None) -> str:
        """Attach a resident SAE feature as a one-channel readout probe.

        Cache invalidation (prefix cache, probe-hash cache, analytics) is
        the session's job at its ``add_probe`` boundary.
        """
        session = self._session
        raw_id = selector.split("/", 1)[1]
        validated = session.validate_sae_feature(raw_id)
        idx = int(validated["id"])
        name = as_name if as_name is not None else f"sae/{idx}"
        with session._model_exclusive(
            "add_probe called while another model operation is in flight; retry shortly"
        ):
            with self.state_lock:
                self.probes[name] = {
                    "feature_id": idx,
                    "layer": int(validated["layer"]),
                    "label": validated.get("label"),
                    "max_act": validated.get("max_act"),
                }
        return name

    def detach(self, name: str) -> None:
        with self.state_lock:
            del self.probes[name]

    def try_detach(self, name: str) -> bool:
        """Atomic membership-check + detach under the registry lock (the
        session's ``remove_probe`` dispatch — a bare check + direct
        delete is two un-serialized registry touches)."""
        with self.state_lock:
            if name not in self.probes:
                return False
            del self.probes[name]
            return True

    def specs(self) -> dict[str, dict[str, Any]]:
        """Snapshot of attached probe specifications (coherent — a
        concurrent detach or backfill cannot tear the iteration)."""
        with self.state_lock:
            return {name: dict(spec) for name, spec in self.probes.items()}

    @property
    def names(self) -> list[str]:
        with self.state_lock:
            return list(self.probes)

    def validate_gate(self, ref: GateRef) -> None:
        validate_gate_channels(ref, self._GATE_CHANNELS, family=self.family)

    # ---------------------------------------------------------------- planning

    def plan(self, prep: InstrumentPrep) -> InstrumentPlan:
        """Declare the SAE family's capture demand for one generation.

        The family reads one resident hook layer: live discovery captures
        the live config's layer; pinned probes capture the resident
        runtime layer whenever a finalize aggregate or an active gate will
        read them (the resident layer is session-side runtime state shared
        with steering atoms, consulted only when probes are attached).
        The prep carries the request; unlike the lens there is no
        registry↔source identity coupling to snapshot, so demand reads the
        live registry (a racing detach only shrinks or widens capture — the
        binding still freezes what the run measures).
        """
        if prep.family != self.family:
            raise TypeError(
                "SaeInstrument.plan takes the InstrumentPrep its own "
                f"prepare() returned, got family={prep.family!r}"
            )
        request = prep.request
        live = self.live if request.live else None
        probes = self.probes
        gate_keys = frozenset(
            key for key in request.gate_keys
            if parse_gate_ref(key).probe in probes
        )
        latest: set[int] = set()
        tail: set[int] = set()
        if live is not None:
            latest.add(int(live["layer"]))
        if probes and (request.final_aggregate or gate_keys):
            layer = self._session._sae_layer
            if layer is not None:
                latest.add(int(layer))
                if request.final_aggregate:
                    tail.add(int(layer))
        return InstrumentPlan(
            family=self.family,
            latest_layers=frozenset(latest),
            tail_layers=frozenset(tail),
            tail_depth=AGG_TAIL_DEPTH if tail else 0,
            per_step=bool(live is not None or gate_keys),
            gate_keys=gate_keys,
            final_aggregate=bool(probes and request.final_aggregate),
            batch_aggregate=bool(
                request.batch and probes and request.final_aggregate
            ),
            prep_token=prep.token,
        )

    def probe_hash(self, name: str) -> str | None:
        """Readout-channel identity digest.

        v2: the channel is normalized strength (activation / maxActApprox)
        when metadata exists, raw activation otherwise — ``max_act`` is
        part of the channel identity (it sets the unit), so drift
        detection catches a unit change.
        """
        with self.state_lock:
            spec = self.probes.get(name)
            if spec is None:
                return None
            spec = dict(spec)
        session = self._session
        info = session.sae_info or {}
        return hashlib.sha256(repr((
            "sae-readout-v2", session.model_id, info.get("fingerprint"),
            info.get("release"), spec["layer"],
            spec["feature_id"],
            session._sae_max_act(int(spec["feature_id"])),
        )).encode("utf-8")).hexdigest()

    # ---------------------------------------------------------------- scoring

    def score_probes(
        self, hidden: dict[int, torch.Tensor] | None = None,
        *, activations: torch.Tensor | None = None,
        only: "set[str] | None" = None,
        raw_by_fid: Mapping[int, float] | None = None,
    ) -> dict[str, "ProbeReading"]:
        from saklas.core.results import ProbeReading

        session = self._session
        if not self._measurement_specs():
            return {}
        _backend, layer, _width = session._require_sae()
        if activations is None:
            if hidden is None or layer not in hidden:
                return {}
            activations = session._encode_sae_hidden(hidden[layer])
        assert activations is not None
        out: dict[str, ProbeReading] = {}
        for name, _fid, _raw_value, value in self.probe_values(
            activations, only=only, raw_by_fid=raw_by_fid,
        ):
            out[name] = ProbeReading(
                fraction=0.0, nearest=[], coords=(value,), residual=0.0,
                coords_per_layer={layer: (value,)},
                depth_com=(layer / max(len(session._layers) - 1, 1),),
                depth_spread=(0.0,),
            )
        return out

    def probe_values(
        self,
        activations: torch.Tensor,
        *,
        only: "set[str] | None" = None,
        raw_by_fid: Mapping[int, float] | None = None,
    ) -> list[tuple[str, int, float, float]]:
        """Pinned SAE probe values as ``(name, fid, raw, normalized)``.

        ``raw_by_fid`` is a per-forward cache from a caller that has already
        transferred selected feature activations (currently the live top-k
        readout). Normalization reads the **measurement specs**: a bound
        run's bind-time snapshot (a mid-generation Neuronpedia backfill
        cannot change the unit under a running generation), else the live
        registry (between generations a refresh changes the unit
        immediately, as before).
        """
        session = self._session
        specs = self._measurement_specs()
        live_fallback = not self.current_run.bound
        names = [
            name for name in specs
            if only is None or name in only
        ]
        if not names:
            return []
        fids = [int(specs[name]["feature_id"]) for name in names]
        raw_values_by_fid: dict[int, float] = {
            int(fid): float(value)
            for fid, value in (raw_by_fid or {}).items()
        }
        missing_fids = [
            fid for fid in fids
            if fid not in raw_values_by_fid
        ]
        if missing_fids:
            fid_tensor = session._readout_long_tensor(
                missing_fids, activations.device,
            )
            # One host transfer for every not-already-read pinned SAE probe
            # value.  Live readout top-k rows seed ``raw_by_fid`` first, so
            # pinned cards that came from the visible top-k avoid a second
            # selected-feature gather + CPU transfer.
            raw_values = (
                activations.index_select(0, fid_tensor)
                .detach()
                .to("cpu")
                .tolist()
            )
            for fid, raw_value in zip(missing_fids, raw_values, strict=True):
                raw_values_by_fid[int(fid)] = float(raw_value)
        out: list[tuple[str, int, float, float]] = []
        for name, fid in zip(names, fids, strict=True):
            spec = specs[name]
            raw_value = float(raw_values_by_fid[fid])
            value = raw_value
            # The ONE channel is normalized strength — ``activation /
            # maxActApprox`` ∈ ~[0,1], apples-to-apples across features like
            # the lens probes' mean fitted-layer probability. Raw activation
            # only when no metadata exists (offline / not on Neuronpedia).
            # A bound run's unit was resolved into the snapshot at bind; the
            # live cache fallback applies only between generations.
            max_act = spec.get("max_act")
            if not (isinstance(max_act, (int, float)) and float(max_act) > 0):
                max_act = session._sae_max_act(fid) if live_fallback else None
            if max_act is not None:
                value = value / float(max_act)
            out.append((name, fid, raw_value, value))
        return out

    def gate_scalars(
        self, gate_keys: "set[str] | None" = None, *, step_id: int = -1,
    ) -> dict[str, float]:
        """Per-forward SAE gate scalars from the latest capture slice.

        Emits ONLY the channels an SAE probe actually measures — ``<name>``
        and ``<name>[0]`` (the normalized strength axis).  The historical
        fake constants (``:fraction`` 0.0 / ``:membership`` 1.0) are gone:
        a gate referencing a channel this family can never produce is a
        composition-preflight error (:meth:`validate_gate`), never a
        silently-constant comparison.  The encode stash is keyed by
        ``step_id`` — the display step reuses it iff it came from the same
        forward (step identity replaced the ``fresh`` handshake).
        """
        session = self._session
        specs = self._measurement_specs()
        if not specs:
            return {}
        _backend, layer, _width = session._require_sae()
        only = None
        if gate_keys:
            only = {
                key.split("[", 1)[0]
                for key in gate_keys
                if key.split("[", 1)[0] in specs
            }
            if not only:
                return {}
        latest = session._capture.latest_per_layer()
        if layer not in latest:
            return {}
        acts = session._encode_sae_hidden(latest[layer])
        scalars: dict[str, float] = {}
        values = self.probe_values(acts, only=only)
        self.step_stash = {
            "activations": acts,
            "step": step_id,
            "raw_by_fid": {
                int(fid): float(raw_value)
                for _name, fid, raw_value, _value in values
            },
        }
        for name, _fid, _raw_value, value in values:
            scalars[name] = value
            scalars[f"{name}[0]"] = value
        return scalars

    def score_aggregate(
        self,
        generated_ids: list[int],
        *,
        pooled: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, "ProbeReading"]:
        """End-of-gen aggregate pooled at the last content token."""
        session = self._session
        # Binding-authoritative guard: a probe detached mid-generation
        # stays in this generation's aggregate roster (mutations apply
        # next generation).
        if not self._measurement_specs() or not generated_ids:
            return {}
        if pooled is None:
            agg_fwd = session._aggregate_forward_index(generated_ids)
            if agg_fwd is None:
                return {}
            pooled = session._capture.tail_slice_at(agg_fwd)
            if not pooled:
                stacked = session._capture.stacked()
                pooled = {
                    layer: rows[agg_fwd]
                    for layer, rows in stacked.items()
                    if rows.shape[0] > agg_fwd
                }
        return self.current_run.observe_aggregate(pooled) if pooled else {}

    # ----------------------------------------------------------- live readout

    def enable_live(self, *, top_k: int = 8) -> dict[str, Any]:
        """Enable the one-matvec live feature readout at the resident layer."""
        session = self._session
        if not 1 <= int(top_k) <= 100:
            raise ValueError("top_k must be in [1, 100]")
        backend, layer, _width = session._require_sae()
        release = str(backend.release)
        source = (
            release
            if release.startswith(("local:", "saelens:"))
            else f"saelens:{release}"
        )
        self.live = {
            "layer": layer,
            "top_k": int(top_k),
            "source": source,
        }
        return {"layer": layer, "top_k": int(top_k)}

    def disable_live(self) -> None:
        self.live = None

    @property
    def is_live(self) -> bool:
        return self.live is not None

    def live_readout_step(
        self, *, step_id: int = -1,
    ) -> list[tuple[int, float, str | None, float | None]] | None:
        """One decode step's live feature top-k from the latest capture slice.

        Reuses the gate callback's encoded activations + raw values when the
        stash came from THIS forward (``stash["step"] == step_id`` — one
        encode shared by gates, pinned probes, and the live display on a
        step; step identity replaced the ``fresh`` consume-once flag, so
        reuse is idempotent and ``step_id < 0`` never matches).
        """
        session = self._session
        state = self.live
        if state is None or not self.active_for_generation:
            return None
        layer = int(state["layer"])
        buckets = session._capture.per_layer_buckets()
        if not buckets.get(layer):
            return None
        stash = self.step_stash
        stashed_raw_by_fid: dict[int, float] = {}
        if (
            stash is not None
            and step_id >= 0
            and stash.get("step") == step_id
        ):
            acts = stash["activations"]
            stashed_raw_by_fid = {
                int(fid): float(value)
                for fid, value in (stash.get("raw_by_fid") or {}).items()
            }
        else:
            acts = session._encode_sae_hidden(buckets[layer][-1])
        k = min(int(state["top_k"]), int(acts.numel()))
        values, indices = torch.topk(acts, k=k)
        value_list = values.detach().to("cpu").tolist()
        id_list = indices.detach().to("cpu").tolist()
        raw_by_fid = {
            fid: value for fid, value in zip(id_list, value_list, strict=True)
        }
        if stashed_raw_by_fid:
            raw_by_fid = {**stashed_raw_by_fid, **raw_by_fid}
        if self._measurement_specs():
            self.last_step_readings = self.score_probes(
                activations=acts,
                raw_by_fid=raw_by_fid,
            )
            # Full-roster readings — prime the run's observe memo for this
            # forward (never an ``only=`` subset on this path).
            self.current_run.prime_observation(
                step_id, self.last_step_readings,
            )
        else:
            self.last_step_readings = None
        # Rows carry ``max_act`` (cached-only — the decode loop never fetches)
        # so clients can render the normalized 0..1 strength beside the raw
        # activation; ``None`` until the metadata backfill lands.
        return [
            (
                int(idx),
                float(value),
                session._sae_label(int(idx)),
                session._sae_max_act(int(idx)),
            )
            for value, idx in zip(value_list, id_list, strict=True)
        ]

    # ------------------------------------------------------- authored prefill

    def authored_capture(
        self,
        hidden: dict[int, torch.Tensor],
    ) -> tuple[
        list[tuple[int, float, str | None, float | None]],
        dict[str, "ProbeReading"],
    ] | None:
        """Live SAE payload for one retained authored producer row.

        Computation only — the token-matching orchestration stays in the
        session's authored-prefill path.
        """
        session = self._session
        state = self.live
        if state is None or not self.active_for_generation:
            return None
        layer = int(state["layer"])
        if layer not in hidden:
            return None
        activations = session._encode_sae_hidden(hidden[layer])
        k = min(int(state["top_k"]), int(activations.numel()))
        values, indices = torch.topk(activations, k=k)
        value_list = values.detach().to("cpu").tolist()
        id_list = indices.detach().to("cpu").tolist()
        raw_by_fid = {
            int(fid): float(value)
            for fid, value in zip(id_list, value_list, strict=True)
        }
        readings = self.score_probes(
            activations=activations, raw_by_fid=raw_by_fid,
        ) if self._measurement_specs() else {}
        rows = [
            (
                int(fid),
                float(value),
                session._sae_label(int(fid)),
                session._sae_max_act(int(fid)),
            )
            for value, fid in zip(value_list, id_list, strict=True)
        ]
        return rows, readings


__all__ = ["SaeInstrument"]
