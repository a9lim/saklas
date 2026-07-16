"""The J-lens instrument: readout-channel probes, gates, live workspace.

Owns everything lens-probe-shaped that used to live inline in
``SaklasSession`` (the ~675-line lens region): the probe registry, the
live-readout runtime state, the per-forward stash, the per-generation
disk-identity pin, and the six read surfaces (attach / per-step scoring /
gate scalars / finalize aggregate / live display step / authored-prefill
computation).  The session re-exposes the state fields under their
historical private names via delegating properties, so `_begin_capture`,
the steering composer, the token tap, and the wire layers are unchanged
until the plan/run split migrates them.

Division of labor (see ``protocol.py``): shared J-lens *primitives* —
``_jlens_logits_rows`` / depth caches / decode memo / transport stack /
the ``jlens`` disk-identity property — stay on the session (they also
serve steering registration and the offline readouts); this instrument
reaches them through the session back-ref at per-step frequency, exactly
the pattern ``SteeringComposer`` established.  Capture planning and
authored-prefill *orchestration* (token matching, ``j-1`` producer
semantics, loom persistence) stay session-side; ``authored_capture`` here
is only the per-row computation.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence, TYPE_CHECKING, cast

import torch

from saklas.core.instruments.types import (
    AGG_TAIL_DEPTH,
    Axis,
    GateRef,
    InstrumentBinding,
    InstrumentPlan,
    ReadRequest,
    parse_gate_ref,
    validate_gate_channels,
)

if TYPE_CHECKING:
    from saklas.core.results import ProbeReading
    from saklas.core.session import SaklasSession


class LensRun:
    """Per-generation measurement executor for the lens family.

    Owns everything generation-scoped (``protocol.py``): the immutable
    :class:`InstrumentBinding` (probe specs frozen at bind), the resident-
    lens disk-identity pin, the live-active flag, the per-forward stash the
    gate callback and the display step share, and the ``observe`` memo.
    An **idle** run (``bound=False``) backs out-of-generation reads —
    defensive captures, offline readouts — with live-registry passthrough
    semantics, so every read path has a run to consult.
    """

    def __init__(
        self,
        instrument: "LensInstrument",
        binding: InstrumentBinding,
        *,
        lens: Any = None,
        pinned: bool = False,
        active: bool = True,
        bound: bool = False,
    ) -> None:
        self._instrument = instrument
        self.binding = binding
        #: The pinned resident lens (or None — unbound / validated-missing).
        self.lens = lens
        #: True when this generation snapshot-pinned the lens identity.
        self.pinned = pinned
        #: Live-readout activity for this generation.
        self.active = active
        #: True for a generation-bound run (idle runs pass through).
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
        ``step_id`` while bound (the production gate→display reuse is the
        worker stash mechanism, not this method).  An idle run never
        memoizes — it persists indefinitely, so a repeated ``step_id``
        with different hidden states would return stale readings."""
        if not self.bound:
            return self._instrument.score_probes(hidden)
        if self._memo_step == step_id and self._memo_readings is not None:
            return self._memo_readings
        readings = self._instrument.score_probes(hidden)
        self._memo_step = step_id
        self._memo_readings = readings
        return readings

    def gate_scalars(
        self,
        step_id: int,
        hidden: dict[int, Any] | None,
        gate_keys: frozenset[str] | set[str],
    ) -> dict[str, float]:
        """The gate channels' scalars for this step.  The per-forward
        matrix-level reuse (band logits stashed for the display step)
        stays inside the instrument's worker — the stash lives on this
        run, so the reuse is generation-scoped by construction."""
        del step_id, hidden  # worker reads the capture's latest slices
        return self._instrument.gate_scalars(set(gate_keys))

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
        """Release generation-scoped state (stash, memo, pin)."""
        self.step_stash = None
        self.last_step_readings = None
        self._memo_step = None
        self._memo_readings = None
        self.lens = None
        self.pinned = False
        self.active = True


class LensInstrument:
    """Session-lifetime handle for the J-lens read family."""

    family = "lens"

    #: Gate channels a lens probe can produce: the one strength axis.
    _GATE_CHANNELS: tuple[type, ...] = (Axis,)

    def __init__(self, session: "SaklasSession") -> None:
        self._session = session
        # Pinned J-lens token probes: name -> {word, token_id, layers}.  NOT
        # monitor probes — they read the lens readout channel (per-layer
        # softmax probability), not a whitened subspace coordinate.
        self.probes: dict[str, dict[str, Any]] = {}
        # Live workspace readout (enable_live): device-resident J_l subset +
        # settings, or None when off.  Runtime residency, not user intent —
        # disabling drops this dict but transported stacks stay in the
        # session's device cache.
        self.live: dict[str, Any] | None = None
        # The current per-generation run (idle passthrough until bind()).
        # All generation-scoped state — stash, display readings, active
        # flag, disk-identity pin — lives on it.
        self.current_run = LensRun(
            self, InstrumentBinding(family=self.family),
        )

    # ------------------------------------------------------------- run state
    # Historical state names, delegating to the current run so the session's
    # own delegating properties (and every internal read path) are unchanged.

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

    @property
    def generation_lens(self) -> Any:
        return self.current_run.lens

    @generation_lens.setter
    def generation_lens(self, value: Any) -> None:
        self.current_run.lens = value

    @property
    def generation_lens_active(self) -> bool:
        return self.current_run.pinned

    @generation_lens_active.setter
    def generation_lens_active(self, value: bool) -> None:
        self.current_run.pinned = bool(value)

    # ------------------------------------------------------------ run lifecycle

    def bind(
        self,
        plan: InstrumentPlan,
        *,
        live_active: bool = True,
        lens: Any = None,
        pinned: bool = False,
    ) -> LensRun:
        """Bind an immutable per-generation run from a declared plan.

        The **session planner supplies the pin**: the disk-refreshing
        ``session.jlens`` read must happen BEFORE ``plan()`` — its adoption
        path rewrites the live probe layer lists when an external
        replacement lens is adopted, so a plan (and a spec freeze) taken
        earlier would pair the new lens with stale layers (sol's slice-B
        review, finding 2).  ``bind`` therefore only freezes the (already
        refreshed) registry and installs the pin it is handed.  The frozen
        spec copies own their ``layers`` list — the live registry's lists
        are replaced in place by the adoption/eviction paths.
        """
        del plan  # demand consumed by the planner; the pin decision is its
        session = self._session
        run = LensRun(
            self,
            InstrumentBinding(
                family=self.family,
                # ``source`` stays None: the public source label lives in
                # ``active.json`` and reading it here would put a disk hit
                # on every generation; the in-memory sidecar identity below
                # is the binding's cheap fingerprint.
                fingerprint=(
                    str(identity)
                    if (identity := getattr(session, "_jlens_identity", None))
                    is not None else None
                ),
                specs={
                    name: {**spec, "layers": list(spec["layers"])}
                    for name, spec in self.probes.items()
                },
            ),
            lens=lens,
            pinned=pinned,
            active=live_active,
            bound=True,
        )
        self.current_run = run
        return run

    def close_run(self) -> None:
        """Close the current run and restore the idle passthrough run."""
        self.current_run.close()
        self.current_run = LensRun(
            self, InstrumentBinding(family=self.family),
        )

    def _measurement_specs(self) -> "dict[str, Any] | Any":
        """The spec source for measurement: a bound run's frozen binding
        (immune to concurrent spec mutation — e.g. the ``jlens`` getter's
        eviction path zeroing probe layers from another thread), else the
        live registry."""
        run = self.current_run
        return run.binding.specs if run.bound else self.probes

    # -------------------------------------------------------------- registry

    def attach(self, selector: str, *, as_name: str | None = None) -> str:
        """Attach a ``jlens/<word>`` token probe.

        Validates the lens artifact + single-token word (the same
        ``resolve_word_token`` contract steering atoms use), records the
        full fitted layer set, and pre-warms the device transport stack so
        the first decode step doesn't hitch on the J_l transfer.  The probe
        reads ONE channel — ``coords = (strength,)``, the mean layer
        probability ``mean_l p_l(v)``.  Cache invalidation (prefix cache,
        probe-hash cache, analytics) is the session's job at its
        ``add_probe`` boundary.
        """
        from saklas.core.jlens import resolve_word_token

        session = self._session
        word = selector.split("/", 1)[1]
        if not word:
            raise ValueError("empty jlens probe word")
        name = as_name if as_name is not None else selector
        with session._model_exclusive(
            "add_probe called while another model operation is in "
            "flight; retry shortly"
        ):
            lens = session._require_jlens()
            token_id = resolve_word_token(session._tokenizer, word)
            readout_layers = [int(l) for l in lens.source_layers]
            session._jlens_transport_stack(
                lens, sorted(readout_layers), session._device,
            )
            self.probes[name] = {
                "word": word,
                "token_id": int(token_id),
                "layers": readout_layers,
            }
        return name

    def detach(self, name: str) -> None:
        del self.probes[name]

    def specs(self) -> dict[str, dict[str, Any]]:
        """Snapshot of attached probe specifications."""
        return {name: dict(spec) for name, spec in self.probes.items()}

    @property
    def names(self) -> list[str]:
        return list(self.probes)

    def probe_layers(self, names: set[str] | None = None) -> set[int]:
        """Union of the attached probes' fitted layers."""
        out: set[int] = set()
        probes = self.probes
        if names is not None:
            probes = {name: probes[name] for name in names if name in probes}
        for spec in probes.values():
            out.update(spec["layers"])
        return out

    def validate_gate(self, ref: GateRef) -> None:
        validate_gate_channels(ref, self._GATE_CHANNELS, family=self.family)

    # ---------------------------------------------------------------- planning

    def plan(self, request: ReadRequest) -> InstrumentPlan:
        """Declare the lens family's capture demand for one generation.

        ``latest_layers`` — the live workspace readout's layer set plus the
        pinned probes' fitted band (full band when a finalize aggregate
        will pool it; the gated probes' band when gates are the only
        per-step consumer with final readings disabled).  ``tail_layers``
        is the finalize-pooling demand: pinned probes' aggregates pool the
        last content token from the capture tail ring, which must span the
        probe band at ring depth ``AGG_TAIL_DEPTH``.
        """
        live = self.live if request.live else None
        probes = self.probes
        gate_keys = frozenset(
            key for key in request.gate_keys
            if parse_gate_ref(key).probe in probes
        )
        latest: set[int] = set()
        tail: set[int] = set()
        if live is not None:
            latest.update(int(layer) for layer in live["layers"])
        if probes and request.final_aggregate:
            band = {int(layer) for layer in self.probe_layers()}
            latest.update(band)
            tail.update(band)
        elif gate_keys:
            # Gate-only pinned probes need per-step latest slices, but
            # dormant probes must not keep capture alive when the caller
            # disabled final probe readings.
            gated_names = {parse_gate_ref(key).probe for key in gate_keys}
            latest.update(
                int(layer) for layer in self.probe_layers(gated_names)
            )
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
        )

    def probe_hash(self, name: str) -> str | None:
        """Readout-channel identity digest (no baked tensor exists).

        v2: single strength axis; v1 carried a salience axis; the depth-CoM
        mass moved salience→probability within v2 — display-only, the
        coords channel is bit-identical, so no bump.
        """
        spec = self.probes.get(name)
        if spec is None:
            return None
        return hashlib.sha256(
            repr(
                (
                    "jlens-readout-v2", self._session.model_id, spec["word"],
                    spec["token_id"], tuple(spec["layers"]),
                )
            ).encode("utf-8")
        ).hexdigest()

    # ------------------------------------------------------------- resolution

    def _resident_lens(self) -> Any:
        """The generation-pinned lens when a generation is active, else the
        session's disk-validated resident lens."""
        session = self._session
        return (
            self.generation_lens
            if self.generation_lens_active
            else session.jlens
        )

    # ---------------------------------------------------------------- scoring

    def score_probes(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        logits: torch.Tensor | None = None,
        probabilities: torch.Tensor | None = None,
        layers: "Sequence[int] | None" = None,
        only: "set[str] | None" = None,
    ) -> dict[str, "ProbeReading"]:
        """Score every attached probe from hidden slices (or reuse
        precomputed lens ``logits``/``probabilities`` rows aligned with
        ``layers``).

        Returns ``{name: ProbeReading}`` with ``coords = (strength,)`` —
        the ONE readout channel, mean layer probability —
        ``coords_per_layer[l] = (p_l,)``, and the depth CoM — the
        readout-channel synthesis of the unified reading shape (geometry
        fields defaulted).  Empty when no probe layer is available.
        """
        from saklas.core.jlens import (
            token_readout_stats,
            token_readout_stats_from_probabilities,
        )
        from saklas.core.results import ProbeReading

        session = self._session
        probes = self._measurement_specs()
        if not probes:
            return {}
        lens = self._resident_lens()
        if lens is None:
            return {}
        readout_layers: set[int] = set()
        for spec in probes.values():
            readout_layers.update(spec["layers"])
        if logits is not None and probabilities is not None:
            raise ValueError("pass lens logits or probabilities, not both")
        if logits is None and probabilities is None:
            layers = sorted(l for l in readout_layers if l in hidden)
            if not layers:
                return {}
            logits = session._jlens_logits_rows(
                lens, [(l, hidden[l]) for l in layers],
            )
        else:
            assert layers is not None
            # Restrict precomputed rows (e.g. a custom live-lens layer set)
            # to the probes' fitted layer set.
            keep = [i for i, l in enumerate(layers) if l in readout_layers]
            if not keep:
                return {}
            if len(keep) != len(layers):
                if logits is not None:
                    logits = logits[keep]
                if probabilities is not None:
                    probabilities = probabilities[keep]
            layers = [layers[i] for i in keep]
        names = [
            name for name in probes
            if only is None or name in only
        ]
        if not names:
            return {}
        token_ids = [probes[n]["token_id"] for n in names]
        prob_device = (
            probabilities.device
            if probabilities is not None
            else cast(torch.Tensor, logits).device
        )
        token_ids_tensor = session._readout_long_tensor(token_ids, prob_device)
        depth_tensor = session._jlens_depth_tensor(layers, prob_device)
        if probabilities is None:
            assert logits is not None
            stats = token_readout_stats(
                logits,
                session._jlens_depths(layers),
                token_ids,
                token_ids_tensor=token_ids_tensor,
                depth_tensor=depth_tensor,
            )
        else:
            stats = token_readout_stats_from_probabilities(
                probabilities,
                session._jlens_depths(layers),
                token_ids,
                token_ids_tensor=token_ids_tensor,
                depth_tensor=depth_tensor,
            )
        out: dict[str, ProbeReading] = {}
        for name, (strength, com, spread, per_layer) in zip(names, stats):
            out[name] = ProbeReading(
                fraction=0.0,
                nearest=[],
                coords=(strength,),
                residual=0.0,
                coords_per_layer={
                    l: (p_l,) for l, p_l in zip(layers, per_layer)
                },
                depth_com=(com,),
                depth_spread=(spread,),
            )
        return out

    def gate_scalars(
        self, gate_keys: "set[str] | None" = None,
    ) -> dict[str, float]:
        """Per-forward gate scalars from the latest capture slices.

        Called from the gating score callback (once per decode forward,
        before the token tap). Computes the referenced lens logits, stashes
        them for the display step to reuse (``step_stash``), and flattens
        the synthesized readings through :meth:`Monitor.flat_scalars` so the
        gate key space is uniform. Gate-only calls score exact
        selected-token softmax columns; live display calls still calibrate
        the full matrix once for downstream card/aggregate reuse.  Empty
        when nothing is capturable yet.
        """
        from saklas.core.monitor import Monitor

        session = self._session
        probes = self._measurement_specs()
        if not probes:
            return {}
        lens = self._resident_lens()
        if lens is None:
            return {}
        latest = session._capture.latest_per_layer()
        if not latest:
            return {}
        only = None
        if gate_keys:
            only = {
                key.split("[", 1)[0]
                for key in gate_keys
                if key.split("[", 1)[0] in probes
            }
            if not only:
                return {}
        live_display_needs_full_probs = bool(
            self.live is not None and self.active_for_generation
        )
        # When the token tap will immediately need every pinned probe reading
        # for the live payload, compute that superset once in the gate
        # callback and let the display reuse it.  Gate-only calls stay on the
        # narrower requested subset.
        probe_read_only = None if live_display_needs_full_probs else only
        readout_layers: set[int] = set()
        for name, spec in probes.items():
            if probe_read_only is None or name in probe_read_only:
                readout_layers.update(spec["layers"])
        layers = sorted(l for l in readout_layers if l in latest)
        if not layers:
            return {}

        logits = session._jlens_logits_rows(
            lens, [(l, latest[l]) for l in layers],
        )
        probabilities = None
        if live_display_needs_full_probs:
            from saklas.core.jlens import readout_probabilities

            probabilities = readout_probabilities(logits)
            live_stash: dict[str, Any] = {
                "layers": tuple(layers),
                "logits": logits,
                "probabilities": probabilities,
                "fresh": True,
            }
            self.step_stash = live_stash
        else:
            self.step_stash = {
                "layers": tuple(layers),
                "logits": logits,
                "fresh": True,
            }
        if probabilities is not None:
            readings = self.score_probes(
                {},
                probabilities=probabilities,
                layers=layers,
                only=probe_read_only,
            )
            live_stash = cast("dict[str, Any]", self.step_stash)
            live_stash["readings"] = readings
            live_stash["readings_layers"] = tuple(layers)
            live_stash["readings_fresh"] = True
            self.last_step_readings = readings
        else:
            readings = self.score_probes(
                {}, logits=logits, layers=layers, only=probe_read_only,
            )
        if only is None:
            return Monitor.flat_scalars(readings)
        return Monitor.flat_scalars({
            name: reading for name, reading in readings.items() if name in only
        })

    def score_aggregate(
        self,
        generated_ids: list[int],
        *,
        pooled: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, "ProbeReading"]:
        """End-of-gen aggregate pooled at the last content token.

        Mirrors the monitor's ``_score_aggregate_only``: one readout at the
        pooled slice from the capture tail ring (or the FULL-mode stack), so
        the aggregate semantics match the monitor probes' exactly.
        """
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
                    l: t[agg_fwd]
                    for l, t in stacked.items()
                    if t.shape[0] > agg_fwd
                }
        if not pooled:
            return {}
        return self.current_run.observe_aggregate(pooled)

    # ----------------------------------------------------------- live readout

    def enable_live(
        self,
        *,
        layers: "Sequence[int] | None" = None,
    ) -> list[int]:
        """Stream the J-lens readout live during generation.

        The selected layers' ``J_l`` move device-resident here, once;
        ``layers`` defaults to every fitted lens layer.  Attaches no
        forward hooks (the reader consumes the capture's existing
        latest-slice buffers post-forward), so steering fast-path /
        compile eligibility is untouched.  Returns the resolved layer list.
        """
        from saklas.core.model import get_final_norm, get_unembedding

        session = self._session
        if session._device.type == "cuda":
            torch.set_float32_matmul_precision("high")

        lens = session._require_jlens()
        uses_all_layers = layers is None
        if layers is None:
            layers = sorted(int(layer) for layer in lens.source_layers)
        else:
            layers = sorted(set(int(l) for l in layers))
            missing = [l for l in layers if l not in lens.jacobians]
            if missing:
                raise ValueError(
                    f"layers {missing} not in the fitted lens "
                    f"(fitted: {lens.source_layers[0]}..{lens.source_layers[-1]})"
                )
        device = session._device
        layer_list = list(layers)
        if layer_list:
            j_stack = session._jlens_transport_stack(lens, layer_list, device)
        else:
            sample = next(iter(lens.jacobians.values()))
            j_stack = torch.empty(
                (0, *sample.shape), device=device, dtype=torch.float32,
            )
        self.live = {
            "layers": layer_list,
            "uses_all_layers": uses_all_layers,
            "J_stack": j_stack,
            "layer_rows": {l: i for i, l in enumerate(layer_list)},
            "unembed": get_unembedding(session._model),
            "norm": get_final_norm(session._model),
            "source": session._active_jlens_source_label(),
        }
        return list(layers)

    def disable_live(self) -> None:
        """Stop streaming the live readout and free the device J_l copies."""
        self.live = None

    @property
    def live_layers(self) -> list[int] | None:
        """The live readout's layer list, or ``None`` when it's off."""
        if self.live is None:
            return None
        return list(self.live["layers"])

    def live_readout_step(
        self, *, top_k: int = 8,
    ) -> (
        tuple[
            dict[int, list[tuple[str, float]]],
            list[tuple[str, float, float, float]],
            dict[int, list[int]],
        ]
        | None
    ):
        """One decode step's lens readout from the capture's latest slices.

        Runs post-forward at the token tap (never inside a hook).  Returns
        ``(per_layer, aggregate, token_ids)`` — top-k tokens per selected
        layer scored by per-layer softmax probability (the one strength
        unit every lens surface reports), the layer-aggregated chip list,
        and the vocabulary ids already selected by ``topk``.  Reuses the
        gate callback's stash rows when the layer sets overlap.
        """
        session = self._session
        state = self.live
        if state is None or not self.active_for_generation:
            return None
        buckets = session._capture.per_layer_buckets()
        unembed = state["unembed"]
        layers_present: list[int] = []
        hidden_rows: list[torch.Tensor] = []
        transport_rows: list[int] = []
        layer_rows: dict[int, int] = state["layer_rows"]
        for layer in state["layers"]:
            bucket = buckets.get(layer)
            if not bucket:
                continue
            layers_present.append(layer)
            # Keep the raw bucket reference until after stash reuse is
            # resolved: an exact gate+live cache hit never needs these hidden
            # rows, so it should not pay a dtype/device conversion just to
            # discard them.
            hidden_rows.append(bucket[-1])
            transport_rows.append(layer_rows[layer])
        if not layers_present:
            return None
        stash = self.step_stash
        logits: torch.Tensor | None = None
        probabilities: torch.Tensor | None = None
        cached_logits: dict[int, torch.Tensor] = {}
        cached_probs: dict[int, torch.Tensor] = {}
        if stash is not None and stash.get("fresh"):
            stash_layers = tuple(int(layer) for layer in (stash.get("layers") or ()))
            if stash_layers == tuple(layers_present):
                # The common gate+live path: exact row-set match.  Keep the
                # existing zero-copy reuse of the full matrix rather than
                # restacking full-vocab rows.
                stash["fresh"] = False
                logits = stash["logits"]
                probabilities = stash.get("probabilities")
            else:
                # The gate callback may already have computed this forward's
                # band logits before the token tap.  Reuse any overlapping
                # rows rather than requiring the live display layer set to
                # match exactly.  Softmax calibration is per-layer, so cached
                # probability rows compose exactly with newly computed rows.
                for row, layer in enumerate(stash_layers):
                    if layer in layers_present:
                        cached_logits[int(layer)] = stash["logits"][row]
                        probs = stash.get("probabilities")
                        if probs is not None:
                            cached_probs[int(layer)] = probs[row]
                if cached_logits:
                    stash["fresh"] = False
        computed_logits: dict[int, torch.Tensor] = {}
        if logits is None:
            missing = [
                (layer, hidden, transport_row)
                for layer, hidden, transport_row in zip(
                    layers_present, hidden_rows, transport_rows, strict=True,
                )
                if layer not in cached_logits
            ]
            if missing:
                J_stack: torch.Tensor = state["J_stack"]
                # Instance-attribute lookup (not a class-qualified call) so a
                # duck-typed test stub that borrows the session helper works.
                J = session._select_tensor_rows(
                    J_stack,
                    [row for _layer, _hidden, row in missing],
                )
                H = torch.stack(
                    [
                        hidden.to(torch.float32)
                        for _layer, hidden, _row in missing
                    ],
                    dim=0,
                ).to(J.device)
                transported = torch.bmm(J, H.unsqueeze(-1)).squeeze(-1)
                normed = state["norm"](transported)
                computed = normed.to(unembed.dtype) @ unembed.T
                if not cached_logits and len(missing) == len(layers_present):
                    logits = computed
                else:
                    computed_logits = {
                        layer: computed[row]
                        for row, (layer, _hidden, _transport) in enumerate(missing)
                    }
            if logits is None:
                logits = torch.stack(
                    [
                        cached_logits[layer]
                        if layer in cached_logits else computed_logits[layer]
                        for layer in layers_present
                    ],
                    dim=0,
                )
        from saklas.core.jlens import readout_probabilities

        if probabilities is None:
            if not cached_probs and not computed_logits:
                probabilities = readout_probabilities(logits)
            else:
                probability_rows: dict[int, torch.Tensor] = dict(cached_probs)
                uncached_prob_layers = [
                    layer for layer in layers_present
                    if layer not in probability_rows
                ]
                if uncached_prob_layers:
                    uncached_logits = torch.stack(
                        [
                            computed_logits[layer]
                            if layer in computed_logits else cached_logits[layer]
                            for layer in uncached_prob_layers
                        ],
                        dim=0,
                    )
                    uncached_probs = readout_probabilities(uncached_logits)
                    probability_rows.update({
                        layer: uncached_probs[row]
                        for row, layer in enumerate(uncached_prob_layers)
                    })
                probabilities = torch.stack(
                    [probability_rows[layer] for layer in layers_present], dim=0,
                )
        # Pinned lens probes ride the same calibrated matrix — per-step
        # readout-channel readings for the payload merge.  A gate callback
        # may already have computed the same readings from these exact rows;
        # reuse them to avoid a second selected-token host sync on the
        # pinned+gated+live path.
        if self._measurement_specs():
            readings_reused = False
            if stash is not None and stash.get("readings_fresh"):
                reading_layers = tuple(
                    int(layer) for layer in (stash.get("readings_layers") or ())
                )
                if reading_layers == tuple(layers_present):
                    self.last_step_readings = cast(
                        "dict[str, ProbeReading]",
                        stash.get("readings") or {},
                    )
                    stash["readings_fresh"] = False
                    readings_reused = True
                elif reading_layers:
                    stash["readings_fresh"] = False
            if not readings_reused:
                self.last_step_readings = self.score_probes(
                    {}, probabilities=probabilities, layers=list(layers_present),
                )
        else:
            self.last_step_readings = None
        # Display scores are per-layer softmax probabilities — the one
        # strength unit every lens surface reports (softmax is monotone, so
        # the top-k selection is unchanged from the raw-logit ranking).
        from saklas.core.jlens import (
            aggregate_readout_tensors_from_probabilities,
            pack_readout_rows_to_host,
        )

        k = min(max(int(top_k), 0), int(probabilities.shape[-1]))
        vals, idxs = probabilities.topk(k, dim=-1)
        depth_tensor = session._jlens_depth_tensor(
            layers_present, probabilities.device,
        )
        agg_idxs, agg_stats = aggregate_readout_tensors_from_probabilities(
            probabilities,
            session._jlens_depths(layers_present),
            top_k=k,
            depth_tensor=depth_tensor,
        )
        # Pack the tiny K-wide result through the shared MPS-safe helper and
        # synchronize once.
        n_layers = len(layers_present)
        host_rows = pack_readout_rows_to_host(
            vals,
            idxs,
            agg_stats,
            agg_idxs.reshape(1, -1),
        ).tolist()
        all_vals = host_rows[:n_layers]
        all_idxs = host_rows[n_layers:2 * n_layers]
        agg_host = host_rows[2 * n_layers:]
        out: dict[int, list[tuple[str, float]]] = {}
        token_ids: dict[int, list[int]] = {}
        for row, layer in enumerate(layers_present):
            pairs: list[tuple[str, float]] = []
            for v, i in zip(all_vals[row], all_idxs[row]):
                pairs.append((session._jlens_decode_id(int(i)), float(v)))
            out[layer] = pairs
            token_ids[layer] = [int(i) for i in all_idxs[row]]
        agg = [
            (
                session._jlens_decode_id(int(agg_host[3][j])),
                float(agg_host[0][j]),
                float(agg_host[1][j]),
                float(agg_host[2][j]),
            )
            for j in range(k)
        ]
        return out, agg, token_ids

    # ------------------------------------------------------- authored prefill

    def authored_capture(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        top_k: int,
    ) -> tuple[
        dict[int, list[tuple[str, float]]],
        list[tuple[str, float, float, float]],
        dict[int, list[int]],
        dict[str, "ProbeReading"],
    ] | None:
        """Live J-LENS payload for one retained authored producer row.

        Computation only — the token matching, ``j-1`` producer semantics,
        row ordering, and loom persistence stay in the session's
        authored-prefill orchestration.
        """
        session = self._session
        state = self.live
        if state is None or not self.active_for_generation:
            return None
        layers = [int(layer) for layer in state["layers"] if int(layer) in hidden]
        if not layers:
            return None
        lens = self._resident_lens()
        if lens is None:
            return None
        logits = session._jlens_logits_rows(
            lens, [(layer, hidden[layer]) for layer in layers],
        )
        from saklas.core.jlens import readout_probabilities

        probabilities = readout_probabilities(logits)
        readings = self.score_probes(
            {}, probabilities=probabilities, layers=layers,
        ) if self._measurement_specs() else {}
        k = min(max(int(top_k), 0), int(probabilities.shape[-1]))
        values, indices = probabilities.topk(k, dim=-1)
        value_rows = values.detach().to("cpu").tolist()
        id_rows = indices.detach().to("cpu").tolist()
        per_layer: dict[int, list[tuple[str, float]]] = {}
        token_ids: dict[int, list[int]] = {}
        for row, layer in enumerate(layers):
            per_layer[layer] = [
                (session._jlens_decode_id(int(tid)), float(value))
                for value, tid in zip(value_rows[row], id_rows[row], strict=True)
            ]
            token_ids[layer] = [int(tid) for tid in id_rows[row]]
        aggregate = session._jlens_aggregate_rows(
            None, layers, top_k=k, probabilities=probabilities,
        )
        return per_layer, aggregate, token_ids, readings


__all__ = ["LensInstrument"]
