"""The instrument protocol — one contract over the three read families.

An **Instrument** is the persistent, session-lifetime object for one read
family: it owns the attached-probe registry, validates gate references
against the channels the family can actually produce, prepares its source
at the generation boundary (``prepare`` — the disk refresh + pin
decision), declares capture demand (``plan``), and binds an immutable
per-generation **InstrumentRun**.  A capture transaction is the uniform
sequence ``close_run → prepare → plan → bind`` — source refresh strictly
precedes planning because adoption may rewrite live probe specs.

An **InstrumentRun** owns the instrument-side generation-scoped state:
the immutable source/spec binding, step stashes, and per-generation
active flags (geometry's curved warm feet stay in the Monitor engine).
Repeated ``observe`` calls for the same ``step_id`` are memoized while
the run is bound — an idle run never memoizes, since it persists
indefinitely and a repeated step id with different hidden states must
not read stale.  The hot paths are wired through step identity: the
decode loop owns one ``step_id`` per forward (``len(generated_ids)``
pre-forward) and hands the SAME value to the capture sink
(``step_callback``), the gate callback (``score_callback``), and the
token tap (the internal ``StepTokenCallback``).  Full-roster reads —
the FULL-incremental sink, the lens/SAE display readings — call
``prime_observation`` so one forward's gate and payload reads share a
single scoring pass; partial reads (gating scalar subsets, lean
``coords_only`` rows, ``only=`` restrictions) NEVER prime.  The
matrix-granular gate→display reuse (band logits, encoded activations)
stays the workers' stash mechanism, now step-keyed
(``stash["step"] == step_id`` — structural staleness, idempotent reuse
— replaced the ``fresh`` consume-once flags): a reading-level memo
cannot carry logit-level partial-row reuse without a second host sync.

Division of labor the protocol deliberately does NOT own:

* **Capture planning** — instruments declare demand
  (:class:`~saklas.core.instruments.types.InstrumentPlan`); the session
  planner unions demands and picks physical retention.  The
  ``INCREMENTAL -> set_tail_with_sink`` upgrade is cross-instrument
  resource sharing and stays session-side.
* **Authored-prefill orchestration** — token matching, the ``j-1``
  producer-position semantics, ordering, and loom persistence stay in the
  session; instruments only ``observe`` the hidden rows handed to them.
* **The wire** — runs return readings; serialization into the versioned
  measurement envelope and its compatibility channels is the payload layer's
  job.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, Union, runtime_checkable

import torch

from saklas.core.instruments.types import (
    GateRef,
    InstrumentBinding,
    InstrumentPlan,
    InstrumentPrep,
    ReadRequest,
    ScalarReading,
)

# The discriminated reading union: geometry emits the full ProbeReading,
# lens/SAE emit the honest single-channel ScalarReading.  (Annotated as
# Any-compatible mapping values in the protocol methods so the geometry
# adapter can keep returning ProbeReading without an import cycle.)
Reading = Union["ScalarReading", Any]


@runtime_checkable
class Instrument(Protocol):
    """Session-lifetime handle for one read family."""

    family: str
    #: The per-generation run currently in effect.  An **idle** run
    #: (``bound=False``, live-registry passthrough) backs out-of-generation
    #: reads; ``bind`` replaces it with a generation-bound run and
    #: ``close_run``-style teardown restores an idle one.
    current_run: "InstrumentRun"

    def specs(self) -> Mapping[str, Mapping[str, Any]]:
        """Attached probe specs keyed by registered name."""
        ...

    def attach(self, name: str, spec: Mapping[str, Any]) -> None: ...

    def detach(self, name: str) -> None: ...

    def validate_gate(self, ref: GateRef) -> None:
        """Raise ``UnsupportedProbeChannelError`` when the referenced
        channel is one this family can never produce."""
        ...

    def probe_hash(self, name: str) -> str | None:
        """Deterministic identity digest for the named probe (loom
        recipe stamping), or None when not attached."""
        ...

    def prepare(self, request: ReadRequest) -> InstrumentPrep:
        """Generation-boundary source preparation — the first protocol
        step of a capture transaction, run after the prior run is
        closed.

        The one step allowed to touch disk-backed source identity: the
        lens family refreshes/adopts the on-disk artifact here, decides
        source pinning, and snapshots its probe specs, live config, and
        sidecar fingerprint against the prepared identity — the whole
        snapshot one atomic transaction under the family's lens-state
        lock, the same boundary the getter's refresh/adopt/evict and the
        registry/live mutations hold, so it cannot tear mid-``prepare``.
        ``plan`` and ``bind`` consume
        the prep — never the live registry — so a registry mutation
        landing inside the prepare→bind window (adoption rewrites live
        probe layer lists; the un-locked ``has_compatible_jlens`` can
        trigger it from another thread) cannot desynchronize what the
        run measures from the source it pinned.  Families without a
        source lifecycle return the bare prep so the session boundary
        stays uniform.  Every prep carries a per-preparation ``token``
        that the plan it derives echoes; ``bind`` compares them, so a
        plan cannot be bound with a prep from a different prepare()
        call.  Raises ``RuntimeError`` on a still-bound run — a
        stale pin would short-circuit the very refresh this step exists
        for, so callers must ``close_run`` first."""
        ...

    def plan(self, prep: InstrumentPrep) -> InstrumentPlan:
        """Declared capture demand, derived solely from the prep (its
        ``request`` + frozen snapshot) — never from the live registry."""
        ...

    def bind(
        self, plan: InstrumentPlan, prep: InstrumentPrep,
    ) -> "InstrumentRun": ...

    def close_run(self) -> None:
        """Close the current run and restore an idle passthrough run.

        Part of the protocol because every ``bind`` needs a matching
        teardown the session can invoke uniformly — the bind/close
        asymmetry is what let a standalone capture path leak a bound
        run past its generation."""
        ...


@runtime_checkable
class InstrumentRun(Protocol):
    """Immutable-per-generation measurement executor for one family."""

    binding: InstrumentBinding
    #: True for a generation-bound run (its binding froze the specs);
    #: False for the idle passthrough run that backs out-of-generation
    #: reads against the live registry.
    bound: bool

    def observe(
        self,
        step_id: int,
        hidden: Mapping[int, torch.Tensor],
    ) -> dict[str, Reading]:
        """Readings for every attached probe at this step.  Repeated
        calls with the same ``step_id`` are memoized while the run is
        bound (never while idle).  ``step_id`` is the decode loop's
        forward index — the hot paths are wired through it: the
        FULL-incremental sink and the workers' full-roster reads
        ``prime_observation`` the memo, so the gate callback's
        full-roster read and the token tap's payload read of one forward
        share a single scoring pass.  The matrix-granular gate→display
        reuse (band logits / encoded activations) stays the workers'
        step-keyed stash — a reading-level memo cannot carry it."""
        ...

    def prime_observation(
        self,
        step_id: int,
        readings: dict[str, Reading],
    ) -> None:
        """Prime the ``observe`` memo with FULL-roster readings a hot-path
        worker already computed for this forward.  Bound runs only (idle
        runs never memoize); callers MUST never prime a gating scalar
        subset or a ``coords_only`` lean row — a partial reading served as
        the full ``observe`` result is the completeness trap."""
        ...

    def gate_scalars(
        self,
        step_id: int,
        hidden: Mapping[int, torch.Tensor] | None,
        gate_keys: frozenset[str] | set[str] | None,
    ) -> dict[str, float]:
        """The gate channels' scalar values for this step, keyed by
        verbatim scalar key.  ``step_id`` keys the worker's per-forward
        stash so the display step reuses this forward's rows (step
        identity replaced the ``fresh`` handshake).  ``hidden=None`` means
        "read the capture's latest slices yourself" (the lens/SAE workers
        always do; geometry fetches them); ``gate_keys=None`` scores the
        full attached roster — the session gate forwarders' bare-call
        shape."""
        ...

    def observe_aggregate(
        self,
        pooled: Mapping[int, torch.Tensor],
    ) -> dict[str, Reading]:
        """The end-of-generation aggregate reading at the pooled
        last-content slice (bit-identical to a live read at that token)."""
        ...

    def observe_many(
        self,
        pooled_rows: Sequence[Mapping[int, torch.Tensor]],
    ) -> list[dict[str, Reading]]:
        """Batch-generation aggregates: one reading set per row."""
        ...

    def close(self) -> None:
        """Release generation-scoped state (stashes, warm feet, flags)."""
        ...


__all__ = ["Instrument", "InstrumentRun", "Reading"]
