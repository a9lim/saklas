"""The instrument contract — one shape over the three read families.

An **Instrument** is the persistent, session-lifetime object for one read
family: it owns the attached-probe registry, prepares its source at the
generation boundary (``prepare`` — the disk refresh + pin decision), declares
capture demand (``plan``), and binds an immutable per-generation
**InstrumentRun**.  A capture transaction is the uniform sequence
``close_run → prepare → plan → bind`` — source refresh strictly precedes
planning because adoption may rewrite live probe specs.  Every ``prepare``
raises on a still-bound run (a stale pin short-circuits the very refresh the
step exists for), every prep carries a per-preparation ``token`` its plan
echoes, and ``bind`` refuses a plan/prep pair whose tokens or families
disagree.

An **InstrumentRun** owns the instrument-side generation-scoped state: the
immutable source/spec binding (``binding``), step stashes, and per-generation
active flags (geometry's curved warm feet stay in the Monitor engine).
``bound`` distinguishes a generation-bound run from the idle passthrough run
that backs out-of-generation reads.  Repeated ``observe`` calls for the same
``step_id`` are memoized while the run is bound — an idle run never memoizes,
since it persists indefinitely and a repeated step id with different hidden
states must not read stale.  The hot paths are wired through step identity:
the decode loop owns one ``step_id`` per forward (``len(generated_ids)``
pre-forward) and hands the SAME value to the capture sink (``step_callback``),
the gate callback (``score_callback``), and the token tap (the internal
``StepTokenCallback``).  Full-roster reads — the FULL-incremental sink, the
lens/SAE display readings — call ``prime_observation`` so one forward's gate
and payload reads share a single scoring pass; partial reads (gating scalar
subsets, lean ``coords_only`` rows, ``only=`` restrictions) NEVER prime.  The
matrix-granular gate→display reuse (band logits, encoded activations) stays
the workers' stash mechanism, step-keyed (``stash["step"] == step_id`` —
structural staleness, idempotent reuse): a reading-level memo cannot carry
logit-level partial-row reuse without a second host sync.

**The protocols below are the contract the session registry
(``session.instruments``), the steering composer's gate preflight, and the
server's family dispatch actually exercise.**  Only uniform surface lives
here: a method one family cannot answer the same way as its siblings (the
geometry attach's ``top_n``, the lens's ``probe_layers``, the SAE's
``probe_values``) stays family-native and off the interface — an aspirational
member nobody can call uniformly is exactly the drift the deleted Protocols
carried.  ``@runtime_checkable`` makes the structural conformance testable
(``isinstance`` verifies member presence), and the concrete families
(``geometry.py`` / ``lens.py`` / ``sae.py``) remain the definition of the
behavior.

Division of labor the contract deliberately does NOT own:

* **Capture planning** — instruments declare demand
  (:class:`~saklas.core.instruments.types.InstrumentPlan`); the session
  planner unions demands and picks physical retention.  The
  ``INCREMENTAL -> set_tail_with_sink`` upgrade is cross-instrument
  resource sharing and stays session-side.
* **Authored-prefill orchestration** — token matching, the ``j-1``
  producer-position semantics, ordering, and loom persistence stay in the
  session; instruments only ``observe`` the hidden rows handed to them.
* **The wire framing** — ``token_readout`` returns the finished versioned
  measurement envelope (``core/measurements.py``), but the HTTP status
  mapping, locking, and request validation stay in the server.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    import threading

    import torch

    from saklas.core.instruments.types import (
        GateRef,
        InstrumentBinding,
        InstrumentFamily,
        InstrumentPlan,
        InstrumentPrep,
        LiveState,
        ReadRequest,
    )


@runtime_checkable
class InstrumentRun(Protocol):
    """One family's per-generation measurement executor."""

    #: Immutable snapshot of what this run measures (source + frozen specs).
    binding: "InstrumentBinding"
    #: False for the idle passthrough run that backs out-of-generation reads.
    bound: bool

    def observe(
        self, step_id: int, hidden: "dict[int, torch.Tensor]",
    ) -> "dict[str, Any]":
        """Measure one forward, memoized by ``step_id`` while bound."""
        ...

    def prime_observation(
        self, step_id: int, readings: "dict[str, Any]",
    ) -> None:
        """Seed the step memo with COMPLETE readings a hot-path scorer
        already computed this forward.  Partial reads never prime."""
        ...

    def observe_aggregate(
        self, pooled: "dict[int, torch.Tensor]",
    ) -> "dict[str, Any]":
        """One full-roster read at the pooled last-content slice."""
        ...

    def close(self) -> None:
        """Release generation-scoped state (idempotent)."""
        ...


@runtime_checkable
class Instrument(Protocol):
    """One read family's session-lifetime handle."""

    #: The family key — also this instrument's key in ``session.instruments``.
    family: "InstrumentFamily"
    #: The family's reentrant leaf state lock (roster/source coherence).
    state_lock: "threading.RLock"
    #: The current run — idle passthrough until ``bind``.
    current_run: InstrumentRun

    # ------------------------------------------------------- run lifecycle

    def prepare(self, request: "ReadRequest") -> "InstrumentPrep":
        """Generation-boundary source snapshot.  Raises on a bound run."""
        ...

    def plan(self, prep: "InstrumentPrep") -> "InstrumentPlan":
        """Declare capture demand from the prep (never the live registry)."""
        ...

    def bind(
        self, plan: "InstrumentPlan", prep: "InstrumentPrep",
    ) -> InstrumentRun:
        """Freeze an immutable per-generation run from a matching pair."""
        ...

    def close_run(self) -> None:
        """Close the bound run and restore the idle passthrough run."""
        ...

    # ------------------------------------------------------------ registry

    @property
    def names(self) -> list[str]:
        """Attached probe names."""
        ...

    def specs(self) -> dict[str, dict[str, Any]]:
        """Coherent snapshot of the attached probes' specs."""
        ...

    def try_detach(self, name: str) -> bool:
        """Detach ``name`` if this family owns it; False when it doesn't."""
        ...

    def probe_hash(self, name: str) -> str | None:
        """Deterministic identity digest for drift detection."""
        ...

    def validate_gate(self, ref: "GateRef") -> None:
        """Raise ``UnsupportedProbeChannelError`` when this family can never
        produce ``ref``'s channel (geometry produces every channel, so its
        implementation accepts unconditionally)."""
        ...

    # -------------------------------------------------------------- source

    @property
    def active_source(self) -> str | None:
        """The active source label in the public source syntax, or ``None``
        for a family with no source lifecycle."""
        ...

    # ---------------------------------------------------------------- live

    @property
    def live_state(self) -> "LiveState":
        """The family's live-readout intent."""
        ...

    def set_live(self, enabled: bool, **kwargs: Any) -> "LiveState":
        """Toggle the live readout; returns the resolved state.  Family
        extras arrive as keyword arguments (lens ``layers``); an unsupported
        keyword raises ``TypeError``."""
        ...

    # -------------------------------------------------------------- replay

    def token_readout(
        self,
        node_id: str,
        raw_index: int,
        *,
        top_k: int = 8,
        layers: "list[int] | str | None" = None,
        apply_steering: bool = True,
        raw: bool = False,
    ) -> dict[str, Any]:
        """The loom-anchored replay: rebuild the producing forward and
        return the FINISHED ``scope="replay"`` measurement envelope
        (``{"measurements": …}``).  A family that cannot honor a knob
        rejects it with ``ValueError`` rather than ignoring it."""
        ...


__all__ = ["Instrument", "InstrumentRun"]
