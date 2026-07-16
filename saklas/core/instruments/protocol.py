"""The instrument protocol — one contract over the three read families.

An **Instrument** is the persistent, session-lifetime object for one read
family: it owns the attached-probe registry, validates gate references
against the channels the family can actually produce, declares capture
demand for a generation (``plan``), and binds an immutable per-generation
**InstrumentRun**.

An **InstrumentRun** owns everything generation-scoped: the immutable
source/spec binding, step stashes, warm state (curved feet live in the
geometry family), and per-generation active flags.  ``observe`` may be
called more than once for the same step — first by the gating score
callback (before the token tap), later by the display step — and MUST
memoize by ``step_id`` so the second caller reuses the first result's
computation (the one-forward-one-computation contract the per-family
stashes implemented implicitly).

Division of labor the protocol deliberately does NOT own:

* **Capture planning** — instruments declare demand
  (:class:`~saklas.core.instruments.types.InstrumentPlan`); the session
  planner unions demands and picks physical retention.  The
  ``INCREMENTAL -> set_tail_with_sink`` upgrade is cross-instrument
  resource sharing and stays session-side.
* **Authored-prefill orchestration** — token matching, the ``j-1``
  producer-position semantics, ordering, and loom persistence stay in the
  session; instruments only ``observe`` the hidden rows handed to them.
* **The wire** — runs return readings; serialization (the phase-2
  measurement envelope, today's compat channels) is the payload layer's
  job.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, Union, runtime_checkable

import torch

from saklas.core.instruments.types import (
    GateRef,
    InstrumentBinding,
    InstrumentPlan,
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

    def plan(self, request: ReadRequest) -> InstrumentPlan: ...

    def bind(self, plan: InstrumentPlan) -> "InstrumentRun": ...


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
        """Readings for every attached probe at this step.  Memoized by
        ``step_id`` — gate and display callers share one computation."""
        ...

    def gate_scalars(
        self,
        step_id: int,
        hidden: Mapping[int, torch.Tensor],
        gate_keys: frozenset[str],
    ) -> dict[str, float]:
        """The gate channels' scalar values for this step (the subset the
        active gates reference), keyed by verbatim scalar key."""
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
