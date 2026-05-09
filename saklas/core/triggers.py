"""Token-window triggers for selective steering activation.

A ``Trigger`` gates when a steering contribution fires during generation:
only during prompt prefill, only during response tokens, only within
thinking sections, only within the first/last N generated tokens, only
when a probe reading crosses a threshold, etc.  Attach one to a steering
entry (``Steering.alphas["name"] = (alpha, trigger)``) or at the
per-``Steering`` default (``Steering(..., trigger=Trigger.AFTER_THINKING)``).

Triggers compose at the hook layer: entries sharing a trigger pre-compose
into one tensor, distinct triggers get distinct tensors, and the hook sums
only those groups whose trigger is active at the current generation step.
The default ``Trigger.BOTH`` (steer every prompt + response + thinking
token) short-circuits the per-step ``.active()`` check.

**Probe gates (v2.1)**.  ``Trigger.gate`` carries an optional
:class:`ProbeGate` that consults the live monitor reading for a named
probe at the current generation step — e.g. "fire calm-steering only
when the angry probe reads above 0.4".  Probe scores are populated
into :class:`TriggerContext` per-step by the generation loop's score
callback (see ``saklas.core.generation.generate_steered``); when a
gate is present and the score isn't yet available (prefill, or
``probe_scores`` empty for any reason) the trigger reports inactive
— prefill gating on probe scores is meaningless.

``TriggerContext`` is a tiny mutable struct shared between the generation
loop and the hooks — the loop mutates it at lifecycle boundaries (prefill
→ decode, thinking transitions, per-step counter) and the hooks read it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal


# Comparison operators a :class:`ProbeGate` accepts.  Stored as the raw
# string so format/parse round-trip cleanly through the steering grammar.
ProbeGateOp = Literal[">", ">=", "<", "<="]


@dataclass(frozen=True)
class ProbeGate:
    """Threshold gate on a live probe reading.

    Evaluated at hook fire time against
    ``TriggerContext.probe_scores[probe]``.  Fires iff the score
    satisfies ``score <op> threshold``.  When the score is missing —
    during prompt prefill (no monitor reading yet) or before the first
    decode step — the gate reports inactive; never raises.

    ``probe`` is the canonical concept name as it appears in the
    session's monitor (e.g. ``"angry.calm"``, ``"deer.wolf"``).  No
    namespace prefix in v2.1 — saklas's monitor stores probes by
    canonical name and the grammar matches that.

    Frozen so two ``Trigger`` instances with identical gates compare
    equal under dataclass equality, which is what the hook's
    per-trigger grouping in :meth:`SteeringHook.recompose` relies on.
    """

    probe: str
    op: ProbeGateOp
    threshold: float

    def evaluate(self, score: float) -> bool:
        """Return True iff ``score <op> self.threshold``."""
        op = self.op
        t = self.threshold
        if op == ">":
            return score > t
        if op == ">=":
            return score >= t
        if op == "<":
            return score < t
        return score <= t  # "<="


@dataclass(frozen=True)
class Trigger:
    """When a steering contribution should fire during generation.

    prompt:    apply during prompt prefill (single forward over input_ids).
    generated: apply during decode (one forward per generated token).
    thinking:  apply when the current decode step is emitting thinking
               content (``ThinkingState.THINKING``). Ignored during prefill.
    response:  apply when the current decode step is in the non-thinking
               response region (all post-thinking states). Ignored during
               prefill. A preamble/channel-delimiter step is treated as
               response-like — ``thinking`` is False unless actively inside
               the thinking section.
    first_n:   if set, apply only when ``gen_step < first_n``. Counts
               generated token positions from 0; applies to decode only.
    after_n:   if set, apply only when ``gen_step >= after_n``. Same
               semantics as ``first_n``.

    Presets are defined as class-level constants at the bottom of this
    module (``Trigger.BOTH``, ``Trigger.GENERATED_ONLY``, etc.). Build
    custom triggers by constructing the dataclass directly.
    """

    prompt: bool = True
    generated: bool = True
    thinking: bool = True
    response: bool = True
    first_n: int | None = None
    after_n: int | None = None
    gate: ProbeGate | None = None

    # Preset slots — assigned at module load below the class.  ClassVar
    # keeps the dataclass machinery from treating them as instance fields.
    BOTH: ClassVar["Trigger"]
    GENERATED_ONLY: ClassVar["Trigger"]
    PROMPT_ONLY: ClassVar["Trigger"]
    AFTER_THINKING: ClassVar["Trigger"]
    THINKING_ONLY: ClassVar["Trigger"]

    def active(self, ctx: "TriggerContext") -> bool:
        """Return True iff this trigger should fire at the current step.

        Called once per layer per forward pass (via the hook) when the
        trigger is not ``Trigger.BOTH``. Hot-path discipline: pure Python
        attribute reads and int comparisons, no allocation.

        Probe gate (v2.1): when ``self.gate`` is set, the trigger fires
        only if the named probe's last-step reading satisfies the
        threshold.  During prefill the gate reports inactive (no probe
        reading yet); on later steps a missing score (probe not
        registered, or ``probe_scores`` empty for any reason) likewise
        reports inactive — gating on a probe that doesn't exist should
        never inject.
        """
        if ctx.is_prefill:
            # Probe gates can't fire during prefill — there's no
            # post-forward score to read against (capture happens *during*
            # the forward, score happens *after*).  Without a gate, fall
            # through to the prompt slot like before.
            if self.gate is not None:
                return False
            return self.prompt
        if not self.generated:
            return False
        if ctx.thinking:
            if not self.thinking:
                return False
        else:
            if not self.response:
                return False
        fn = self.first_n
        if fn is not None and ctx.gen_step >= fn:
            return False
        an = self.after_n
        if an is not None and ctx.gen_step < an:
            return False
        gate = self.gate
        if gate is not None:
            score = ctx.probe_scores.get(gate.probe)
            if score is None:
                return False
            if not gate.evaluate(score):
                return False
        return True

    @classmethod
    def first(cls, n: int) -> "Trigger":
        """Apply only to the first ``n`` generated tokens (prefill off)."""
        return cls(prompt=False, first_n=n)

    @classmethod
    def after(cls, n: int) -> "Trigger":
        """Apply only after the first ``n`` generated tokens (prefill off)."""
        return cls(prompt=False, after_n=n)

    @classmethod
    def when(
        cls, probe: str, op: ProbeGateOp, threshold: float,
    ) -> "Trigger":
        """Probe-gated trigger: fire only on decode steps where the named
        probe's last reading satisfies ``score <op> threshold``.

        The constructed trigger has ``prompt=False`` (probe scores
        aren't available during prefill), ``thinking=True`` and
        ``response=True`` so the gate can fire across the whole
        generation window.  Compose with explicit field overrides to
        narrow further, e.g. ``replace(Trigger.when("angry.calm", ">",
        0.4), thinking=False)`` to fire only on response tokens.
        """
        return cls(
            prompt=False,
            gate=ProbeGate(probe=probe, op=op, threshold=float(threshold)),
        )


# Preset constants. Assigned after the class so they themselves are
# Trigger instances — users pass them directly: ``Trigger.AFTER_THINKING``.
#
# ``BOTH`` is the default (steer every prompt + response + thinking token).
# Hook hot path short-circuits on ``is BOTH`` to skip the per-step
# ``.active()`` call, so default generations pay zero added cost from the
# trigger machinery.
Trigger.BOTH = Trigger()
Trigger.GENERATED_ONLY = Trigger(prompt=False)
Trigger.PROMPT_ONLY = Trigger(generated=False)
Trigger.AFTER_THINKING = Trigger(prompt=False, thinking=False)
Trigger.THINKING_ONLY = Trigger(prompt=False, response=False)


@dataclass
class TriggerContext:
    """Mutable state read by hooks to decide whether to apply each trigger.

    One instance per generation call, owned by the ``SteeringManager`` and
    mutated by ``generate_steered`` at lifecycle boundaries:

    * ``is_prefill`` — True during the first forward pass (the one that
      processes the full prompt); False thereafter.
    * ``thinking`` — True while the decode step is inside a thinking
      section (``ThinkingState.THINKING``); False in prompt, preamble,
      response, and post-termination states.
    * ``gen_step`` — monotonically increasing index of the generated
      token position the current forward is about to produce. 0 during
      prefill and during the first decode; incremented after the sample.
    * ``probe_scores`` — last-step monitor scores keyed by probe name
      (canonical form, e.g. ``"angry.calm"``).  Populated by the
      generation loop's score callback when at least one steering
      :class:`Trigger` has a probe gate; empty otherwise (the
      no-allocation default in the no-gate path).  Read by
      :meth:`Trigger.active` for gate evaluation.

    Hot-path discipline: the only allocation introduced for probe
    gates is the dict update on each gated step, and a single
    ``dict.get`` per gated trigger per layer per step at hook fire.
    Sessions without any gated trigger pay zero — the score callback
    never fires.
    """

    is_prefill: bool = False
    thinking: bool = False
    gen_step: int = 0
    probe_scores: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.is_prefill = False
        self.thinking = False
        self.gen_step = 0
        self.probe_scores = {}


# Re-exported to keep internal imports tidy.
__all__ = ["ProbeGate", "ProbeGateOp", "Trigger", "TriggerContext"]
