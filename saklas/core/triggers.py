"""Token-window triggers for selective steering activation.

A ``Trigger`` gates when a steering contribution fires during generation:
only during prompt prefill, only during response tokens, only within
thinking sections, only within the first/last N generated tokens, etc.
Attach one to a steering entry (``Steering.alphas["name"] = (alpha,
trigger)``) or at the per-``Steering`` default (``Steering(..., trigger=
Trigger.AFTER_THINKING)``).

Triggers compose at the hook layer: entries sharing a trigger pre-compose
into one tensor, distinct triggers get distinct tensors, and the hook sums
only those groups whose trigger is active at the current generation step.
The default ``Trigger.BOTH`` (steer every prompt + response + thinking
token) short-circuits the per-step ``.active()`` check.

``TriggerContext`` is a tiny mutable struct shared between the generation
loop and the hooks — the loop mutates it at lifecycle boundaries (prefill
→ decode, thinking transitions, per-step counter) and the hooks read it.
"""

from __future__ import annotations

from dataclasses import dataclass


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

    def active(self, ctx: "TriggerContext") -> bool:
        """Return True iff this trigger should fire at the current step.

        Called once per layer per forward pass (via the hook) when the
        trigger is not ``Trigger.BOTH``. Hot-path discipline: pure Python
        attribute reads and int comparisons, no allocation.
        """
        if ctx.is_prefill:
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
        return True

    @classmethod
    def first(cls, n: int) -> "Trigger":
        """Apply only to the first ``n`` generated tokens (prefill off)."""
        return cls(prompt=False, first_n=n)

    @classmethod
    def after(cls, n: int) -> "Trigger":
        """Apply only after the first ``n`` generated tokens (prefill off)."""
        return cls(prompt=False, after_n=n)


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

    Fields are plain Python primitives so hook reads are zero-allocation
    attribute access.
    """

    is_prefill: bool = False
    thinking: bool = False
    gen_step: int = 0

    def reset(self) -> None:
        self.is_prefill = False
        self.thinking = False
        self.gen_step = 0


# Re-exported to keep internal imports tidy.
__all__ = ["Trigger", "TriggerContext"]
